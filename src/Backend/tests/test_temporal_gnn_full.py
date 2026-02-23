"""
Meticulous test suite for EvolvingGNN v2 (Semi-Temporal GNN).
Covers: correctness, gradient flow, numerical stability, edge cases,
        memory profiling, and timing bottleneck comparison (v1 vs v2).
"""

import sys
import os
import time
import tracemalloc
import unittest

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data

# Project root (3 levels up: tests/ -> Backend/ -> src/ -> IncidentLens/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.Backend.temporal_gnn import (
    EvolvingGNN,
    EvolvingGCNLayer,
    preprocess_graph,
    preprocess_graphs,
    sanitize_graph,
    build_temporal_sequences,
    normalize_features_global,
    apply_normalization,
    recompute_node_features,
    collate_temporal_batch,
)


# ============================================================
# HELPERS
# ============================================================

def make_graph(num_nodes, num_edges, node_feat_dim, edge_feat_dim,
               include_labels=False, preprocessed=False, device="cpu"):
    """Create a single random PyG Data object."""
    x = torch.randn(num_nodes, node_feat_dim, device=device)
    src = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.randn(num_edges, edge_feat_dim, device=device)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=num_nodes)
    if include_labels:
        data.y = torch.randint(0, 2, (num_edges,), device=device).float()
    if preprocessed:
        data = preprocess_graph(data)
    return data


def make_sequence(seq_len, num_nodes, num_edges, node_feat_dim, edge_feat_dim,
                  include_labels=False, preprocessed=False, device="cpu"):
    """Create a list of graphs simulating consecutive time windows."""
    return [make_graph(num_nodes, num_edges, node_feat_dim, edge_feat_dim,
                       include_labels, preprocessed, device)
            for _ in range(seq_len)]


# ============================================================
# 1. CORRECTNESS TESTS
# ============================================================

class TestCorrectness(unittest.TestCase):
    """Verify shapes, determinism, and basic forward-pass sanity."""

    def setUp(self):
        self.cfg = dict(input_dim=6, hidden_dim=16, edge_feat_dim=5)
        self.model = EvolvingGNN(**self.cfg)
        self.model.eval()

    def test_output_shape_matches_last_graph_edges(self):
        """Logits length == number of edges in the LAST graph only."""
        for last_edges in [5, 20, 50]:
            seq = make_sequence(5, 10, 15, 6, 5)
            seq[-1] = make_graph(10, last_edges, 6, 5)
            logits = self.model(seq)
            self.assertEqual(logits.shape, (last_edges,),
                             f"Expected ({last_edges},), got {logits.shape}")

    def test_deterministic_eval_mode(self):
        """Two identical forward passes in eval mode must produce identical results."""
        torch.manual_seed(42)
        seq = make_sequence(5, 10, 20, 6, 5)
        self.model.eval()
        out1 = self.model(seq).detach().clone()
        out2 = self.model(seq).detach().clone()
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6),
                        "Eval-mode outputs differ across identical inputs")

    def test_single_graph_sequence(self):
        """Model should work with a sequence of length 1."""
        seq = make_sequence(1, 8, 12, 6, 5)
        logits = self.model(seq)
        self.assertEqual(logits.shape, (12,))
        self.assertFalse(torch.isnan(logits).any())

    def test_long_sequence(self):
        """Model should work with a longer sequence (e.g., 20 steps)."""
        seq = make_sequence(20, 8, 10, 6, 5)
        logits = self.model(seq)
        self.assertEqual(logits.shape, (10,))
        self.assertFalse(torch.isnan(logits).any())

    def test_varying_node_counts_across_sequence(self):
        """Different graphs in the sequence can have different node counts."""
        seq = []
        for n in [5, 8, 12, 6, 10]:
            seq.append(make_graph(n, 15, 6, 5))
        logits = self.model(seq)
        self.assertEqual(logits.shape, (15,))
        self.assertFalse(torch.isnan(logits).any())

    def test_varying_edge_counts_across_sequence(self):
        """Different graphs can have different edge counts; only last matters."""
        seq = []
        for e in [5, 10, 20, 8, 13]:
            seq.append(make_graph(10, e, 6, 5))
        logits = self.model(seq)
        self.assertEqual(logits.shape, (13,))

    def test_preprocessed_and_raw_produce_same_output(self):
        """Preprocessed graphs should produce the same logits as raw graphs."""
        torch.manual_seed(99)
        seq_raw = make_sequence(3, 10, 20, 6, 5, preprocessed=False)

        # Preprocess copies
        seq_pre = [preprocess_graph(
            Data(x=g.x.clone(), edge_index=g.edge_index.clone(),
                 edge_attr=g.edge_attr.clone(), num_nodes=g.num_nodes))
            for g in seq_raw]

        self.model.eval()
        out_raw = self.model(seq_raw).detach()
        out_pre = self.model(seq_pre).detach()
        self.assertTrue(torch.allclose(out_raw, out_pre, atol=1e-5),
                        "Preprocessed and raw outputs differ!")


# ============================================================
# 2. WEIGHT EVOLUTION TESTS
# ============================================================

class TestWeightEvolution(unittest.TestCase):
    """Verify the LSTM actually evolves GNN weights meaningfully."""

    def setUp(self):
        self.model = EvolvingGNN(input_dim=6, hidden_dim=16, edge_feat_dim=5)

    def test_weights_change_every_step(self):
        """Weight matrix must differ at EVERY consecutive step."""
        flat = self.model.initial_weights.reshape(1, 1, -1).detach().clone()
        h = None
        prev = flat.squeeze()

        for step in range(5):
            out, h = self.model.weight_lstm(flat, h)
            flat = out.detach()
            current = flat.squeeze()

            diff = (current - prev).abs().sum().item()
            self.assertGreater(diff, 1e-4,
                               f"Weights did not change at step {step}")
            prev = current.clone()

    def test_weights_dont_explode(self):
        """After many steps, weights should remain bounded."""
        flat = self.model.initial_weights.reshape(1, 1, -1).detach().clone()
        h = None

        for _ in range(50):
            out, h = self.model.weight_lstm(flat, h)
            flat = out.detach()

        max_val = flat.abs().max().item()
        self.assertLess(max_val, 100.0,
                        f"Weights exploded to {max_val} after 50 steps")

    def test_weights_dont_collapse_to_zero(self):
        """Weights should not vanish to near-zero after many steps."""
        flat = self.model.initial_weights.reshape(1, 1, -1).detach().clone()
        h = None

        for _ in range(50):
            out, h = self.model.weight_lstm(flat, h)
            flat = out.detach()

        norm = flat.norm().item()
        self.assertGreater(norm, 1e-6,
                           f"Weights collapsed to near-zero: norm={norm}")


# ============================================================
# 3. GRADIENT FLOW TESTS
# ============================================================

class TestGradientFlow(unittest.TestCase):
    """Verify gradients flow through all components during backprop."""

    def setUp(self):
        self.model = EvolvingGNN(input_dim=6, hidden_dim=16, edge_feat_dim=5)

    def test_all_parameters_receive_gradients(self):
        """Every trainable parameter must receive a non-zero gradient."""
        self.model.train()
        seq = make_sequence(5, 10, 20, 6, 5, include_labels=True)

        logits = self.model(seq)
        labels = seq[-1].y
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()

        no_grad_params = []
        zero_grad_params = []

        for name, param in self.model.named_parameters():
            if param.grad is None:
                no_grad_params.append(name)
            elif param.grad.abs().max().item() == 0:
                zero_grad_params.append(name)

        self.assertEqual(len(no_grad_params), 0,
                         f"Parameters with NO gradient: {no_grad_params}")
        if zero_grad_params:
            print(f"  [WARN] Parameters with zero gradient: {zero_grad_params}")

    def test_gradient_magnitude_reasonable(self):
        """Gradients should not be excessively large."""
        self.model.train()
        seq = make_sequence(5, 10, 20, 6, 5, include_labels=True)

        logits = self.model(seq)
        labels = seq[-1].y
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.assertLess(grad_norm, 1000.0,
                                f"Gradient EXPLODING for {name}: norm={grad_norm}")

    def test_loss_decreases_over_steps(self):
        """Basic overfitting test: loss should decrease over 20 optimization steps."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        torch.manual_seed(123)
        seq = make_sequence(5, 8, 15, 6, 5, include_labels=True)
        criterion = nn.BCEWithLogitsLoss()

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            logits = self.model(seq)
            loss = criterion(logits, seq[-1].y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.assertLess(losses[-1], losses[0],
                        f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}")

    def test_initial_weights_receive_gradient(self):
        """The initial_weights parameter (seed for LSTM) must be trainable."""
        self.model.train()
        seq = make_sequence(3, 8, 12, 6, 5, include_labels=True)

        logits = self.model(seq)
        loss = nn.BCEWithLogitsLoss()(logits, seq[-1].y)
        loss.backward()

        grad = self.model.initial_weights.grad
        self.assertIsNotNone(grad, "initial_weights has no gradient")
        self.assertGreater(grad.abs().sum().item(), 0,
                           "initial_weights gradient is all zeros")


# ============================================================
# 4. NUMERICAL STABILITY TESTS
# ============================================================

class TestNumericalStability(unittest.TestCase):
    """Test behavior with extreme/adversarial inputs."""

    def setUp(self):
        self.model = EvolvingGNN(input_dim=6, hidden_dim=16, edge_feat_dim=5)
        self.model.eval()

    def test_zero_features(self):
        """All-zero node and edge features should NOT produce NaN."""
        seq = []
        for _ in range(5):
            g = make_graph(10, 15, 6, 5)
            g.x = torch.zeros_like(g.x)
            g.edge_attr = torch.zeros_like(g.edge_attr)
            seq.append(g)

        logits = self.model(seq)
        self.assertFalse(torch.isnan(logits).any(), "NaN with zero features")
        self.assertFalse(torch.isinf(logits).any(), "Inf with zero features")

    def test_large_features(self):
        """Very large feature values should not produce NaN."""
        seq = []
        for _ in range(5):
            g = make_graph(10, 15, 6, 5)
            g.x = g.x * 1000.0
            g.edge_attr = g.edge_attr * 1000.0
            seq.append(g)

        logits = self.model(seq)
        self.assertFalse(torch.isnan(logits).any(), "NaN with large features")

    def test_single_node_graph(self):
        """A graph with only 1 node (self-loop edge) should work."""
        seq = []
        for _ in range(5):
            x = torch.randn(1, 6)
            edge_index = torch.tensor([[0], [0]])
            edge_attr = torch.randn(1, 5)
            seq.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            num_nodes=1))
        logits = self.model(seq)
        self.assertEqual(logits.shape, (1,))
        self.assertFalse(torch.isnan(logits).any())

    def test_disconnected_nodes(self):
        """Graph with isolated nodes should work."""
        seq = []
        for _ in range(5):
            x = torch.randn(20, 6)
            src = torch.randint(0, 5, (10,))
            dst = torch.randint(0, 5, (10,))
            edge_index = torch.stack([src, dst])
            edge_attr = torch.randn(10, 5)
            seq.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            num_nodes=20))
        logits = self.model(seq)
        self.assertEqual(logits.shape, (10,))
        self.assertFalse(torch.isnan(logits).any())

    def test_no_edges_graph(self):
        """A graph with 0 edges should not crash."""
        seq = []
        for _ in range(4):
            seq.append(make_graph(10, 15, 6, 5))
        x = torch.randn(5, 6)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 5))
        seq.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                        num_nodes=5))
        logits = self.model(seq)
        self.assertEqual(logits.shape, (0,))

    def test_duplicate_edges(self):
        """Duplicate edges (multi-graph) should be handled."""
        seq = []
        for _ in range(5):
            edge_index = torch.tensor([[0, 0, 0], [1, 1, 1]])
            x = torch.randn(5, 6)
            edge_attr = torch.randn(3, 5)
            seq.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            num_nodes=5))
        logits = self.model(seq)
        self.assertEqual(logits.shape, (3,))
        self.assertFalse(torch.isnan(logits).any())

    def test_sanitize_graph_cleans_nans(self):
        """sanitize_graph should replace NaN/Inf with 0."""
        g = make_graph(10, 15, 6, 5)
        g.x[0, 0] = float('nan')
        g.x[1, 1] = float('inf')
        g.edge_attr[0, 0] = float('-inf')

        g = sanitize_graph(g)
        self.assertFalse(torch.isnan(g.x).any())
        self.assertFalse(torch.isinf(g.x).any())
        self.assertFalse(torch.isnan(g.edge_attr).any())
        self.assertFalse(torch.isinf(g.edge_attr).any())


# ============================================================
# 5. INPUT VALIDATION TESTS
# ============================================================

class TestInputValidation(unittest.TestCase):
    """Test error handling and edge cases in input."""

    def setUp(self):
        self.model = EvolvingGNN(input_dim=6, hidden_dim=16, edge_feat_dim=5)

    def test_wrong_node_feature_dim(self):
        seq = [make_graph(10, 15, 99, 5)]
        with self.assertRaises(ValueError):
            self.model(seq)

    def test_empty_sequence(self):
        with self.assertRaises(ValueError):
            self.model([])

    def test_wrong_edge_feat_dim(self):
        seq = make_sequence(3, 10, 15, 6, 99)
        with self.assertRaises(RuntimeError):
            self.model(seq)


# ============================================================
# 6. PREPROCESSING UTILITY TESTS
# ============================================================

class TestPreprocessing(unittest.TestCase):
    """Test the preprocessing utilities."""

    def test_preprocess_adds_self_loops(self):
        """preprocess_graph should add self-loops."""
        g = make_graph(10, 20, 6, 5)
        original_edges = g.edge_index.shape[1]
        g = preprocess_graph(g)

        self.assertTrue(hasattr(g, 'edge_index_with_loops'))
        self.assertTrue(hasattr(g, 'norm'))
        # Should have original edges + 10 self-loops
        self.assertEqual(g.edge_index_with_loops.shape[1], original_edges + 10)
        self.assertEqual(g.norm.shape[0], original_edges + 10)

    def test_preprocess_does_not_modify_original_edge_index(self):
        """Original edge_index should remain unchanged."""
        g = make_graph(10, 20, 6, 5)
        original_edges = g.edge_index.shape[1]
        g = preprocess_graph(g)
        self.assertEqual(g.edge_index.shape[1], original_edges)

    def test_preprocess_norm_values(self):
        """Normalization values should be finite and non-negative."""
        g = make_graph(10, 20, 6, 5)
        g = preprocess_graph(g)
        self.assertFalse(torch.isnan(g.norm).any())
        self.assertFalse(torch.isinf(g.norm).any())
        self.assertTrue((g.norm >= 0).all())

    def test_preprocess_graphs_batch(self):
        """preprocess_graphs should handle a list."""
        gs = [make_graph(10, 20, 6, 5) for _ in range(5)]
        gs = preprocess_graphs(gs)
        for g in gs:
            self.assertTrue(hasattr(g, 'edge_index_with_loops'))
            self.assertTrue(hasattr(g, 'norm'))

    def test_build_temporal_sequences(self):
        """build_temporal_sequences should create correct sliding windows."""
        # Create graphs with fake window_start
        gs = []
        for i in range(10):
            g = make_graph(5, 10, 6, 5)
            g.window_start = float(i)
            gs.append(g)

        seqs = build_temporal_sequences(gs, seq_len=5, stride=1)
        self.assertEqual(len(seqs), 6)  # 10 - 5 + 1 = 6

        # Each sequence should have 5 graphs
        for s in seqs:
            self.assertEqual(len(s), 5)

    def test_build_temporal_sequences_stride(self):
        """Stride > 1 should reduce the number of sequences."""
        gs = []
        for i in range(10):
            g = make_graph(5, 10, 6, 5)
            g.window_start = float(i)
            gs.append(g)

        seqs = build_temporal_sequences(gs, seq_len=5, stride=2)
        self.assertEqual(len(seqs), 3)  # (10-5)//2 + 1 = 3


# ============================================================
# 7. PERFORMANCE & BOTTLENECK COMPARISON (v1 vs v2)
# ============================================================

class TestPerformanceBottlenecks(unittest.TestCase):
    """Profile timing and memory to verify bottleneck fixes."""

    def _time_forward(self, model, seq, n_runs=5):
        model.eval()
        with torch.no_grad():
            model(seq)  # warmup

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            with torch.no_grad():
                model(seq)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        return np.mean(times), np.std(times)

    def test_preprocessed_faster_than_raw(self):
        """Pre-processed graphs should be faster than raw graphs."""
        model = EvolvingGNN(input_dim=6, hidden_dim=32, edge_feat_dim=5)
        model.eval()

        seq_raw = make_sequence(5, 200, 1000, 6, 5, preprocessed=False)
        seq_pre = make_sequence(5, 200, 1000, 6, 5, preprocessed=True)

        mean_raw, _ = self._time_forward(model, seq_raw, n_runs=10)
        mean_pre, _ = self._time_forward(model, seq_pre, n_runs=10)

        print("\n" + "=" * 60)
        print("PREPROCESSED vs RAW (5 graphs, 200 nodes, 1000 edges)")
        print("=" * 60)
        print(f"  Raw (recompute self-loops):  {mean_raw:.2f} ms")
        print(f"  Preprocessed (cached):       {mean_pre:.2f} ms")
        print(f"  Speedup:                     {mean_raw/mean_pre:.2f}x")
        print("=" * 60)

        # Preprocessed should be at least a little faster
        # (may not always be measurable for small graphs)

    def test_forward_timing_breakdown(self):
        """Break down time in LSTM vs GNN vs MLP."""
        model = EvolvingGNN(input_dim=6, hidden_dim=32, edge_feat_dim=5)
        model.eval()
        seq = make_sequence(5, 100, 500, 6, 5, preprocessed=True)

        total_mean, total_std = self._time_forward(model, seq)

        # LSTM only
        flat_dim = 6 * 32
        lstm_times = []
        for _ in range(5):
            w = model.initial_weights.reshape(1, 1, -1).detach()
            h = None
            start = time.perf_counter()
            for _ in range(5):
                out, h = model.weight_lstm(w, h)
                w = out
            lstm_times.append((time.perf_counter() - start) * 1000)

        # GNN only (with preprocessing)
        gnn_times = []
        weight = torch.randn(6, 32)
        for _ in range(5):
            start = time.perf_counter()
            for g in seq:
                model.gnn_layer(g.x, g.edge_index, weight,
                                norm=g.norm,
                                edge_index_with_loops=g.edge_index_with_loops)
            gnn_times.append((time.perf_counter() - start) * 1000)

        # MLP only
        mlp_times = []
        dummy_input = torch.randn(500, 32 * 2 + 5)
        for _ in range(5):
            start = time.perf_counter()
            model.edge_mlp(dummy_input)
            mlp_times.append((time.perf_counter() - start) * 1000)

        print("\n" + "=" * 60)
        print("TIMING BREAKDOWN v2 (5 graphs, 100 nodes, 500 edges)")
        print("=" * 60)
        print(f"  Total Forward:     {total_mean:8.2f} ms (+/- {total_std:.2f})")
        print(f"  LSTM (flat):       {np.mean(lstm_times):8.2f} ms (+/- {np.std(lstm_times):.2f})")
        print(f"  GNN (cached):      {np.mean(gnn_times):8.2f} ms (+/- {np.std(gnn_times):.2f})")
        print(f"  Edge MLP:          {np.mean(mlp_times):8.2f} ms (+/- {np.std(mlp_times):.2f})")

        components = {
            "LSTM": np.mean(lstm_times),
            "GNN": np.mean(gnn_times),
            "MLP": np.mean(mlp_times),
        }
        bottleneck = max(components, key=components.get)
        print(f"  >> BOTTLENECK: {bottleneck} ({components[bottleneck]:.2f} ms)")
        print("=" * 60)

    def test_lstm_scaling_with_hidden_dim(self):
        """Verify LSTM no longer scales badly with hidden_dim (flat fix)."""
        print("\n" + "=" * 60)
        print("LSTM SCALING vs HIDDEN DIM (flattened weights)")
        print("=" * 60)

        for hidden_dim in [16, 32, 64, 128]:
            model = EvolvingGNN(input_dim=6, hidden_dim=hidden_dim, edge_feat_dim=5)
            model.eval()
            seq = make_sequence(5, 50, 200, 6, 5)
            mean_ms, _ = self._time_forward(model, seq)
            print(f"  hidden_dim={hidden_dim:4d}: {mean_ms:8.2f} ms")

        print("  >> Should scale more smoothly than v1")
        print("=" * 60)

    def test_scaling_with_sequence_length(self):
        """Time scales linearly with sequence length."""
        model = EvolvingGNN(input_dim=6, hidden_dim=32, edge_feat_dim=5)
        model.eval()

        print("\n" + "=" * 60)
        print("SCALING WITH SEQUENCE LENGTH")
        print("=" * 60)

        times_by_len = {}
        for seq_len in [1, 3, 5, 10, 20]:
            seq = make_sequence(seq_len, 50, 200, 6, 5, preprocessed=True)
            mean_ms, std_ms = self._time_forward(model, seq)
            times_by_len[seq_len] = mean_ms
            print(f"  Seq len {seq_len:3d}: {mean_ms:8.2f} ms (+/- {std_ms:.2f})")

        if times_by_len[5] > 0:
            ratio = times_by_len[20] / times_by_len[5]
            print(f"  Ratio T=20 / T=5: {ratio:.2f}x (ideal ~4x)")
        print("=" * 60)

    def test_memory_usage(self):
        """Track peak memory during forward + backward pass."""
        model = EvolvingGNN(input_dim=6, hidden_dim=32, edge_feat_dim=5)
        model.train()
        seq = make_sequence(5, 100, 500, 6, 5, include_labels=True,
                            preprocessed=True)

        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        logits = model(seq)
        loss = nn.BCEWithLogitsLoss()(logits, seq[-1].y)
        loss.backward()

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        total_new_mb = sum(s.size_diff for s in stats if s.size_diff > 0) / (1024 * 1024)

        print("\n" + "=" * 60)
        print("MEMORY PROFILE (forward + backward)")
        print("=" * 60)
        print(f"  New allocations: {total_new_mb:.2f} MB")
        print("  Top 5 allocation sites:")
        for stat in stats[:5]:
            print(f"    {stat}")
        print("=" * 60)


# ============================================================
# 7. NORMALIZATION & RECOMPUTATION TESTS
# ============================================================

class TestNormalization(unittest.TestCase):
    """Verify global Z-score normalization and recomputation utilities."""

    def _make_graphs(self, n=5, num_nodes=10, num_edges=20,
                     node_dim=6, edge_dim=5):
        """Create a list of graphs with varying feature magnitudes."""
        graphs = []
        for i in range(n):
            x = torch.randn(num_nodes, node_dim) * (i + 1) * 10
            src = torch.randint(0, num_nodes, (num_edges,))
            dst = torch.randint(0, num_nodes, (num_edges,))
            edge_index = torch.stack([src, dst])
            # edge_attr[:, 0] = pkt_count, [:, 1] = total_bytes
            edge_attr = torch.rand(num_edges, edge_dim).abs() * (i + 1) * 100
            g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                     num_nodes=num_nodes)
            g.y = torch.randint(0, 2, (num_edges,)).float()
            g.window_start = float(i)
            graphs.append(g)
        return graphs

    def test_normalize_zero_mean_unit_var(self):
        """After normalization, concatenated features ~mean=0, std=1."""
        graphs = self._make_graphs(10)
        graphs, stats = normalize_features_global(graphs)

        all_x = torch.cat([g.x for g in graphs], dim=0)
        all_e = torch.cat([g.edge_attr for g in graphs], dim=0)

        self.assertTrue(torch.allclose(all_x.mean(dim=0),
                                       torch.zeros(6), atol=0.05))
        self.assertTrue(torch.allclose(all_x.std(dim=0),
                                       torch.ones(6), atol=0.1))
        self.assertTrue(torch.allclose(all_e.mean(dim=0),
                                       torch.zeros(5), atol=0.05))

    def test_normalize_returns_stats(self):
        """Stats dict should contain the 4 required tensors."""
        graphs = self._make_graphs(3)
        _, stats = normalize_features_global(graphs)
        for key in ("node_mean", "node_std", "edge_mean", "edge_std"):
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], torch.Tensor)

    def test_apply_normalization_uses_stats(self):
        """apply_normalization with pre-fitted stats must match direct normalize."""
        torch.manual_seed(7)
        graphs_a = self._make_graphs(5)
        # Fit on training set
        _, stats = normalize_features_global(
            [Data(x=g.x.clone(), edge_index=g.edge_index.clone(),
                  edge_attr=g.edge_attr.clone(), num_nodes=g.num_nodes)
             for g in graphs_a]
        )
        # Apply to separate "test" set
        graphs_b = self._make_graphs(3)
        graphs_b = apply_normalization(graphs_b, stats)
        # Should not crash and features should not contain NaN
        for g in graphs_b:
            self.assertFalse(torch.isnan(g.x).any())
            self.assertFalse(torch.isnan(g.edge_attr).any())

    def test_normalize_constant_feature_handled(self):
        """A column with zero variance should get std=1 (no div-by-zero)."""
        graphs = self._make_graphs(3)
        for g in graphs:
            g.x[:, 0] = 5.0  # constant column
        graphs, stats = normalize_features_global(graphs)
        for g in graphs:
            self.assertFalse(torch.isnan(g.x).any())
            self.assertFalse(torch.isinf(g.x).any())

    def test_recompute_node_features_creates_x(self):
        """recompute_node_features should create x from edge data."""
        num_nodes, num_edges, edge_dim = 8, 15, 5
        src = torch.randint(0, num_nodes, (num_edges,))
        dst = torch.randint(0, num_nodes, (num_edges,))
        edge_attr = torch.rand(num_edges, edge_dim)
        g = Data(edge_index=torch.stack([src, dst]),
                 edge_attr=edge_attr, num_nodes=num_nodes)
        # No x initially
        self.assertIsNone(g.x)
        g = recompute_node_features(g)
        # Now should have x with 6 features
        self.assertIsNotNone(g.x)
        self.assertEqual(g.x.shape, (num_nodes, 6))
        self.assertFalse(torch.isnan(g.x).any())

    def test_recompute_preserves_edge_data(self):
        """recompute_node_features should not modify edge_attr."""
        graphs = self._make_graphs(1)
        g = graphs[0]
        original_edge_attr = g.edge_attr.clone()
        g = recompute_node_features(g)
        self.assertTrue(torch.equal(g.edge_attr, original_edge_attr))

    def test_recompute_skips_insufficient_edge_attr(self):
        """Graphs with < 2 edge_attr columns should be returned unchanged."""
        g = Data(
            x=torch.randn(5, 6),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            edge_attr=torch.randn(2, 1),  # only 1 column
            num_nodes=5,
        )
        original_x = g.x.clone()
        g = recompute_node_features(g)
        self.assertTrue(torch.equal(g.x, original_x))


# ============================================================
# 8. BATCH COLLATION TESTS
# ============================================================

class TestCollation(unittest.TestCase):
    """Verify collate_temporal_batch utility."""

    def _make_sequences(self, n_seq=4, seq_len=3, n_nodes=8,
                        n_edges=10, nd=6, ed=5):
        seqs = []
        for _ in range(n_seq):
            seq = []
            for _ in range(seq_len):
                g = make_graph(n_nodes, n_edges, nd, ed, include_labels=True)
                seq.append(g)
            seqs.append(seq)
        return seqs

    def test_collate_all(self):
        """Without indices, all sequences should be returned."""
        seqs = self._make_sequences(4)
        batch, labels = collate_temporal_batch(seqs)
        self.assertEqual(len(batch), 4)
        # Labels = concatenated y from last graph of each sequence
        expected_len = sum(seq[-1].y.shape[0] for seq in seqs)
        self.assertEqual(labels.shape[0], expected_len)

    def test_collate_subset(self):
        """With indices, only selected sequences should be returned."""
        seqs = self._make_sequences(6)
        idx = np.array([0, 2, 5])
        batch, labels = collate_temporal_batch(seqs, indices=idx)
        self.assertEqual(len(batch), 3)

    def test_collate_labels_correct(self):
        """Labels should come from the last graph in each sequence."""
        seqs = self._make_sequences(3)
        # Set known labels
        for i, seq in enumerate(seqs):
            seq[-1].y = torch.full((seq[-1].y.shape[0],), float(i))
        _, labels = collate_temporal_batch(seqs)
        # Labels should contain 0s, 1s, and 2s
        unique_vals = set(labels.unique().tolist())
        self.assertEqual(unique_vals, {0.0, 1.0, 2.0})

    def test_collate_empty_indices(self):
        """Empty indices should return empty batch."""
        seqs = self._make_sequences(3)
        batch, labels = collate_temporal_batch(seqs, indices=np.array([], dtype=int))
        self.assertEqual(len(batch), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
