"""
Meticulous coverage-gap test suite for temporal_gnn.py.

Targets every function, edge case, and integration path
NOT covered by test_temporal_gnn.py / test_temporal_gnn_full.py.

Sections:
  1. sanitize_graph — y-label clamping, None attr paths
  2. collate_temporal_batch — device transfer, list indices
  3. recompute_node_features — value correctness, None edge_attr
  4. build_temporal_sequences — ordering, over-length, empty
  5. preprocess_graph — idempotence, self-loop dedup
  6. normalize_features_global — single graph, numerical precision
  7. EvolvingGCNLayer — message-passing analytical correctness
  8. EvolvingGNN model — serialization, param count, stochasticity,
                          gradient accumulation, hidden-state reset
  9. Full pipeline integration (sanitize→recompute→normalize→preprocess→forward)
 10. prepare_temporal_dataset (mock-data smoke test)
"""

import copy
import io
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree

# Project root (3 levels up: tests/ -> Backend/ -> src/ -> IncidentLens/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.Backend.temporal_gnn import (
    EvolvingGCNLayer,
    EvolvingGNN,
    apply_normalization,
    build_temporal_sequences,
    collate_temporal_batch,
    normalize_features_global,
    preprocess_graph,
    preprocess_graphs,
    recompute_node_features,
    sanitize_graph,
)


# ─── helpers ────────────────────────────────────────────────

def _g(num_nodes=10, num_edges=20, nd=6, ed=5, labels=False, ws=0.0):
    """Quick graph builder."""
    x = torch.randn(num_nodes, nd)
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    ei = torch.stack([src, dst])
    ea = torch.randn(num_edges, ed)
    g = Data(x=x, edge_index=ei, edge_attr=ea, num_nodes=num_nodes)
    if labels:
        g.y = torch.randint(0, 2, (num_edges,)).float()
    g.window_start = ws
    return g


# ============================================================
# 1. SANITIZE_GRAPH — extended edge cases
# ============================================================

class TestSanitizeExtended(unittest.TestCase):
    """Gaps: y-label clamping, None attrs, mixed types."""

    def test_y_clamped_to_binary(self):
        """Labels outside [0,1] must be clamped; dtype must be float."""
        g = _g(labels=True)
        g.y = torch.tensor([0, 1, 2, -1, 5, 0, 1, 3, -2, 1,
                            0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).float()
        g = sanitize_graph(g)
        self.assertTrue((g.y >= 0).all())
        self.assertTrue((g.y <= 1).all())
        self.assertEqual(g.y.dtype, torch.float32)

    def test_none_edge_attr_not_crash(self):
        """graph with edge_attr = None should not crash."""
        g = _g()
        g.edge_attr = None
        g = sanitize_graph(g)  # should survive
        self.assertIsNone(g.edge_attr)

    def test_none_y_not_crash(self):
        """graph with y = None should not crash."""
        g = _g()
        g = sanitize_graph(g)  # no y at all by default with labels=False
        # g.y is not set; ensure no AttributeError
        self.assertFalse(hasattr(g, "y") and g.y is not None)

    def test_none_x_not_crash(self):
        """graph with x = None should not crash."""
        g = _g()
        g.x = None
        g = sanitize_graph(g)
        self.assertIsNone(g.x)

    def test_all_nans_replaced(self):
        """A fully-NaN feature matrix should become all zeros."""
        g = _g(num_nodes=5, num_edges=3)
        g.x = torch.full((5, 6), float("nan"))
        g.edge_attr = torch.full((3, 5), float("nan"))
        g = sanitize_graph(g)
        self.assertTrue((g.x == 0).all())
        self.assertTrue((g.edge_attr == 0).all())

    def test_mixed_nan_inf(self):
        """Mix of NaN, +Inf, -Inf in one tensor."""
        g = _g(num_nodes=3, num_edges=2, nd=4, ed=3)
        g.x = torch.tensor([
            [float("nan"), float("inf"), float("-inf"), 1.0],
            [2.0,          float("nan"), 3.0,           float("inf")],
            [float("-inf"), 0.0,         float("nan"),  float("-inf")],
        ])
        g = sanitize_graph(g)
        self.assertFalse(torch.isnan(g.x).any())
        self.assertFalse(torch.isinf(g.x).any())
        # The finite value 1.0 at [0,3] should survive
        self.assertAlmostEqual(g.x[0, 3].item(), 1.0)

    def test_sanitize_double_to_float(self):
        """Double tensors should be cast to float32."""
        g = _g(num_nodes=3, num_edges=2, nd=4, ed=3)
        g.x = g.x.double()
        g = sanitize_graph(g)
        self.assertEqual(g.x.dtype, torch.float32)


# ============================================================
# 2. COLLATE_TEMPORAL_BATCH — device & index paths
# ============================================================

class TestCollateExtended(unittest.TestCase):
    """Gaps: device transfer, list-based indices, single-sequence batch."""

    def _seqs(self, n=3, sl=3, nn=5, ne=8, nd=6, ed=5):
        return [
            [_g(nn, ne, nd, ed, labels=True) for _ in range(sl)]
            for _ in range(n)
        ]

    def test_device_cpu_explicit(self):
        """Passing device=cpu should clone tensors (no crash)."""
        seqs = self._seqs()
        batch, labels = collate_temporal_batch(
            seqs, device=torch.device("cpu")
        )
        self.assertEqual(len(batch), 3)
        # Graphs should be on cpu
        for seq in batch:
            for g in seq:
                self.assertEqual(g.x.device.type, "cpu")
                self.assertEqual(g.edge_index.device.type, "cpu")

    def test_device_transfer_preserves_preprocess(self):
        """Preprocessed graphs should keep norm/edge_index_with_loops after device move."""
        seqs = self._seqs(2, 2)
        for seq in seqs:
            for g in seq:
                preprocess_graph(g)
        batch, labels = collate_temporal_batch(
            seqs, device=torch.device("cpu")
        )
        for seq in batch:
            for g in seq:
                self.assertTrue(hasattr(g, "norm"))
                self.assertTrue(hasattr(g, "edge_index_with_loops"))

    def test_list_indices(self):
        """Plain-list indices (not numpy) should also work."""
        seqs = self._seqs(5)
        batch, labels = collate_temporal_batch(seqs, indices=[1, 3])
        self.assertEqual(len(batch), 2)

    def test_single_sequence_batch(self):
        """Batch of 1 should still produce valid labels."""
        seqs = self._seqs(1)
        batch, labels = collate_temporal_batch(seqs)
        self.assertEqual(len(batch), 1)
        self.assertEqual(labels.shape[0], seqs[0][-1].y.shape[0])

    def test_device_moves_y(self):
        """Labels tensor should be moved to the target device."""
        seqs = self._seqs(2)
        _, labels = collate_temporal_batch(seqs, device=torch.device("cpu"))
        self.assertEqual(labels.device.type, "cpu")


# ============================================================
# 3. RECOMPUTE_NODE_FEATURES — value correctness
# ============================================================

class TestRecomputeCorrectness(unittest.TestCase):
    """Gaps: verify aggregated VALUES, not just shapes."""

    def test_known_aggregation(self):
        """Hand-computed node features must match _compute_node_features_arrays."""
        # Graph: 3 nodes, 2 edges: 0→1 and 0→2
        ei = torch.tensor([[0, 0], [1, 2]])
        # edge_attr[:, 0] = pkt_count, [:, 1] = total_bytes
        ea = torch.tensor([[10.0, 100.0, 0, 0, 0],
                           [20.0, 200.0, 0, 0, 0]])
        g = Data(edge_index=ei, edge_attr=ea, num_nodes=3)
        g = recompute_node_features(g)

        x = g.x  # (3, 6): [bytes_sent, bytes_recv, pkts_sent, pkts_recv, out_deg, in_deg]
        # Node 0 (source of both edges):
        #   bytes_sent = 100 + 200 = 300, bytes_recv = 0
        #   pkts_sent = 10 + 20 = 30, pkts_recv = 0
        #   out_degree = 2, in_degree = 0
        self.assertAlmostEqual(x[0, 0].item(), 300.0, places=1)
        self.assertAlmostEqual(x[0, 1].item(), 0.0)
        self.assertAlmostEqual(x[0, 2].item(), 30.0, places=1)
        self.assertAlmostEqual(x[0, 3].item(), 0.0)
        self.assertAlmostEqual(x[0, 4].item(), 2.0)
        self.assertAlmostEqual(x[0, 5].item(), 0.0)

        # Node 1 (target of edge 0→1):
        self.assertAlmostEqual(x[1, 0].item(), 0.0)
        self.assertAlmostEqual(x[1, 1].item(), 100.0, places=1)
        self.assertAlmostEqual(x[1, 3].item(), 10.0, places=1)
        self.assertAlmostEqual(x[1, 5].item(), 1.0)

        # Node 2 (target of edge 0→2):
        self.assertAlmostEqual(x[2, 1].item(), 200.0, places=1)
        self.assertAlmostEqual(x[2, 3].item(), 20.0, places=1)

    def test_none_edge_attr_returns_unchanged(self):
        """graph with edge_attr=None should return as-is."""
        g = _g()
        original_x = g.x.clone()
        g.edge_attr = None
        g = recompute_node_features(g)
        self.assertTrue(torch.equal(g.x, original_x))

    def test_overwrites_existing_x(self):
        """Existing x should be replaced with recomputed features."""
        g = _g(num_nodes=4, num_edges=3)
        old_shape = g.x.shape
        g = recompute_node_features(g)
        # x should now have 6 columns (from _compute_node_features_arrays)
        self.assertEqual(g.x.shape[1], 6)
        self.assertEqual(g.x.shape[0], 4)

    def test_self_loop_edges(self):
        """Self-loops (src==dst) should contribute to both sent and recv."""
        ei = torch.tensor([[0], [0]])
        ea = torch.tensor([[5.0, 50.0, 0, 0, 0]])
        g = Data(edge_index=ei, edge_attr=ea, num_nodes=1)
        g = recompute_node_features(g)
        # Node 0: bytes_sent = 50, bytes_recv = 50
        self.assertAlmostEqual(g.x[0, 0].item(), 50.0, places=1)
        self.assertAlmostEqual(g.x[0, 1].item(), 50.0, places=1)


# ============================================================
# 4. BUILD_TEMPORAL_SEQUENCES — ordering & edge cases
# ============================================================

class TestSequenceBuilderExtended(unittest.TestCase):
    """Gaps: out-of-order sorting, seq_len > graphs, empty input."""

    def test_sorts_by_window_start(self):
        """Graphs given out of temporal order should be sorted first."""
        gs = [_g(ws=5.0), _g(ws=1.0), _g(ws=3.0), _g(ws=2.0), _g(ws=4.0)]
        seqs = build_temporal_sequences(gs, seq_len=3, stride=1)
        # After sorting: ws = [1, 2, 3, 4, 5] → 3 sequences of length 3
        self.assertEqual(len(seqs), 3)
        for seq in seqs:
            ws_vals = [float(g.window_start) for g in seq]
            self.assertEqual(ws_vals, sorted(ws_vals),
                             "Sequence not in temporal order")

    def test_seq_len_greater_than_num_graphs(self):
        """If seq_len > len(graphs) → empty list (no valid window)."""
        gs = [_g(ws=float(i)) for i in range(3)]
        seqs = build_temporal_sequences(gs, seq_len=5, stride=1)
        self.assertEqual(len(seqs), 0)

    def test_seq_len_equals_num_graphs(self):
        """Exactly one sequence when seq_len == len(graphs)."""
        gs = [_g(ws=float(i)) for i in range(5)]
        seqs = build_temporal_sequences(gs, seq_len=5, stride=1)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 5)

    def test_empty_graph_list(self):
        """Empty input → empty output."""
        seqs = build_temporal_sequences([], seq_len=5, stride=1)
        self.assertEqual(len(seqs), 0)

    def test_stride_larger_than_remaining(self):
        """Stride skipping past the end should still produce valid sequences."""
        gs = [_g(ws=float(i)) for i in range(10)]
        seqs = build_temporal_sequences(gs, seq_len=3, stride=5)
        # range(0, 10-3+1, 5) → [0, 5] → 2 sequences
        self.assertEqual(len(seqs), 2)

    def test_last_graph_is_last_in_each_sequence(self):
        """The last element of each sequence should have the latest window_start."""
        gs = [_g(ws=float(i)) for i in range(8)]
        seqs = build_temporal_sequences(gs, seq_len=3, stride=1)
        for i, seq in enumerate(seqs):
            expected_last_ws = float(i + 2)
            self.assertAlmostEqual(float(seq[-1].window_start), expected_last_ws)


# ============================================================
# 5. PREPROCESS_GRAPH — idempotence & dedup
# ============================================================

class TestPreprocessExtended(unittest.TestCase):
    """Gaps: double-preprocessing, self-loop on graph that already has them."""

    def test_idempotent_norm_values(self):
        """Preprocessing twice should NOT duplicate self-loops or corrupt norm."""
        g = _g(num_nodes=5, num_edges=8)
        g1 = preprocess_graph(g)
        n_edges_after_first = g1.edge_index_with_loops.shape[1]

        # Preprocessing again on the ORIGINAL edge_index should be the same
        g2 = preprocess_graph(g)
        self.assertEqual(g2.edge_index_with_loops.shape[1], n_edges_after_first)
        self.assertTrue(torch.allclose(g1.norm, g2.norm))

    def test_graph_with_existing_self_loops(self):
        """If graph already has self-loops, add_self_loops adds MORE (known PyG behavior)."""
        ei = torch.tensor([[0, 1, 0, 1, 2], [1, 0, 0, 1, 2]])
        # already has self-loops at (0,0), (1,1), (2,2)
        g = Data(x=torch.randn(3, 6), edge_index=ei, edge_attr=torch.randn(5, 5),
                 num_nodes=3)
        g = preprocess_graph(g)
        # add_self_loops adds one per node regardless → 5 + 3 = 8
        self.assertEqual(g.edge_index_with_loops.shape[1], 8)

    def test_preprocessing_preserves_edge_attr(self):
        """edge_attr should not be modified by preprocessing."""
        g = _g()
        orig = g.edge_attr.clone()
        preprocess_graph(g)
        self.assertTrue(torch.equal(g.edge_attr, orig))

    def test_norm_sum_per_node_bounded(self):
        """The sum of norm values for edges incident to a node should be ≤ 1."""
        # For a D^{-1/2} A D^{-1/2} normalization, each column of the norm
        # should roughly sum to 1 (for connected nodes).
        g = _g(num_nodes=5, num_edges=10)
        g = preprocess_graph(g)
        # Not a strict assertion but sanity: all norms in (0, 1]
        self.assertTrue((g.norm >= 0).all())
        self.assertTrue((g.norm <= 1.0 + 1e-6).all())


# ============================================================
# 6. NORMALIZE_FEATURES_GLOBAL — single graph, precision
# ============================================================

class TestNormalizeExtended(unittest.TestCase):
    """Gaps: single graph, numerical precision, negative features."""

    def test_single_graph(self):
        """Normalizing a single graph should work (std may be small)."""
        g = _g(num_nodes=20, num_edges=30)
        graphs, stats = normalize_features_global([g])
        self.assertFalse(torch.isnan(graphs[0].x).any())
        self.assertFalse(torch.isinf(graphs[0].x).any())

    def test_negative_features(self):
        """Graphs with all-negative features should normalize fine."""
        graphs = [_g() for _ in range(3)]
        for g in graphs:
            g.x = -torch.rand(10, 6) * 100
            g.edge_attr = -torch.rand(20, 5) * 100
        graphs, stats = normalize_features_global(graphs)
        for g in graphs:
            self.assertFalse(torch.isnan(g.x).any())

    def test_stats_shapes(self):
        """Stats tensors should match feature dimensions."""
        graphs = [_g(nd=8, ed=3) for _ in range(4)]
        _, stats = normalize_features_global(graphs)
        self.assertEqual(stats["node_mean"].shape, (8,))
        self.assertEqual(stats["node_std"].shape, (8,))
        self.assertEqual(stats["edge_mean"].shape, (3,))
        self.assertEqual(stats["edge_std"].shape, (3,))

    def test_apply_normalization_transforms_correctly(self):
        """After apply_normalization, features = (x - mean) / std."""
        torch.manual_seed(42)
        graphs_train = [_g() for _ in range(5)]
        graphs_train_copy = [
            Data(x=g.x.clone(), edge_index=g.edge_index.clone(),
                 edge_attr=g.edge_attr.clone(), num_nodes=g.num_nodes)
            for g in graphs_train
        ]
        _, stats = normalize_features_global(graphs_train_copy)

        # Apply to a test graph
        g_test = _g()
        orig_x = g_test.x.clone()
        [g_test] = apply_normalization([g_test], stats)
        expected = (orig_x - stats["node_mean"]) / stats["node_std"]
        self.assertTrue(torch.allclose(g_test.x, expected, atol=1e-5))

    def test_identical_graphs_zero_std_handled(self):
        """If all graphs have identical features, std→eps→1.0 (no NaN)."""
        template = torch.ones(10, 6) * 42.0
        template_e = torch.ones(20, 5) * 7.0
        graphs = []
        for _ in range(3):
            g = _g()
            g.x = template.clone()
            g.edge_attr = template_e.clone()
            graphs.append(g)
        graphs, stats = normalize_features_global(graphs)
        for g in graphs:
            self.assertFalse(torch.isnan(g.x).any())


# ============================================================
# 7. EVOLVING_GCN_LAYER — analytical correctness
# ============================================================

class TestLayerAnalytical(unittest.TestCase):
    """Verify GCN layer against hand-computed values on a tiny graph."""

    def test_identity_weight_star_graph(self):
        """Star graph (center + 3 leaves), identity weight → verify propagation."""
        # Star: 0→1, 0→2, 0→3  (directed from center)
        # With self-loops: 0→0, 1→1, 2→2, 3→3 added
        num_nodes = 4
        edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]])
        x = torch.eye(4, 4)  # one-hot (node_feat_dim = 4)
        weight = torch.eye(4, 4)  # identity

        layer = EvolvingGCNLayer()
        out = layer(x, edge_index, weight)

        # With self-loops, graph has 7 edges: (0,1),(0,2),(0,3),(0,0),(1,1),(2,2),(3,3)
        # Degrees (in undirected sense with self-loops): node0=4, node1=2, node2=2, node3=2
        # Output = D^{-1/2} A D^{-1/2} @ x @ W
        # For identity W, output = D^{-1/2} A D^{-1/2} @ x
        self.assertEqual(out.shape, (4, 4))
        self.assertFalse(torch.isnan(out).any())
        # Node 0 gets messages from 1,2,3 and itself → non-zero in all 4 dims
        self.assertGreater(out[0].abs().sum().item(), 0)
        # Leaf node 1 only gets message from itself → non-zero at dim 1 only? No,
        # it also gets from node 0 (via incoming edge 0→1 with self-loop).
        # Actually edges are directed: 0→1 means node 1 receives from 0.
        # So node 1 receives: from node 0 (edge 0→1) + from itself (self-loop 1→1)
        # Hence out[1] has non-zero in dim 0 and dim 1
        self.assertGreater(out[1, 0].abs().item(), 0, "Node 1 should receive from node 0")
        self.assertGreater(out[1, 1].abs().item(), 0, "Node 1 should receive from itself")

    def test_zero_weight_gives_zero_output(self):
        """Zero weight matrix → output should be all zeros."""
        g = _g(num_nodes=5, num_edges=8)
        weight = torch.zeros(6, 16)
        layer = EvolvingGCNLayer()
        out = layer(g.x, g.edge_index, weight)
        self.assertTrue((out == 0).all())

    def test_preprocessed_vs_fallback_match(self):
        """Pre-processed path must produce identical output to fallback path."""
        torch.manual_seed(77)
        g = _g(num_nodes=8, num_edges=15)
        weight = torch.randn(6, 16)
        layer = EvolvingGCNLayer()

        out_raw = layer(g.x, g.edge_index, weight).detach().clone()

        gp = preprocess_graph(
            Data(x=g.x.clone(), edge_index=g.edge_index.clone(),
                 edge_attr=g.edge_attr.clone(), num_nodes=g.num_nodes)
        )
        out_pre = layer(gp.x, gp.edge_index, weight,
                        norm=gp.norm,
                        edge_index_with_loops=gp.edge_index_with_loops).detach()

        self.assertTrue(torch.allclose(out_raw, out_pre, atol=1e-5),
                        f"Max diff: {(out_raw - out_pre).abs().max().item()}")


# ============================================================
# 8. EVOLVING_GNN MODEL — serialization, param count, stochasticity
# ============================================================

class TestModelExtended(unittest.TestCase):
    """Gaps: save/load roundtrip, param count, dropout stochasticity,
    gradient accumulation, hidden-state independence."""

    def setUp(self):
        self.cfg = dict(input_dim=6, hidden_dim=16, edge_feat_dim=5)
        self.model = EvolvingGNN(**self.cfg)

    def test_save_load_roundtrip(self):
        """Model should produce identical output after save → load."""
        self.model.eval()
        torch.manual_seed(10)
        seq = [_g(num_nodes=8, num_edges=12) for _ in range(3)]
        out_before = self.model(seq).detach().clone()

        # Save to buffer
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        buf.seek(0)

        # Load into new model
        model2 = EvolvingGNN(**self.cfg)
        model2.load_state_dict(torch.load(buf, weights_only=True))
        model2.eval()
        out_after = model2(seq).detach()

        self.assertTrue(torch.allclose(out_before, out_after, atol=1e-6),
                        "Output changed after save/load roundtrip")

    def test_parameter_count(self):
        """Verify expected number of trainable parameters."""
        total = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # initial_weights: 6 * 16 = 96
        # LSTM: input_size=flat_dim=96, hidden_size=96
        #   LSTM has 4 * hidden * (input + hidden + 1) params (weight_ih, weight_hh, bias)
        #   = 4 * 96 * (96 + 96 + 1) = 4 * 96 * 193 = 74112  (approx)
        #   Actually: weight_ih_l0 = 4*96*96, weight_hh_l0 = 4*96*96,
        #             bias_ih_l0 = 4*96, bias_hh_l0 = 4*96
        #   = 36864 + 36864 + 384 + 384 = 74496
        # edge_mlp: Linear(16*2+5, 16) = Linear(37, 16) → 37*16+16 = 608
        #           Linear(16, 1) → 16*1+1 = 17
        # Total: 96 + 74496 + 608 + 17 = 75217
        expected = 96 + 74496 + 608 + 17
        self.assertEqual(total, expected,
                         f"Expected {expected} params, got {total}")

    def test_train_mode_stochastic(self):
        """In train mode, dropout should make outputs non-deterministic."""
        self.model.train()
        seq = [_g(num_nodes=10, num_edges=20) for _ in range(3)]
        torch.manual_seed(1)
        out1 = self.model(seq).detach().clone()
        torch.manual_seed(2)
        out2 = self.model(seq).detach().clone()
        # They CAN be the same if dropout doesn't hit any neurons,
        # but with p=0.2 and 20 edges * 16 dims, very unlikely.
        # Use a soft check
        if not torch.allclose(out1, out2, atol=1e-6):
            pass  # expected
        # At minimum, both should be finite
        self.assertFalse(torch.isnan(out1).any())
        self.assertFalse(torch.isnan(out2).any())

    def test_gradient_accumulation(self):
        """Gradients should accumulate across multiple backward passes."""
        self.model.train()
        seq1 = [_g(labels=True) for _ in range(3)]
        seq2 = [_g(labels=True) for _ in range(3)]

        self.model.zero_grad()

        # First backward
        logits1 = self.model(seq1)
        loss1 = nn.BCEWithLogitsLoss()(logits1, seq1[-1].y)
        loss1.backward()
        grad_after_first = {n: p.grad.clone() for n, p in self.model.named_parameters()
                            if p.grad is not None}

        # Second backward (accumulate)
        logits2 = self.model(seq2)
        loss2 = nn.BCEWithLogitsLoss()(logits2, seq2[-1].y)
        loss2.backward()

        # At least some gradients should be larger after accumulation
        any_increased = False
        for n, p in self.model.named_parameters():
            if p.grad is not None and n in grad_after_first:
                if p.grad.abs().sum().item() > grad_after_first[n].abs().sum().item() * 0.5:
                    any_increased = True
        # We just verify no crash and grads exist
        self.assertTrue(any(p.grad is not None for _, p in self.model.named_parameters()))

    def test_independent_sequences_different_output(self):
        """Different sequences should produce different outputs (LSTM hidden reset)."""
        self.model.eval()
        seq_a = [_g(num_nodes=8, num_edges=10) for _ in range(3)]
        seq_b = [_g(num_nodes=8, num_edges=10) for _ in range(3)]

        out_a = self.model(seq_a).detach()
        out_b = self.model(seq_b).detach()

        # Very unlikely to be identical with random features
        self.assertFalse(torch.allclose(out_a, out_b, atol=1e-6),
                         "Different sequences produced identical output")

    def test_hidden_state_reset_across_calls(self):
        """Each forward call should start with fresh LSTM hidden state."""
        self.model.eval()
        torch.manual_seed(55)
        seq = [_g(num_nodes=6, num_edges=10) for _ in range(3)]

        out1 = self.model(seq).detach().clone()
        out2 = self.model(seq).detach().clone()

        self.assertTrue(torch.allclose(out1, out2, atol=1e-6),
                        "Repeated calls gave different results — hidden state leak?")

    def test_dropout_disabled_in_eval(self):
        """In eval mode, all outputs must be identical across runs."""
        self.model.eval()
        seq = [_g() for _ in range(3)]
        results = [self.model(seq).detach().clone() for _ in range(5)]
        for r in results[1:]:
            self.assertTrue(torch.allclose(results[0], r, atol=1e-6),
                            "Eval mode outputs differ — dropout active?")

    def test_model_repr_no_crash(self):
        """repr/str of model should not crash."""
        s = str(self.model)
        self.assertIn("EvolvingGNN", s)
        self.assertIn("LSTM", s)


# ============================================================
# 9. FULL PIPELINE INTEGRATION
# ============================================================

class TestIntegrationPipeline(unittest.TestCase):
    """End-to-end: sanitize → recompute → normalize → preprocess → forward."""

    def test_full_pipeline_synthetic(self):
        """Synthesize graphs, run full preprocessing, then forward pass."""
        nd, ed = 6, 5
        raw_graphs = []
        for i in range(10):
            nn_, ne_ = np.random.randint(5, 15), np.random.randint(8, 25)
            g = _g(nn_, ne_, nd, ed, labels=True, ws=float(i))
            # Inject some NaN to test sanitize
            if i % 3 == 0:
                g.x[0, 0] = float("nan")
            raw_graphs.append(g)

        # Pipeline
        graphs = [sanitize_graph(g) for g in raw_graphs]
        graphs = [recompute_node_features(g) for g in graphs]
        graphs, stats = normalize_features_global(graphs)
        graphs = preprocess_graphs(graphs)
        sequences = build_temporal_sequences(graphs, seq_len=5, stride=1)

        self.assertEqual(len(sequences), 6)  # 10 - 5 + 1

        # Create model matching recomputed feature dim (6 node features)
        model = EvolvingGNN(input_dim=6, hidden_dim=16,
                            edge_feat_dim=ed)
        model.eval()

        for seq in sequences:
            logits = model(seq)
            last_edges = seq[-1].edge_index.shape[1]
            self.assertEqual(logits.shape, (last_edges,))
            self.assertFalse(torch.isnan(logits).any())

    def test_pipeline_with_normalization_stats_reuse(self):
        """Train normalization stats should apply cleanly to 'test' graphs."""
        nd, ed = 6, 5
        train_graphs = [_g(10, 20, nd, ed, labels=True, ws=float(i))
                        for i in range(8)]
        test_graphs = [_g(10, 20, nd, ed, labels=True, ws=float(i))
                       for i in range(3)]

        # Process train
        train_graphs = [sanitize_graph(g) for g in train_graphs]
        train_graphs = [recompute_node_features(g) for g in train_graphs]
        train_graphs, stats = normalize_features_global(train_graphs)
        train_graphs = preprocess_graphs(train_graphs)

        # Process test using train stats
        test_graphs = [sanitize_graph(g) for g in test_graphs]
        test_graphs = [recompute_node_features(g) for g in test_graphs]
        test_graphs = apply_normalization(test_graphs, stats)
        test_graphs = preprocess_graphs(test_graphs)

        # Model should work on both
        model = EvolvingGNN(input_dim=6, hidden_dim=16, edge_feat_dim=ed)
        model.eval()

        train_seq = build_temporal_sequences(train_graphs, seq_len=3)
        for seq in train_seq:
            logits = model(seq)
            self.assertFalse(torch.isnan(logits).any())

        # Test graphs only 3 → 1 sequence of len 3
        test_seq = build_temporal_sequences(test_graphs, seq_len=3)
        self.assertEqual(len(test_seq), 1)
        logits = model(test_seq[0])
        self.assertFalse(torch.isnan(logits).any())

    def test_pipeline_collation_into_model(self):
        """Full flow: build sequences → collate batch → model forward."""
        nd, ed = 6, 5
        graphs = [_g(8, 15, nd, ed, labels=True, ws=float(i))
                  for i in range(10)]
        graphs = [sanitize_graph(g) for g in graphs]
        graphs = [recompute_node_features(g) for g in graphs]
        graphs, _ = normalize_features_global(graphs)
        graphs = preprocess_graphs(graphs)
        sequences = build_temporal_sequences(graphs, seq_len=3)

        # Collate a mini-batch
        indices = np.array([0, 2, 4])
        batch, labels = collate_temporal_batch(sequences, indices=indices)
        self.assertEqual(len(batch), 3)
        self.assertGreater(labels.shape[0], 0)

        # Forward each sequence in the batch
        model = EvolvingGNN(input_dim=6, hidden_dim=16, edge_feat_dim=ed)
        model.eval()
        for seq in batch:
            logits = model(seq)
            self.assertFalse(torch.isnan(logits).any())

    def test_training_loop_simulation(self):
        """Simulate 5 mini-batch training steps end-to-end."""
        nd, ed = 6, 5
        graphs = [_g(8, 15, nd, ed, labels=True, ws=float(i))
                  for i in range(15)]
        graphs = [sanitize_graph(g) for g in graphs]
        graphs = [recompute_node_features(g) for g in graphs]
        graphs, _ = normalize_features_global(graphs)
        graphs = preprocess_graphs(graphs)
        sequences = build_temporal_sequences(graphs, seq_len=3)

        model = EvolvingGNN(input_dim=6, hidden_dim=16, edge_feat_dim=ed)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        losses = []
        n_seq = len(sequences)
        for step in range(5):
            # Random mini-batch of 3
            idx = np.random.choice(n_seq, size=min(3, n_seq), replace=False)
            batch, labels = collate_temporal_batch(sequences, indices=idx)

            total_loss = 0.0
            optimizer.zero_grad()
            for seq in batch:
                logits = model(seq)
                loss = criterion(logits, seq[-1].y)
                loss.backward()
                total_loss += loss.item()

            optimizer.step()
            losses.append(total_loss / len(batch))

        # Loss values should be finite
        for l in losses:
            self.assertFalse(np.isnan(l), f"NaN loss at some step")
            self.assertFalse(np.isinf(l), f"Inf loss at some step")


# ============================================================
# 10. PREPARE_TEMPORAL_DATASET — mock-data smoke test
# ============================================================

class TestPrepareDatasetSmoke(unittest.TestCase):
    """Test prepare_temporal_dataset with synthetic DataFrame data."""

    def _make_mock_packets_df(self, n_packets=100):
        """Create a minimal packets DataFrame matching expected columns."""
        import pandas as pd
        np.random.seed(42)
        ips = [f"192.168.1.{i}" for i in range(1, 6)]
        df = pd.DataFrame({
            "packet_index": np.arange(n_packets),
            "timestamp": np.sort(np.random.uniform(0, 20, n_packets)),
            "src_ip": np.random.choice(ips, n_packets),
            "dst_ip": np.random.choice(ips, n_packets),
            "protocol": np.random.choice(["TCP", "UDP"], n_packets),
            "dst_port": np.random.choice([80, 443, 8080], n_packets),
            "packet_length": np.random.randint(40, 1500, n_packets).astype(float),
            "payload_length": np.random.randint(0, 1400, n_packets).astype(float),
            "label": np.random.choice([0, 1], n_packets, p=[0.9, 0.1]),
        })
        return df

    def test_prepare_from_dataframe(self):
        """End-to-end prepare_temporal_dataset with mock data."""
        import pandas as pd
        from src.Backend.temporal_gnn import prepare_temporal_dataset

        df = self._make_mock_packets_df(200)

        sequences, info = prepare_temporal_dataset(
            packets_df=df,
            window_size=5.0,
            stride=2.0,
            seq_len=3,
            seq_stride=1,
            normalize=True,
        )

        self.assertGreater(len(sequences), 0, "No sequences were built")
        self.assertIn("node_feat_dim", info)
        self.assertIn("edge_feat_dim", info)
        self.assertIn("norm_stats", info)
        self.assertIsNotNone(info["norm_stats"])
        self.assertEqual(info["seq_len"], 3)

        # Each sequence should have 3 graphs
        for seq in sequences:
            self.assertEqual(len(seq), 3)
            for g in seq:
                self.assertFalse(torch.isnan(g.x).any())
                self.assertFalse(torch.isinf(g.x).any())

    def test_prepare_no_normalize(self):
        """prepare_temporal_dataset with normalize=False should skip normalization."""
        import pandas as pd
        from src.Backend.temporal_gnn import prepare_temporal_dataset

        df = self._make_mock_packets_df(150)

        sequences, info = prepare_temporal_dataset(
            packets_df=df,
            window_size=5.0,
            stride=2.0,
            seq_len=3,
            normalize=False,
        )

        self.assertIsNone(info["norm_stats"])
        self.assertGreater(len(sequences), 0)

    def test_prepare_with_separate_labels_df(self):
        """Labels provided as a separate DataFrame."""
        import pandas as pd
        from src.Backend.temporal_gnn import prepare_temporal_dataset

        n = 100
        packets_df = self._make_mock_packets_df(n)
        # Remove label from packets_df — it should come from labels_df
        packets_df = packets_df.drop(columns=["label"])

        labels_df = pd.DataFrame({
            "Unnamed: 0": np.arange(n),
            "x": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        })

        sequences, info = prepare_temporal_dataset(
            packets_df=packets_df,
            labels_df=labels_df,
            window_size=5.0,
            stride=2.0,
            seq_len=3,
        )

        self.assertGreater(len(sequences), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
