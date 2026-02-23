"""
Edge-case and regression tests for the GNN classification/prediction pipeline.

Covers: gnn_interface, graph_data_wrapper, graph.py
Run:  python -m pytest test_gnn_edge_cases.py -v
"""

import sys, os
# Project root (3 levels up: tests/ -> Backend/ -> src/ -> IncidentLens/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from torch_geometric.data import Data

# ──────────────────────────────────────────────
# Imports from the project
# ──────────────────────────────────────────────
from src.Backend.gnn_interface import BaseGNNEncoder, create_dataloaders, compute_class_weights
from src.Backend.graph_data_wrapper import (
    build_sliding_window_graphs,
    _aggregate_flows_numpy,
    _assign_window_ids,
    _compute_node_features_arrays,
    analyze_graphs,
    edge_perturbation_counterfactual,
    compare_graph_windows,
    find_most_anomalous_window,
    find_most_normal_window,
)
from src.Backend.graph import network, node, build_sample_graph, build_snapshot_dataset


# ═══════════════════════════════════════════════
# CONCRETE GNN SUBCLASS FOR TESTING
# ═══════════════════════════════════════════════

class DummyGNN(BaseGNNEncoder):
    """Minimal GNN that concatenates src+dst node features → MLP → edge emb."""

    def __init__(self, node_in=6, edge_in=5, emb_dim=8, num_classes=2):
        super().__init__(node_in_dim=node_in, edge_in_dim=edge_in, embedding_dim=emb_dim)
        self.node_mlp = nn.Linear(node_in, emb_dim)
        self.edge_mlp = nn.Linear(emb_dim * 2 + edge_in, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        if data.edge_index.numel() == 0:
            return torch.zeros(0, self.embedding_dim)
        node_emb = F.relu(self.node_mlp(data.x.float()))
        src, dst = data.edge_index
        edge_input = torch.cat([node_emb[src], node_emb[dst], data.edge_attr.float()], dim=1)
        return self.edge_mlp(edge_input)

    def predict(self, data: Data) -> torch.Tensor:
        if data.edge_index.numel() == 0:
            return torch.zeros(0, 2)
        emb = self.forward(data)
        return self.classifier(emb)


class BinaryScoreGNN(BaseGNNEncoder):
    """GNN that returns (E,) scores instead of (E,2) logits — tests the else branch."""

    def __init__(self):
        super().__init__(node_in_dim=6, edge_in_dim=5, embedding_dim=4)
        self.lin = nn.Linear(5, 4)
        self.out = nn.Linear(4, 1)

    def forward(self, data: Data) -> torch.Tensor:
        if data.edge_attr is None or data.edge_attr.numel() == 0:
            return torch.zeros(0, self.embedding_dim)
        return F.relu(self.lin(data.edge_attr.float()))

    def predict(self, data: Data) -> torch.Tensor:
        emb = self.forward(data)
        return self.out(emb).squeeze(-1)  # (E,) — binary scores path


# ═══════════════════════════════════════════════
# HELPER: build synthetic PyG Data
# ═══════════════════════════════════════════════

def make_graph(n_nodes=4, n_edges=5, node_dim=6, edge_dim=5, label_val=0):
    """Create a synthetic PyG graph with the expected feature shapes."""
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    edge_index = torch.stack([src, dst])
    x = torch.randn(n_nodes, node_dim)
    edge_attr = torch.rand(n_edges, edge_dim)
    y = torch.full((n_edges,), label_val, dtype=torch.long)
    return Data(edge_index=edge_index, x=x, edge_attr=edge_attr, y=y, num_nodes=n_nodes)


def make_packet_df(n_packets=20, n_ips=4, seed=42):
    """Create a synthetic packet DataFrame for graph building."""
    rng = np.random.default_rng(seed)
    ips = [f"10.0.0.{i}" for i in range(n_ips)]
    return pd.DataFrame({
        "timestamp": np.sort(rng.uniform(0, 10, n_packets)),
        "src_ip": rng.choice(ips, n_packets),
        "dst_ip": rng.choice(ips, n_packets),
        "protocol": rng.choice(["TCP", "UDP"], n_packets),
        "dst_port": rng.choice([80, 443, 53, 8080], n_packets),
        "packet_length": rng.integers(40, 1500, n_packets),
        "payload_length": rng.integers(0, 1400, n_packets),
        "label": rng.choice([0, 1], n_packets, p=[0.8, 0.2]),
    })


# ═══════════════════════════════════════════════
# TEST SUITE: BaseGNNEncoder (gnn_interface.py)
# ═══════════════════════════════════════════════

class TestBaseGNNEncoder:
    """Tests for the abstract base class + concrete subclass."""

    def test_forward_normal(self):
        gnn = DummyGNN()
        g = make_graph()
        emb = gnn(g)
        assert emb.shape == (5, 8), f"Expected (5,8), got {emb.shape}"

    def test_encode_l2_normalized(self):
        gnn = DummyGNN()
        g = make_graph()
        emb = gnn.encode(g)
        norms = emb.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            "Encoded embeddings should be L2-normalised"

    def test_encode_no_grad(self):
        gnn = DummyGNN()
        g = make_graph()
        emb = gnn.encode(g)
        assert not emb.requires_grad, "encode() should produce no-grad tensors"

    def test_predict_labels_binary(self):
        gnn = DummyGNN()
        g = make_graph(n_edges=10)
        labels, scores = gnn.predict_labels(g)
        assert labels.shape == (10,), f"Labels shape: {labels.shape}"
        assert scores.shape == (10,), f"Scores shape: {scores.shape}"
        assert set(labels.unique().tolist()).issubset({0, 1})
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_predict_labels_single_edge(self):
        """Edge case: graph with exactly 1 edge — was buggy before fix (0-dim tensor)."""
        gnn = DummyGNN()
        g = make_graph(n_nodes=2, n_edges=1)
        labels, scores = gnn.predict_labels(g)
        assert labels.shape == (1,), f"Single-edge labels shape: {labels.shape}"
        assert scores.shape == (1,), f"Single-edge scores shape: {scores.shape}"

    def test_predict_labels_binary_score_path(self):
        """Tests the (E,) output branch of predict_labels (sigmoid path)."""
        gnn = BinaryScoreGNN()
        g = make_graph(n_edges=7)
        labels, scores = gnn.predict_labels(g)
        assert labels.shape == (7,)
        assert scores.shape == (7,)

    def test_predict_labels_single_edge_binary_score(self):
        """Single edge with (E,) logits — previously produced 0-dim after squeeze."""
        gnn = BinaryScoreGNN()
        g = make_graph(n_nodes=2, n_edges=1)
        labels, scores = gnn.predict_labels(g)
        assert labels.dim() == 1 and labels.shape[0] == 1

    def test_encode_zero_edges(self):
        """Zero-edge graph should return empty tensor gracefully."""
        gnn = DummyGNN()
        g = Data(
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            x=torch.randn(3, 6),
            edge_attr=torch.zeros(0, 5),
            num_nodes=3,
        )
        emb = gnn.encode(g)
        assert emb.shape[0] == 0

    def test_predict_labels_zero_edges(self):
        """Zero-edge graph should return empty labels/scores."""
        gnn = DummyGNN()
        g = Data(
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            x=torch.randn(3, 6),
            edge_attr=torch.zeros(0, 5),
            num_nodes=3,
        )
        labels, scores = gnn.predict_labels(g)
        assert labels.shape[0] == 0
        assert scores.shape[0] == 0

    def test_save_and_load(self, tmp_path):
        gnn = DummyGNN()
        g = make_graph()
        emb_before = gnn.encode(g)
        path = tmp_path / "model.pt"
        gnn.save(path)
        assert path.exists()

        gnn2 = DummyGNN()
        gnn2.load(path)
        emb_after = gnn2.encode(g)
        assert torch.allclose(emb_before, emb_after, atol=1e-6)

    def test_load_missing_file(self):
        gnn = DummyGNN()
        with pytest.raises(FileNotFoundError):
            gnn.load("/nonexistent/path/model.pt")

    def test_large_graph(self):
        """Stress test: 1000 nodes, 5000 edges — should not crash or OOM."""
        gnn = DummyGNN()
        g = make_graph(n_nodes=1000, n_edges=5000)
        emb = gnn.encode(g)
        assert emb.shape == (5000, 8)
        labels, scores = gnn.predict_labels(g)
        assert labels.shape == (5000,)


# ═══════════════════════════════════════════════
# TEST SUITE: create_dataloaders (gnn_interface.py)
# ═══════════════════════════════════════════════

class TestDataLoaders:

    def test_normal_split(self):
        graphs = [make_graph() for _ in range(20)]
        train, val, test, info = create_dataloaders(graphs, batch_size=4)
        assert train is not None
        assert val is not None
        assert info["total"] == 20
        assert info["train"] + info["val"] + info["test"] == 20

    def test_too_few_graphs(self):
        """< 3 graphs → all go to train, val and test are None."""
        graphs = [make_graph(), make_graph()]
        train, val, test, info = create_dataloaders(graphs)
        assert val is None
        assert test is None
        assert info["train"] == 2

    def test_single_graph(self):
        graphs = [make_graph()]
        train, val, test, info = create_dataloaders(graphs)
        assert val is None
        assert info["train"] == 1

    def test_exactly_three_graphs(self):
        graphs = [make_graph() for _ in range(3)]
        train, val, test, info = create_dataloaders(graphs)
        assert info["train"] >= 1
        assert info["val"] >= 1

    def test_reproducible_seed(self):
        graphs = [make_graph() for _ in range(10)]
        _, _, _, info1 = create_dataloaders(graphs, seed=123)
        _, _, _, info2 = create_dataloaders(graphs, seed=123)
        assert info1 == info2


# ═══════════════════════════════════════════════
# TEST SUITE: compute_class_weights (gnn_interface.py)
# ═══════════════════════════════════════════════

class TestClassWeights:

    def test_balanced_labels(self):
        g1 = make_graph(n_edges=10, label_val=0)
        g2 = make_graph(n_edges=10, label_val=1)
        w = compute_class_weights([g1, g2])
        assert w.shape[0] >= 2
        # Balanced → weights ≈ equal
        assert abs(w[0].item() - w[1].item()) < 0.01

    def test_imbalanced_labels(self):
        g0 = make_graph(n_edges=100, label_val=0)
        g1 = make_graph(n_edges=10, label_val=1)
        w = compute_class_weights([g0, g1])
        # Malicious class (1) should have higher weight
        assert w[1].item() > w[0].item()

    def test_only_class_1(self):
        """Only malicious edges — was buggy before fix (wrong class count)."""
        g = make_graph(n_edges=10, label_val=1)
        w = compute_class_weights([g])
        assert w.shape[0] >= 2  # should still have weights for both classes
        assert not torch.isnan(w).any()

    def test_only_class_0(self):
        g = make_graph(n_edges=10, label_val=0)
        w = compute_class_weights([g])
        assert w.shape[0] >= 2
        assert not torch.isnan(w).any()

    def test_no_labels(self):
        """Graphs with y=None → fallback equal weights."""
        g = make_graph()
        g.y = None
        w = compute_class_weights([g])
        assert torch.allclose(w, torch.ones(2))

    def test_empty_graph_list(self):
        w = compute_class_weights([])
        assert torch.allclose(w, torch.ones(2))


# ═══════════════════════════════════════════════
# TEST SUITE: _aggregate_flows_numpy (graph_data_wrapper.py)
# ═══════════════════════════════════════════════

class TestAggregateFlows:

    def test_empty_input(self):
        """Empty arrays — was crashing with IndexError before fix."""
        empty = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        result = _aggregate_flows_numpy(
            empty, empty, empty, empty, empty,
            empty_f, empty_f, empty_f, empty,
        )
        assert len(result) == 11
        assert all(len(r) == 0 for r in result)

    def test_single_packet(self):
        """Single packet → 1 flow, std_iat=0."""
        result = _aggregate_flows_numpy(
            np.array([0]), np.array([0]), np.array([1]),
            np.array([0]), np.array([0]),
            np.array([1.0]), np.array([100.0]), np.array([50.0]),
            np.array([1]),
        )
        pkt_count = result[5]
        std_iat = result[9]
        assert pkt_count[0] == 1.0
        assert std_iat[0] == 0.0  # single-packet → std = 0

    def test_two_packets_same_flow(self):
        """Two packets in the same flow → 1 flow with correct IAT."""
        result = _aggregate_flows_numpy(
            np.array([0, 0]), np.array([0, 0]), np.array([1, 1]),
            np.array([0, 0]), np.array([0, 0]),
            np.array([1.0, 2.0]), np.array([100.0, 200.0]), np.array([50.0, 60.0]),
            np.array([0, 1]),
        )
        pkt_count = result[5]
        total_bytes = result[6]
        label_max = result[10]
        assert pkt_count[0] == 2.0
        assert total_bytes[0] == 300.0
        assert label_max[0] == 1  # max of (0, 1)

    def test_multiple_flows(self):
        """Two distinct flows → 2 entries in output."""
        result = _aggregate_flows_numpy(
            np.array([0, 0]),           # same window
            np.array([0, 1]),           # different src
            np.array([1, 2]),           # different dst
            np.array([0, 0]),
            np.array([0, 0]),
            np.array([1.0, 2.0]),
            np.array([100.0, 200.0]),
            np.array([50.0, 60.0]),
            np.array([0, 0]),
        )
        assert len(result[5]) == 2  # 2 flows

    def test_large_cardinality_safe_path(self):
        """Force the safe path (>16-bit cardinalities)."""
        n = 100
        rng = np.random.default_rng(0)
        result = _aggregate_flows_numpy(
            rng.integers(0, 100000, n, dtype=np.int64),
            rng.integers(0, 100000, n, dtype=np.int64),
            rng.integers(0, 100000, n, dtype=np.int64),
            rng.integers(0, 300, n, dtype=np.int64),
            rng.integers(0, 300, n, dtype=np.int64),
            np.sort(rng.uniform(0, 100, n)),
            rng.uniform(40, 1500, n),
            rng.uniform(0, 1400, n),
            rng.choice([0, 1], n),
        )
        assert len(result[5]) > 0  # at least some flows


# ═══════════════════════════════════════════════
# TEST SUITE: _assign_window_ids (graph_data_wrapper.py)
# ═══════════════════════════════════════════════

class TestAssignWindowIds:

    def test_single_timestamp(self):
        """All packets at the same timestamp."""
        ts = np.array([5.0, 5.0, 5.0])
        result, starts = _assign_window_ids(ts, 5.0, 5.0, 2.0, 1.0)
        assert len(starts) >= 1

    def test_empty_timestamps(self):
        ts = np.array([])
        result, starts = _assign_window_ids(ts, 0.0, 0.0, 2.0, 1.0)
        assert len(starts) >= 1

    def test_large_dataset_searchsorted_path(self):
        """Force the searchsorted path (N*W > 5M)."""
        ts = np.linspace(0, 1000, 10000)
        result, starts = _assign_window_ids(ts, 0, 1000, 1.0, 0.5)
        # Should return tuple of (pkt_idx, win_idx)
        assert isinstance(result, tuple)

    def test_broadcast_path(self):
        """Small dataset that uses the broadcast path."""
        ts = np.array([0.5, 1.5, 2.5])
        result, starts = _assign_window_ids(ts, 0, 3, 2.0, 1.0)
        assert isinstance(result, np.ndarray)  # broadcast returns mask


# ═══════════════════════════════════════════════
# TEST SUITE: build_sliding_window_graphs
# ═══════════════════════════════════════════════

class TestBuildSlidingWindowGraphs:

    def test_normal_build(self):
        df = make_packet_df(n_packets=50)
        graphs = build_sliding_window_graphs(df)
        assert len(graphs) > 0
        g = graphs[0]
        assert hasattr(g, "edge_index")
        assert hasattr(g, "x")
        assert hasattr(g, "edge_attr")
        assert hasattr(g, "y")
        assert g.x.shape[1] == 6  # node features
        assert g.edge_attr.shape[1] == 5  # edge features

    def test_single_packet(self):
        """Just one packet → should produce at least 1 graph."""
        df = pd.DataFrame({
            "timestamp": [1.0],
            "src_ip": ["10.0.0.1"],
            "dst_ip": ["10.0.0.2"],
            "protocol": ["TCP"],
            "dst_port": [80],
            "packet_length": [100],
            "payload_length": [50],
            "label": [0],
        })
        graphs = build_sliding_window_graphs(df)
        assert len(graphs) >= 1
        assert graphs[0].edge_index.shape[1] == 1

    def test_all_same_timestamp(self):
        """All packets at the exact same time."""
        n = 10
        df = pd.DataFrame({
            "timestamp": [5.0] * n,
            "src_ip": [f"10.0.0.{i % 3}" for i in range(n)],
            "dst_ip": [f"10.0.0.{(i + 1) % 3}" for i in range(n)],
            "protocol": ["TCP"] * n,
            "dst_port": [80] * n,
            "packet_length": [100] * n,
            "payload_length": [50] * n,
            "label": [0] * n,
        })
        graphs = build_sliding_window_graphs(df)
        assert len(graphs) >= 1

    def test_all_malicious(self):
        df = make_packet_df()
        df["label"] = 1
        graphs = build_sliding_window_graphs(df)
        for g in graphs:
            assert (g.y == 1).all()

    def test_all_normal(self):
        df = make_packet_df()
        df["label"] = 0
        graphs = build_sliding_window_graphs(df)
        for g in graphs:
            assert (g.y == 0).all()

    def test_very_small_window(self):
        """Tiny window that might produce many small graphs."""
        df = make_packet_df(n_packets=100)
        graphs = build_sliding_window_graphs(df, window_size=0.1, stride=0.05)
        # Should not crash; may produce many windows
        assert isinstance(graphs, list)

    def test_very_large_window(self):
        """Window larger than data range → single graph."""
        df = make_packet_df(n_packets=20)
        graphs = build_sliding_window_graphs(df, window_size=1000.0, stride=1000.0)
        assert len(graphs) == 1

    def test_two_ips_only(self):
        """Minimal 2-IP network."""
        df = pd.DataFrame({
            "timestamp": [0.0, 0.5, 1.0, 1.5],
            "src_ip": ["A", "B", "A", "B"],
            "dst_ip": ["B", "A", "B", "A"],
            "protocol": ["TCP"] * 4,
            "dst_port": [80] * 4,
            "packet_length": [100, 200, 300, 400],
            "payload_length": [50, 100, 150, 200],
            "label": [0, 0, 1, 1],
        })
        graphs = build_sliding_window_graphs(df, window_size=2.0, stride=1.0)
        assert len(graphs) >= 1

    def test_network_object_attached(self):
        """Each graph should have a .network attribute of type network."""
        df = make_packet_df(n_packets=30)
        graphs = build_sliding_window_graphs(df)
        for g in graphs:
            assert hasattr(g, "network")
            assert isinstance(g.network, network)

    def test_node_count_consistent(self):
        """num_nodes should match x.shape[0]."""
        df = make_packet_df(n_packets=50)
        graphs = build_sliding_window_graphs(df)
        for g in graphs:
            assert g.num_nodes == g.x.shape[0]

    def test_edge_count_consistent(self):
        """edge_index columns == edge_attr rows == y length."""
        df = make_packet_df(n_packets=50)
        graphs = build_sliding_window_graphs(df)
        for g in graphs:
            n_edges = g.edge_index.shape[1]
            assert g.edge_attr.shape[0] == n_edges
            assert g.y.shape[0] == n_edges


# ═══════════════════════════════════════════════
# TEST SUITE: graph.py — network, node
# ═══════════════════════════════════════════════

class TestNetworkGraph:

    def test_build_sample_graph(self):
        g = build_sample_graph()
        assert g.num_nodes == 4
        ei = g.build_edge_index()
        assert ei.shape[1] == 5  # 5 edges

    def test_degree(self):
        g = build_sample_graph()
        out = g.out_degree()
        in_ = g.in_degree()
        assert out.shape[0] == 4
        assert in_.shape[0] == 4
        assert out[0].item() == 2  # 0→1, 0→2

    def test_add_duplicate_node(self):
        g = network(num_nodes=0)
        g.add_node(node("1.2.3.4", 0, torch.zeros(1)))
        with pytest.raises(ValueError, match="already exists"):
            g.add_node(node("5.6.7.8", 0, torch.zeros(1)))

    def test_add_edge_invalid_nodes(self):
        g = network(num_nodes=2)
        g.add_node(node("A", 0, torch.zeros(1)))
        with pytest.raises(ValueError):
            g.add_edge(0, 99)  # node 99 doesn't exist

    def test_empty_network(self):
        g = network(num_nodes=0)
        ei = g.build_edge_index()
        assert ei.shape == (2, 0)

    def test_sparse_adjacency(self):
        g = build_sample_graph()
        adj = g.build_sparse_adjacency()
        assert adj.is_sparse
        assert adj.shape == (4, 4)

    def test_from_edge_list(self):
        g = network.from_edge_list(3, [(0, 1), (1, 2)])
        assert g.num_nodes == 3
        assert g.build_edge_index().shape[1] == 2

    def test_to_pyg_data(self):
        g = build_sample_graph()
        data = g.to_pyg_data()
        assert isinstance(data, Data)
        assert data.edge_index.shape[1] == 5

    def test_single_node_no_edges(self):
        g = network(num_nodes=0)
        g.add_node(node("10.0.0.1", 0, torch.ones(3)))
        ei = g.build_edge_index()
        assert ei.shape == (2, 0)
        assert g.out_degree().tolist() == [0]

    def test_self_loop(self):
        g = network(num_nodes=0)
        g.add_node(node("A", 0, torch.zeros(1)))
        g.add_edge(0, 0)
        ei = g.build_edge_index()
        assert ei[:, 0].tolist() == [0, 0]

    def test_degree_cache_invalidation(self):
        """Adding an edge should invalidate degree cache."""
        g = network(num_nodes=0)
        g.add_node(node("A", 0, torch.zeros(1)))
        g.add_node(node("B", 1, torch.zeros(1)))
        assert g.out_degree()[0].item() == 0
        g.add_edge(0, 1)
        assert g.out_degree()[0].item() == 1  # cache was cleared


# ═══════════════════════════════════════════════
# TEST SUITE: Analysis & Counterfactual tools
# ═══════════════════════════════════════════════

class TestAnalysisTools:

    def _make_graphs(self):
        df = make_packet_df(n_packets=100)
        return build_sliding_window_graphs(df)

    def test_analyze_graphs(self, capsys):
        graphs = self._make_graphs()
        analyze_graphs(graphs)
        captured = capsys.readouterr()
        assert "GRAPH ANALYSIS" in captured.out

    def test_analyze_empty(self, capsys):
        analyze_graphs([])
        captured = capsys.readouterr()
        assert "No graphs" in captured.out

    def test_edge_perturbation_cf(self):
        g = make_graph(n_nodes=10, n_edges=20, label_val=1)
        results = edge_perturbation_counterfactual(g, max_removals=3)
        assert isinstance(results, list)
        assert len(results) <= 3
        for r in results:
            assert "structural_impact" in r
            assert "feature_impact" in r

    def test_edge_perturbation_no_attr(self):
        """Graph with no edge_attr — should return empty."""
        g = make_graph()
        g.edge_attr = None
        results = edge_perturbation_counterfactual(g)
        assert results == []

    def test_edge_perturbation_no_edges(self):
        g = Data(
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            x=torch.randn(3, 6),
            edge_attr=torch.zeros(0, 5),
            y=torch.zeros(0, dtype=torch.long),
            num_nodes=3,
        )
        results = edge_perturbation_counterfactual(g)
        assert results == []

    def test_compare_graph_windows(self):
        g_a = make_graph(n_nodes=5, n_edges=8, label_val=0)
        g_b = make_graph(n_nodes=7, n_edges=15, label_val=1)
        g_a.window_start = 0.0
        g_b.window_start = 2.0
        result = compare_graph_windows(g_a, g_b)
        assert "feature_diffs" in result
        assert "edge_diff" in result

    def test_find_most_anomalous(self):
        graphs = self._make_graphs()
        if len(graphs) == 0:
            pytest.skip("No graphs built")
        idx, g, stats = find_most_anomalous_window(graphs)
        assert 0 <= idx < len(graphs)
        assert "malicious_ratio" in stats

    def test_find_most_normal(self):
        graphs = self._make_graphs()
        if len(graphs) == 0:
            pytest.skip("No graphs built")
        idx, g, stats = find_most_normal_window(graphs)
        assert 0 <= idx < len(graphs)

    def test_find_anomalous_empty(self):
        with pytest.raises(ValueError, match="No graphs"):
            find_most_anomalous_window([])

    def test_find_normal_empty(self):
        with pytest.raises(ValueError, match="No graphs"):
            find_most_normal_window([])


# ═══════════════════════════════════════════════
# TEST SUITE: GNN + graph_data_wrapper integration
# ═══════════════════════════════════════════════

class TestGNNIntegration:

    def test_gnn_on_real_graphs(self):
        """Build graphs from synthetic data, then run GNN forward + predict."""
        df = make_packet_df(n_packets=80, n_ips=6)
        graphs = build_sliding_window_graphs(df)
        gnn = DummyGNN()
        for g in graphs:
            emb = gnn.encode(g)
            assert emb.shape[0] == g.edge_index.shape[1]
            assert emb.shape[1] == 8
            labels, scores = gnn.predict_labels(g)
            assert labels.shape[0] == g.edge_index.shape[1]

    def test_class_weights_on_real_graphs(self):
        df = make_packet_df(n_packets=80, n_ips=6)
        graphs = build_sliding_window_graphs(df)
        w = compute_class_weights(graphs)
        assert w.shape[0] >= 2
        assert not torch.isnan(w).any()
        assert not torch.isinf(w).any()

    def test_dataloaders_on_real_graphs(self):
        df = make_packet_df(n_packets=200, n_ips=8)
        graphs = build_sliding_window_graphs(df, window_size=1.0, stride=0.5)
        if len(graphs) < 3:
            pytest.skip("Not enough graphs for splitting")
        train_loader, val_loader, test_loader, info = create_dataloaders(graphs, batch_size=4)
        assert info["total"] == len(graphs)
        # Iterate one batch from train
        batch = next(iter(train_loader))
        gnn = DummyGNN()
        emb = gnn(batch)
        assert emb.shape[0] == batch.edge_index.shape[1]


# ═══════════════════════════════════════════════
# TEST SUITE: _compute_node_features_arrays
# ═══════════════════════════════════════════════

class TestNodeFeatures:

    def test_basic(self):
        feats = _compute_node_features_arrays(
            np.array([0, 1]), np.array([1, 2]),
            np.array([100.0, 200.0]), np.array([5.0, 10.0]),
            n_nodes=3,
        )
        assert feats.shape == (3, 6)
        # Node 0: bytes_sent=100, out_degree=1 (sent 1 flow)
        assert feats[0, 0] == 100.0
        assert feats[0, 4] == 1  # out_degree

    def test_zero_nodes(self):
        feats = _compute_node_features_arrays(
            np.array([], dtype=np.int64), np.array([], dtype=np.int64),
            np.array([], dtype=np.float32), np.array([], dtype=np.float32),
            n_nodes=0,
        )
        assert feats.shape == (0, 6)

    def test_single_node_self_loop(self):
        feats = _compute_node_features_arrays(
            np.array([0]), np.array([0]),
            np.array([100.0]), np.array([5.0]),
            n_nodes=1,
        )
        assert feats.shape == (1, 6)
        assert feats[0, 0] == 100.0  # bytes_sent
        assert feats[0, 1] == 100.0  # bytes_recv (self-loop)


# ═══════════════════════════════════════════════
# TEST SUITE: build_snapshot_dataset (graph.py)
# ═══════════════════════════════════════════════

class TestBuildSnapshotDataset:

    def test_basic(self):
        df = pd.DataFrame({
            "timestamp": [0, 1, 2, 3, 4, 5],
            "src_ip": ["A", "B", "A", "C", "A", "B"],
            "dst_ip": ["B", "A", "C", "A", "B", "C"],
            "protocol": ["TCP"] * 6,
            "packet_length": [100] * 6,
            "payload_length": [50] * 6,
            "label": [0, 0, 1, 0, 1, 0],
        })
        data_list, node_map, flows_df = build_snapshot_dataset(df, delta_t=3.0)
        assert len(data_list) > 0
        assert len(node_map) == 3  # A, B, C


# ═══════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
