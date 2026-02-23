import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
from typing import Optional

from src.Backend.graph_data_wrapper import (
    build_sliding_window_graphs,
    analyze_graphs,
    _compute_node_features_arrays,
    load_graph_dataset,
)


# ============================================================
# BOTTLENECK FIX 1: Pre-process self-loops & normalization
# Instead of recomputing add_self_loops + degree normalization
# on EVERY forward pass for EVERY graph, we do it ONCE upfront.
# ============================================================

def preprocess_graph(graph: Data) -> Data:
    """Pre-add self-loops and cache degree normalization on a single graph.

    This eliminates the repeated add_self_loops + degree computation
    inside EvolvingGCNLayer.forward(), which was the #1 bottleneck
    for large graphs.

    Adds to graph:
        - .edge_index_with_loops: edge_index with self-loops added
        - .norm: pre-computed D^{-1/2} normalization per edge
    """
    edge_index = graph.edge_index
    num_nodes = (
        graph.num_nodes
        if graph.num_nodes is not None
        else int(edge_index.max()) + 1
    )

    # Add self-loops ONCE
    edge_index_sl, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # Compute degree normalization ONCE
    row, col = edge_index_sl
    deg = degree(col, num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    graph.edge_index_with_loops = edge_index_sl
    graph.norm = norm
    return graph


def preprocess_graphs(graphs: list[Data]) -> list[Data]:
    """Pre-process a list of graphs (add self-loops + normalization)."""
    return [preprocess_graph(g) for g in graphs]


# ============================================================
# BOTTLENECK FIX 2: Sanitize tensors (NaN/Inf safety)
# Reuses the same logic from train.py but applied at data-prep time.
# ============================================================

def sanitize_graph(graph: Data) -> Data:
    """Replace NaN/Inf in node features and edge attributes with 0."""
    if graph.x is not None:
        graph.x = torch.nan_to_num(
            graph.x.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        graph.edge_attr = torch.nan_to_num(
            graph.edge_attr.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
    if hasattr(graph, "y") and graph.y is not None:
        graph.y = graph.y.long().clamp(0, 1).float()
    return graph


# ============================================================
# FEATURE NORMALIZATION: vectorised Z-score across all graphs
# ============================================================

def normalize_features_global(
    graphs: list[Data],
    eps: float = 1e-8,
) -> tuple[list[Data], dict]:
    """Z-score normalise node & edge features across ALL graphs at once.

    Vectorised with numpy — a single pass over concatenated features is
    both faster and more statistically stable than per-graph normalisation.
    Returns ``(graphs, stats)`` so the same transform can be applied to
    unseen/test data via ``apply_normalization()``.
    """
    all_x = np.concatenate([g.x.detach().cpu().numpy() for g in graphs], axis=0)
    all_e = np.concatenate([g.edge_attr.detach().cpu().numpy() for g in graphs], axis=0)

    node_mean = all_x.mean(axis=0).astype(np.float32)
    node_std = all_x.std(axis=0).astype(np.float32)
    node_std[node_std < eps] = 1.0

    edge_mean = all_e.mean(axis=0).astype(np.float32)
    edge_std = all_e.std(axis=0).astype(np.float32)
    edge_std[edge_std < eps] = 1.0

    nm, ns = torch.from_numpy(node_mean), torch.from_numpy(node_std)
    em, es = torch.from_numpy(edge_mean), torch.from_numpy(edge_std)

    for g in graphs:
        g.x = (g.x - nm) / ns
        g.edge_attr = (g.edge_attr - em) / es

    stats = {"node_mean": nm, "node_std": ns, "edge_mean": em, "edge_std": es}
    return graphs, stats


def apply_normalization(graphs: list[Data], stats: dict) -> list[Data]:
    """Apply pre-fitted normalisation stats to new/test graphs."""
    nm, ns = stats["node_mean"], stats["node_std"]
    em, es = stats["edge_mean"], stats["edge_std"]
    for g in graphs:
        g.x = (g.x - nm) / ns
        g.edge_attr = (g.edge_attr - em) / es
    return graphs


# ============================================================
# NODE FEATURE RECOMPUTATION: uses _compute_node_features_arrays
# to guarantee consistency after edge modification / sanitisation
# ============================================================

def recompute_node_features(graph: Data) -> Data:
    """Recompute node features from edge topology & edge attributes.

    Uses ``graph_data_wrapper._compute_node_features_arrays`` for
    vectorised aggregation.  Call this after sanitisation or edge
    augmentation to guarantee that node features are consistent
    with the (possibly modified) edge data.

    Also useful for graphs that have no node features yet (e.g.
    those from ``load_graph_dataset`` which only carry edge_attr).

    Expects edge_attr layout: ``[pkt_count, total_bytes, ...]``.
    """
    if graph.edge_attr is None or graph.edge_attr.shape[1] < 2:
        return graph  # cannot recompute without standard edge features

    src = graph.edge_index[0].detach().cpu().numpy()
    dst = graph.edge_index[1].detach().cpu().numpy()
    pkt_count = graph.edge_attr[:, 0].detach().cpu().numpy()
    total_bytes = graph.edge_attr[:, 1].detach().cpu().numpy()
    n_nodes = graph.num_nodes or int(graph.edge_index.max().item()) + 1

    feats = _compute_node_features_arrays(src, dst, total_bytes, pkt_count, n_nodes)
    graph.x = torch.from_numpy(feats)
    return graph


# ============================================================
# SEQUENCE BUILDER: creates sliding windows of consecutive graphs
# for the temporal model
# ============================================================

def build_temporal_sequences(
    graphs: list[Data],
    seq_len: int = 5,
    stride: int = 1,
) -> list[list[Data]]:
    """Build overlapping sequences of consecutive graphs for temporal training.

    Args:
        graphs: List of PyG Data objects sorted by time (window_start).
        seq_len: Number of consecutive graphs per sequence.
        stride: Step size between sequences.

    Returns:
        List of sequences, where each sequence is a list of ``seq_len``
        Data objects.  Labels come from the LAST graph in each sequence.
    """
    # Sort by window_start to guarantee temporal order
    graphs = sorted(graphs, key=lambda g: float(g.window_start))

    sequences: list[list[Data]] = []
    for start in range(0, len(graphs) - seq_len + 1, stride):
        seq = graphs[start : start + seq_len]
        sequences.append(seq)

    return sequences


# ============================================================
# SHARED POST-PROCESSING: sanitize → recompute → normalize →
# preprocess → build sequences → info dict
# ============================================================

def _postprocess_graphs(
    graphs: list[Data],
    seq_len: int,
    seq_stride: int,
    normalize: bool,
) -> tuple[list[list[Data]], dict]:
    """Shared back-end for both pipeline entry points.

    Steps (applied to an already-built list of Data objects):
        1. Analyze (print summary via graph_data_wrapper)
        2. Sanitize (NaN/Inf removal)
        3. Recompute node features (_compute_node_features_arrays)
        4. Global Z-score normalisation (optional)
        5. Pre-process (self-loops + normalization cache)
        6. Build temporal sequences

    Returns:
        (sequences, info_dict) where info_dict includes ``norm_stats``.
    """
    analyze_graphs(graphs)

    graphs = [sanitize_graph(g) for g in graphs]
    graphs = [recompute_node_features(g) for g in graphs]

    norm_stats = None
    if normalize:
        graphs, norm_stats = normalize_features_global(graphs)

    graphs = preprocess_graphs(graphs)
    sequences = build_temporal_sequences(
        graphs, seq_len=seq_len, stride=seq_stride
    )

    info = {
        "node_feat_dim": int(graphs[0].x.shape[1]),
        "edge_feat_dim": int(graphs[0].edge_attr.shape[1]),
        "num_graphs": len(graphs),
        "num_sequences": len(sequences),
        "seq_len": seq_len,
        "norm_stats": norm_stats,
    }
    return sequences, info


# ============================================================
# FULL DATA PIPELINE: CSV -> preprocessed temporal sequences
# Uses graph_data_wrapper functions end-to-end
# ============================================================

def prepare_temporal_dataset(
    packets_df: pd.DataFrame,
    labels_df: Optional[pd.DataFrame] = None,
    window_size: float = 2.0,
    stride: float = 1.0,
    seq_len: int = 5,
    seq_stride: int = 1,
    bytes_col: str = "packet_length",
    label_col: str = "label",
    normalize: bool = True,
) -> tuple[list[list[Data]], dict]:
    """End-to-end pipeline: raw packets -> temporal sequences ready for training.

    Steps:
        1. Merge labels (using graph_data_wrapper conventions)
        2. Build sliding-window graphs (graph_data_wrapper.build_sliding_window_graphs)
        3-8. Shared post-processing (sanitize, recompute, normalize, preprocess, sequences)

    Returns:
        (sequences, info_dict) where info_dict includes ``norm_stats``.
    """
    # 1. Merge labels if provided
    if labels_df is not None:
        labels_df = labels_df.rename(
            columns={
                "Unnamed: 0": "packet_index",
                "x": "label",
            }
        )
        packets_df = packets_df.merge(labels_df, on="packet_index", how="left")
        packets_df[label_col] = packets_df[label_col].fillna(0).astype(int)

    # 2. Build sliding-window graphs (uses graph_data_wrapper pipeline)
    graphs = build_sliding_window_graphs(
        packets_df,
        window_size=window_size,
        stride=stride,
        bytes_col=bytes_col,
        label_col=label_col,
    )

    if not graphs:
        raise ValueError(
            "No graphs were built. Check your data and window parameters."
        )

    # 3-8. Shared post-processing
    sequences, info = _postprocess_graphs(
        graphs, seq_len=seq_len, seq_stride=seq_stride, normalize=normalize
    )

    print(
        f"\n[TEMPORAL PIPELINE] Built {info['num_sequences']} sequences "
        f"of length {seq_len} from {info['num_graphs']} graphs."
    )
    print(f"  Node feature dim: {info['node_feat_dim']}")
    print(f"  Edge feature dim: {info['edge_feat_dim']}")

    return sequences, info


# ============================================================
# CSV CONVENIENCE PIPELINE: uses load_graph_dataset
# ============================================================

def prepare_temporal_dataset_from_csv(
    packet_csv: str,
    label_csv: str,
    delta_t: float = 5.0,
    seq_len: int = 5,
    seq_stride: int = 1,
    normalize: bool = True,
) -> tuple[list[list[Data]], dict]:
    """End-to-end pipeline from CSV files using ``load_graph_dataset``.

    Uses the graph.py snapshot pipeline (delta-t based windowing)
    instead of the sliding-window builder, providing an alternative
    entry point for datasets structured around fixed time intervals.

    Graphs from ``load_graph_dataset`` have edge_attr but NO node
    features — ``recompute_node_features`` creates them from edge
    statistics via ``_compute_node_features_arrays``.

    Returns:
        (sequences, info_dict) — info_dict includes ``norm_stats``.
    """
    graphs = load_graph_dataset(packet_csv, label_csv, delta_t=delta_t)
    if not graphs:
        raise ValueError("No graphs were built from CSVs.")

    # Set window_start from window_id for temporal ordering
    for g in graphs:
        if not hasattr(g, "window_start") or g.window_start is None:
            g.window_start = float(getattr(g, "window_id", 0))

    # Shared post-processing
    sequences, info = _postprocess_graphs(
        graphs, seq_len=seq_len, seq_stride=seq_stride, normalize=normalize
    )

    print(
        f"\n[CSV PIPELINE] Built {info['num_sequences']} sequences "
        f"of length {seq_len} from {info['num_graphs']} snapshot graphs."
    )
    return sequences, info


# ============================================================
# BATCH COLLATION: efficient extraction for training loops
# ============================================================

def collate_temporal_batch(
    sequences: list[list[Data]],
    indices: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> tuple[list[list[Data]], torch.Tensor]:
    """Extract a batch of temporal sequences and their labels.

    Args:
        sequences: All temporal sequences.
        indices: Batch indices (numpy array or list). If None, use all.
        device: Optional device to move all graph tensors to.

    Returns:
        (batch_sequences, labels) where labels are the concatenated
        edge labels from the LAST graph in each sequence.
    """
    if indices is None:
        batch = sequences
    else:
        batch = [sequences[int(i)] for i in indices]

    if not batch:
        return [], torch.tensor([], dtype=torch.float)

    labels = torch.cat([seq[-1].y for seq in batch])

    if device is not None:
        # Use PyG Data.to(device) — moves ALL tensor attributes (x,
        # edge_index, edge_attr, y, norm, edge_index_with_loops, etc.)
        # in a single call, avoiding manual per-attribute transfers and
        # the risk of missing newly-added tensor attributes.
        batch = [
            [g.clone().to(device) for g in seq]
            for seq in batch
        ]
        labels = labels.to(device)

    return batch, labels


# ============================================================
# EVOLVING GCN LAYER (OPTIMIZED)
# Now uses pre-cached self-loops and normalization when available.
# Falls back to recomputation for non-preprocessed graphs.
# ============================================================

class EvolvingGCNLayer(MessagePassing):
    """Graph Convolution Layer with dynamic (externally-provided) weights.

    Optimization: If graphs are pre-processed with ``preprocess_graph()``,
    self-loop addition and degree normalization are skipped (already cached).
    """

    def __init__(self):
        super().__init__(aggr="add")

    def forward(
        self, x, edge_index, weight, norm=None, edge_index_with_loops=None
    ):
        """
        Args:
            x: Node features [N, in_channels]
            edge_index: Original edge index (used as fallback)
            weight: Dynamic weight matrix [in_channels, out_channels]
            norm: Pre-computed normalization (from preprocess_graph). Optional.
            edge_index_with_loops: Edge index with self-loops already added. Optional.
        """
        # Use pre-computed data if available (BOTTLENECK FIX)
        if edge_index_with_loops is not None and norm is not None:
            ei = edge_index_with_loops
            n = norm
        else:
            # Fallback: compute on the fly (for non-preprocessed graphs)
            ei, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            row, col = ei
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            n = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Linear transformation with DYNAMIC weight
        x = torch.matmul(x, weight)

        # Message passing with pre-computed normalization
        return self.propagate(ei, x=x, norm=n)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


# ============================================================
# BOTTLENECK FIX 3: Optimized LSTM Weight Evolution
# Old: LSTM batch = hidden_dim (scales poorly with hidden_dim)
# New: Flatten W to 1D, process as single sequence step.
#      LSTM input/hidden = input_dim * hidden_dim (flat).
#      This removes the hidden_dim scaling bottleneck entirely.
# ============================================================

class EvolvingGNN(nn.Module):
    """Semi-Temporal GNN where GNN weights are evolved by an LSTM.

    Architecture (EvolveGCN-O style):
    - An LSTM takes the flattened GNN weight matrix as input and evolves it.
    - The evolved weight matrix is used by EvolvingGCNLayer per time step.
    - Sequences of consecutive graphs are processed; prediction is on the last graph.

    Optimizations over v1:
    - LSTM operates on flattened weights (batch=1) instead of batch=hidden_dim.
    - GNN layer uses pre-cached self-loops and normalization when available.
    - Input validation is done upfront.
    """

    def __init__(self, input_dim, hidden_dim, edge_feat_dim, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # The GNN Layer (stateless/dynamic weights)
        self.gnn_layer = EvolvingGCNLayer()

        # BOTTLENECK FIX: Flatten the weight matrix for the LSTM
        # Old: LSTM(input_size=input_dim, hidden_size=input_dim) with batch=hidden_dim
        # New: LSTM(input_size=flat_dim, hidden_size=flat_dim) with batch=1
        flat_dim = input_dim * hidden_dim
        self.weight_lstm = nn.LSTM(
            input_size=flat_dim,
            hidden_size=flat_dim,
            batch_first=True,
        )

        # Initial static weights (the "seed" for evolution)
        self.initial_weights = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.initial_weights)

        # Edge Classifier MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_sequence):
        """
        Args:
            graph_sequence: A list of PyG Data objects (length T).
                            Graphs should be pre-processed with
                            ``preprocess_graph()`` for best performance.
        Returns:
            logits: Edge classification logits for the LAST graph in the sequence.
        """
        if not graph_sequence:
            raise ValueError("graph_sequence must contain at least one graph")

        # 1. Flatten initial weights: [input_dim, hidden_dim] -> [1, 1, flat_dim]
        current_weight_flat = self.initial_weights.reshape(1, 1, -1)

        # LSTM hidden states
        h_state = None

        final_embeddings = None
        final_graph = None

        # 2. Process each graph in the temporal sequence
        for i, graph in enumerate(graph_sequence):
            # --- EVOLVE WEIGHTS (optimized: single LSTM step, batch=1) ---
            evolved_flat, h_state = self.weight_lstm(
                current_weight_flat, h_state
            )
            current_weight_flat = evolved_flat

            # Reshape back to [input_dim, hidden_dim]
            current_weight = evolved_flat.squeeze(0).squeeze(0).reshape(
                self.input_dim, self.hidden_dim
            )

            # --- APPLY GNN ---
            x = graph.x
            edge_index = graph.edge_index

            # Validate input dimensions
            if x.shape[1] != self.input_dim:
                raise ValueError(
                    f"Graph feature dim {x.shape[1]} doesn't match "
                    f"model input_dim {self.input_dim}"
                )

            # Use pre-cached self-loops and normalization if available
            norm = getattr(graph, "norm", None)
            ei_loops = getattr(graph, "edge_index_with_loops", None)

            # GCN convolution with evolved weights
            x = self.gnn_layer(
                x, edge_index, current_weight,
                norm=norm, edge_index_with_loops=ei_loops,
            )
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Only keep the last graph's embeddings (saves memory)
            if i == len(graph_sequence) - 1:
                final_embeddings = x
                final_graph = graph

        # 3. Edge classification on the LAST graph
        # Use original edge_index (without self-loops) for classification
        src, dst = final_graph.edge_index
        h_src = final_embeddings[src]
        h_dst = final_embeddings[dst]
        edge_attr = final_graph.edge_attr

        edge_input = torch.cat([h_src, h_dst, edge_attr], dim=1)
        logits = self.edge_mlp(edge_input).squeeze(-1)

        return logits
