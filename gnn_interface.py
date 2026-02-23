"""
GNN Integration Interface for IncidentLens
===========================================
Defines the contract that any GNN model must satisfy to plug into the
IncidentLens pipeline.  Your GNN class should subclass ``BaseGNNEncoder``
and implement ``forward()`` + ``predict()``.

The pipeline flow:
    1. ``graph_data_wrapper.build_sliding_window_graphs()`` produces
       ``list[Data]`` with:
           - ``edge_index``  : (2, E) long    — directed edges
           - ``x``           : (N, 6) float   — node features
             [bytes_sent, bytes_recv, pkts_sent, pkts_recv, out_degree, in_degree]
           - ``edge_attr``   : (E, 5) float   — edge (flow) features
             [packet_count, total_bytes, mean_payload, mean_iat, std_iat]
           - ``y``           : (E,)   long    — ground-truth labels (0/1)
           - ``num_nodes``   : int
           - ``window_start``: float
           - ``network``     : graph.network object

    2. ``wrappers.generate_embeddings()`` calls your GNN's ``encode()``
       to get per-edge embeddings (E, D) used for kNN counterfactual
       retrieval in Elasticsearch.

    3. ``wrappers.index_pyg_graph()`` stores ``prediction`` and
       ``prediction_score`` fields from your GNN's ``predict()`` output.

Key design decision — **edge-level embeddings**:
    GNNs typically produce node-level embeddings.  For counterfactual
    analysis, we need one embedding per *flow* (= one edge).  The
    recommended approach is:
        edge_emb[i] = concat(node_emb[src[i]], node_emb[dst[i]])
    or use an edge-aware architecture (EGNN, PNA with edge features).

Usage example:
    >>> from gnn_interface import BaseGNNEncoder
    >>> class MyGNN(BaseGNNEncoder):
    ...     def __init__(self):
    ...         super().__init__(
    ...             node_in_dim=6,
    ...             edge_in_dim=5,
    ...             embedding_dim=32,
    ...         )
    ...         self.conv1 = GCNConv(6, 64)
    ...         self.conv2 = GCNConv(64, 32)
    ...         self.edge_mlp = nn.Linear(32*2 + 5, 32)
    ...         self.classifier = nn.Linear(32, 2)
    ...
    ...     def forward(self, data):
    ...         x = F.relu(self.conv1(data.x, data.edge_index))
    ...         node_emb = self.conv2(x, data.edge_index)
    ...         src, dst = data.edge_index
    ...         edge_emb = torch.cat([node_emb[src], node_emb[dst], data.edge_attr], dim=1)
    ...         edge_emb = self.edge_mlp(edge_emb)
    ...         return edge_emb   # (E, 32)
    ...
    ...     def predict(self, data):
    ...         edge_emb = self.forward(data)
    ...         logits = self.classifier(edge_emb)
    ...         return logits     # (E, 2)

Then register it:
    >>> import wrappers
    >>> wrappers.set_gnn_encoder(MyGNN())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data


class BaseGNNEncoder(ABC, nn.Module):
    """Abstract base class for GNN encoders in IncidentLens.

    Subclass this and implement ``forward()`` and ``predict()``.
    The pipeline calls ``encode()`` (which wraps ``forward()``) and
    ``predict()`` at inference time.

    Parameters
    ----------
    node_in_dim : int
        Dimension of input node features (default 6 from graph_data_wrapper).
    edge_in_dim : int
        Dimension of input edge features (default 5 from graph_data_wrapper).
    embedding_dim : int
        Dimension of output per-edge embeddings for kNN retrieval.
    """

    def __init__(
        self,
        node_in_dim: int = 6,
        edge_in_dim: int = 5,
        embedding_dim: int = 32,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        """Produce per-edge embeddings from a single PyG Data graph.

        Parameters
        ----------
        data : Data
            A single graph snapshot with ``edge_index``, ``x``,
            ``edge_attr``, and ``num_nodes``.

        Returns
        -------
        torch.Tensor of shape ``(E, embedding_dim)``
            One embedding vector per edge (flow).
        """
        ...

    @abstractmethod
    def predict(self, data: Data) -> torch.Tensor:
        """Produce per-edge classification logits / probabilities.

        Parameters
        ----------
        data : Data
            Same as ``forward()``.

        Returns
        -------
        torch.Tensor of shape ``(E, num_classes)`` or ``(E,)``
            Raw logits or probabilities.  If shape is ``(E, 2)``,
            column 1 is treated as the malicious probability.
            If shape is ``(E,)``, values are treated as malicious scores.
        """
        ...

    @torch.no_grad()
    def encode(self, data: Data) -> torch.Tensor:
        """Inference-mode wrapper around ``forward()`` with L2 normalisation.

        This is what ``wrappers.generate_embeddings()`` calls.
        Returns (E, embedding_dim) numpy-compatible float32 tensor.
        """
        self.eval()
        emb = self.forward(data)
        if emb.numel() == 0:
            return emb.float()
        # L2-normalize for cosine similarity in Elasticsearch kNN
        norms = emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return (emb / norms).float()

    @torch.no_grad()
    def predict_labels(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience: return (predicted_labels, confidence_scores).

        Returns
        -------
        labels : LongTensor (E,) — 0=normal, 1=malicious
        scores : FloatTensor (E,) — probability of malicious class
        """
        self.eval()
        logits = self.predict(data)
        if logits.numel() == 0:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0)
        if logits.dim() == 2 and logits.shape[1] >= 2:
            probs = torch.softmax(logits, dim=1)
            scores = probs[:, 1]
            labels = (scores >= 0.5).long()
        else:
            scores = torch.sigmoid(logits.view(-1))
            labels = (scores >= 0.5).long()
        return labels, scores

    def save(self, path: str | Path) -> None:
        """Save model weights + config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "node_in_dim": self.node_in_dim,
                "edge_in_dim": self.edge_in_dim,
                "embedding_dim": self.embedding_dim,
                "class_name": self.__class__.__name__,
            },
            path,
        )

    def load(self, path: str | Path, strict: bool = True) -> None:
        """Load model weights from a checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(checkpoint["state_dict"], strict=strict)


# ──────────────────────────────────────────────
# Utility: PyG DataLoader setup for training
# ──────────────────────────────────────────────

def create_dataloaders(
    graphs: list[Data],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple:
    """Split graphs into train/val/test and return PyG DataLoaders.

    Parameters
    ----------
    graphs : list[Data] — output from ``graph_data_wrapper.build_sliding_window_graphs()``
    train_ratio : fraction for training
    val_ratio : fraction for validation (rest goes to test)
    batch_size : mini-batch size
    shuffle : whether to shuffle training data
    seed : random seed for reproducible splits

    Returns
    -------
    (train_loader, val_loader, test_loader, split_info)
    """
    from torch_geometric.loader import DataLoader

    n = len(graphs)
    if n < 3:
        # Too few graphs — return all as train
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)
        info = {"train": n, "val": 0, "test": 0, "total": n}
        return loader, None, None, info

    # Deterministic shuffle
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()

    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = n - n_train - n_val

    train_graphs = [graphs[i] for i in perm[:n_train]]
    val_graphs = [graphs[i] for i in perm[n_train:n_train + n_val]]
    test_graphs = [graphs[i] for i in perm[n_train + n_val:]]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False) if test_graphs else None

    info = {
        "train": n_train,
        "val": n_val,
        "test": n_test,
        "total": n,
        "batch_size": batch_size,
    }
    return train_loader, val_loader, test_loader, info


def compute_class_weights(graphs: list[Data]) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced edge labels.

    Returns a FloatTensor of shape ``(num_classes,)`` suitable for
    ``nn.CrossEntropyLoss(weight=...)``.
    """
    label_tensors = [
        g.y for g in graphs
        if g.y is not None and g.y.numel() > 0
    ]
    if not label_tensors:
        return torch.ones(2)  # fallback: equal weights
    all_labels = torch.cat(label_tensors).long()
    num_classes = max(int(all_labels.max()) + 1, 2)  # at least 2 classes
    counts = torch.bincount(all_labels, minlength=num_classes).float()
    counts[counts == 0] = 1.0  # avoid division by zero
    present = (counts > 0).sum().item()
    weights = counts.sum() / (present * counts)
    return weights
