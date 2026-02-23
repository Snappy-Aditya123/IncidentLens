import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


from src.Backend.graph_data_wrapper import build_sliding_window_graphs


# ============================================================
# MODEL (GraphSAGE + Edge MLP)
# ============================================================

class EdgeGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim,
                 hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(node_feat_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.dropout = dropout

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        # NOTE: tensors must be finite here
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Node encoder
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Edge classifier
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]

        edge_input = torch.cat([h_src, h_dst, edge_attr], dim=1)
        logits = self.edge_mlp(edge_input).squeeze(-1)
        return logits


# ============================================================
# DATA NUMERICAL SAFETY (paper-grade handling of missing values)
# ============================================================

def tensor_make_finite_(t: torch.Tensor) -> torch.Tensor:
    """Convert NaN/Inf to finite values *without removing data*."""
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def sanitize_graphs_inplace(graphs):
    """
    Make graphs numerically safe:
    - keep NaNs in raw pandas if you like, but Data tensors must be finite
    - do NOT drop edges/rows
    """
    fixed = 0
    for g in graphs:
        if g.x is not None:
            g.x = g.x.float()
            if torch.isnan(g.x).any() or torch.isinf(g.x).any():
                g.x = tensor_make_finite_(g.x)
                fixed += 1

        if hasattr(g, "edge_attr") and g.edge_attr is not None:
            g.edge_attr = g.edge_attr.float()
            if torch.isnan(g.edge_attr).any() or torch.isinf(g.edge_attr).any():
                g.edge_attr = tensor_make_finite_(g.edge_attr)
                fixed += 1

        if hasattr(g, "y") and g.y is not None:
            # enforce 0/1 float labels
            g.y = g.y.long().clamp(0, 1).float()

    print(f"[SANITIZE] NaN/Inf fixed in {fixed} tensor(s).")
    return graphs


# ============================================================
# SPLIT / NORMALIZE / DIAGNOSTICS
# ============================================================

def time_split(graphs, train_ratio=0.7, val_ratio=0.15):
    graphs = sorted(graphs, key=lambda g: float(g.window_start))
    n = len(graphs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return graphs[:n_train], graphs[n_train:n_train+n_val], graphs[n_train+n_val:]


def normalize_edge_features(train_graphs, val_graphs, test_graphs):
    """
    Train-only z-score normalization.
    Must be called AFTER sanitize_graphs_inplace().
    """
    all_train = torch.cat([g.edge_attr for g in train_graphs], dim=0).float()
    all_train = tensor_make_finite_(all_train)

    mean = all_train.mean(dim=0, keepdim=True)
    std = all_train.std(dim=0, keepdim=True)
    std[std == 0] = 1.0

    for gs in (train_graphs, val_graphs, test_graphs):
        for g in gs:
            ea = tensor_make_finite_(g.edge_attr.float())
            g.edge_attr = (ea - mean) / std

    return train_graphs, val_graphs, test_graphs


def label_stats(graphs, name="SET"):
    ys = torch.cat([g.y for g in graphs]).long()
    pos = (ys == 1).sum().item()
    neg = (ys == 0).sum().item()
    total = pos + neg
    rate = pos / (total + 1e-9)
    print(f"[LABELS:{name}] pos={pos} neg={neg} pos_rate={rate:.8f}")
    return pos, neg


def compute_pos_weight(graphs, cap=1000.0):
    ys = torch.cat([g.y for g in graphs]).long()
    pos = (ys == 1).sum().item()
    neg = (ys == 0).sum().item()
    print(f"[CLASS BALANCE] Pos={pos} Neg={neg}")

    if pos == 0:
        print("[WARN] No positives in TRAIN. pos_weight set to 1.0.")
        return torch.tensor(1.0)

    w = neg / pos
    if w > cap:
        print(f"[WARN] pos_weight={w:.1f} too large; capping to {cap}.")
        w = cap

    return torch.tensor(w, dtype=torch.float32)


def safe_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


# ============================================================
# EVALUATION (robust, prints confusion matrix etc.)
# ============================================================

def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # NOTE: batch.x and batch.edge_attr are already sanitized by
            # sanitize_graphs_inplace() at data-prep time — skipping
            # redundant tensor_make_finite_ here saves a tensor copy per batch.

            logits = model(batch)
            logits = tensor_make_finite_(logits)  # model output may have NaN

            loss = criterion(logits, batch.y.float())
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_labels.append(batch.y.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    pr_auc = average_precision_score(labels, probs)
    roc_auc = safe_roc_auc(labels, probs)

    preds = (probs >= threshold).astype(int)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)

    return total_loss / max(1, len(loader)), pr_auc, roc_auc, f1, precision, recall, cm, probs, labels


def find_best_threshold(probs, labels):
    """Vectorised threshold search using numpy broadcasting."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    thresholds = np.linspace(0.05, 0.95, 91)
    # (91, N) boolean matrix of predictions at each threshold
    preds = probs[np.newaxis, :] >= thresholds[:, np.newaxis]
    # Compute TP, FP, FN in bulk
    tp = (preds & (labels == 1)).sum(axis=1).astype(np.float64)
    fp = (preds & (labels == 0)).sum(axis=1).astype(np.float64)
    fn = (~preds & (labels == 1)).sum(axis=1).astype(np.float64)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1_arr = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1_arr))
    return float(thresholds[best_idx]), float(f1_arr[best_idx])


# ============================================================
# TRAIN LOOP (paper-grade diagnostics)
# ============================================================

def train_edge_gnn(
    graphs,
    device="cpu",
    epochs=30,
    batch_size=32,
    lr=1e-3,
    hidden_dim=64,
    num_layers=2,
    dropout=0.2,
):

    print("\n========== FULL OVERFIT TRAINING ==========")

    # Sanitize once
    sanitize_graphs_inplace(graphs)

    # Normalize using ALL data (hackathon mode)
    graphs, _, _ = normalize_edge_features(graphs, graphs, graphs)

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    node_feat_dim = graphs[0].x.shape[1]
    edge_feat_dim = graphs[0].edge_attr.shape[1]

    model = EdgeGNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    # Compute imbalance from ALL data
    labels_all = torch.cat([g.y for g in graphs])
    pos = (labels_all == 1).sum().item()
    neg = (labels_all == 0).sum().item()

    print(f"[TRAINING ON ALL DATA] Pos={pos} | Neg={neg}")

    if pos == 0:
        pos_weight = torch.tensor(1.0).to(device)
    else:
        pos_weight = torch.tensor(neg / pos).to(device)

    print(f"Using pos_weight={pos_weight.item():.4f}")

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"\nDevice: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print("===========================================\n")

    for epoch in range(1, epochs + 1):

        model.train()
        running_loss = 0
        grad_norms = []
        all_probs = []
        all_labels = []

        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(device)

            # batch.x / edge_attr already sanitized at data-prep time

            optimizer.zero_grad()

            logits = model(batch)
            logits = tensor_make_finite_(logits)  # model output may have NaN

            loss = criterion(logits, batch.y.float())

            if torch.isnan(loss):
                print("NaN LOSS DETECTED — STOPPING")
                return model

            loss.backward()

            # clip_grad_norm_ returns the total norm before clipping
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            grad_norms.append(total_norm.item())

            optimizer.step()

            running_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu())
            all_labels.append(batch.y.detach().cpu())

        # ===== Epoch Metrics =====

        probs = torch.cat(all_probs).numpy()
        labels = torch.cat(all_labels).numpy()

        pr_auc = average_precision_score(labels, probs)
        roc_auc = roc_auc_score(labels, probs)

        preds = (probs >= 0.5).astype(int)
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        cm = confusion_matrix(labels, preds)

        print(f"\nEpoch {epoch:03d}")
        print(f"Loss: {running_loss / len(loader):.6f}")
        print(f"PR-AUC: {pr_auc:.6f}")
        print(f"ROC-AUC: {roc_auc:.6f}")
        print(f"F1: {f1:.6f}")
        print(f"Precision: {precision:.6f}")
        print(f"Recall: {recall:.6f}")
        print("Confusion Matrix:")
        print(cm)
        print(f"Avg Grad Norm: {np.mean(grad_norms):.6f}")

    print("\n========== OVERFIT TRAINING COMPLETE ==========\n")

    return model

# ============================================================
# MAIN (loads csv, builds graphs, runs training)
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train EdgeGNN on sliding-window graphs")
    parser.add_argument("--packets", required=True, help="Path to ssdp_packets_rich.csv")
    parser.add_argument("--labels", required=True, help="Path to SSDP_Flood_labels.csv")
    parser.add_argument("--window-size", type=float, default=1.0)
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading CSV files...")
    packets = pd.read_csv(args.packets)
    labels = pd.read_csv(args.labels)

    labels = labels.rename(columns={"Unnamed: 0": "packet_index", "x": "label"})
    packets = packets.merge(labels, on="packet_index", how="left")
    packets["label"] = packets["label"].fillna(0).astype(int)

    raw_pos = int((packets["label"] == 1).sum())
    raw_neg = int((packets["label"] == 0).sum())
    print(f"[RAW PACKET LABELS] pos={raw_pos} neg={raw_neg} pos_rate={raw_pos/(raw_pos+raw_neg+1e-9):.8f}")

    print("Building sliding window graphs...")
    graphs = build_sliding_window_graphs(
        packets,
        window_size=args.window_size,
        stride=args.stride,
        bytes_col="packet_length",
        label_col="label",
    )

    print(f"Built {len(graphs)} graphs with window_size={args.window_size}, stride={args.stride}")

    model = train_edge_gnn(
        graphs,
        device=DEVICE,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
