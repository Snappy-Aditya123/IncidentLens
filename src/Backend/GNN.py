# ──────────────────────────────────────────────
# DEPRECATED: This file is a standalone duplicate of EdgeGNN.
# The canonical implementation lives in  train.py.
# This file is kept only for reference and will be removed.
# ──────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class EdgeGNN(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(node_feat_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.dropout = dropout

        # Edge classifier
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # ---- Node embedding ----
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- Edge classification ----
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]

        edge_input = torch.cat([h_src, h_dst, edge_attr], dim=1)
        logits = self.edge_mlp(edge_input).squeeze(-1)

        return logits
