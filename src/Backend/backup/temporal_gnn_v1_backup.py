# ──────────────────────────────────────────────
# DEPRECATED: V1 backup of the Evolving-GCN implementation.
# The current version lives in  temporal_gnn.py.
# This file is kept only for reference and will be removed.
# ──────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class EvolvingGCNLayer(MessagePassing):
    """
    A Graph Convolution Layer where the weight matrix is provided dynamically 
    during the forward pass, rather than being a static parameter.
    """
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5)

    def forward(self, x, edge_index, weight):
        # x has shape [N, in_channels]
        # weight has shape [in_channels, out_channels] (dynamic)
        
        # 1. Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 2. Linear transformation using the DYNAMIC weight
        x = torch.matmul(x, weight)

        # 3. Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 4. Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        return norm.view(-1, 1) * x_j


class EvolvingGNN(nn.Module):
    """
    Semi-Temporal GNN where GNN weights are evolved by an LSTM.
    
    Architecture:
    - An LSTM takes the previous GNN weight matrix as input and predicts the new one.
    - The evolved weight matrix is used by the EvolvingGCNLayer to process the graph.
    - We process a sequence of graphs (window) and return the embeddings of the last one.
    """
    def __init__(self, input_dim, hidden_dim, edge_feat_dim, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # The GNN Layer (stateless/dynamic weights)
        self.gnn_layer = EvolvingGCNLayer()

        # The LSTM to evolve the weights
        # We treat the weight matrix columns as the "features" for the LSTM
        # Input size: input_dim (rows of W), Hidden size: input_dim (to match rows of W)
        # Note: In EvolveGCN-O, the LSTM is: W_t = LSTM(W_{t-1})
        self.weight_lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=input_dim, 
            batch_first=True
        )

        # Initial static weights (the "seed" for the evolution)
        # Shape: [input_dim, hidden_dim]
        # We wrap it in nn.Parameter so it's trainable
        self.initial_weights = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.initial_weights)

        # Edge Classifier (MLP)
        # Takes [Source Node Emb, Dest Node Emb, Edge Attr] -> [1] (logit)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, graph_sequence):
        """
        Args:
            graph_sequence: A list of PyG Data objects (length T).
                            Example: [G_{t-4}, G_{t-3}, ..., G_{t}]
        Returns:
            logits: Edge classification logits for the LAST graph in the sequence.
        """
        
        # Guard: empty sequence
        if not graph_sequence:
            raise ValueError("graph_sequence must contain at least one graph")
        
        # 1. Prepare "Static" Initial Weight Matrix
        # [input_dim, hidden_dim]
        current_weight = self.initial_weights
        
        # LSTM hidden states (h_0, c_0)
        # We initialize them to zeros for each sequence
        h_state = None 

        final_embeddings = None
        final_graph = None

        # 2. Iterate through the sequence of graphs
        for i, graph in enumerate(graph_sequence):
            # --- EVOLVE WEIGHTS ---
            # We reshape weights to be compatible with LSTM: [Batch=hidden_dim, Seq=1, Features=input_dim]
            # We are treating each output channel (column of W) as an independent sequence
            
            # Transpose: [hidden_dim, input_dim] -> Treated as Batch=hidden_dim, Feat=input_dim
            w_input = current_weight.t().unsqueeze(1) 
            
            # LSTM Step: Evolve the weights
            out, h_state = self.weight_lstm(w_input, h_state)
            
            # Reshape back to [input_dim, hidden_dim]
            # out shape: [hidden_dim, 1, input_dim]
            current_weight = out.squeeze(1).t()
            
            # --- APPLY GNN ---
            x, edge_index = graph.x, graph.edge_index
            
            # Ensure features match input_dim
            if x.shape[1] != self.input_dim:
                raise ValueError(f"Graph feature dim {x.shape[1]} doesn't match model input dim {self.input_dim}")

            # Convolution with evolved weights
            x = self.gnn_layer(x, edge_index, current_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if i == len(graph_sequence) - 1:
                final_embeddings = x
                final_graph = graph

        # 3. Edge Classification on the LAST graph
        # We only care about predicting anomalies in the most recent snapshot
        src, dst = final_graph.edge_index
        h_src = final_embeddings[src]
        h_dst = final_embeddings[dst]
        edge_attr = final_graph.edge_attr
        
        # Concatenate: [Source Emb, Dest Emb, Edge Features]
        edge_input = torch.cat([h_src, h_dst, edge_attr], dim=1)
        
        # Apply MLP
        logits = self.edge_mlp(edge_input).squeeze(-1)
        
        return logits
