import pandas as pd
from graph import build_snapshot_dataset

def load_graph_dataset(packet_csv, label_csv, delta_t=5.0):

    packets_df = pd.read_csv(packet_csv)
    labels_df = pd.read_csv(label_csv)

    # Rename columns properly
    labels_df = labels_df.rename(columns={
        "Unnamed: 0": "packet_index",
        "x": "label"
    })

    # Merge on packet_index
    packets_df = packets_df.merge(
        labels_df,
        on="packet_index",
        how="left"
    )

    # Fill missing labels as 0
    packets_df["label"] = packets_df["label"].fillna(0)

    data_list, node_map, flows_df = build_snapshot_dataset(
        packets_df,
        delta_t=delta_t,
        ts_col="timestamp",
        src_col="src_ip",
        dst_col="dst_ip",
        proto_col="protocol",
    )
    print("Packet-level label distribution:")
    print(packets_df["label"].value_counts())

    return data_list


import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data


# ===============================
# SLIDING WINDOW GRAPH BUILDER
# ===============================

def build_sliding_window_graphs(
    packets_df,
    window_size=2.0,
    stride=1.0,
    bytes_col="packet_length",
    label_col="label",
):
    graphs = []

    t_min = packets_df["timestamp"].min()
    t_max = packets_df["timestamp"].max()

    current_start = t_min

    while current_start + window_size <= t_max:
        current_end = current_start + window_size

        window_df = packets_df[
            (packets_df["timestamp"] >= current_start)
            & (packets_df["timestamp"] < current_end)
        ]

        if len(window_df) == 0:
            current_start += stride
            continue

        # -------- FLOW AGGREGATION --------
        keys = ["src_ip", "dst_ip", "protocol", "dst_port"]

        grouped = window_df.groupby(keys).agg(
            packet_count=("timestamp", "count"),
            total_bytes=(bytes_col, "sum"),
            mean_payload=("payload_length", "mean"),
            mean_iat=("timestamp", lambda x: x.diff().fillna(0).mean()),
            std_iat=("timestamp", lambda x: x.diff().fillna(0).std()),
            flow_label=(label_col, lambda x: x.mode()[0])
        ).reset_index()

        grouped["std_iat"] = grouped["std_iat"].fillna(0.0)

        # -------- NODE MAPPING --------
        ips = pd.concat([grouped["src_ip"], grouped["dst_ip"]]).unique()
        node_map = {ip: i for i, ip in enumerate(ips)}

        src_ids = grouped["src_ip"].map(node_map).values
        dst_ids = grouped["dst_ip"].map(node_map).values

        edge_index = torch.from_numpy(
            np.vstack((src_ids, dst_ids))
        ).long()

        edge_attr = torch.tensor(
            grouped[
                ["packet_count", "total_bytes",
                 "mean_payload", "mean_iat", "std_iat"]
            ].values,
            dtype=torch.float
        )

        y = torch.tensor(grouped["flow_label"].values, dtype=torch.long)

        data = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=len(node_map)
        )

        data.window_start = current_start
        graphs.append(data)

        current_start += stride

    return graphs


# ===============================
# ANALYSIS FUNCTION
# ===============================

def analyze_graphs(graphs):

    total_edges = 0
    total_nodes = 0
    all_labels = []

    for g in graphs:
        total_edges += g.edge_index.shape[1]
        total_nodes += g.num_nodes
        all_labels.append(g.y)

    all_labels = torch.cat(all_labels)

    num_graphs = len(graphs)
    num_edges = total_edges
    num_nodes_avg = total_nodes / num_graphs

    num_classes = torch.unique(all_labels)

    print("\n====== GRAPH ANALYSIS ======")
    print("Number of graphs:", num_graphs)
    print("Total edges:", num_edges)
    print("Average edges per graph:", num_edges / num_graphs)
    print("Average nodes per graph:", num_nodes_avg)

    print("Unique classes:", num_classes.tolist())

    for c in num_classes:
        count = (all_labels == c).sum().item()
        print(f"Class {int(c)} count:", count,
              f"({100*count/len(all_labels):.2f}%)")

    print("============================\n")
