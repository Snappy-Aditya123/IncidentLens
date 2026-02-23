import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from graph import network, node, build_snapshot_dataset


# ===============================
# CSV LOADER (uses graph.py pipeline)
# ===============================

def load_graph_dataset(packet_csv: str, label_csv: str, delta_t: float = 5.0):

    packets_df = pd.read_csv(packet_csv)
    labels_df = pd.read_csv(label_csv)

    labels_df = labels_df.rename(columns={
        "Unnamed: 0": "packet_index",
        "x": "label",
    })

    packets_df = packets_df.merge(labels_df, on="packet_index", how="left")
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


# ===============================
# VECTORISED HELPERS
# ===============================

def _assign_window_ids(
    timestamps: np.ndarray,
    t_min: float, t_max: float,
    window_size: float, stride: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each timestamp to its overlapping windows.

    For small W: broadcast (N, W) bool mask.
    For large W: searchsorted + range expansion (avoids O(N*W) memory).
    Returns (pkt_indices, win_indices) parallel arrays and window_starts.
    """
    window_starts = np.arange(t_min, t_max - window_size + 1e-9, stride)
    if len(window_starts) == 0:
        window_starts = np.array([t_min])
    W = len(window_starts)
    N = len(timestamps)

    # Choose strategy based on matrix size
    if N * W <= 5_000_000:
        # Small enough for broadcast
        ws = window_starts[np.newaxis, :]        # (1, W)
        ts = timestamps[:, np.newaxis]           # (N, 1)
        mask = (ts >= ws) & (ts < ws + window_size)
        return mask, window_starts
    else:
        # Large: use searchsorted to find valid window range per packet
        # For each packet, find the range of windows it belongs to
        # Window w contains packet if: window_starts[w] <= t < window_starts[w] + window_size
        # Equivalently: t - window_size < window_starts[w] <= t
        low = np.searchsorted(window_starts, timestamps - window_size, side="right")
        high = np.searchsorted(window_starts, timestamps, side="right")
        # Counts per packet
        counts = high - low
        total = int(counts.sum())
        pkt_idx = np.repeat(np.arange(N), counts)
        # Build win_idx: concatenate range(low[i], high[i]) for each i
        # Vectorised: offsets within each packet's range
        offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        win_idx = np.repeat(low, counts) + offsets
        return (pkt_idx, win_idx), window_starts


def _aggregate_flows_numpy(
    win_ids: np.ndarray,
    src_codes: np.ndarray,
    dst_codes: np.ndarray,
    proto_codes: np.ndarray,
    port_codes: np.ndarray,
    timestamps: np.ndarray,
    bytes_vals: np.ndarray,
    payload_vals: np.ndarray,
    label_vals: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Pure-numpy flow aggregation — avoids pandas groupby entirely.

    Returns parallel arrays: (flow_win, flow_src, flow_dst, flow_proto,
        flow_port, packet_count, total_bytes, mean_payload, mean_iat,
        std_iat, flow_label) all length F (number of unique flows).
    """
    n = len(win_ids)
    if n == 0:
        empty = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float32)
        return (empty, empty, empty, empty, empty,
                empty_f, empty_f, empty_f, empty_f, empty_f, empty)

    # Encode 5 groupby keys into a single int64 composite key for fast sort.
    # Bit allocation: win(16) | src(16) | dst(16) | proto(8) | port(8) = 64 bits.
    # Guard: if any dimension overflows its bit-width, fall back to
    # wider packing (split into two keys) — keeps correctness for large datasets.
    max_win = int(win_ids.max()) if n > 0 else 0
    max_src = int(src_codes.max()) if n > 0 else 0
    max_dst = int(dst_codes.max()) if n > 0 else 0
    max_proto = int(proto_codes.max()) if n > 0 else 0
    max_port = int(port_codes.max()) if n > 0 else 0

    if max_win < 65536 and max_src < 65536 and max_dst < 65536 and max_proto < 256 and max_port < 256:
        # Fast path: everything fits in 64 bits
        composite = (win_ids.astype(np.int64) << 48 |
                     src_codes.astype(np.int64) << 32 |
                     dst_codes.astype(np.int64) << 16 |
                     proto_codes.astype(np.int64) << 8 |
                     port_codes.astype(np.int64))
    else:
        # Safe path: use two int64 keys (handles arbitrarily large cardinalities)
        key_a = (win_ids.astype(np.int64) * (max_src + 1) + src_codes.astype(np.int64))
        key_b = (dst_codes.astype(np.int64) * (max_proto + 1) * (max_port + 1) +
                 proto_codes.astype(np.int64) * (max_port + 1) +
                 port_codes.astype(np.int64))
        # Combine into single key via unique pair mapping
        _, composite = np.unique(
            np.column_stack([key_a, key_b]), axis=0, return_inverse=True
        )
        # Remap pkt order to match composite sort
        composite = composite.astype(np.int64)

    # Sort by composite key then timestamp for IAT
    order = np.lexsort((timestamps, composite))
    comp_sorted = composite[order]
    ts = timestamps[order]; by = bytes_vals[order]
    pl = payload_vals[order]; lb = label_vals[order]

    # Detect group boundaries
    diff = np.empty(n, dtype=bool)
    diff[0] = True
    diff[1:] = comp_sorted[1:] != comp_sorted[:-1]
    group_ids = np.cumsum(diff) - 1
    n_groups = int(group_ids[-1]) + 1

    # --- aggregations via np.bincount (one pass each) ---
    pkt_count = np.bincount(group_ids, minlength=n_groups).astype(np.float32)
    total_bytes = np.bincount(group_ids, weights=by, minlength=n_groups).astype(np.float32)
    payload_sum = np.bincount(group_ids, weights=pl, minlength=n_groups).astype(np.float32)
    mean_payload = payload_sum / pkt_count

    # IAT: diff within each group, first packet = 0
    iat = np.empty(n, dtype=np.float64)
    iat[0] = 0.0
    iat[1:] = ts[1:] - ts[:-1]
    iat[diff] = 0.0  # first of each group → 0

    iat_sum = np.bincount(group_ids, weights=iat, minlength=n_groups)
    mean_iat = (iat_sum / pkt_count).astype(np.float32)

    # std_iat: sample std (ddof=1) to match pandas .std() default
    # Formula: sqrt( (sum(x²) - n*mean²) / (n-1) )
    iat_sq_sum = np.bincount(group_ids, weights=iat * iat, minlength=n_groups)
    denom = pkt_count - 1.0
    denom[denom < 1.0] = 1.0  # avoid /0 for single-packet flows
    variance = (iat_sq_sum - pkt_count * mean_iat.astype(np.float64) ** 2) / denom
    np.maximum(variance, 0.0, out=variance)  # numerical guard
    std_iat = np.sqrt(variance).astype(np.float32)
    # single-packet flows: std = NaN → 0 (matches pandas .fillna(0))
    std_iat[pkt_count == 1] = 0.0

    # label: max per group
    label_max = np.full(n_groups, -1, dtype=lb.dtype)
    np.maximum.at(label_max, group_ids, lb)

    # Extract first-row-per-group for key columns (use original arrays via order)
    first = np.nonzero(diff)[0]
    first_orig = order[first]  # indices into original input arrays
    flow_win = win_ids[first_orig]
    flow_src = src_codes[first_orig]
    flow_dst = dst_codes[first_orig]
    flow_proto = proto_codes[first_orig]
    flow_port = port_codes[first_orig]

    return (flow_win, flow_src, flow_dst, flow_proto, flow_port,
            pkt_count, total_bytes, mean_payload, mean_iat, std_iat,
            label_max)


def _compute_node_features_arrays(
    src_ids: np.ndarray,
    dst_ids: np.ndarray,
    total_bytes: np.ndarray,
    pkt_count: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Build (n_nodes, 6) node features from pre-extracted arrays.

    Features: [bytes_sent, bytes_recv, pkts_sent, pkts_recv,
               out_degree, in_degree]
    """
    feats = np.zeros((n_nodes, 6), dtype=np.float32)
    np.add.at(feats[:, 0], src_ids, total_bytes)
    np.add.at(feats[:, 1], dst_ids, total_bytes)
    np.add.at(feats[:, 2], src_ids, pkt_count)
    np.add.at(feats[:, 3], dst_ids, pkt_count)
    np.add.at(feats[:, 4], src_ids, 1)
    np.add.at(feats[:, 5], dst_ids, 1)
    return feats


def _build_network_fast(
    local_ids: np.ndarray,
    local_src: np.ndarray,
    local_dst: np.ndarray,
    id_to_ip: dict[int, str],
    node_feats: torch.Tensor,
) -> network:
    """Build a graph.network object without per-element cache invalidation.

    Pre-builds neighbor lists with numpy groupby, then assigns in bulk.
    """
    n_local = len(local_ids)
    n_edges = len(local_src)

    # Pre-build neighbor lists using numpy sort+split (no per-edge Python loop)
    out_order = np.argsort(local_src, kind="mergesort")
    out_splits = np.searchsorted(local_src[out_order], np.arange(n_local))
    in_order = np.argsort(local_dst, kind="mergesort")
    in_splits = np.searchsorted(local_dst[in_order], np.arange(n_local))

    out_targets = local_dst[out_order]  # destinations for each source
    in_sources = local_src[in_order]    # sources for each destination

    net = object.__new__(network)
    net.num_nodes = n_local
    net.device = torch.device("cpu")
    net._degree_cache = {}
    net.nodes = {}

    # Batch-create all nodes with pre-computed neighbor lists
    for lid in range(n_local):
        gid = int(local_ids[lid])
        n = object.__new__(node)
        n.IPaddress = id_to_ip[gid]
        n.node_id = lid
        n.features = node_feats[lid]
        # Slice neighbor lists from pre-sorted arrays
        out_lo = int(out_splits[lid])
        out_hi = int(out_splits[lid + 1]) if lid + 1 < n_local else n_edges
        n.out_neighbors = out_targets[out_lo:out_hi].tolist()
        in_lo = int(in_splits[lid])
        in_hi = int(in_splits[lid + 1]) if lid + 1 < n_local else n_edges
        n.in_neighbors = in_sources[in_lo:in_hi].tolist()
        net.nodes[lid] = n

    # Pre-build edge_index cache so later queries are free
    ei = torch.from_numpy(np.vstack((local_src, local_dst)).astype(np.int64))
    net._edge_index_cache = ei
    net.x = node_feats
    return net


# ===============================
# SLIDING WINDOW GRAPH BUILDER
# Builds graph.network per window → PyG Data with node features
# ===============================

def build_sliding_window_graphs(
    packets_df: pd.DataFrame,
    window_size: float = 2.0,
    stride: float = 1.0,
    bytes_col: str = "packet_length",
    label_col: str = "label",
) -> list[Data]:

    ts_vals = packets_df["timestamp"].values.astype(np.float64)
    t_min, t_max = ts_vals.min(), ts_vals.max()

    # ---- vectorised window assignment ----
    result, window_starts = _assign_window_ids(ts_vals, t_min, t_max,
                                               window_size, stride)
    if isinstance(result, tuple):
        # searchsorted path: already (pkt_idx, win_idx)
        pkt_idx, win_idx = result
    else:
        # broadcast mask path
        pkt_idx, win_idx = np.nonzero(result)
    if len(pkt_idx) == 0:
        return []

    # ---- encode IPs / protocol / port to integers ONCE (pd.factorize is ~20x faster than np.unique on strings) ----
    src_raw = packets_df["src_ip"].values
    dst_raw = packets_df["dst_ip"].values
    bytes_raw = packets_df[bytes_col].values.astype(np.float64)
    payload_raw = packets_df["payload_length"].values.astype(np.float64)
    label_raw = packets_df[label_col].values.astype(np.int64)

    # pd.factorize on concatenated src+dst gives consistent codes
    all_ips_arr = np.concatenate([src_raw, dst_raw])
    ip_codes_all, unique_ips = pd.factorize(all_ips_arr, sort=True)
    ip_to_id: dict[str, int] = {ip: int(i) for i, ip in enumerate(unique_ips)}
    id_to_ip: dict[int, str] = {int(i): str(ip) for i, ip in enumerate(unique_ips)}
    n_total = len(packets_df)
    src_codes_full = ip_codes_all[:n_total]
    dst_codes_full = ip_codes_all[n_total:]

    proto_codes_full, _ = pd.factorize(packets_df["protocol"].values, sort=True)
    port_codes_full, _ = pd.factorize(packets_df["dst_port"].values, sort=True)

    # ---- explode to per-(packet, window) rows using integer arrays only ----
    e_win = win_idx
    e_src = src_codes_full[pkt_idx]
    e_dst = dst_codes_full[pkt_idx]
    e_proto = proto_codes_full[pkt_idx]
    e_port = port_codes_full[pkt_idx]
    e_ts = ts_vals[pkt_idx]
    e_bytes = bytes_raw[pkt_idx]
    e_payload = payload_raw[pkt_idx]
    e_label = label_raw[pkt_idx]

    # ---- pure-numpy flow aggregation ----
    (flow_win, flow_src, flow_dst, flow_proto, flow_port,
     pkt_count, total_bytes, mean_payload, mean_iat, std_iat,
     flow_label) = _aggregate_flows_numpy(
        e_win, e_src, e_dst, e_proto, e_port,
        e_ts, e_bytes, e_payload, e_label)

    # ---- edge feature + label arrays ----
    feat_arr = np.column_stack([pkt_count, total_bytes, mean_payload,
                                mean_iat, std_iat]).astype(np.float32)
    label_arr = flow_label.astype(np.int64)
    src_arr = flow_src.astype(np.int64)
    dst_arr = flow_dst.astype(np.int64)
    wid_arr = flow_win

    # ---- split indices by window_id (already sorted by win from aggregation) ----
    order = np.argsort(wid_arr, kind="mergesort")
    wid_sorted = wid_arr[order]
    split_pts = np.searchsorted(wid_sorted,
                                np.arange(wid_sorted[-1] + 1),
                                side="left")
    split_pts = np.append(split_pts, len(wid_sorted))
    num_global = len(unique_ips)

    # ---- per-window: build network + node features → PyG Data ----
    graphs: list[Data] = []

    for w in range(len(split_pts) - 1):
        lo, hi = split_pts[w], split_pts[w + 1]
        if lo == hi:
            continue
        idx = order[lo:hi]

        w_src = src_arr[idx]
        w_dst = dst_arr[idx]
        w_bytes = total_bytes[idx]
        w_pkts = pkt_count[idx]

        # Preserve first-occurrence order (src then dst) to match original
        combined = np.concatenate([w_src, w_dst])
        _, first_occ = np.unique(combined, return_index=True)
        local_ids = combined[np.sort(first_occ)]
        n_local = len(local_ids)

        # Remap global IDs → local 0..n-1 via lookup array (no dict)
        remap = np.empty(num_global, dtype=np.int64)
        remap[local_ids] = np.arange(n_local, dtype=np.int64)
        local_src = remap[w_src]
        local_dst = remap[w_dst]

        # Node features directly on local size (no global alloc + slice)
        node_feats_np = _compute_node_features_arrays(
            local_src, local_dst, w_bytes, w_pkts, n_local)
        node_feats = torch.from_numpy(node_feats_np)

        # Build network object (fast path — no per-element cache clears)
        net = _build_network_fast(local_ids, local_src, local_dst,
                                  id_to_ip, node_feats)

        # Build PyG Data directly (skip net.to_pyg_data overhead)
        ei = net._edge_index_cache
        data = Data(edge_index=ei, x=node_feats, num_nodes=n_local)
        data.edge_attr = torch.from_numpy(feat_arr[idx])
        data.y = torch.from_numpy(label_arr[idx])
        data.window_start = float(window_starts[w]) if w < len(window_starts) else 0.0
        data.network = net
        graphs.append(data)

    return graphs


# ===============================
# ANALYSIS (vectorised)
# ===============================

def analyze_graphs(graphs: list[Data]) -> None:
    if not graphs:
        print("No graphs to analyse.")
        return

    edge_counts = np.array([g.edge_index.shape[1] for g in graphs])
    node_counts = np.array([g.num_nodes for g in graphs])
    all_labels = torch.cat([g.y for g in graphs])

    num_graphs = len(graphs)
    classes = torch.unique(all_labels)
    total = len(all_labels)

    print("\n====== GRAPH ANALYSIS ======")
    print("Number of graphs:", num_graphs)
    print("Total edges:", int(edge_counts.sum()))
    print("Average edges per graph:", f"{edge_counts.mean():.1f}")
    print("Average nodes per graph:", f"{node_counts.mean():.1f}")
    print("Node feature dim:", graphs[0].x.shape[1] if graphs[0].x is not None else 0)
    print("Edge feature dim:", graphs[0].edge_attr.shape[1])
    print("Unique classes:", classes.tolist())

    for c in classes:
        count = (all_labels == c).sum().item()
        print(f"Class {int(c)} count: {count} ({100*count/total:.2f}%)")

    print("============================\n")


# ===============================
# GRAPH-LEVEL COUNTERFACTUAL TOOLS
# ===============================

def edge_perturbation_counterfactual(
    graph: Data,
    target_edge_indices: list[int] | None = None,
    max_removals: int = 5,
) -> list[dict]:
    """Compute graph-level counterfactuals via edge perturbation.

    For each edge removed, measures the change in graph-level statistics
    (mean edge features, node degree distribution) to identify which
    connections have the greatest structural impact on the anomaly.

    Parameters
    ----------
    graph : PyG Data with edge_index, edge_attr, y
    target_edge_indices : specific edges to test (default: top malicious)
    max_removals : max edges to perturb

    Returns
    -------
    list of dicts with keys: edge_idx, src, dst, removed_label,
    feature_impact (dict of feature -> delta), structural_impact (float)
    """
    ei = graph.edge_index.cpu().numpy()
    ea = graph.edge_attr.cpu().numpy() if graph.edge_attr is not None else None
    labels = graph.y.cpu().numpy() if graph.y is not None else None
    n_edges = ei.shape[1]

    if ea is None or n_edges == 0:
        return []

    # Baseline graph statistics
    baseline_mean = ea.mean(axis=0)
    baseline_degree = np.bincount(ei[0], minlength=graph.num_nodes) + \
                      np.bincount(ei[1], minlength=graph.num_nodes)
    baseline_degree_std = float(baseline_degree.std())

    # Select target edges (prefer malicious-labeled edges)
    if target_edge_indices is not None:
        targets = target_edge_indices[:max_removals]
    elif labels is not None:
        mal_mask = labels == 1
        mal_idx = np.where(mal_mask)[0]
        if len(mal_idx) > 0:
            # Sort by edge feature magnitude (total_bytes) descending
            magnitudes = ea[mal_idx, 1] if ea.shape[1] > 1 else ea[mal_idx, 0]
            top_order = np.argsort(-magnitudes)
            targets = mal_idx[top_order[:max_removals]].tolist()
        else:
            # No malicious edges; pick edges with highest total_bytes
            magnitudes = ea[:, 1] if ea.shape[1] > 1 else ea[:, 0]
            top_order = np.argsort(-magnitudes)
            targets = top_order[:max_removals].tolist()
    else:
        targets = list(range(min(max_removals, n_edges)))

    feat_names = ["packet_count", "total_bytes", "mean_payload", "mean_iat", "std_iat"]
    results = []

    for edge_idx in targets:
        # Remove this edge and recompute stats
        keep_mask = np.ones(n_edges, dtype=bool)
        keep_mask[edge_idx] = False
        new_ea = ea[keep_mask]

        if len(new_ea) == 0:
            continue

        new_mean = new_ea.mean(axis=0)
        delta = new_mean - baseline_mean

        new_ei = ei[:, keep_mask]
        new_degree = np.bincount(new_ei[0], minlength=graph.num_nodes) + \
                     np.bincount(new_ei[1], minlength=graph.num_nodes)
        new_degree_std = float(new_degree.std())

        feature_impact = {}
        for j, fname in enumerate(feat_names[:ea.shape[1]]):
            feature_impact[fname] = {
                "delta": round(float(delta[j]), 4),
                "pct_change": round(
                    abs(float(delta[j])) / max(abs(float(baseline_mean[j])), 1e-9) * 100, 2
                ),
            }

        structural_impact = abs(new_degree_std - baseline_degree_std) / max(baseline_degree_std, 1e-9)

        results.append({
            "edge_idx": int(edge_idx),
            "src": int(ei[0, edge_idx]),
            "dst": int(ei[1, edge_idx]),
            "removed_label": int(labels[edge_idx]) if labels is not None else 0,
            "edge_features": {
                fname: round(float(ea[edge_idx, j]), 4)
                for j, fname in enumerate(feat_names[:ea.shape[1]])
            },
            "feature_impact": feature_impact,
            "structural_impact": round(float(structural_impact), 4),
        })

    # Sort by structural impact descending
    results.sort(key=lambda r: r["structural_impact"], reverse=True)
    return results


def compare_graph_windows(
    graph_a: Data,
    graph_b: Data,
) -> dict:
    """Compare two graph snapshots (e.g., a malicious window vs a normal one).

    Computes structural and feature-level differences to identify what
    changed between a normal and anomalous time window.

    Returns
    -------
    dict with keys: node_diff, edge_diff, feature_diffs,
    degree_distribution_shift, label_distribution_shift
    """
    ei_a = graph_a.edge_index.cpu().numpy()
    ei_b = graph_b.edge_index.cpu().numpy()
    ea_a = graph_a.edge_attr.cpu().numpy() if graph_a.edge_attr is not None else None
    ea_b = graph_b.edge_attr.cpu().numpy() if graph_b.edge_attr is not None else None
    y_a = graph_a.y.cpu().numpy() if graph_a.y is not None else np.zeros(ei_a.shape[1])
    y_b = graph_b.y.cpu().numpy() if graph_b.y is not None else np.zeros(ei_b.shape[1])

    feat_names = ["packet_count", "total_bytes", "mean_payload", "mean_iat", "std_iat"]

    # Feature-level comparison
    feature_diffs = {}
    if ea_a is not None and ea_b is not None:
        mean_a = ea_a.mean(axis=0)
        mean_b = ea_b.mean(axis=0)
        for j, fname in enumerate(feat_names[:min(ea_a.shape[1], ea_b.shape[1])]):
            diff = float(mean_b[j] - mean_a[j])
            pct = abs(diff) / max(abs(float(mean_a[j])), 1e-9) * 100
            feature_diffs[fname] = {
                "window_a_mean": round(float(mean_a[j]), 4),
                "window_b_mean": round(float(mean_b[j]), 4),
                "delta": round(diff, 4),
                "pct_change": round(pct, 2),
                "direction": "increase" if diff > 0 else "decrease" if diff < 0 else "unchanged",
            }

    # Degree distribution shift
    deg_a = np.bincount(ei_a[0], minlength=max(graph_a.num_nodes, 1)) + \
            np.bincount(ei_a[1], minlength=max(graph_a.num_nodes, 1))
    deg_b = np.bincount(ei_b[0], minlength=max(graph_b.num_nodes, 1)) + \
            np.bincount(ei_b[1], minlength=max(graph_b.num_nodes, 1))

    # Label distribution
    label_dist_a = {
        "normal": int((y_a == 0).sum()),
        "malicious": int((y_a == 1).sum()),
    }
    label_dist_b = {
        "normal": int((y_b == 0).sum()),
        "malicious": int((y_b == 1).sum()),
    }

    return {
        "window_a": {
            "num_nodes": int(graph_a.num_nodes),
            "num_edges": int(ei_a.shape[1]),
            "window_start": float(getattr(graph_a, "window_start", 0)),
            "label_distribution": label_dist_a,
            "mean_degree": round(float(deg_a.mean()), 2),
        },
        "window_b": {
            "num_nodes": int(graph_b.num_nodes),
            "num_edges": int(ei_b.shape[1]),
            "window_start": float(getattr(graph_b, "window_start", 0)),
            "label_distribution": label_dist_b,
            "mean_degree": round(float(deg_b.mean()), 2),
        },
        "node_diff": int(graph_b.num_nodes) - int(graph_a.num_nodes),
        "edge_diff": int(ei_b.shape[1]) - int(ei_a.shape[1]),
        "feature_diffs": feature_diffs,
        "degree_shift": round(float(deg_b.mean()) - float(deg_a.mean()), 4),
        "label_distribution_shift": {
            "normal_delta": label_dist_b["normal"] - label_dist_a["normal"],
            "malicious_delta": label_dist_b["malicious"] - label_dist_a["malicious"],
        },
    }


def find_most_anomalous_window(
    graphs: list[Data],
) -> tuple[int, Data, dict]:
    """Find the graph window with the highest proportion of malicious edges.

    Returns (window_index, graph, stats_dict).
    """
    if not graphs:
        raise ValueError("No graphs provided")

    best_idx = 0
    best_score = -1.0
    stats_list = []

    for i, g in enumerate(graphs):
        y = g.y.cpu().numpy() if g.y is not None else np.zeros(g.edge_index.shape[1])
        n_edges = len(y)
        n_mal = int((y == 1).sum())
        ratio = n_mal / max(n_edges, 1)
        stats = {
            "window_index": i,
            "window_start": float(getattr(g, "window_start", 0)),
            "num_edges": n_edges,
            "num_malicious": n_mal,
            "malicious_ratio": round(ratio, 4),
        }
        stats_list.append(stats)
        if ratio > best_score:
            best_score = ratio
            best_idx = i

    return best_idx, graphs[best_idx], stats_list[best_idx]


def find_most_normal_window(
    graphs: list[Data],
) -> tuple[int, Data, dict]:
    """Find the graph window with the lowest proportion of malicious edges.

    Returns (window_index, graph, stats_dict).
    """
    if not graphs:
        raise ValueError("No graphs provided")

    best_idx = 0
    best_score = 2.0  # > 1.0 so any ratio wins

    for i, g in enumerate(graphs):
        y = g.y.cpu().numpy() if g.y is not None else np.zeros(g.edge_index.shape[1])
        n_edges = len(y)
        n_mal = int((y == 1).sum())
        ratio = n_mal / max(n_edges, 1)
        if ratio < best_score:
            best_score = ratio
            best_idx = i

    g = graphs[best_idx]
    y = g.y.cpu().numpy() if g.y is not None else np.zeros(g.edge_index.shape[1])
    stats = {
        "window_index": best_idx,
        "window_start": float(getattr(g, "window_start", 0)),
        "num_edges": len(y),
        "num_malicious": int((y == 1).sum()),
        "malicious_ratio": round(best_score, 4),
    }
    return best_idx, graphs[best_idx], stats
