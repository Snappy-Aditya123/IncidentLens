"""
IncidentLens ingestion + counterfactual analysis pipeline.

Loads NDJSON data (produced by ``csv_to_json.py``), builds temporal
graphs, indexes flows + embeddings into Elasticsearch, and runs
counterfactual analysis using the wrappers in ``wrappers.py``.

Usage:
    # Step 1 — convert CSV → JSON  (run once)
    python csv_to_json.py

    # Step 2 — ingest + analyse
    python ingest_pipeline.py

    # Or just a quick test on a small subset
    python ingest_pipeline.py --max-rows 10000 --skip-graphs
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# single-module API — all graph building, ES indexing, and analysis
import src.Backend.wrappers as wrappers

# Populate graph cache for graph-level CF tools
from src.Backend.agent_tools import set_graph_cache

logger = logging.getLogger(__name__)

# Project root: IncidentLens/ (3 levels up from src/Backend/ingest_pipeline.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = str(_PROJECT_ROOT / "data")


# ===================================================================
# 0. LOAD PRETRAINED TEMPORAL GNN (if checkpoint exists)
# ===================================================================

def _try_load_temporal_gnn(checkpoint_path: str | None = None) -> bool:
    """Attempt to load a pretrained TemporalGNNEncoder and register it.

    Returns True if a model was loaded, False otherwise (falls back to
    random-projection embeddings).
    """
    from src.Backend.temporal_gnn import TemporalGNNEncoder, get_default_checkpoint

    path = Path(checkpoint_path) if checkpoint_path else get_default_checkpoint()
    if not path.exists():
        logger.info(
            "No pretrained GNN checkpoint at %s — using random-projection fallback. "
            "Run 'python main.py train' to train a model first.",
            path,
        )
        return False

    try:
        encoder = TemporalGNNEncoder.from_checkpoint(path)
        wrappers.set_gnn_encoder(encoder)
        print(f"  [GNN] Loaded pretrained TemporalGNN from {path}")
        return True
    except Exception as exc:
        logger.warning("Failed to load GNN checkpoint %s: %s", path, exc)
        return False


# ===================================================================
# 1. LOAD NDJSON DATA
# ===================================================================

def load_ndjson_files(
    data_dir: str = DATA_DIR,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Read all NDJSON chunk files from ``data_dir`` into a single DataFrame.

    Parameters
    ----------
    data_dir : directory containing ``packets_*.json`` files and ``metadata.json``
    max_rows : optional cap on total rows loaded (for quick testing)
    """
    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"metadata.json not found in {data_dir}. "
            "Run csv_to_json.py first to generate NDJSON data."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    files = sorted(
        [os.path.join(data_dir, fn) for fn in meta["files"]],
    )
    print(f"Found {len(files)} NDJSON files, {meta['total_rows']:,} total rows")

    # Use pd.read_json(lines=True) — C-level parser, 5-20x faster than
    # line-by-line json.loads()
    chunks: list[pd.DataFrame] = []
    rows_loaded = 0
    for fpath in files:
        chunk = pd.read_json(fpath, lines=True)
        if max_rows and rows_loaded + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - rows_loaded]
        chunks.append(chunk)
        rows_loaded += len(chunk)
        if max_rows and rows_loaded >= max_rows:
            break

    df = pd.concat(chunks, ignore_index=True)

    # pd.read_json auto-parses Unix-epoch floats as datetime64[ns].
    # Convert back to float64 so downstream code (ES indexing, graph
    # building sliding-window math) sees numeric seconds.
    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = df["timestamp"].astype("int64") / 1e9

    print(f"Loaded {len(df):,} rows, columns: {df.columns.tolist()}")
    return df


# ===================================================================
# 2. RAW PACKET INDEXING (individual packets → ES)
# ===================================================================

# Index name for raw packets — shared constant from wrappers
RAW_PACKETS_INDEX = wrappers.RAW_PACKETS_INDEX

RAW_PACKETS_MAPPING = {
    "mappings": {
        "properties": {
            "packet_index":         {"type": "integer"},
            "timestamp":            {"type": "double"},
            "inter_arrival_time":   {"type": "float"},
            "src_ip":               {"type": "ip"},
            "dst_ip":               {"type": "ip"},
            "src_port":             {"type": "integer"},
            "dst_port":             {"type": "integer"},
            "protocol":             {"type": "integer"},
            "ttl":                  {"type": "integer"},
            "ip_header_len":        {"type": "integer"},
            "tcp_flags":            {"type": "float"},
            "udp_length":           {"type": "float"},
            "payload_length":       {"type": "integer"},
            "packet_length":        {"type": "integer"},
            "label":                {"type": "integer"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}


def setup_raw_packet_index(es=None, delete_existing: bool = False) -> bool:
    """Create the raw-packets index."""
    es = es or wrappers.get_client()
    return wrappers.create_index(
        RAW_PACKETS_INDEX, RAW_PACKETS_MAPPING, es, delete_existing
    )


def _clean_doc(doc: dict) -> dict:
    """Replace NaN / inf with None for JSON-safe ES indexing."""
    clean = {}
    for k, v in doc.items():
        if v is None:
            clean[k] = None
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[k] = None
        else:
            clean[k] = v
    return clean


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN/inf at the DataFrame level (vectorised, much faster
    than per-row _clean_doc for large datasets)."""
    df = df.copy()
    # Replace inf/-inf with NaN first
    df = df.replace([np.inf, -np.inf], np.nan)
    # Convert float columns to object dtype so NaN → None sticks
    # (.where with other=None on float cols casts None back to NaN)
    for col in df.columns:
        if df[col].dtype.kind == 'f':          # float columns
            df[col] = df[col].astype(object)
    # Now .where correctly sets NaN positions to None
    return df.where(df.notna(), other=None)


def index_raw_packets(
    df: pd.DataFrame,
    es=None,
    batch_size: int = 2000,
) -> int:
    """Bulk-index raw packet rows into ES.

    Returns total successfully indexed.
    """
    from elasticsearch import helpers

    es = es or wrappers.get_client()
    total_ok = 0
    n = len(df)

    # Vectorised NaN/inf cleaning at DataFrame level (much faster than per-row)
    clean_df = _clean_dataframe(df)
    records = clean_df.to_dict(orient="records")

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        actions = [
            {
                "_index": RAW_PACKETS_INDEX,
                "_id": str(rec["packet_index"]) if "packet_index" in rec else str(start + i),
                "_source": rec,
            }
            for i, rec in enumerate(records[start:end])
        ]
        ok, errors = helpers.bulk(es, actions, chunk_size=batch_size, raise_on_error=False)
        total_ok += ok
        if errors:
            logger.warning("Batch %d-%d: %d errors", start, end, len(errors))

        if (start // batch_size) % 20 == 0:
            print(f"  Indexed {total_ok:,} / {n:,} raw packets ...", flush=True)

    print(f"  Raw packets indexed: {total_ok:,} / {n:,}")
    return total_ok


# ===================================================================
# 3. GRAPH BUILDING + FLOW INDEXING
# ===================================================================

def build_graphs_from_df(
    df: pd.DataFrame,
    window_size: float = 2.0,
    stride: float = 1.0,
) -> tuple[list[Data], dict[int, str]]:
    """Build sliding-window graphs via wrappers (graph_data_wrapper)."""
    print(f"\nBuilding sliding-window graphs (window={window_size}s, stride={stride}s) ...")
    graphs, id_to_ip = wrappers.build_graphs(
        df,
        window_size=window_size,
        stride=stride,
        bytes_col="packet_length",
        label_col="label",
    )
    print(f"  Built {len(graphs)} graph snapshots")
    return graphs, id_to_ip


def index_all_graphs(
    graphs: list[Data],
    id_to_ip: dict[int, str],
    es=None,
) -> int:
    """Index all graph snapshots as aggregated flows into ES."""
    es = es or wrappers.get_client()
    print(f"\nIndexing {len(graphs)} graph snapshots into ES ...")
    total = wrappers.index_graphs_bulk(graphs, id_to_ip, es=es)
    print(f"  Flow docs indexed: {total:,}")
    return total


# ===================================================================
# 4. SYNTHETIC EMBEDDINGS (via wrappers.generate_embeddings)
# ===================================================================

def generate_feature_embeddings(
    graphs: list[Data],
    id_to_ip: dict[int, str],
    embedding_dim: int = 16,
) -> tuple[list[str], np.ndarray, list[int]]:
    """Generate embeddings via wrappers (graph_data_wrapper graphs)."""
    flow_ids, embeddings, labels = wrappers.generate_embeddings(
        graphs, id_to_ip, embedding_dim=embedding_dim,
    )
    print(f"  Generated {len(flow_ids)} embeddings (dim={embedding_dim})")
    return flow_ids, embeddings, labels


# ===================================================================
# 5. COUNTERFACTUAL ANALYSIS
# ===================================================================

def run_counterfactual_analysis(
    flow_ids: list[str],
    embeddings: np.ndarray,
    labels: list[int],
    max_counterfactuals: int = 20,
    es=None,
) -> list[dict]:
    """For each malicious flow (up to ``max_counterfactuals``), find
    the nearest normal neighbour and index the counterfactual diff.

    Returns a list of counterfactual result dicts.
    """
    es = es or wrappers.get_client()

    # find malicious flow indices
    mal_indices = [i for i, lbl in enumerate(labels) if lbl == 1]
    if not mal_indices:
        print("  No malicious flows found — skipping CF analysis.")
        return []

    n_cf = min(len(mal_indices), max_counterfactuals)
    print(f"\nRunning counterfactual analysis on {n_cf} malicious flows ...")

    results: list[dict] = []
    for idx in mal_indices[:n_cf]:
        fid = flow_ids[idx]
        emb = embeddings[idx]

        cf = wrappers.build_and_index_counterfactual(
            anomalous_flow_id=fid,
            query_embedding=emb,
            es=es,
            k=1,
        )
        if cf:
            results.append(cf)

    print(f"  Indexed {len(results)} counterfactual explanations")
    return results


def print_counterfactual_report(results: list[dict], max_show: int = 5) -> None:
    """Pretty-print a summary of counterfactual results."""
    if not results:
        print("\nNo counterfactuals to report.")
        return

    print(f"\n{'='*60}")
    print(f"COUNTERFACTUAL ANALYSIS REPORT ({len(results)} total)")
    print(f"{'='*60}")

    for i, cf in enumerate(results[:max_show]):
        print(f"\n--- CF #{i+1}: flow {cf['flow_id']} ---")
        print(f"    Nearest normal: {cf['nearest_normal_id']}")
        print(f"    Similarity:     {cf['similarity_score']:.4f}")
        print(f"    Feature diffs:")
        for d in cf["feature_diffs"]:
            print(
                f"      {d['feature']:20s}: {d['original_value']:12.2f} -> "
                f"{d['cf_value']:12.2f}  ({d['direction']}, {d['pct_change']:.1f}%)"
            )

    if len(results) > max_show:
        print(f"\n  ... and {len(results) - max_show} more.")


# ===================================================================
# 6. FEATURE STATS + SIGNIFICANT TERMS
# ===================================================================

def run_feature_analysis(es=None) -> dict:
    """Run feature distribution and significant-terms analysis.

    Returns dict with stats and significant terms results.
    """
    es = es or wrappers.get_client()
    report: dict[str, Any] = {}

    # Feature stats
    print("\n--- Feature Stats by Label ---")
    stats = wrappers.feature_stats_by_label(es=es)
    report["feature_stats"] = stats
    for feat, by_label in stats.items():
        for lbl, s in by_label.items():
            print(f"  {feat:20s} [{lbl}]: avg={s['avg']:12.4f}, std={s['std_deviation']:12.4f}")

    # Percentiles for packet_count
    print("\n--- Packet Count Percentiles ---")
    pctls = wrappers.feature_percentiles_by_label("packet_count", es=es)
    report["packet_count_percentiles"] = pctls
    for lbl, vals in pctls.items():
        print(f"  {lbl}: {vals}")

    # Significant terms on src_ip (which IPs dominate attacks)
    print("\n--- Significant Source IPs in Attack Traffic ---")
    try:
        sig_src = wrappers.significant_terms_by_label("src_ip", foreground_label=1, es=es)
        report["significant_src_ips"] = sig_src
        for s in sig_src[:10]:
            print(f"  {s['term']:20s}  docs={s['doc_count']:6d}  bg={s['bg_count']:6d}  score={s['score']:.4f}")
    except Exception as e:
        print(f"  Skipped (field might need keyword type): {e}")
        report["significant_src_ips"] = []

    # Significant terms on dst_ip
    print("\n--- Significant Dest IPs in Attack Traffic ---")
    try:
        sig_dst = wrappers.significant_terms_by_label("dst_ip", foreground_label=1, es=es)
        report["significant_dst_ips"] = sig_dst
        for s in sig_dst[:10]:
            print(f"  {s['term']:20s}  docs={s['doc_count']:6d}  bg={s['bg_count']:6d}  score={s['score']:.4f}")
    except Exception as e:
        print(f"  Skipped: {e}")
        report["significant_dst_ips"] = []

    return report


# ===================================================================
# 7. FULL PIPELINE
# ===================================================================

def run_pipeline(
    data_dir: str = DATA_DIR,
    max_rows: int | None = None,
    skip_graphs: bool = False,
    skip_raw_index: bool = False,
    window_size: float = 2.0,
    stride: float = 1.0,
    embedding_dim: int = 16,
    max_counterfactuals: int = 20,
    delete_existing: bool = False,
) -> dict:
    """Run the full ingestion + analysis pipeline.

    Steps:
        1. Load NDJSON data
        1b. Load pretrained Temporal GNN (if checkpoint exists)
        2. Create ES indices
        3. Index raw packets (optional)
        4. Build temporal graphs
        5. Index aggregated flows
        6. Generate + index embeddings (using GNN if available)
        7. Run counterfactual analysis
        8. Run feature analysis
    """
    t0 = time.time()
    es = wrappers.get_client()
    report: dict[str, Any] = {}

    # --- Health check ---
    assert wrappers.ping(es), "Cannot reach Elasticsearch"
    h = wrappers.health_check(es)
    print(f"[OK] ES cluster: {h['status']}")

    # --- 1b. Load pretrained GNN ---
    has_gnn = _try_load_temporal_gnn()

    # --- 1. Load data ---
    print("\n[1/8] Loading NDJSON data ...")
    df = load_ndjson_files(data_dir, max_rows=max_rows)
    report["total_rows"] = len(df)

    # --- 2. Create indices ---
    print("\n[2/8] Setting up ES indices ...")
    # If a GNN is loaded, use its actual embedding dimension for the ES index
    if has_gnn:
        encoder = wrappers.get_gnn_encoder()
        if encoder is not None:
            embedding_dim = encoder.embedding_dim
            print(f"  [GNN] Using GNN embedding dim: {embedding_dim}")
    wrappers.setup_all_indices(es, embedding_dim=embedding_dim, delete_existing=delete_existing)
    setup_raw_packet_index(es, delete_existing=delete_existing)
    print("  Indices ready.")

    # --- 3. Raw packet indexing ---
    if not skip_raw_index:
        print("\n[3/8] Indexing raw packets ...")
        raw_count = index_raw_packets(df, es)
        report["raw_packets_indexed"] = raw_count
    else:
        print("\n[3/8] Skipping raw packet indexing.")
        report["raw_packets_indexed"] = 0

    # --- 4. Build graphs ---
    graphs = []
    id_to_ip: dict[int, str] = {}
    if not skip_graphs:
        print("\n[4/8] Building temporal graphs (via graph_data_wrapper) ...")
        graphs, id_to_ip = build_graphs_from_df(df, window_size, stride)
        wrappers.analyze_graph_dataset(graphs)
        report["num_graphs"] = len(graphs)
        # Populate in-memory graph cache for graph-level CF tools
        set_graph_cache(graphs, id_to_ip)

        # If a GNN is loaded, preload the graph sequence for batch inference
        if has_gnn:
            encoder = wrappers.get_gnn_encoder()
            if hasattr(encoder, "set_graph_sequence"):
                # Preprocess graphs for the temporal model (sanitize + norm)
                from src.Backend.temporal_gnn import (
                    sanitize_graph, recompute_node_features,
                    preprocess_graphs as preprocess_gnn_graphs,
                    apply_normalization,
                )
                gnn_graphs = [sanitize_graph(g.clone()) for g in graphs]
                gnn_graphs = [recompute_node_features(g) for g in gnn_graphs]
                if encoder.norm_stats is not None:
                    gnn_graphs = apply_normalization(gnn_graphs, encoder.norm_stats)
                gnn_graphs = preprocess_gnn_graphs(gnn_graphs)
                encoder.set_graph_sequence(gnn_graphs)
                print(f"  [GNN] Preloaded {len(gnn_graphs)} graphs for temporal inference")
    else:
        print("\n[4/8] Skipping graph building.")
        report["num_graphs"] = 0

    # --- 5. Index flows ---
    if graphs:
        print("\n[5/8] Indexing aggregated flows ...")
        flow_count = index_all_graphs(graphs, id_to_ip, es)
        report["flows_indexed"] = flow_count
        # Index graph-level summaries (spectral + structural metrics)
        try:
            n_summaries = wrappers.index_graph_summaries(graphs, es=es)
            print(f"  Graph summaries indexed: {n_summaries}")
            report["graph_summaries_indexed"] = n_summaries
        except Exception as exc:
            print(f"  [WARN] Graph summary indexing failed: {exc}")
            report["graph_summaries_indexed"] = 0
    else:
        print("\n[5/8] Skipping flow indexing (no graphs).")
        report["flows_indexed"] = 0
        report["graph_summaries_indexed"] = 0

    # --- 6. Embeddings ---
    flow_ids: list[str] = []
    embeddings = np.empty((0, embedding_dim))
    labels_list: list[int] = []

    if graphs:
        print("\n[6/8] Generating + indexing embeddings ...")
        flow_ids, embeddings, labels_list = generate_feature_embeddings(
            graphs, id_to_ip, embedding_dim
        )
        emb_count = wrappers.index_embeddings(flow_ids, embeddings, labels_list, es=es)
        report["embeddings_indexed"] = emb_count
    else:
        print("\n[6/8] Skipping embeddings (no graphs).")
        report["embeddings_indexed"] = 0

    # Let ES refresh before queries
    es.indices.refresh(index="incidentlens-*")
    time.sleep(1)

    # --- 7. Counterfactual analysis ---
    if flow_ids and any(l == 1 for l in labels_list):
        print("\n[7/8] Running counterfactual analysis ...")
        cf_results = run_counterfactual_analysis(
            flow_ids, embeddings, labels_list,
            max_counterfactuals=max_counterfactuals, es=es,
        )
        print_counterfactual_report(cf_results)
        report["counterfactuals"] = len(cf_results)
    else:
        print("\n[7/8] Skipping counterfactual analysis (no malicious flows or no embeddings).")
        report["counterfactuals"] = 0

    # --- 8. Feature analysis ---
    if report.get("flows_indexed", 0) > 0:
        print("\n[8/8] Running feature analysis ...")
        feat_report = run_feature_analysis(es=es)
        report["feature_analysis"] = feat_report
    else:
        print("\n[8/8] Skipping feature analysis (no flows indexed).")

    elapsed = time.time() - t0
    report["elapsed_seconds"] = round(elapsed, 1)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  Rows loaded:      {report['total_rows']:,}")
    print(f"  Raw packets:      {report['raw_packets_indexed']:,}")
    print(f"  Graphs built:     {report['num_graphs']}")
    print(f"  Flows indexed:    {report['flows_indexed']:,}")
    print(f"  Embeddings:       {report['embeddings_indexed']:,}")
    print(f"  Counterfactuals:  {report['counterfactuals']}")
    print(f"{'='*60}")

    return report


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IncidentLens ingestion + counterfactual analysis pipeline"
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR,
        help="Directory containing NDJSON files from csv_to_json.py",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Limit rows loaded (for quick testing)",
    )
    parser.add_argument(
        "--skip-graphs", action="store_true",
        help="Skip graph building + flow indexing (just index raw packets)",
    )
    parser.add_argument(
        "--skip-raw-index", action="store_true",
        help="Skip raw packet indexing",
    )
    parser.add_argument(
        "--window-size", type=float, default=2.0,
        help="Sliding window size in seconds",
    )
    parser.add_argument(
        "--stride", type=float, default=1.0,
        help="Sliding window stride in seconds",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=16,
        help="Embedding vector dimension",
    )
    parser.add_argument(
        "--max-cf", type=int, default=20,
        help="Max counterfactual explanations to generate",
    )
    parser.add_argument(
        "--delete-existing", action="store_true",
        help="Delete and recreate ES indices",
    )
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        max_rows=args.max_rows,
        skip_graphs=args.skip_graphs,
        skip_raw_index=args.skip_raw_index,
        window_size=args.window_size,
        stride=args.stride,
        embedding_dim=args.embedding_dim,
        max_counterfactuals=args.max_cf,
        delete_existing=args.delete_existing,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
