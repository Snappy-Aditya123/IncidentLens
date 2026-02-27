from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

import src.Backend.wrappers as wrappers
from src.Backend.simulation import StreamSimulator
from src.Backend.temporal_gnn import TemporalGNNEncoder, get_default_checkpoint

logger = logging.getLogger(__name__)


# =========================
# Configuration
# =========================

@dataclass
class PipelineConfig:
    window_size: float = 5.0
    stride: float = 5.0
    embedding_dim: int = 16
    delete_existing_indices: bool = False

    source: str = "simulate"  # simulate | ndjson | es
    ndjson_path: str = "data/packets_sample.json"
    es_query_size: int = 5000

    print_top_anomalies: int = 5
    anomaly_threshold: float = 0.7

    checkpoint_path: Optional[str] = None


# =========================
# Data Loading
# =========================

def load_ndjson(path: str, max_rows: Optional[int] = None) -> List[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NDJSON not found: {p}")

    out = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_packets_from_es(es, size: int = 5000) -> List[dict]:
    idx = wrappers.RAW_PACKETS_INDEX
    if not es.indices.exists(index=idx):
        logger.warning("RAW_PACKETS_INDEX does not exist: %s", idx)
        return []

    resp = es.search(
        index=idx,
        body={
            "size": size,
            "query": {"match_all": {}},
            "sort": [{"timestamp": {"order": "asc"}}],
        },
    )
    hits = resp.get("hits", {}).get("hits", [])
    return [h["_source"] for h in hits]


# =========================
# Real-Time Engine
# =========================

class RealTimeIncidentLens:

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.window_size = cfg.window_size
        self.stride = cfg.stride
        self.embedding_dim = cfg.embedding_dim

        self.es = wrappers.get_client()
        assert wrappers.ping(self.es), "Elasticsearch not reachable"

        wrappers.setup_all_indices(
            self.es,
            embedding_dim=self.embedding_dim,
            delete_existing=cfg.delete_existing_indices,
        )

        self._load_gnn()

    def _load_gnn(self):
        path = Path(self.cfg.checkpoint_path) if self.cfg.checkpoint_path else get_default_checkpoint()
        if not path.exists():
            print("[RT] No trained GNN found — using fallback embeddings.")
            return

        try:
            encoder = TemporalGNNEncoder.from_checkpoint(path)
            wrappers.set_gnn_encoder(encoder)

            if getattr(encoder, "embedding_dim", None):
                self.embedding_dim = encoder.embedding_dim

            print(f"[RT] Loaded Temporal GNN ({self.embedding_dim} dim)")
        except Exception as exc:
            logger.warning("Failed to load GNN checkpoint %s: %s", path, exc)

    async def process_window(self, window_id: int, window_start: float, flows: List[dict]):

        if not flows:
            return

        print(f"\n[RT] Window {window_id} ({len(flows)} flows)")

        df = pd.DataFrame(flows)

        # Always use window_start for graph windowing — simulation flows
        # carry a wall-clock "timestamp" (ms) that would corrupt window
        # assignment in build_sliding_window_graphs.
        df["timestamp"] = window_start

        # Map simulation flow fields → graph builder expected columns
        if "packet_length" not in df.columns:
            df["packet_length"] = df.get("total_bytes", 0)

        if "payload_length" not in df.columns:
            df["payload_length"] = 0

        if "dst_port" not in df.columns:
            df["dst_port"] = 0

        if "protocol" not in df.columns:
            df["protocol"] = 0

        if "label" not in df.columns:
            df["label"] = 0

        # NOTE: Since flows are pre-aggregated per window, all rows share
        # the same timestamp → mean_iat and std_iat edge features will be
        # zero.  This is an accepted limitation of the streaming path;
        # the GNN's remaining 3 features (packet_count, total_bytes,
        # mean_payload) still carry signal.

        graphs, id_to_ip = wrappers.build_graphs(
            df,
            window_size=self.window_size,
            stride=self.stride,
            bytes_col="packet_length",
            label_col="label",
        )

        if not graphs:
            print("[RT] No graphs produced.")
            return

        graph = graphs[0]

        wrappers.index_graphs_bulk(graphs, id_to_ip, es=self.es)

        try:
            wrappers.index_graph_summaries(graphs, es=self.es)
        except Exception as exc:
            logger.warning("Graph summary indexing failed: %s", exc)

        flow_ids, embeddings, labels = wrappers.generate_embeddings(
            graphs,
            id_to_ip,
            embedding_dim=self.embedding_dim,
        )

        wrappers.index_embeddings(flow_ids, embeddings, labels, es=self.es)

        # index_graphs_bulk already ran predict_labels() on each graph,
        # storing sigmoid-applied per-edge scores in graph.pred_scores.
        # Reuse that instead of running a redundant forward pass.
        pred_scores = getattr(graph, "pred_scores", None)
        if pred_scores is not None and pred_scores.numel() > 0:
            score = float(pred_scores.mean().item())
            print(f"[RT] GNN anomaly score: {score:.4f}")

        try:
            breakdown = wrappers.aggregate_severity_breakdown(es=self.es)
            print(f"[RT] Severity dist: {breakdown.get('severity')} | "
                  f"Volume dist: {breakdown.get('volume')}")
        except Exception as exc:
            logger.warning("Severity aggregation failed: %s", exc)

        try:
            top = wrappers.search_anomalous_flows(
                min_prediction_score=self.cfg.anomaly_threshold,
                size=self.cfg.print_top_anomalies,
                es=self.es,
            )

            if top:
                print(f"[RT] Top {len(top)} anomalous flows:")
                for d in top:
                    print(
                        f" flow_id={d.get('flow_id')} "
                        f"{d.get('src_ip')} -> {d.get('dst_ip')} "
                        f"bytes={d.get('total_bytes')} "
                        f"score={d.get('prediction_score')}"
                    )
            else:
                print("[RT] No anomalous flows above threshold.")
        except Exception as exc:
            logger.warning("Anomaly search failed: %s", exc)


# =========================
# Runner
# =========================

async def run_realtime(cfg: PipelineConfig, packets: List[dict], rate: float):
    engine = RealTimeIncidentLens(cfg)

    simulator = StreamSimulator(
        packets=packets,
        rate=rate,
        window_size=cfg.window_size,
        mode="rate",
        window_callback=engine.process_window,
    )

    await simulator.run()


def main():
    parser = argparse.ArgumentParser("IncidentLens real-time pipeline")

    parser.add_argument("--source", choices=["simulate", "ndjson", "es"], default="simulate")
    parser.add_argument("--ndjson", default="data/packets_sample.json")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--rate", type=float, default=500.0)
    parser.add_argument("--window-size", type=float, default=5.0)
    parser.add_argument("--stride", type=float, default=5.0)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--delete-existing", action="store_true")
    parser.add_argument("--anomaly-threshold", type=float, default=0.7)
    parser.add_argument("--top-anomalies", type=int, default=5)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--es-query-size", type=int, default=5000)

    args = parser.parse_args()

    cfg = PipelineConfig(
        window_size=args.window_size,
        stride=args.stride,
        embedding_dim=args.embedding_dim,
        delete_existing_indices=args.delete_existing,
        source=args.source,
        ndjson_path=args.ndjson,
        es_query_size=args.es_query_size,
        print_top_anomalies=args.top_anomalies,
        anomaly_threshold=args.anomaly_threshold,
        checkpoint_path=args.checkpoint,
    )

    es = wrappers.get_client()
    assert wrappers.ping(es), "Elasticsearch not reachable"

    if cfg.source in ("simulate", "ndjson"):
        packets = load_ndjson(cfg.ndjson_path, max_rows=args.max_rows)
        print(f"[MAIN] Loaded {len(packets)} packets from {cfg.ndjson_path}")

    elif cfg.source == "es":
        packets = load_packets_from_es(es, size=cfg.es_query_size)
        print(f"[MAIN] Loaded {len(packets)} packets from ES")

        if not packets:
            print("No packets found in ES.")
            return
    else:
        raise ValueError(f"Unknown source: {cfg.source}")

    asyncio.run(run_realtime(cfg, packets, rate=args.rate))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
