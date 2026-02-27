from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from src.Backend import wrappers
from src.Backend.simulation import StreamSimulator
from src.Backend.temporal_gnn import TemporalGNNEncoder, get_default_checkpoint
from src.Backend.llm_reasoner import LLMReasoner


# ============================================================
# Real-Time Engine
# ============================================================

class RealTimeIncidentLens:

    def __init__(
        self,
        window_size: float = 5.0,
        stride: float = 5.0,
        embedding_dim: int = 16,
        delete_existing_indices: bool = False,
        debug: bool = False,
    ):
        self.window_size = window_size
        self.stride = stride
        self.embedding_dim = embedding_dim
        self.debug = debug
        
        # Elasticsearch
        self.es = wrappers.get_client()
        assert wrappers.ping(self.es), "Elasticsearch not reachable"
        self.reasoner = LLMReasoner(self.es)
        wrappers.setup_all_indices(
            self.es,
            embedding_dim=embedding_dim,
            delete_existing=delete_existing_indices,
        )

        self._load_gnn()

    # --------------------------------------------------------

    def _load_gnn(self):
        path = get_default_checkpoint()

        if not path.exists():
            print("[RT] No trained GNN found — using fallback embeddings.")
            return

        try:
            encoder = TemporalGNNEncoder.from_checkpoint(path)
            wrappers.set_gnn_encoder(encoder)

            self.embedding_dim = encoder.embedding_dim
            print(f"[RT] Loaded Temporal GNN ({encoder.embedding_dim} dim)")
        except Exception as exc:
            print(f"[RT] Failed to load GNN checkpoint {path}: {exc}")
            print("[RT] Falling back to synthetic embeddings.")

    # --------------------------------------------------------

    async def process_window(
        self,
        window_id: int,
        window_start: float,
        flows: List[dict],
        ) -> Dict[str, Any] | None:

        if not flows:
            return None

        print(f"\n========== WINDOW {window_id} ==========")
        print(f"Flows in window: {len(flows)}")

        df = pd.DataFrame(flows)

        # Map simulation flow fields → graph builder expected columns.
        # Always use window_start for graph windowing — simulation flows
        # carry a wall-clock "timestamp" (ms) that would corrupt window
        # assignment.
        df["timestamp"] = df["window_start"]
        df["packet_length"] = df["total_bytes"]

        if "payload_length" not in df.columns:
            df["payload_length"] = 0
        if "dst_port" not in df.columns:
            df["dst_port"] = 0
        if "protocol" not in df.columns:
            df["protocol"] = 0
        if "label" not in df.columns:
            df["label"] = 0

        # NOTE: Pre-aggregated flows → mean_iat / std_iat will be zero.
        # Accepted limitation; the GNN's remaining features still carry signal.

        # ---------------------------------
        # Build + Index Graphs
        # ---------------------------------
        graphs, id_to_ip, n_indexed = wrappers.build_and_index_graphs(
            df,
            window_size=self.window_size,
            stride=self.stride,
            es=self.es,
        )

        if not graphs:
            print("[RT] No graph generated.")
            return None

        graph = graphs[0]

        print(f"[RT] Graph Nodes: {graph.num_nodes}")
        print(f"[RT] Graph Edges: {graph.num_edges}")

        # ---------------------------------
        # Generate + Index Embeddings
        # ---------------------------------
        flow_ids, embeddings, labels = wrappers.generate_embeddings(
            graphs,
            id_to_ip,
            embedding_dim=self.embedding_dim,
        )

        n_emb = wrappers.index_embeddings(
            flow_ids,
            embeddings,
            labels,
            es=self.es,
        )

        print(f"[RT] Indexed {n_indexed} flows")
        print(f"[RT] Indexed {n_emb} embeddings")

        # ---------------------------------
        # GNN Anomaly Score
        # ---------------------------------
        # index_graphs_bulk already ran predict_labels() on each graph,
        # storing sigmoid-applied per-edge scores in graph.pred_scores.
        # Reuse that instead of running a redundant forward pass.
        anomaly_score = 0.0
        pred_scores = getattr(graph, "pred_scores", None)
        if pred_scores is not None and pred_scores.numel() > 0:
            anomaly_score = float(pred_scores.mean().item())
        else:
            # Fallback: no GNN encoder was registered, so no scores available
            pass

        print(f"[RT] GNN Anomaly Score: {anomaly_score:.4f}")

        # ---------------------------------
        # Severity Breakdown (ES Aggregation)
        # ---------------------------------
        breakdown = {}
        try:
            breakdown = wrappers.aggregate_severity_breakdown(es=self.es)
            print("[RT] Severity Distribution:", breakdown.get("severity"))
            print("[RT] Volume Distribution:", breakdown.get("volume"))
        except Exception:
            print("[RT] Severity aggregation unavailable.")

        # ---------------------------------
        # Top Anomalous Flows
        # ---------------------------------
        top = []
        try:
            top = wrappers.search_anomalous_flows(
                min_prediction_score=0.7,
                size=5,
                es=self.es,
            )
        except Exception:
            pass

        # ---------------------------------
        # LLM Reasoning (only for elevated anomaly scores)
        # ---------------------------------
        insight = None
        if anomaly_score >= 0.5:
            context = {
                "window_id": window_id,
                "gnn_score": anomaly_score,
                "severity_breakdown": breakdown,
                "top_anomalies": top,
            }

            try:
                insight = await self.reasoner.analyze_window(context)
                print("\n=== LLM INVESTIGATION ===")
                print(json.dumps(insight, indent=2))
            except Exception as exc:
                print(f"[RT] LLM reasoning failed: {exc}")

        return {
            "window_id": window_id,
            "num_flows": len(flows),
            "num_indexed": n_indexed,
            "num_embeddings": n_emb,
            "anomaly_score": anomaly_score,
        }


# ============================================================
# Simulation Runner
# ============================================================

async def run_realtime_simulation(
    packets: List[dict],
    rate: float,
    window_size: float,
    debug: bool = False,
    mode: str = "rate",
    time_scale: float = 1.0,
):

    engine = RealTimeIncidentLens(
        window_size=window_size,
        stride=window_size,
        embedding_dim=16,
        debug=debug,
    )

    simulator = StreamSimulator(
        packets=packets,
        rate=rate,
        window_size=window_size,
        mode=mode,
        time_scale=time_scale,
        window_callback=engine.process_window,
    )

    await simulator.run()


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser("Real-Time IncidentLens")

    parser.add_argument("--ndjson", required=True, help="Path to NDJSON packet file")
    parser.add_argument("--rate", type=float, default=500.0)
    parser.add_argument("--window-size", type=float, default=5.0)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    data_path = Path(args.ndjson)

    if not data_path.exists():
        print(f"File not found: {data_path}")
        return

    packets = []
    with open(data_path, "r") as f:
        for line in f:
            packets.append(json.loads(line))

    print(f"Loaded {len(packets)} packets.")

    asyncio.run(
        run_realtime_simulation(
            packets=packets,
            rate=args.rate,
            window_size=args.window_size,
            debug=args.debug,
        )
    )


if __name__ == "__main__":
    main()
