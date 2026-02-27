from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import torch

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

        encoder = TemporalGNNEncoder.from_checkpoint(path)
        wrappers.set_gnn_encoder(encoder)

        self.embedding_dim = encoder.embedding_dim
        print(f"[RT] Loaded Temporal GNN ({encoder.embedding_dim} dim)")

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

        df["payload_length"] = 0
        df["packet_length"] = df["total_bytes"]
        df["timestamp"] = df["window_start"]

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
        anomaly_score = 0.0
        gnn = wrappers.get_gnn_encoder()

        if gnn is not None:
            with torch.no_grad():
                output = gnn(graph)

                if isinstance(output, dict):
                    anomaly_score = output.get("prediction_score", 0.0)
                elif isinstance(output, (tuple, list)):
                    anomaly_score = output[-1]
                else:
                    anomaly_score = output

                if isinstance(anomaly_score, torch.Tensor):
                    anomaly_score = anomaly_score.item()

        anomaly_score = float(anomaly_score)
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
        # LLM Trigger
        # ---------------------------------
        trigger = anomaly_score > 0.75

        if trigger:
            context = {
                "window_id": window_id,
                "gnn_score": anomaly_score,
                "severity_breakdown": breakdown,
                "top_anomalies": top,
            }

            insight = await self.reasoner.analyze_window(context)

            print("\n=== LLM INVESTIGATION ===")
            print(json.dumps(insight, indent=2))

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
    debug: bool,
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
        mode="rate",
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