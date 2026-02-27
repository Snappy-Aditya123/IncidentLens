# src/Backend/realtime_pipeline.py

from __future__ import annotations

import asyncio
from typing import List

import pandas as pd
import numpy as np
import torch

import src.Backend.wrappers as wrappers
from src.Backend.simulation import StreamSimulator
from src.Backend.temporal_gnn import TemporalGNNEncoder, get_default_checkpoint


# 1. REAL-TIME ENGINE (Inference + Indexing Only)


class RealTimeIncidentLens:

    def __init__(
        self,
        window_size: float = 5.0,
        stride: float = 5.0,
        embedding_dim: int = 16,
        delete_existing_indices: bool = False,
    ):
        self.window_size = window_size
        self.stride = stride
        self.embedding_dim = embedding_dim

        # ---------------------------
        # Elasticsearch setup
        # ---------------------------
        self.es = wrappers.get_client()
        assert wrappers.ping(self.es), "Elasticsearch not reachable"

        wrappers.setup_all_indices(
            self.es,
            embedding_dim=embedding_dim,
            delete_existing=delete_existing_indices,
        )

        # ---------------------------
        # Load GNN
        # ---------------------------
        self._load_gnn()

    # --------------------------------------------------------

    def _load_gnn(self):

        path = get_default_checkpoint()
        if not path.exists():
            print("[RT] No trained GNN found — using fallback embeddings.")
            return

        encoder = TemporalGNNEncoder.from_checkpoint(path)
        wrappers.set_gnn_encoder(encoder)

        print(f"[RT] Loaded Temporal GNN ({encoder.embedding_dim} dim)")

    # --------------------------------------------------------

    async def process_window(
        self,
        window_id: int,
        window_start: float,
        flows: List[dict],
    ):

        if not flows:
            return

        print(f"\n[RT] Window {window_id} ({len(flows)} flows)")

        # ----------------------------------------------------
        # Convert flows → DataFrame
        # ----------------------------------------------------

        df = pd.DataFrame(flows)

        # Required fields for graph builder
        df["payload_length"] = 0
        df["packet_length"] = df["total_bytes"]
        df["timestamp"] = df["window_start"]

        # ----------------------------------------------------
        # Build graph snapshot
        # ----------------------------------------------------

        graphs, id_to_ip = wrappers.build_sliding_window_graphs(
            df,
            window_size=self.window_size,
            stride=self.stride,
        )

        if not graphs:
            return

        graph = graphs[0]

        # ----------------------------------------------------
        # Generate embeddings (GNN auto-used)
        # ----------------------------------------------------

        flow_ids, embeddings, labels = wrappers.generate_embeddings(
            graphs,
            id_to_ip,
            embedding_dim=self.embedding_dim,
        )

        # ----------------------------------------------------
        # Index flows
        # ----------------------------------------------------

        wrappers.index_graphs_bulk(
            graphs,
            id_to_ip,
            es=self.es,
        )

        # ----------------------------------------------------
        # Index embeddings
        # ----------------------------------------------------

        wrappers.index_embeddings(
            flow_ids,
            embeddings,
            labels,
            es=self.es,
        )

        # ----------------------------------------------------
        # Print anomaly score
        # ----------------------------------------------------

        if wrappers.get_gnn_encoder() is not None:

            with torch.no_grad():
                output = wrappers.get_gnn_encoder()(graph)

                if isinstance(output, dict):
                    score = output.get("prediction_score", 0.0)
                elif isinstance(output, (tuple, list)):
                    score = output[-1]
                else:
                    score = output

                if isinstance(score, torch.Tensor):
                    score = score.item()

                print(f"[RT] GNN anomaly score: {float(score):.4f}")


# ============================================================
# 2. RUN REAL-TIME SIMULATION
# ============================================================

async def run_realtime_simulation(
    packets: List[dict],
    rate: float = 500.0,
    window_size: float = 5.0,
):

    engine = RealTimeIncidentLens(
        window_size=window_size,
        stride=window_size,
        embedding_dim=16,
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
# CLI ENTRY
# ============================================================

def main():

    import json
    from pathlib import Path

    data_path = Path("data/packets_sample.json")

    if not data_path.exists():
        print("Provide a packet JSON file for simulation.")
        return

    packets = []
    with open(data_path, "r") as f:
        for line in f:
            packets.append(json.loads(line))

    asyncio.run(run_realtime_simulation(packets))


if __name__ == "__main__":
    main()