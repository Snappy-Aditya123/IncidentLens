#!/usr/bin/env python3
"""
IncidentLens - CLI entry point
==============================

Unified command-line interface for all IncidentLens operations:

    python main.py ingest          # Run data ingestion + analysis pipeline
    python main.py investigate     # Run AI agent investigation (auto-detect)
    python main.py investigate "why is 192.168.100.5 anomalous?"
    python main.py serve           # Start FastAPI server (REST + WebSocket)
    python main.py convert         # Convert raw CSV -> NDJSON
    python main.py health          # Quick ES health check

All queries go through Elasticsearch at http://localhost:9200.
Kibana dashboard at http://localhost:5601.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def cmd_health(args):
    """Check Elasticsearch connectivity and index status."""
    import src.Backend.wrappers as wrappers

    es = wrappers.get_client()
    if not wrappers.ping(es):
        print("[FAIL] Cannot reach Elasticsearch at http://localhost:9200")
        print("       Run: docker compose up -d")
        sys.exit(1)

    h = wrappers.health_check(es)
    print(f"[OK] Cluster: {h['status']}, nodes: {h.get('number_of_nodes', '?')}")

    for idx in [wrappers.FLOWS_INDEX, wrappers.EMBEDDINGS_INDEX, wrappers.COUNTERFACTUALS_INDEX]:
        exists = es.indices.exists(index=idx)
        if exists:
            count = es.count(index=idx)["count"]
            print(f"  {idx}: {count:,} docs")
        else:
            print(f"  {idx}: (not created)")


def cmd_ingest(args):
    """Run the full ingestion + analysis pipeline."""
    from src.Backend.ingest_pipeline import run_pipeline

    run_pipeline(
        data_dir=args.data_dir,
        max_rows=args.max_rows,
        skip_graphs=args.skip_graphs,
        skip_raw_index=args.skip_raw,
        window_size=args.window_size,
        stride=args.stride,
        embedding_dim=args.embedding_dim,
        max_counterfactuals=args.max_cf,
        delete_existing=args.delete_existing,
    )


def cmd_investigate(args):
    """Run the LLM agent to investigate anomalies."""
    from src.Backend.agent import IncidentAgent, AgentConfig

    config = AgentConfig()
    if not config.api_key:
        print("Set OPENAI_API_KEY environment variable, or for local models:")
        print("  set OPENAI_BASE_URL=http://localhost:11434/v1")
        print("  set OPENAI_API_KEY=ollama")
        sys.exit(1)

    agent = IncidentAgent(config)
    query = " ".join(args.query) if args.query else None
    if not query:
        query = input("Enter investigation query (or press Enter for auto-detect): ").strip()

    if not query:
        print("[Auto-detect mode]")
        events = agent.investigate_auto()
    else:
        events = agent.investigate(query)

    for event in events:
        etype = event["type"]
        content = event.get("content", "")
        if etype == "status":
            print(f"\n--- {content} ---")
        elif etype == "thinking":
            print(f"\n[Thinking] {content}")
        elif etype == "tool_call":
            print(f"\n>> Calling: {event['tool']}({json.dumps(event['arguments'])})")
        elif etype == "tool_result":
            print(f"<< Result: {event.get('result', '')[:500]}")
        elif etype == "conclusion":
            print(f"\n{'=' * 60}\n{content}\n{'=' * 60}")
        elif etype == "error":
            print(f"\n[ERROR] {content}")


def cmd_serve(args):
    """Start the FastAPI server for REST + WebSocket access."""
    import uvicorn

    print(f"Starting IncidentLens server on port {args.port} ...")
    print(f"  REST API:   http://localhost:{args.port}/api/")
    print(f"  WebSocket:  ws://localhost:{args.port}/ws/investigate")
    print(f"  Health:     http://localhost:{args.port}/health")
    print(f"  ES backend: http://localhost:9200")
    uvicorn.run(
        "src.Backend.server:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


def cmd_convert(args):
    """Convert raw CSV files to NDJSON for ingestion."""
    from src.Backend.csv_to_json import convert

    kwargs = {"outdir": args.outdir, "chunk_size": args.chunk_size}
    if args.packets is not None:
        kwargs["packets_csv"] = args.packets
    if args.labels is not None:
        kwargs["labels_csv"] = args.labels
    if args.max_rows is not None:
        kwargs["max_rows"] = args.max_rows
    convert(**kwargs)


def cmd_train(args):
    """Train the Temporal GNN and save a checkpoint.

    This should be run ONCE (or whenever data changes) to produce a
    pretrained model.  Subsequent ``ingest`` and ``serve`` commands load
    the checkpoint for inference — no retraining needed.
    """
    import pandas as pd
    from src.Backend.temporal_gnn import (
        prepare_temporal_dataset,
        prepare_temporal_dataset_from_csv,
        train_temporal_gnn,
        get_default_checkpoint,
    )

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    checkpoint = args.checkpoint or str(get_default_checkpoint())

    if args.packets and args.labels:
        # Direct CSV mode
        print(f"Loading CSVs: packets={args.packets}, labels={args.labels}")
        packets_df = pd.read_csv(args.packets)
        labels_df = pd.read_csv(args.labels)

        sequences, info = prepare_temporal_dataset(
            packets_df,
            labels_df,
            window_size=args.window_size,
            stride=args.stride,
            seq_len=args.seq_len,
            normalize=True,
        )
    elif args.data_dir:
        # NDJSON mode (post csv_to_json)
        from src.Backend.ingest_pipeline import load_ndjson_files, build_graphs_from_df
        from src.Backend.temporal_gnn import (
            sanitize_graph, recompute_node_features,
            normalize_features_global, preprocess_graphs,
            build_temporal_sequences,
        )

        print(f"Loading NDJSON from {args.data_dir} ...")
        df = load_ndjson_files(args.data_dir, max_rows=args.max_rows)

        print("Building graphs ...")
        from src.Backend.graph_data_wrapper import build_sliding_window_graphs
        graphs, id_to_ip = build_graphs_from_df(df, args.window_size, args.stride)

        # Pipeline: sanitize -> recompute -> normalize -> preprocess -> sequences
        graphs = [sanitize_graph(g) for g in graphs]
        graphs = [recompute_node_features(g) for g in graphs]
        graphs, norm_stats = normalize_features_global(graphs)
        graphs = preprocess_graphs(graphs)
        sequences = build_temporal_sequences(graphs, seq_len=args.seq_len)

        info = {
            "node_feat_dim": int(graphs[0].x.shape[1]),
            "edge_feat_dim": int(graphs[0].edge_attr.shape[1]),
            "num_graphs": len(graphs),
            "num_sequences": len(sequences),
            "seq_len": args.seq_len,
            "norm_stats": norm_stats,
        }
    else:
        print("Provide either --packets + --labels (CSV mode) or --data-dir (NDJSON mode)")
        sys.exit(1)

    if not sequences:
        print("[ERROR] No sequences built — check your data and window parameters.")
        sys.exit(1)

    train_temporal_gnn(
        sequences,
        info,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        use_ode=getattr(args, "ode", False),
        checkpoint_path=checkpoint,
        norm_stats=info.get("norm_stats"),
        seq_len=args.seq_len,
    )
    print(f"\nModel saved to: {checkpoint}")
    print("You can now run: python main.py ingest  (will auto-load this checkpoint)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="incidentlens",
        description="IncidentLens - AI-powered network incident investigation",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- health ---
    sub.add_parser("health", help="Check ES connectivity and index status")

    # --- ingest ---
    p_ingest = sub.add_parser("ingest", help="Run data ingestion + analysis pipeline")
    p_ingest.add_argument("--data-dir", default=str(Path(__file__).resolve().parent.parent.parent / "data"))
    p_ingest.add_argument("--max-rows", type=int, default=None)
    p_ingest.add_argument("--skip-graphs", action="store_true")
    p_ingest.add_argument("--skip-raw", action="store_true")
    p_ingest.add_argument("--window-size", type=float, default=2.0)
    p_ingest.add_argument("--stride", type=float, default=1.0)
    p_ingest.add_argument("--embedding-dim", type=int, default=16)
    p_ingest.add_argument("--max-cf", type=int, default=20)
    p_ingest.add_argument("--delete-existing", action="store_true")

    # --- investigate ---
    p_inv = sub.add_parser("investigate", help="Run AI agent investigation")
    p_inv.add_argument("query", nargs="*", help="Investigation query (empty = auto-detect)")

    # --- serve ---
    p_serve = sub.add_parser("serve", help="Start API server (REST + WebSocket)")
    p_serve.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    p_serve.add_argument("--reload", action="store_true", help="Enable hot-reload")

    # --- convert ---
    p_conv = sub.add_parser("convert", help="Convert CSV -> NDJSON")
    p_conv.add_argument("--packets", default=None, help="Path to ssdp_packets_rich.csv")
    p_conv.add_argument("--labels", default=None, help="Path to SSDP_Flood_labels.csv")
    p_conv.add_argument("--outdir", default=str(Path(__file__).resolve().parent.parent.parent / "data"))
    p_conv.add_argument("--chunk-size", type=int, default=100_000)
    p_conv.add_argument("--max-rows", type=int, default=None)

    # --- train ---
    p_train = sub.add_parser("train", help="Train Temporal GNN and save checkpoint")
    p_train.add_argument("--packets", default=None, help="Path to ssdp_packets_rich.csv")
    p_train.add_argument("--labels", default=None, help="Path to SSDP_Flood_labels.csv")
    p_train.add_argument("--data-dir", default=None, help="NDJSON data directory (alternative to --packets/--labels)")
    p_train.add_argument("--max-rows", type=int, default=None)
    p_train.add_argument("--window-size", type=float, default=2.0)
    p_train.add_argument("--stride", type=float, default=1.0)
    p_train.add_argument("--seq-len", type=int, default=5, help="Temporal sequence length")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--hidden-dim", type=int, default=64)
    p_train.add_argument("--dropout", type=float, default=0.2)
    p_train.add_argument("--batch-size", type=int, default=8, help="Sequences per gradient update")
    p_train.add_argument("--ode", action="store_true", help="Use Neural ODE weight evolution (requires torchdiffeq)")
    p_train.add_argument("--checkpoint", default=None, help="Where to save model (default: models/temporal_gnn.pt)")
        # --- simulate ---
    p_sim = sub.add_parser("simulate", help="Run real-time packet simulation")
    p_sim.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent.parent.parent / "data"),
        help="NDJSON data directory"
    )
    p_sim.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit number of packets"
    )
    p_sim.add_argument(
        "--rate",
        type=float,
        default=200.0,
        help="Packets per second"
    )
    p_sim.add_argument(
        "--window-size",
        type=float,
        default=5.0,
        help="Window duration in seconds"
    )
    p_sim.add_argument(
    "--mode",
    choices=["rate", "realtime"],
    default="rate",
    help="Replay mode"
)

    p_sim.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Scale realtime replay (0.1 = 10x faster)"
    )

    return parser
async def pipeline(window_id, window_start, flows):
    from src.Backend.wrappers import index_flows_bulk

    print(f"[PIPE] Processing window {window_id}")

    if flows:
        index_flows_bulk(flows)


def cmd_simulate(args):
    import asyncio
    from src.Backend.ingest_pipeline import load_ndjson_files
    from src.Backend.simulation import StreamSimulator

    print(
        f"[Simulation] mode={args.mode} | "
        f"rate={args.rate}pps | "
        f"window={args.window_size}s | "
        f"time_scale={args.time_scale}"
    )

    # Load packets
    df = load_ndjson_files(args.data_dir, max_rows=args.max_rows)
    packets = df.to_dict(orient="records")

    simulator = StreamSimulator(
        packets=packets,
        rate=args.rate,
        window_size=args.window_size,
        mode=args.mode,
        time_scale=args.time_scale,
        window_callback=pipeline,
    )

    asyncio.run(simulator.run())

def main():
    parser = build_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    commands = {
        "health": cmd_health,
        "ingest": cmd_ingest,
        "investigate": cmd_investigate,
        "serve": cmd_serve,
        "convert": cmd_convert,
        "train": cmd_train,
        "simulate": cmd_simulate,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        parser.print_help()
        sys.exit(1)



    if args.command is None:
        print("[INFO] No command provided — starting simulation mode")

        # Set defaults for simulation
        args.command = "simulate"
        args.mode = "realtime"
        args.time_scale = 0.2
        args.rate = 200.0
        args.window_size = 5.0
        args.max_rows = None
        args.data_dir = str(Path(__file__).resolve().parent.parent.parent / "data")

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
