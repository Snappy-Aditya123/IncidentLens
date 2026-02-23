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
    p_ingest.add_argument("--data-dir", default=str(Path(__file__).resolve().parent.parent / "data"))
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
    p_conv.add_argument("--outdir", default=str(Path(__file__).resolve().parent.parent / "data"))
    p_conv.add_argument("--chunk-size", type=int, default=100_000)
    p_conv.add_argument("--max-rows", type=int, default=None)

    return parser


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
    }

    if args.command is None:
        parser.print_help()
        print("\nQuick start:")
        print("  1. docker compose up -d          # Start ES + Kibana")
        print("  2. python main.py health          # Verify ES is up")
        print("  3. python main.py convert          # CSV -> NDJSON (if needed)")
        print("  4. python main.py ingest           # Index data + build graphs")
        print("  5. python main.py investigate      # Run AI agent")
        sys.exit(0)

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
