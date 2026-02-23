"""
CSV → JSON converter for IncidentLens.

Reads the SSDP Flood packet CSV + label CSV, merges them, and writes
newline-delimited JSON (NDJSON) files into ``data/`` for consumption by
the ES ingestion pipeline.

Large dataset (~4M rows) is chunked into manageable files.

Usage:
    python csv_to_json.py
    python csv_to_json.py --packets <path> --labels <path> --outdir data --chunk-size 100000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Defaults — paths relative to workspace
# ──────────────────────────────────────────────

# Project root: IncidentLens/ (3 levels up from src/Backend/csv_to_json.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DATA_ROOT = Path(os.environ.get(
    "INCIDENTLENS_DATA_ROOT",
    str(_PROJECT_ROOT / "data"),
))
DEFAULT_PACKETS_CSV = os.environ.get(
    "INCIDENTLENS_PACKETS_CSV",
    str(_DATA_ROOT / "ssdp_packets_rich.csv"),
)
DEFAULT_LABELS_CSV = os.environ.get(
    "INCIDENTLENS_LABELS_CSV",
    str(_DATA_ROOT / "SSDP_Flood_labels.csv"),
)
DEFAULT_OUTDIR = str(_PROJECT_ROOT / "data")
DEFAULT_CHUNK_SIZE = 100_000


# ──────────────────────────────────────────────
# Core conversion
# ──────────────────────────────────────────────

def _safe_val(v):
    """Convert numpy / pandas scalars to JSON-safe Python types."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def load_and_merge(
    packets_csv: str,
    labels_csv: str,
) -> pd.DataFrame:
    """Load packet + label CSVs and merge into a single DataFrame.

    Returns a DataFrame with all original packet columns plus ``label``.
    """
    print(f"Loading packets from {packets_csv} ...")
    packets = pd.read_csv(packets_csv)
    print(f"  -> {len(packets):,} rows, {len(packets.columns)} columns")

    print(f"Loading labels from {labels_csv} ...")
    labels = pd.read_csv(labels_csv)
    labels = labels.rename(columns={"Unnamed: 0": "packet_index", "x": "label"})
    print(f"  -> {len(labels):,} rows")

    # merge on packet_index
    merged = packets.merge(labels[["packet_index", "label"]], on="packet_index", how="left")
    merged["label"] = merged["label"].fillna(0).astype(int)
    print(f"  -> Merged: {len(merged):,} rows, label distribution:")
    print(f"    normal={int((merged['label'] == 0).sum()):,}  "
          f"malicious={int((merged['label'] == 1).sum()):,}")
    return merged


def dataframe_to_ndjson_chunks(
    df: pd.DataFrame,
    outdir: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    prefix: str = "packets",
) -> list[str]:
    """Write DataFrame to NDJSON files in ``outdir``, chunked.

    Each file is named ``<prefix>_<chunk_number>.json``.
    Returns list of written file paths.
    """
    os.makedirs(outdir, exist_ok=True)
    n_chunks = math.ceil(len(df) / chunk_size)
    written: list[str] = []

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(df))
        chunk = df.iloc[start:end]

        fname = f"{prefix}_{i:04d}.json"
        fpath = os.path.join(outdir, fname)

        # Use to_dict('records') + batch json.dumps — 10-100x faster than iterrows()
        records = chunk.to_dict(orient="records")
        with open(fpath, "w", encoding="utf-8") as f:
            for doc in records:
                cleaned = {k: _safe_val(v) for k, v in doc.items()}
                f.write(json.dumps(cleaned) + "\n")

        written.append(fpath)
        print(f"  Wrote {fpath}  ({end - start:,} records)")

    return written


def write_metadata(outdir: str, total_rows: int, chunk_size: int, files: list[str]) -> str:
    """Write a metadata JSON with info about the export."""
    meta = {
        "source": "SSDP_Flood dataset (Kitsune)",
        "total_rows": total_rows,
        "chunk_size": chunk_size,
        "num_files": len(files),
        "files": [os.path.basename(f) for f in files],
        "columns": [
            "packet_index", "timestamp", "inter_arrival_time",
            "src_ip", "dst_ip", "src_port", "dst_port",
            "protocol", "ttl", "ip_header_len", "tcp_flags",
            "udp_length", "payload_length", "packet_length", "label",
        ],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = os.path.join(outdir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote {meta_path}")
    return meta_path


def convert(
    packets_csv: str = DEFAULT_PACKETS_CSV,
    labels_csv: str = DEFAULT_LABELS_CSV,
    outdir: str = DEFAULT_OUTDIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_rows: int | None = None,
) -> dict:
    """End-to-end: load CSVs → merge → write NDJSON chunks + metadata.

    Returns a summary dict.
    """
    t0 = time.time()

    df = load_and_merge(packets_csv, labels_csv)

    if max_rows is not None:
        df = df.head(max_rows)
        print(f"  Capped to {len(df):,} rows (--max-rows {max_rows})")

    print(f"\nWriting NDJSON chunks (chunk_size={chunk_size:,}) ...")
    files = dataframe_to_ndjson_chunks(df, outdir, chunk_size)

    meta_path = write_metadata(outdir, len(df), chunk_size, files)

    elapsed = time.time() - t0
    summary = {
        "total_rows": len(df),
        "num_files": len(files),
        "output_dir": outdir,
        "metadata": meta_path,
        "elapsed_seconds": round(elapsed, 1),
    }
    print(f"\nDone in {elapsed:.1f}s. {len(files)} files in {outdir}/")
    return summary


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert SSDP Flood CSVs to NDJSON for Elasticsearch ingestion"
    )
    parser.add_argument(
        "--packets", default=DEFAULT_PACKETS_CSV,
        help="Path to ssdp_packets_rich.csv",
    )
    parser.add_argument(
        "--labels", default=DEFAULT_LABELS_CSV,
        help="Path to SSDP_Flood_labels.csv",
    )
    parser.add_argument(
        "--outdir", default=DEFAULT_OUTDIR,
        help="Output directory for NDJSON files",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help="Rows per NDJSON file (default: 100000)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Limit total rows exported (for quick testing)",
    )
    args = parser.parse_args()
    convert(args.packets, args.labels, args.outdir, args.chunk_size, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
