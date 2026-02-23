"""
IncidentLens Agent Tools
========================
Defines every investigative tool the LLM agent can call.

Each tool wraps a function in ``wrappers.py`` and is exposed as an
OpenAI-function-calling-compatible JSON schema.  The ``dispatch()``
function maps tool names to their implementations.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from typing import Any

import numpy as np

import src.Backend.wrappers as wrappers

# ──────────────────────────────────────────────
# Tool registry (name -> callable)
# ──────────────────────────────────────────────

_REGISTRY: dict[str, Any] = {}

# TTL-based cache for feature stats (avoids repeated ES agg queries)
_STATS_CACHE: dict[str, Any] = {"data": None, "ts": 0.0}
_STATS_TTL = 30.0  # seconds


def _register(name: str):
    """Decorator to register a tool function."""
    def decorator(fn):
        _REGISTRY[name] = fn
        return fn
    return decorator


# ──────────────────────────────────────────────
# 1. ES Cluster Health
# ──────────────────────────────────────────────

@_register("es_health_check")
def _tool_health_check(**kwargs) -> dict:
    h = wrappers.health_check()
    return {"cluster_status": h["status"], "number_of_nodes": h.get("number_of_nodes", 0)}


# ──────────────────────────────────────────────
# 2. Search Flows (general ES query on flows index)
# ──────────────────────────────────────────────

@_register("search_flows")
def _tool_search_flows(
    label: int | None = None,
    src_ip: str | None = None,
    dst_ip: str | None = None,
    min_packet_count: float | None = None,
    size: int = 20,
    **kwargs,
) -> dict:
    """Search the incidentlens-flows index with optional filters."""
    must_clauses: list[dict] = []
    if label is not None:
        must_clauses.append({"term": {"label": label}})
    if src_ip:
        must_clauses.append({"term": {"src_ip": src_ip}})
    if dst_ip:
        must_clauses.append({"term": {"dst_ip": dst_ip}})
    if min_packet_count is not None:
        must_clauses.append({"range": {"packet_count": {"gte": min_packet_count}}})

    query = {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}}
    flows = wrappers.search_flows(query=query, size=size)
    return {"count": len(flows), "flows": flows}


# ──────────────────────────────────────────────
# 3. Get single flow by ID
# ──────────────────────────────────────────────

@_register("get_flow")
def _tool_get_flow(flow_id: str, **kwargs) -> dict:
    flow = wrappers.get_flow(flow_id)
    if flow is None:
        return {"error": f"Flow {flow_id} not found"}
    return flow


# ──────────────────────────────────────────────
# 4. Detect anomalous flows (trigger)
# ──────────────────────────────────────────────

@_register("detect_anomalies")
def _tool_detect_anomalies(
    method: str = "label",
    threshold: float = 0.5,
    size: int = 50,
    **kwargs,
) -> dict:
    """Detect anomalous flows.

    method="label"  -> flows with label=1 (ground-truth or GNN output)
    method="score"  -> flows with prediction_score >= threshold
    method="stats"  -> flows whose packet_count is > 2 std above normal mean
    """
    if method == "score":
        flows = wrappers.search_anomalous_flows(
            min_prediction_score=threshold, size=size,
        )
        return {
            "method": "prediction_score",
            "threshold": threshold,
            "count": len(flows),
            "flows": flows,
        }

    if method == "stats":
        # Statistical anomaly: packet_count > mean + 2*std of normal
        stats = wrappers.feature_stats_by_label(features=["packet_count"])
        normal = stats.get("packet_count", {}).get("label_0", {})
        avg = normal.get("avg", 0)
        std = normal.get("std_deviation", 0)
        cutoff = avg + 2 * std
        flows = wrappers.search_flows(
            query={"range": {"packet_count": {"gte": cutoff}}},
            size=size,
        )
        return {
            "method": "statistical",
            "normal_avg": round(avg, 2),
            "normal_std": round(std, 2),
            "cutoff": round(cutoff, 2),
            "count": len(flows),
            "flows": flows,
        }

    # default: label-based
    flows = wrappers.search_flows(
        query={"term": {"label": 1}}, size=size,
    )
    return {
        "method": "label",
        "count": len(flows),
        "flows": flows,
    }


# ──────────────────────────────────────────────
# 5. Feature statistics by label
# ──────────────────────────────────────────────

@_register("feature_stats")
def _tool_feature_stats(**kwargs) -> dict:
    stats = wrappers.feature_stats_by_label()
    return stats


# ──────────────────────────────────────────────
# 6. Feature percentiles
# ──────────────────────────────────────────────

@_register("feature_percentiles")
def _tool_feature_percentiles(feature: str = "packet_count", **kwargs) -> dict:
    return wrappers.feature_percentiles_by_label(feature)


# ──────────────────────────────────────────────
# 7. Significant terms (discriminating features)
# ──────────────────────────────────────────────

@_register("significant_terms")
def _tool_significant_terms(
    field: str = "src_ip",
    foreground_label: int = 1,
    size: int = 10,
    **kwargs,
) -> dict:
    try:
        terms = wrappers.significant_terms_by_label(
            field=field, foreground_label=foreground_label, size=size,
        )
        return {"field": field, "terms": terms}
    except Exception as e:
        return {"field": field, "error": str(e)}


# ──────────────────────────────────────────────
# 8. Run counterfactual analysis for a flow
# ──────────────────────────────────────────────

@_register("counterfactual_analysis")
def _tool_counterfactual(flow_id: str, **kwargs) -> dict:
    """Find the nearest normal flow and compute feature diffs."""
    es = wrappers.get_client()

    # get the flow's embedding
    body = {"query": {"term": {"flow_id": flow_id}}, "size": 1}
    resp = es.search(index=wrappers.EMBEDDINGS_INDEX, body=body)
    hits = resp["hits"]["hits"]
    if not hits:
        return {"error": f"No embedding found for flow {flow_id}"}

    emb = hits[0]["_source"]["embedding"]
    cf = wrappers.build_and_index_counterfactual(
        anomalous_flow_id=flow_id,
        query_embedding=emb,
        es=es,
    )
    if cf is None:
        return {"error": "No normal neighbour found"}
    return cf


# ──────────────────────────────────────────────
# 9. Get counterfactual narrative (human-readable)
# ──────────────────────────────────────────────

@_register("counterfactual_narrative")
def _tool_cf_narrative(flow_id: str, **kwargs) -> dict:
    cfs = wrappers.get_counterfactuals_for_flow(flow_id)
    if not cfs:
        return {"error": f"No counterfactuals found for {flow_id}"}
    narratives = [wrappers.format_counterfactual_narrative(cf) for cf in cfs]
    return {"flow_id": flow_id, "count": len(cfs), "narratives": narratives}


# ──────────────────────────────────────────────
# 10. Explain API (why a doc matched)
# ──────────────────────────────────────────────

@_register("explain_flow")
def _tool_explain(flow_id: str, **kwargs) -> dict:
    try:
        return wrappers.explain_flow_match(flow_id)
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────
# 11. Search raw packets
# ──────────────────────────────────────────────

@_register("search_raw_packets")
def _tool_search_packets(
    src_ip: str | None = None,
    dst_ip: str | None = None,
    protocol: int | None = None,
    label: int | None = None,
    size: int = 20,
    **kwargs,
) -> dict:
    es = wrappers.get_client()
    must: list[dict] = []
    if src_ip:
        must.append({"term": {"src_ip": src_ip}})
    if dst_ip:
        must.append({"term": {"dst_ip": dst_ip}})
    if protocol is not None:
        must.append({"term": {"protocol": protocol}})
    if label is not None:
        must.append({"term": {"label": label}})

    query = {"bool": {"must": must}} if must else {"match_all": {}}
    resp = es.search(
        index="incidentlens-packets",
        body={"query": query, "size": size, "sort": [{"timestamp": "asc"}]},
    )
    packets = [h["_source"] for h in resp["hits"]["hits"]]
    return {"count": len(packets), "packets": packets}


# ──────────────────────────────────────────────
# 12. Historical incident matching (kNN similarity)
# ──────────────────────────────────────────────

@_register("find_similar_incidents")
def _tool_similar(flow_id: str, k: int = 5, **kwargs) -> dict:
    """Find flows most similar to the given one (any label)."""
    es = wrappers.get_client()
    body = {"query": {"term": {"flow_id": flow_id}}, "size": 1}
    resp = es.search(index=wrappers.EMBEDDINGS_INDEX, body=body)
    hits = resp["hits"]["hits"]
    if not hits:
        return {"error": f"No embedding for {flow_id}"}
    emb = hits[0]["_source"]["embedding"]
    results = wrappers.knn_search(emb, k=k)
    return {"query_flow": flow_id, "similar": results}


# ──────────────────────────────────────────────
# 13. Graph-level edge perturbation CF
# ──────────────────────────────────────────────

# Module-level graph cache (set by ingest_pipeline or build_graphs)
_GRAPH_CACHE: dict[str, Any] = {"graphs": None, "id_to_ip": None}


def set_graph_cache(graphs: list, id_to_ip: dict[int, str]) -> None:
    """Store graphs in memory for graph-level CF tools (avoids re-building)."""
    _GRAPH_CACHE["graphs"] = graphs
    _GRAPH_CACHE["id_to_ip"] = id_to_ip


@_register("graph_edge_counterfactual")
def _tool_graph_edge_cf(window_index: int | None = None, max_removals: int = 5, **kwargs) -> dict:
    """Run edge-perturbation counterfactual on a graph window.

    Shows which network connections have the greatest impact on the
    anomaly classification — removing them reveals what connections
    would need to not exist for the window to look normal.
    """
    graphs = _GRAPH_CACHE.get("graphs")
    id_to_ip = _GRAPH_CACHE.get("id_to_ip")
    if graphs is None:
        return {"error": "No graphs loaded. Run ingest pipeline first."}

    if window_index is None:
        # Auto-select most anomalous window
        idx, graph, stats = wrappers.find_anomalous_window(graphs)
    else:
        if window_index < 0 or window_index >= len(graphs):
            return {"error": f"Window {window_index} out of range (0-{len(graphs)-1})"}
        idx = window_index
        graph = graphs[idx]
        stats = {"window_index": idx}

    results = wrappers.graph_edge_perturbation_cf(
        graph, id_to_ip=id_to_ip, max_removals=max_removals,
    )
    return {
        "window": stats,
        "edge_perturbations": results,
        "total_edges_analyzed": len(results),
    }


@_register("graph_window_comparison")
def _tool_graph_window_compare(window_a: int | None = None, window_b: int | None = None, **kwargs) -> dict:
    """Compare two graph windows (structural + feature differences).

    By default compares the most normal window vs the most anomalous one.
    """
    graphs = _GRAPH_CACHE.get("graphs")
    if graphs is None:
        return {"error": "No graphs loaded. Run ingest pipeline first."}
    if window_a is not None and (window_a < 0 or window_a >= len(graphs)):
        return {"error": f"Window A={window_a} out of range (0-{len(graphs)-1})"}
    if window_b is not None and (window_b < 0 or window_b >= len(graphs)):
        return {"error": f"Window B={window_b} out of range (0-{len(graphs)-1})"}

    return wrappers.graph_window_comparison(graphs, window_a=window_a, window_b=window_b)


# ──────────────────────────────────────────────
# 14. Assess severity
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# 15. ES ML anomaly detection — get anomaly records
# ──────────────────────────────────────────────

@_register("get_ml_anomaly_records")
def _tool_ml_anomaly_records(
    job_id: str = "incidentlens-flow-anomaly",
    min_score: float = 75.0,
    size: int = 50,
    **kwargs,
) -> dict:
    """Fetch ES ML anomaly detection records above a score threshold.

    Each record includes influencers — the field values that drove
    the anomaly — providing direct built-in feature attribution from
    Elasticsearch's ML engine, complementing the GNN-based counterfactuals.
    """
    try:
        records = wrappers.get_anomaly_records(
            job_id=job_id, min_score=min_score, size=size,
        )
        return {
            "job_id": job_id,
            "min_score": min_score,
            "count": len(records),
            "records": records,
        }
    except Exception as e:
        return {"error": f"ML anomaly records failed: {e}"}


# ──────────────────────────────────────────────
# 16. ES ML anomaly detection — get influencers
# ──────────────────────────────────────────────

@_register("get_ml_influencers")
def _tool_ml_influencers(
    job_id: str = "incidentlens-flow-anomaly",
    min_score: float = 50.0,
    size: int = 50,
    **kwargs,
) -> dict:
    """Fetch ES ML influencer results — tells you WHICH src_ip, dst_ip,
    or protocol values contributed most to detected anomalies.

    Provides the raw material for root-cause analysis and complements
    the GNN-based explanation with ES-native feature attribution.
    """
    try:
        influencers = wrappers.get_influencers(
            job_id=job_id, min_score=min_score, size=size,
        )
        return {
            "job_id": job_id,
            "min_score": min_score,
            "count": len(influencers),
            "influencers": influencers,
        }
    except Exception as e:
        return {"error": f"ML influencers failed: {e}"}


# ──────────────────────────────────────────────
# 17. ES severity breakdown (runtime fields)
# ──────────────────────────────────────────────

@_register("severity_breakdown")
def _tool_severity_breakdown(**kwargs) -> dict:
    """Compute severity distribution using ES runtime fields + aggregation.

    Runs the computation entirely on the ES cluster using Painless scripts —
    zero Python iteration.  Returns bucket counts for critical/high/medium/low
    and traffic volume categories.
    """
    try:
        return wrappers.aggregate_severity_breakdown()
    except Exception as e:
        return {"error": f"Severity breakdown failed: {e}"}


# ──────────────────────────────────────────────
# 18. Full-text search over counterfactual narratives
# ──────────────────────────────────────────────

@_register("search_counterfactuals")
def _tool_search_counterfactuals(query: str, size: int = 10, **kwargs) -> dict:
    """Full-text search over counterfactual explanation narratives.

    Uses ES text analysis (tokenisation, stemming, fuzzy matching) on the
    ``explanation_text`` field — searches like "high packet count" work
    naturally.
    """
    try:
        results = wrappers.full_text_search_counterfactuals(query, size=size)
        return {"query": query, "count": len(results), "results": results}
    except Exception as e:
        return {"error": f"Counterfactual search failed: {e}"}

@_register("assess_severity")
def _tool_severity(flow_id: str, **kwargs) -> dict:
    """Compute a severity level for a flow based on feature deviation
    from normal baselines.  Uses a TTL cache for stats to avoid
    repeated ES aggregation queries (millisecond-fast on subsequent calls)."""
    flow = wrappers.get_flow(flow_id)
    if not flow:
        return {"error": f"Flow {flow_id} not found"}

    # Use cached stats if fresh (avoids ES round-trip on every severity call)
    now = time.time()
    if _STATS_CACHE["data"] is not None and (now - _STATS_CACHE["ts"]) < _STATS_TTL:
        stats = _STATS_CACHE["data"]
    else:
        stats = wrappers.feature_stats_by_label()
        _STATS_CACHE["data"] = stats
        _STATS_CACHE["ts"] = now

    # Vectorised z-score computation via numpy (single array pass)
    feats = wrappers.FEATURE_FIELDS
    vals = np.array([flow.get(f, np.nan) for f in feats], dtype=np.float64)
    avgs = np.array([stats.get(f, {}).get("label_0", {}).get("avg", 0) for f in feats], dtype=np.float64)
    stds = np.array([stats.get(f, {}).get("label_0", {}).get("std_deviation", 1) for f in feats], dtype=np.float64)
    stds[stds <= 0] = 1.0
    valid = ~np.isnan(vals)
    z = np.abs((vals - avgs) / stds)
    z[~valid] = 0.0
    scores = {f: round(float(z[i]), 2) for i, f in enumerate(feats) if valid[i]}
    max_z = float(z[valid].max()) if valid.any() else 0.0
    if max_z >= 3:
        level = "high"
    elif max_z >= 1.5:
        level = "medium"
    else:
        level = "low"

    return {
        "flow_id": flow_id,
        "severity": level,
        "max_z_score": round(max_z, 2),
        "feature_z_scores": scores,
        "flow": flow,
    }


# ══════════════════════════════════════════════
# TOOL SCHEMAS (OpenAI function-calling format)
# ══════════════════════════════════════════════

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "es_health_check",
            "description": "Check Elasticsearch cluster health and connectivity.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_flows",
            "description": "Search aggregated network flows in Elasticsearch. Filter by label (0=normal, 1=malicious), src_ip, dst_ip, or minimum packet_count.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "integer", "description": "0=normal, 1=malicious"},
                    "src_ip": {"type": "string", "description": "Source IP filter"},
                    "dst_ip": {"type": "string", "description": "Destination IP filter"},
                    "min_packet_count": {"type": "number", "description": "Minimum packets in flow"},
                    "size": {"type": "integer", "description": "Max results (default 20)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_flow",
            "description": "Retrieve a single flow document by its flow_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flow_id": {"type": "string", "description": "The flow ID"},
                },
                "required": ["flow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Detect anomalous network flows. Use method='label' for ground-truth labels, 'score' for model prediction scores, or 'stats' for statistical outliers (packet_count > mean + 2*std).",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["label", "score", "stats"], "description": "Detection method"},
                    "threshold": {"type": "number", "description": "Score threshold (for method=score)"},
                    "size": {"type": "integer", "description": "Max results"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "feature_stats",
            "description": "Get extended statistics (avg, std, min, max) for each flow feature (packet_count, total_bytes, mean_payload, mean_iat, std_iat) grouped by label (normal vs malicious).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "feature_percentiles",
            "description": "Get percentile distribution (5th, 25th, 50th, 75th, 95th, 99th) for a specific feature grouped by label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "feature": {"type": "string", "description": "Feature name, e.g. packet_count, total_bytes"},
                },
                "required": ["feature"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "significant_terms",
            "description": "Find terms (IPs, protocols) significantly overrepresented in attack traffic vs normal traffic using Elasticsearch significant_terms aggregation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "description": "Field to analyze, e.g. src_ip, dst_ip, protocol"},
                    "foreground_label": {"type": "integer", "description": "Label for foreground set (default 1=malicious)"},
                    "size": {"type": "integer", "description": "Number of top terms"},
                },
                "required": ["field"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "counterfactual_analysis",
            "description": "Run counterfactual analysis on a specific anomalous flow. Finds the nearest normal flow via kNN embedding search, then computes per-feature diffs showing what would need to change for the flow to be classified as normal. This is a core explainability tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flow_id": {"type": "string", "description": "The anomalous flow ID to explain"},
                },
                "required": ["flow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "counterfactual_narrative",
            "description": "Get a human-readable narrative explanation of a flow's counterfactual analysis. Shows what features would need to change for the flow to be classified as normal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flow_id": {"type": "string", "description": "The flow ID"},
                },
                "required": ["flow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_flow",
            "description": "Use Elasticsearch _explain API to understand why a flow document matched a malicious query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flow_id": {"type": "string", "description": "The flow ID"},
                },
                "required": ["flow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_raw_packets",
            "description": "Search individual raw packet records in Elasticsearch. Filter by src_ip, dst_ip, protocol number, or label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src_ip": {"type": "string"},
                    "dst_ip": {"type": "string"},
                    "protocol": {"type": "integer", "description": "IP protocol number (6=TCP, 17=UDP)"},
                    "label": {"type": "integer", "description": "0=normal, 1=malicious"},
                    "size": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_incidents",
            "description": "Find flows most similar to a given flow using kNN embedding search. Useful for historical incident matching and pattern recognition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flow_id": {"type": "string", "description": "The flow ID to find similar incidents for"},
                    "k": {"type": "integer", "description": "Number of similar flows to return (default 5)"},
                },
                "required": ["flow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_severity",
            "description": "Assess the severity level (low/medium/high) of an anomalous flow by computing z-scores of its features against the normal baseline distribution. Returns per-feature deviation scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flow_id": {"type": "string", "description": "The flow ID to assess"},
                },
                "required": ["flow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_edge_counterfactual",
            "description": "Run graph-level edge perturbation counterfactual analysis. Shows which network connections (edges) have the greatest structural impact on the anomaly — what connections would need to not exist for the window to look normal. Core tool for graph-based explainability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "window_index": {"type": "integer", "description": "Graph window index to analyze (default: most anomalous)"},
                    "max_removals": {"type": "integer", "description": "Max edges to perturb (default 5)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_window_comparison",
            "description": "Compare two graph snapshots (time windows) to identify structural and feature differences between normal and anomalous periods. Shows node/edge count changes, feature distribution shifts, and degree distribution changes. Core tool for temporal anomaly explanation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "window_a": {"type": "integer", "description": "First window index (default: most normal)"},
                    "window_b": {"type": "integer", "description": "Second window index (default: most anomalous)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ml_anomaly_records",
            "description": "Fetch Elasticsearch ML anomaly detection records above a score threshold. Each record includes influencers — the field values that drove the anomaly. This is ES-native feature attribution that complements the GNN-based counterfactuals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "ML job ID (default: incidentlens-flow-anomaly)"},
                    "min_score": {"type": "number", "description": "Minimum anomaly score (default: 75)"},
                    "size": {"type": "integer", "description": "Max records to return"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ml_influencers",
            "description": "Fetch ES ML influencer results — tells you WHICH src_ip, dst_ip, or protocol values contributed most to detected anomalies. Provides root-cause signal that complements GNN-based explanations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "ML job ID (default: incidentlens-flow-anomaly)"},
                    "min_score": {"type": "number", "description": "Minimum influencer score (default: 50)"},
                    "size": {"type": "integer", "description": "Max influencers to return"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "severity_breakdown",
            "description": "Compute severity distribution across all flows using ES runtime fields and aggregations. Runs entirely server-side via Painless scripts — zero Python iteration. Returns counts for critical/high/medium/low severity and traffic volume categories.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_counterfactuals",
            "description": "Full-text search over counterfactual explanation narratives. Uses ES text analysis (tokenisation, stemming, fuzzy matching) on explanation_text — natural language queries like 'high packet count from source' work naturally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query"},
                    "size": {"type": "integer", "description": "Max results (default 10)"},
                },
                "required": ["query"],
            },
        },
    },
]


# ══════════════════════════════════════════════
# DISPATCHER
# ══════════════════════════════════════════════

def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None for JSON safety."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def dispatch(tool_name: str, arguments: dict[str, Any]) -> str:
    """Call a tool by name with the given arguments.

    Returns JSON-serialised result string (for insertion into LLM messages).
    """
    fn = _REGISTRY.get(tool_name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        result = fn(**arguments)
        clean = _sanitize_for_json(result)
        return json.dumps(clean, default=str)
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


def list_tools() -> list[str]:
    """Return names of all registered tools."""
    return list(_REGISTRY.keys())
