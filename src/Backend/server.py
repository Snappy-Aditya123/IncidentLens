"""
IncidentLens Server
===================
FastAPI backend with WebSocket streaming for the LLM investigation agent.

**Architecture note — direct wrappers vs dispatch:**
    REST endpoints call ``wrappers.*`` functions directly (via asyncio.to_thread)
    instead of routing through ``agent_tools.dispatch()``.  This eliminates the
    unnecessary JSON serialise→deserialise round-trip on every request and keeps
    the event loop unblocked.  ``agent_tools.dispatch()`` is reserved for the
    LLM agent's tool-calling loop where JSON serialisation is required.

Endpoints:
    GET  /health                          - Server + ES health
    POST /api/investigate                 - Start investigation (returns JSON)
    WS   /ws/investigate                  - Stream investigation events in real time
    POST /api/detect                      - Quick anomaly detection (no LLM)
    GET  /api/flows                       - List flows from ES
    GET  /api/stats                       - Feature statistics
    POST /api/counterfactual              - Run counterfactual for a flow
    GET  /api/severity/{flow_id}          - Assess severity of a flow
    GET  /api/similar/{flow_id}           - Find similar historical incidents
    GET  /api/tools                       - List available agent tools
    GET  /api/incidents                   - List incidents (anomalous flows)
    GET  /api/incidents/{id}              - Single incident detail
    GET  /api/incidents/{id}/graph        - Network graph for D3 vis
    GET  /api/incidents/{id}/logs         - ES-style log entries
    --- NEW ES-native endpoints ---
    GET  /api/severity-breakdown          - Runtime-field severity distribution
    GET  /api/flows/search                - Paginated flow search (search_after + PIT)
    GET  /api/flows/severity              - Query flows by runtime severity level
    GET  /api/counterfactuals/search      - Full-text search over CF narratives
    GET  /api/aggregate/{field}           - Composite aggregation (paginated buckets)
    GET  /api/ml/anomalies                - ES ML anomaly detection records
    GET  /api/ml/influencers              - ES ML influencer results
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import time
from datetime import timezone
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import src.Backend.wrappers as wrappers
import src.Backend.agent_tools as agent_tools
from src.Backend.agent import IncidentAgent, AgentConfig

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="IncidentLens API",
    description="AI-powered network incident investigation agent with Elasticsearch-native analytics",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # lock down in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agent singleton (reused across requests)
_agent: IncidentAgent | None = None


def _get_agent() -> IncidentAgent:
    global _agent
    if _agent is None:
        _agent = IncidentAgent(AgentConfig())
    return _agent


# ──────────────────────────────────────────────
# TTL cache for detect results (avoids repeated ES hits)
# ──────────────────────────────────────────────
_DETECT_CACHE: dict[str, Any] = {"data": None, "ts": 0.0, "key": ""}
_DETECT_TTL = 15.0  # seconds


def _cached_detect(method: str = "label", threshold: float = 0.5, size: int = 50) -> dict:
    """Detect anomalies with a short TTL cache.

    The detect endpoint is called by multiple routes (/api/incidents,
    /api/incidents/{id}/graph, /api/detect).  Caching avoids redundant
    ES queries within a 15-second window.
    """
    cache_key = f"{method}:{threshold}:{size}"
    now = time.time()
    if _DETECT_CACHE["data"] is not None and _DETECT_CACHE["key"] == cache_key and (now - _DETECT_CACHE["ts"]) < _DETECT_TTL:
        return _DETECT_CACHE["data"]

    # Call the tool directly (returns dict, not JSON string)
    result = agent_tools._REGISTRY["detect_anomalies"](method=method, threshold=threshold, size=size)
    _DETECT_CACHE["data"] = result
    _DETECT_CACHE["ts"] = now
    _DETECT_CACHE["key"] = cache_key
    return result


# ──────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────

class InvestigateRequest(BaseModel):
    query: str = ""
    """User's investigation question.  Empty = auto-detect anomalies."""


class DetectRequest(BaseModel):
    method: str = "label"
    threshold: float = 0.5
    size: int = 50


class CounterfactualRequest(BaseModel):
    flow_id: str


class PaginatedSearchRequest(BaseModel):
    query: dict | None = None
    size: int = 20
    search_after: list | None = None
    pit_id: str | None = None


# ──────────────────────────────────────────────
# REST endpoints — direct wrappers calls (no dispatch overhead)
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    """Server + Elasticsearch health."""
    try:
        def _check():
            es = wrappers.get_client()
            es_health = wrappers.health_check(es)
            return {
                "server": "ok",
                "elasticsearch": es_health["status"],
                "indices": {
                    "flows": es.indices.exists(index=wrappers.FLOWS_INDEX),
                    "embeddings": es.indices.exists(index=wrappers.EMBEDDINGS_INDEX),
                    "counterfactuals": es.indices.exists(index=wrappers.COUNTERFACTUALS_INDEX),
                },
            }
        return await asyncio.to_thread(_check)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"server": "ok", "elasticsearch": "unreachable", "error": str(e)},
        )


@app.post("/api/investigate")
async def investigate(req: InvestigateRequest):
    """Run a full investigation (non-streaming, returns all events)."""
    agent = _get_agent()
    query = req.query or (
        "Detect any anomalous network flows, investigate the top anomalies, "
        "run counterfactual analysis, assess severity, and provide a full "
        "investigation summary."
    )

    events: list[dict] = []
    def _run():
        for event in agent.investigate(query):
            events.append(event)

    await asyncio.to_thread(_run)
    return {"events": events}


@app.post("/api/detect")
async def detect(req: DetectRequest):
    """Quick anomaly detection (no LLM agent).

    Calls wrappers directly — no dispatch JSON round-trip.
    Results are cached for 15 seconds to reduce ES load.
    """
    try:
        result = await asyncio.to_thread(
            _cached_detect, req.method, req.threshold, req.size,
        )
        if "error" in result:
            return JSONResponse(status_code=502, content=result)
        return result
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/flows")
async def list_flows(
    label: int | None = Query(None),
    src_ip: str | None = Query(None),
    dst_ip: str | None = Query(None),
    size: int = Query(20),
):
    """List flows with optional filters.  Calls wrappers.search_flows directly."""
    try:
        must_clauses: list[dict] = []
        if label is not None:
            must_clauses.append({"term": {"label": label}})
        if src_ip:
            must_clauses.append({"term": {"src_ip": src_ip}})
        if dst_ip:
            must_clauses.append({"term": {"dst_ip": dst_ip}})

        query = {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}}
        flows = await asyncio.to_thread(wrappers.search_flows, query, size)
        return {"count": len(flows), "flows": flows}
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/stats")
async def feature_stats():
    """Feature statistics grouped by label.  Uses ES extended_stats aggregation."""
    try:
        stats = await asyncio.to_thread(wrappers.feature_stats_by_label)
        return stats
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.post("/api/counterfactual")
async def counterfactual(req: CounterfactualRequest):
    """Run counterfactual analysis for a specific flow.

    Uses ES kNN vector search to find nearest normal flow, then computes
    per-feature diffs.
    """
    try:
        def _run_cf():
            es = wrappers.get_client()
            body = {"query": {"term": {"flow_id": req.flow_id}}, "size": 1}
            resp = es.search(index=wrappers.EMBEDDINGS_INDEX, body=body)
            hits = resp["hits"]["hits"]
            if not hits:
                return {"error": f"No embedding found for flow {req.flow_id}"}
            emb = hits[0]["_source"]["embedding"]
            cf = wrappers.build_and_index_counterfactual(
                anomalous_flow_id=req.flow_id, query_embedding=emb, es=es,
            )
            if cf is None:
                return {"error": "No normal neighbour found"}
            return cf
        result = await asyncio.to_thread(_run_cf)
        if "error" in result:
            return JSONResponse(status_code=502, content=result)
        return result
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/severity/{flow_id}")
async def severity(flow_id: str):
    """Assess severity of a flow using z-score deviation from baseline."""
    try:
        result = await asyncio.to_thread(
            agent_tools._REGISTRY["assess_severity"], flow_id=flow_id,
        )
        if "error" in result:
            return JSONResponse(status_code=502, content=result)
        return result
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/similar/{flow_id}")
async def similar(flow_id: str, k: int = Query(5)):
    """Find similar historical incidents via ES kNN embedding search."""
    try:
        def _find():
            es = wrappers.get_client()
            body = {"query": {"term": {"flow_id": flow_id}}, "size": 1}
            resp = es.search(index=wrappers.EMBEDDINGS_INDEX, body=body)
            hits = resp["hits"]["hits"]
            if not hits:
                return {"error": f"No embedding for {flow_id}"}
            emb = hits[0]["_source"]["embedding"]
            results = wrappers.knn_search(emb, k=k)
            return {"query_flow": flow_id, "similar": results}
        result = await asyncio.to_thread(_find)
        if "error" in result:
            return JSONResponse(status_code=502, content=result)
        return result
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/tools")
async def list_tools():
    """List available agent tools."""
    return {"tools": agent_tools.list_tools()}


# ──────────────────────────────────────────────
# Incident-oriented endpoints (frontend convenience)
# ──────────────────────────────────────────────

def _flow_to_incident(flow: dict) -> dict:
    """Map a raw ES flow doc to a frontend Incident shape.

    Uses the same severity thresholds as the runtime field — keeping
    Python and ES-level severity always in sync.
    """
    score = flow.get("prediction_score", 0.85 if flow.get("label") == 1 else 0.2)
    # Use runtime-computed severity if available, else compute locally
    sev = flow.get("severity_level")
    if not sev:
        if score > 0.9:
            sev = "critical"
        elif score > 0.7:
            sev = "high"
        elif score > 0.5:
            sev = "medium"
        else:
            sev = "low"

    src = flow.get("src_ip", "unknown")
    dst = flow.get("dst_ip", "unknown")
    return {
        "id": flow.get("_id", ""),
        "title": f"Anomalous Flow: {src} → {dst}",
        "severity": sev,
        "status": "investigating",
        "timestamp": flow.get("@timestamp", flow.get("timestamp", "")),
        "affectedSystems": [ip for ip in [src, dst] if ip != "unknown"],
        "description": (
            f"Detected anomalous traffic — {flow.get('packet_count', '?')} packets, "
            f"{flow.get('total_bytes', '?')} bytes. "
            f"Score: {round(score * 100)}%."
        ),
        "anomalyScore": score,
    }


@app.get("/api/incidents")
async def list_incidents(size: int = Query(50)):
    """Return anomalous flows shaped as frontend Incident objects.

    Uses the detect cache to avoid redundant ES queries.
    """
    try:
        data = await asyncio.to_thread(_cached_detect, "label", 0.5, size)
        incidents = [_flow_to_incident(f) for f in data.get("flows", [])]
        return {"count": len(incidents), "incidents": incidents}
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/incidents/{incident_id}")
async def get_incident(incident_id: str):
    """Return a single incident by flow ID.  Direct ES GET — no fetch-all."""
    try:
        flow = await asyncio.to_thread(wrappers.get_flow, incident_id)
        if flow is None:
            return JSONResponse(status_code=404, content={"error": f"Flow {incident_id} not found"})
        flow["_id"] = incident_id
        return _flow_to_incident(flow)
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/incidents/{incident_id}/graph")
async def incident_graph(incident_id: str, size: int = Query(30)):
    """Build a network graph from anomalous flows for the D3 visualisation.

    Scopes by the incident's src_ip/dst_ip to show only relevant traffic.
    """
    try:
        def _build_graph():
            # Fetch the incident to scope the graph to related IPs
            inc_data = wrappers.get_flow(incident_id)
            scope_ips: set[str] = set()
            if inc_data:
                for key in ("src_ip", "dst_ip"):
                    if inc_data.get(key):
                        scope_ips.add(inc_data[key])

            # Get anomalous flows (cached)
            data = _cached_detect("label", 0.5, size)
            all_flows = data.get("flows", [])

            # Scope to incident's IPs
            flows = [
                f for f in all_flows
                if not scope_ips or f.get("src_ip") in scope_ips or f.get("dst_ip") in scope_ips
            ] if scope_ips else all_flows

            # Build node + edge sets
            node_map: dict[str, dict[str, Any]] = {}
            edges: list[dict[str, Any]] = []

            for f in flows:
                src, dst = f.get("src_ip"), f.get("dst_ip")
                score = f.get("prediction_score", 0.8 if f.get("label") == 1 else 0.15)
                for ip in (src, dst):
                    if ip:
                        prev = node_map.get(ip, {"risk": 0, "label": 0})
                        node_map[ip] = {
                            "risk": max(prev["risk"], score),
                            "label": max(prev["label"], f.get("label", 0)),
                        }
                if src and dst:
                    edges.append({
                        "source": src,
                        "target": dst,
                        "type": "data_flow",
                        "weight": min((f.get("packet_count", 1) or 1) / 100, 10),
                        "anomalous": f.get("label", 0) == 1,
                    })

            nodes = []
            for ip, info in node_map.items():
                status = "compromised" if info["risk"] > 0.8 else "suspicious" if info["risk"] > 0.5 else "normal"
                nodes.append({
                    "id": ip, "label": ip, "type": "server",
                    "status": status, "risk": round(info["risk"], 3),
                })

            return {"nodes": nodes, "edges": edges}

        return await asyncio.to_thread(_build_graph)
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/incidents/{incident_id}/logs")
async def incident_logs(incident_id: str, size: int = Query(20)):
    """Return ES-style log entries for the frontend log viewer."""
    try:
        def _build_logs():
            inc_data = wrappers.get_flow(incident_id)
            query_clauses: list[dict] = []
            if inc_data and inc_data.get("src_ip"):
                query_clauses.append({"term": {"src_ip": inc_data["src_ip"]}})
            query = {"bool": {"must": query_clauses}} if query_clauses else {"match_all": {}}
            flows = wrappers.search_flows(query=query, size=size)

            logs = []
            for f in flows[:size]:
                logs.append({
                    "timestamp": f.get("@timestamp", datetime.datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
                    "source": f.get("src_ip", "unknown"),
                    "message": (
                        f"Flow {f.get('src_ip', '?')} → {f.get('dst_ip', '?')}: "
                        f"{f.get('packet_count', 0)} pkts, {f.get('total_bytes', 0)} bytes"
                        + (" [ANOMALOUS]" if f.get("label") == 1 else "")
                    ),
                    "level": "CRITICAL" if f.get("label") == 1 else "INFO",
                })

            return {
                "totalHits": len(logs),
                "logs": logs,
                "query": {
                    "bool": {
                        "must": [{"term": {"label": 1}}],
                        "filter": [{"range": {"@timestamp": {"gte": "now-1h"}}}],
                    }
                },
            }

        return await asyncio.to_thread(_build_logs)
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


# ──────────────────────────────────────────────
# ES-NATIVE ENDPOINTS (runtime fields, aggregations, ML, PIT)
# These showcase Elasticsearch capabilities beyond basic CRUD.
# ──────────────────────────────────────────────

@app.get("/api/severity-breakdown")
async def severity_breakdown():
    """Severity distribution using ES runtime fields + terms aggregation.

    Computes severity_level and traffic_volume_category entirely
    server-side via Painless scripts — zero Python iteration.
    """
    try:
        raw = await asyncio.to_thread(wrappers.aggregate_severity_breakdown)
        # Transform to match SeverityBreakdownResponse contract
        severity_levels = raw.get("severity", {})
        volume_cats = raw.get("volume", {})
        total = sum(severity_levels.values()) if severity_levels else 0
        return {
            "severity_levels": severity_levels,
            "traffic_volume_categories": volume_cats,
            "total_flows": total,
        }
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/flows/severity")
async def flows_by_severity(
    severity: str = Query("critical", description="critical|high|medium|low"),
    size: int = Query(20),
):
    """Query flows by runtime-computed severity level.

    Leverages ES runtime fields — the severity classification is applied
    at query time using Painless scripts, not stored in the document.
    """
    try:
        flows = await asyncio.to_thread(
            wrappers.search_flows_with_severity, None, severity, size,
        )
        return {"severity": severity, "count": len(flows), "flows": flows}
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.post("/api/flows/search")
async def paginated_flow_search(req: PaginatedSearchRequest):
    """Paginated flow search using ES search_after + point-in-time.

    Unlike from+size, this doesn't degrade with depth and provides a
    consistent snapshot of the data — the recommended ES approach for
    deep pagination.
    """
    try:
        result = await asyncio.to_thread(
            wrappers.search_with_pagination,
            req.query, req.size, req.search_after, req.pit_id,
        )
        return result
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/counterfactuals/search")
async def search_counterfactuals(
    q: str = Query(..., description="Natural-language search query"),
    size: int = Query(10),
):
    """Full-text search over counterfactual explanation narratives.

    Uses ES text analysis (tokenisation, stemming, fuzzy matching) on the
    `explanation_text` field — enables natural-language queries like
    "high packet count from source" to find relevant CF explanations.
    """
    try:
        results = await asyncio.to_thread(
            wrappers.full_text_search_counterfactuals, q, size,
        )
        return {"query": q, "count": len(results), "results": results}
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/aggregate/{field}")
async def aggregate_field(
    field: str,
    size: int = Query(100),
):
    """Composite aggregation on any keyword field.

    Uses ES composite aggregation for paginated bucket enumeration —
    handles unbounded cardinality (many unique IPs) where a standard
    terms aggregation would silently truncate.
    """
    try:
        raw_buckets = await asyncio.to_thread(
            wrappers.composite_aggregation, field, None, size,
        )
        # Flatten composite keys: {"key": {"src_ip": "10.0.0.1"}} → {"key": "10.0.0.1"}
        buckets = []
        for b in raw_buckets:
            key_obj = b.get("key", {})
            flat_key = next(iter(key_obj.values()), "") if isinstance(key_obj, dict) else str(key_obj)
            buckets.append({"key": flat_key, "doc_count": b.get("doc_count", 0)})
        return {"field": field, "buckets": buckets, "total_buckets": len(buckets)}
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/ml/anomalies")
async def ml_anomalies(
    job_id: str = Query("incidentlens-flow-anomaly"),
    min_score: float = Query(75.0),
    size: int = Query(50),
):
    """Fetch ES ML anomaly detection records.

    Elasticsearch's ML feature automatically learns normal traffic baselines
    and flags deviations.  Each record includes influencers — the field
    values that drove the anomaly — providing direct feature attribution.
    """
    try:
        records = await asyncio.to_thread(
            wrappers.get_anomaly_records, job_id, min_score, size,
        )
        return {"job_id": job_id, "count": len(records), "records": records}
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/ml/influencers")
async def ml_influencers(
    job_id: str = Query("incidentlens-flow-anomaly"),
    min_score: float = Query(50.0),
    size: int = Query(50),
):
    """Fetch ES ML influencer results.

    Influencers tell you WHICH src_ip, dst_ip, or protocol values
    contributed most to detected anomalies — the raw material for
    root-cause analysis and counterfactual explanations.
    """
    try:
        influencers = await asyncio.to_thread(
            wrappers.get_influencers, job_id, min_score, size,
        )
        return {"job_id": job_id, "count": len(influencers), "influencers": influencers}
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


# ──────────────────────────────────────────────
# WebSocket endpoint (real-time streaming)
# ──────────────────────────────────────────────

@app.websocket("/ws/investigate")
async def ws_investigate(websocket: WebSocket):
    """Stream investigation events to the frontend.

    Protocol:
        1. Client connects.
        2. Client sends JSON: {"query": "..."} or {} for auto-detect.
        3. Server streams events as JSON lines.
        4. Server sends {"type": "done"} and closes.
    """
    await websocket.accept()
    logger.info("WebSocket connected")

    try:
        # Receive the investigation query
        raw = await websocket.receive_text()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"query": raw}

        query = payload.get("query", "").strip()
        if not query:
            query = (
                "Detect any anomalous network flows, investigate the top "
                "anomalies, run counterfactual analysis on them, assess "
                "severity, and provide a full investigation summary with "
                "root-cause analysis and recommended actions."
            )

        agent = _get_agent()

        # Stream events from a background thread
        async def _stream():
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[dict | None] = asyncio.Queue()

            def _producer():
                try:
                    for event in agent.investigate(query):
                        loop.call_soon_threadsafe(queue.put_nowait, event)
                except Exception as e:
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {"type": "error", "content": str(e)},
                    )
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            fut = loop.run_in_executor(None, _producer)

            while True:
                event = await queue.get()
                if event is None:
                    break
                await websocket.send_json(event)

            await fut

        await _stream()
        await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Prefer: python main.py serve
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "src.Backend.server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
