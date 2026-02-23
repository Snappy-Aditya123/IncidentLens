"""
IncidentLens Server
===================
FastAPI backend with WebSocket streaming for the LLM investigation agent.

Endpoints:
    GET  /health           - Server + ES health
    POST /api/investigate   - Start investigation (returns JSON)
    WS   /ws/investigate    - Stream investigation events in real time
    POST /api/detect        - Quick anomaly detection (no LLM)
    GET  /api/flows         - List flows from ES
    GET  /api/stats         - Feature statistics
    POST /api/counterfactual - Run counterfactual for a flow

Flow:
    Frontend opens WebSocket -> sends query ->
    Backend streams reasoning events:
        {"type": "thinking",    "content": "..."}
        {"type": "tool_call",   "tool": "...", "arguments": {...}}
        {"type": "tool_result", "tool": "...", "result": "..."}
        {"type": "conclusion",  "content": "..."}
    Frontend updates UI live
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
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
    description="AI-powered network incident investigation agent",
    version="0.1.0",
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


# ──────────────────────────────────────────────
# REST endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    """Server + Elasticsearch health."""
    try:
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

    events = []
    # run sync generator in thread to not block event loop
    def _run():
        for event in agent.investigate(query):
            events.append(event)

    await asyncio.to_thread(_run)
    return {"events": events}


@app.post("/api/detect")
async def detect(req: DetectRequest):
    """Quick anomaly detection (no LLM agent)."""
    result_str = agent_tools.dispatch("detect_anomalies", {
        "method": req.method,
        "threshold": req.threshold,
        "size": req.size,
    })
    return json.loads(result_str)


@app.get("/api/flows")
async def list_flows(
    label: int | None = Query(None),
    src_ip: str | None = Query(None),
    dst_ip: str | None = Query(None),
    size: int = Query(20),
):
    """List flows with optional filters."""
    args: dict[str, Any] = {"size": size}
    if label is not None:
        args["label"] = label
    if src_ip:
        args["src_ip"] = src_ip
    if dst_ip:
        args["dst_ip"] = dst_ip
    result_str = agent_tools.dispatch("search_flows", args)
    return json.loads(result_str)


@app.get("/api/stats")
async def feature_stats():
    """Feature statistics grouped by label."""
    result_str = agent_tools.dispatch("feature_stats", {})
    return json.loads(result_str)


@app.post("/api/counterfactual")
async def counterfactual(req: CounterfactualRequest):
    """Run counterfactual analysis for a specific flow."""
    result_str = agent_tools.dispatch("counterfactual_analysis", {
        "flow_id": req.flow_id,
    })
    return json.loads(result_str)


@app.get("/api/severity/{flow_id}")
async def severity(flow_id: str):
    """Assess severity of a flow."""
    result_str = agent_tools.dispatch("assess_severity", {"flow_id": flow_id})
    return json.loads(result_str)


@app.get("/api/similar/{flow_id}")
async def similar(flow_id: str, k: int = Query(5)):
    """Find similar historical incidents."""
    result_str = agent_tools.dispatch("find_similar_incidents", {
        "flow_id": flow_id, "k": k,
    })
    return json.loads(result_str)


@app.get("/api/tools")
async def list_tools():
    """List available agent tools."""
    return {"tools": agent_tools.list_tools()}


# ──────────────────────────────────────────────
# Incident-oriented endpoints (frontend convenience)
# ──────────────────────────────────────────────

def _flow_to_incident(flow: dict) -> dict:
    """Map a raw ES flow doc to a frontend Incident shape."""
    score = flow.get("prediction_score", 0.85 if flow.get("label") == 1 else 0.2)
    if score > 0.9:
        severity = "critical"
    elif score > 0.7:
        severity = "high"
    elif score > 0.5:
        severity = "medium"
    else:
        severity = "low"

    src = flow.get("src_ip", "unknown")
    dst = flow.get("dst_ip", "unknown")
    return {
        "id": flow.get("_id", ""),
        "title": f"Anomalous Flow: {src} → {dst}",
        "severity": severity,
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
    """Return anomalous flows shaped as frontend Incident objects."""
    result_str = agent_tools.dispatch("detect_anomalies", {
        "method": "label", "size": size,
    })
    data = json.loads(result_str)
    incidents = [_flow_to_incident(f) for f in data.get("flows", [])]
    return {"count": len(incidents), "incidents": incidents}


@app.get("/api/incidents/{incident_id}")
async def get_incident(incident_id: str):
    """Return a single incident by flow ID."""
    result_str = agent_tools.dispatch("get_flow", {"flow_id": incident_id})
    flow = json.loads(result_str)
    if "error" in flow:
        return JSONResponse(status_code=404, content=flow)
    flow["_id"] = incident_id
    return _flow_to_incident(flow)


@app.get("/api/incidents/{incident_id}/graph")
async def incident_graph(incident_id: str, size: int = Query(30)):
    """Build a network graph from anomalous flows for the D3 visualisation."""
    result_str = agent_tools.dispatch("detect_anomalies", {
        "method": "label", "size": size,
    })
    data = json.loads(result_str)
    flows = data.get("flows", [])

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
            "id": ip,
            "label": ip,
            "type": "server",
            "status": status,
            "risk": round(info["risk"], 3),
        })

    return {"nodes": nodes, "edges": edges}


@app.get("/api/incidents/{incident_id}/logs")
async def incident_logs(incident_id: str, size: int = Query(20)):
    """Return ES-style log entries for the frontend log viewer."""
    import datetime

    result_str = agent_tools.dispatch("search_flows", {"size": size})
    data = json.loads(result_str)
    flows = data.get("flows", [])

    logs = []
    for f in flows[:size]:
        logs.append({
            "timestamp": f.get("@timestamp", datetime.datetime.utcnow().isoformat() + "Z"),
            "source": f.get("src_ip", "unknown"),
            "message": (
                f"Flow {f.get('src_ip', '?')} → {f.get('dst_ip', '?')}: "
                f"{f.get('packet_count', 0)} pkts, {f.get('total_bytes', 0)} bytes"
                + (" [ANOMALOUS]" if f.get("label") == 1 else "")
            ),
            "level": "CRITICAL" if f.get("label") == 1 else "INFO",
        })

    return {
        "totalHits": data.get("count", len(logs)),
        "logs": logs,
        "query": {
            "bool": {
                "must": [{"term": {"label": 1}}],
                "filter": [{"range": {"@timestamp": {"gte": "now-1h"}}}],
            }
        },
    }


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
            # Use a queue to bridge sync generator -> async send
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

            # Run the sync generator in a thread
            fut = loop.run_in_executor(None, _producer)

            while True:
                event = await queue.get()
                if event is None:
                    break
                await websocket.send_json(event)

            # Ensure producer thread completed cleanly
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
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
