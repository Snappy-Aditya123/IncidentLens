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

import wrappers
import agent_tools
from agent import IncidentAgent, AgentConfig

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
        es_health = wrappers.health_check()
        return {
            "server": "ok",
            "elasticsearch": es_health["status"],
            "indices": {
                "flows": wrappers.get_client().indices.exists(index=wrappers.FLOWS_INDEX),
                "embeddings": wrappers.get_client().indices.exists(index=wrappers.EMBEDDINGS_INDEX),
                "counterfactuals": wrappers.get_client().indices.exists(index=wrappers.COUNTERFACTUALS_INDEX),
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
