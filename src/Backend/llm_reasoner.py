from __future__ import annotations

import asyncio
import json
import logging
import os
import smtplib
import time
from email.message import EmailMessage
from typing import Any, Callable, Dict

from openai import AsyncOpenAI

import src.Backend.wrappers as wrappers

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are IncidentLens Autonomous Security Analyst.

You analyze graph-level network anomalies and investigate them using
available diagnostic tools.

Rules:
- Think step-by-step.
- Call tools when more data is required.
- Never hallucinate data.
- Always base conclusions on tool results.
- Return structured JSON with:
  - summary
  - risk_level (low | medium | high | critical)
  - reasoning
  - recommended_action
  - confidence (0.0–1.0)

If risk_level is high or critical, you must call send_email_alert.
Do not send alerts for low or medium risk.
"""


class LLMReasoner:

    def __init__(self, es_client):
        self.es = es_client
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_iterations = 5

        # Lazy init: AsyncOpenAI will read OPENAI_API_KEY at call time
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — LLM reasoning will be unavailable")
        self.client = AsyncOpenAI(api_key=api_key) if api_key else None

        self.tools: Dict[str, Callable[..., Any]] = {
            "get_flow_details": self.get_flow_details,
            "get_graph_summary": self.get_graph_summary,
            "get_severity_breakdown": self.get_severity_breakdown,
            "get_recent_windows": self.get_recent_windows,
            "send_email_alert": self.send_email_alert,
        }

    # ==========================================================
    # Tool Implementations (Controlled Diagnostic Interface)
    # ==========================================================

    def get_flow_details(self, flow_id: str) -> dict:
        return wrappers.get_flow_by_id(flow_id, es=self.es) or {}

    def get_graph_summary(self, window_id: int) -> dict:
        return wrappers.get_graph_summary(window_id, es=self.es) or {}

    def get_severity_breakdown(self) -> dict:
        return wrappers.aggregate_severity_breakdown(es=self.es)

    def get_recent_windows(self, n: int = 5) -> list:
        return wrappers.get_recent_graph_summaries(n=n, es=self.es)

    # ==========================================================
    # Tool Schema for OpenAI
    # ==========================================================

    def tool_schema(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_flow_details",
                    "description": "Retrieve full indexed flow document by flow_id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "flow_id": {"type": "string"}
                        },
                        "required": ["flow_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_graph_summary",
                    "description": "Retrieve graph-level summary stats for a given window_id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "window_id": {"type": "integer"}
                        },
                        "required": ["window_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_severity_breakdown",
                    "description": "Retrieve overall severity distribution statistics.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_recent_windows",
                    "description": "Retrieve recent graph summary stats.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n": {"type": "integer"}
                        }
                    }
                }
            },
            {
            "type": "function",
            "function": {
                "name": "send_email_alert",
                "description": "Send an alert email if risk is high.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "window_id": {"type": "integer"},
                        "summary": {"type": "string"},
                        "risk_level": {"type": "string"}
                    },
                    "required": ["window_id", "summary", "risk_level"]
                    }
                }
            },
        ]

    # ==========================================================
    # Reasoning Loop
    # ==========================================================

    async def analyze_window(self, context: dict) -> dict:
        if self.client is None:
            return {"summary": "LLM reasoning unavailable (no API key).",
                    "risk_level": "unknown", "confidence": 0.0}

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context)}
        ]

        for _ in range(self.max_iterations):

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_schema(),
                tool_choice="auto"
            )

            msg = response.choices[0].message

            # If LLM wants to call tools
            if msg.tool_calls:

                messages.append(msg)

                for call in msg.tool_calls:
                    tool_name = call.function.name
                    args = json.loads(call.function.arguments or "{}")

                    fn = self.tools.get(tool_name)
                    if fn is None:
                        result = {"error": f"Unknown tool: {tool_name}"}
                    else:
                        # ES calls are synchronous — offload to thread pool
                        result = await asyncio.to_thread(fn, **args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result)
                    })

                continue

            # Final reasoning output
            try:
                output = json.loads(msg.content)
            except Exception:
                output = {
                    "summary": msg.content,
                    "risk_level": "unknown",
                    "reasoning": "Unstructured output",
                    "recommended_action": "Manual review required.",
                    "confidence": 0.5
                }

            self.store_insight(context["window_id"], output)

            return output

        return {"summary": "Max reasoning depth reached."}

    # ==========================================================
    # Store Insight in ES
    # ==========================================================

    def store_insight(self, window_id: int, insight: dict):

        doc = {
            "window_id": window_id,
            "timestamp": int(time.time() * 1000),
            **insight
        }

        self.es.index(
            index="incidentlens-llm-insights",
            document=doc
        )

    def send_email_alert(self, window_id: int, summary: str, risk_level: str):
        """
        Send alert email directly via SMTP (no Elasticsearch Watcher).
        """

        sender = os.getenv("ALERT_EMAIL_SENDER")
        password = os.getenv("ALERT_EMAIL_PASSWORD")
        recipient = os.getenv("ALERT_EMAIL_RECIPIENT")

        if not sender or not password or not recipient:
            return {"status": "email_config_missing"}

        msg = EmailMessage()
        msg["Subject"] = f"[IncidentLens] {risk_level.upper()} Risk (Window {window_id})"
        msg["From"] = sender
        msg["To"] = recipient
        msg.set_content(summary)

        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)

            return {"status": "email_sent"}

        except Exception as e:
            return {"status": "email_failed", "error": str(e)}