"""
IncidentLens LLM Agent
======================
Multi-step reasoning agent that investigates network incidents by calling
tools (ES queries, counterfactual analysis, severity assessment) in an
iterative hypothesis -> verification -> conclusion loop.

Supports **streaming** — every reasoning step, tool call, and conclusion
is yielded as a typed event dict so the caller (WebSocket server) can
push updates to the frontend in real time.

The agent uses the OpenAI Chat Completions API (works with any
OpenAI-compatible endpoint: OpenAI, Azure OpenAI, local Ollama, etc.).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from openai import OpenAI

import src.Backend.agent_tools as agent_tools

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

@dataclass
class AgentConfig:
    """Runtime configuration for the agent."""
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str | None = os.getenv("OPENAI_BASE_URL", None)
    max_steps: int = 15          # safety cap on reasoning loop
    temperature: float = 0.1     # low temp for deterministic tool use
    max_tokens: int = 4096


SYSTEM_PROMPT = """\
You are **IncidentLens**, an expert AI agent for investigating network security \
incidents.  You work inside an operational pipeline that combines Elasticsearch \
(logs, metrics, graph data), a GNN-based anomaly prediction model, and \
counterfactual explainability research.

Your job:
1. **Detect** anomalous network flows (via labels, model scores, or statistics).
2. **Investigate** using ES queries, feature analysis, and counterfactual \
   explanations.
3. **Explain** root causes — which features (packet_count, total_bytes, \
   mean_payload, mean_iat, std_iat) deviate from normal and by how much.
4. **Assess severity** (low / medium / high) based on statistical deviation.
5. **Recommend actions** (block IP, rate-limit, investigate further, etc.).

Investigation protocol:
- Start by detecting anomalies (detect_anomalies tool).
- For each anomalous flow, assess severity, then run counterfactual analysis.
- Use graph_edge_counterfactual to identify which connections drive the anomaly.
- Use graph_window_comparison to compare normal vs anomalous time windows.
- Use feature_stats and significant_terms to understand the broader pattern.
- Use find_similar_incidents to check for historical matches.
- Synthesise findings into a structured conclusion.

Output format for your FINAL answer (after all tool calls):
```
## Investigation Summary
**Incident**: <one-line description>
**Severity**: <low|medium|high>
**Root Cause**: <ranked list of contributing factors>
**Evidence**: <key data points from tools>
**Recommendation**: <actionable steps>
**Confidence**: <low|medium|high>
```

Always think step-by-step and explain your reasoning before calling each tool. \
Never fabricate data — only use information returned by tools.
"""


# ──────────────────────────────────────────────
# Event types streamed to the frontend
# ──────────────────────────────────────────────

def _event(event_type: str, **data) -> dict:
    return {"type": event_type, "timestamp": time.time(), **data}


def thinking_event(text: str) -> dict:
    return _event("thinking", content=text)


def tool_call_event(tool_name: str, arguments: dict) -> dict:
    return _event("tool_call", tool=tool_name, arguments=arguments)


def tool_result_event(tool_name: str, result: str) -> dict:
    # Truncate huge results for the stream (full result still goes to LLM)
    preview = result[:2000] + "..." if len(result) > 2000 else result
    return _event("tool_result", tool=tool_name, result=preview)


def conclusion_event(text: str) -> dict:
    return _event("conclusion", content=text)


def error_event(message: str) -> dict:
    return _event("error", content=message)


def status_event(message: str) -> dict:
    return _event("status", content=message)


# ──────────────────────────────────────────────
# Agent class
# ──────────────────────────────────────────────

class IncidentAgent:
    """Multi-step reasoning agent with tool-calling capabilities."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self._client = OpenAI(
            api_key=self.config.api_key or "not-set",
            base_url=self.config.base_url,
        )
        self._tools = agent_tools.TOOL_SCHEMAS

    def investigate(self, user_query: str) -> Iterator[dict]:
        """Run a full multi-step investigation, yielding events.

        This is a synchronous generator so it works easily with both
        sync callers and ``asyncio.to_thread`` for async WebSocket.
        """
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]

        yield status_event("Starting investigation...")

        for step in range(1, self.config.max_steps + 1):
            yield status_event(f"Step {step}/{self.config.max_steps}")

            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    tools=self._tools,
                    tool_choice="auto",
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            except Exception as e:
                yield error_event(f"LLM API error: {e}")
                return

            choice = response.choices[0]
            msg = choice.message

            # Emit any text reasoning
            if msg.content:
                yield thinking_event(msg.content)

            # If no tool calls -> agent is done
            if not msg.tool_calls:
                if msg.content:
                    yield conclusion_event(msg.content)
                else:
                    yield conclusion_event("Investigation complete (no further reasoning).")
                return

            # Append the assistant message (with tool_calls) to history
            messages.append(msg.model_dump())

            # Execute each tool call
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                yield tool_call_event(fn_name, fn_args)

                result_str = agent_tools.dispatch(fn_name, fn_args)

                yield tool_result_event(fn_name, result_str)

                # Append tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

            # Check finish_reason
            if choice.finish_reason == "stop":
                if msg.content:
                    yield conclusion_event(msg.content)
                return

        yield error_event("Max reasoning steps reached. Investigation truncated.")

    def investigate_auto(self) -> Iterator[dict]:
        """Auto-trigger: detect anomalies and investigate without user input."""
        return self.investigate(
            "Detect any anomalous network flows, investigate the top anomalies, "
            "run counterfactual analysis on them, assess severity, and provide "
            "a full investigation summary with root-cause analysis and "
            "recommended actions."
        )


# ──────────────────────────────────────────────
# CLI entry point (for testing without server)
# ──────────────────────────────────────────────

def _run_cli():
    """Interactive CLI for testing the agent."""
    import sys

    config = AgentConfig()
    if not config.api_key:
        print("Set OPENAI_API_KEY environment variable.")
        print("Or for local models: set OPENAI_BASE_URL=http://localhost:11434/v1")
        sys.exit(1)

    agent = IncidentAgent(config)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
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
            preview = event.get("result", "")
            print(f"<< Result: {preview[:500]}")
        elif etype == "conclusion":
            print(f"\n{'='*60}\n{content}\n{'='*60}")
        elif etype == "error":
            print(f"\n[ERROR] {content}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_cli()
