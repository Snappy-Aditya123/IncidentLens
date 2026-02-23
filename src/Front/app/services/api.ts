/**
 * API client for the IncidentLens FastAPI backend.
 *
 * All functions hit the Vite dev-server proxy (same origin) which
 * forwards to http://localhost:8000.  In production the frontend
 * is served alongside the API so no proxy is needed.
 *
 * Every function is `async` and returns typed data or throws.
 */

import type {
  BackendCounterfactualResponse,
  BackendDetectResponse,
  BackendFlow,
  BackendFlowsResponse,
  BackendHealthResponse,
  BackendSeverityResponse,
  InvestigationEvent,
} from "../types";

const BASE = "";  // same-origin — Vite proxy handles /api → backend

/* ─── helpers ─────────────────────────────── */

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

/* ─── health ──────────────────────────────── */

export async function checkHealth(): Promise<BackendHealthResponse> {
  return fetchJson<BackendHealthResponse>("/health");
}

/* ─── flows ───────────────────────────────── */

export async function listFlows(params?: {
  label?: number;
  src_ip?: string;
  dst_ip?: string;
  size?: number;
}): Promise<BackendFlowsResponse> {
  const qs = new URLSearchParams();
  if (params?.label !== undefined) qs.set("label", String(params.label));
  if (params?.src_ip) qs.set("src_ip", params.src_ip);
  if (params?.dst_ip) qs.set("dst_ip", params.dst_ip);
  if (params?.size !== undefined) qs.set("size", String(params.size));
  const query = qs.toString();
  return fetchJson<BackendFlowsResponse>(`/api/flows${query ? `?${query}` : ""}`);
}

/* ─── detection ───────────────────────────── */

export async function detectAnomalies(params?: {
  method?: string;
  threshold?: number;
  size?: number;
}): Promise<BackendDetectResponse> {
  return fetchJson<BackendDetectResponse>("/api/detect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      method: params?.method ?? "label",
      threshold: params?.threshold ?? 0.5,
      size: params?.size ?? 50,
    }),
  });
}

/* ─── single-flow actions ─────────────────── */

export async function getFlowSeverity(flowId: string): Promise<BackendSeverityResponse> {
  return fetchJson<BackendSeverityResponse>(`/api/severity/${encodeURIComponent(flowId)}`);
}

export async function getFlowCounterfactual(flowId: string): Promise<BackendCounterfactualResponse> {
  return fetchJson<BackendCounterfactualResponse>("/api/counterfactual", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ flow_id: flowId }),
  });
}

export async function getSimilarIncidents(flowId: string, k = 5): Promise<{ flow_id: string; similar: BackendFlow[] }> {
  return fetchJson(`/api/similar/${encodeURIComponent(flowId)}?k=${k}`);
}

/* ─── stats ───────────────────────────────── */

export async function getFeatureStats(): Promise<Record<string, unknown>> {
  return fetchJson<Record<string, unknown>>("/api/stats");
}

/* ─── full investigation (REST, non-streaming) */

export async function investigate(query: string): Promise<{ events: InvestigationEvent[] }> {
  return fetchJson<{ events: InvestigationEvent[] }>("/api/investigate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
}

/* ─── investigation WebSocket (streaming) ─── */

/**
 * Opens a WebSocket to `/ws/investigate` and yields events as they arrive.
 *
 * Usage:
 * ```ts
 * for await (const event of investigateStream("Why is 10.0.2.45 anomalous?")) {
 *   console.log(event);
 * }
 * ```
 */
export async function* investigateStream(
  query: string,
  signal?: AbortSignal,
): AsyncGenerator<InvestigationEvent> {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${window.location.host}/ws/investigate`);

  // Abort support
  signal?.addEventListener("abort", () => ws.close(), { once: true });

  // Promise-based message queue
  const queue: InvestigationEvent[] = [];
  let resolve: (() => void) | null = null;
  let done = false;
  let error: Error | null = null;

  ws.onmessage = (ev) => {
    const event: InvestigationEvent = JSON.parse(ev.data);
    if (event.type === "done") {
      done = true;
      resolve?.();
      return;
    }
    queue.push(event);
    resolve?.();
  };

  ws.onerror = () => {
    error = new Error("WebSocket error");
    resolve?.();
  };

  ws.onclose = () => {
    done = true;
    resolve?.();
  };

  // Wait for open, then send the query
  await new Promise<void>((res, rej) => {
    ws.onopen = () => {
      ws.send(JSON.stringify({ query }));
      res();
    };
    ws.onerror = () => rej(new Error("WebSocket failed to connect"));
  });

  // Yield messages as they arrive
  while (!done) {
    if (queue.length > 0) {
      yield queue.shift()!;
      continue;
    }
    if (error) throw error;
    // Wait for next message
    await new Promise<void>((r) => {
      resolve = r;
    });
  }

  // Drain remaining
  while (queue.length > 0) {
    yield queue.shift()!;
  }
}
