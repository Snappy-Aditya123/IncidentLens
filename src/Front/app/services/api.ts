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
  SeverityBreakdownResponse,
  PaginatedFlowsResponse,
  CounterfactualSearchResponse,
  MLAnomaliesResponse,
  MLInfluencersResponse,
  AggregationResponse,
  Incident,
  NetworkGraphData,
  ElasticsearchData,
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

export async function getSimilarIncidents(flowId: string, k = 5): Promise<{ query_flow: string; similar: BackendFlow[] }> {
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

/* ─── incident endpoints (direct wrappers) ─── */

export async function listIncidents(size = 50): Promise<{ count: number; incidents: Incident[] }> {
  return fetchJson<{ count: number; incidents: Incident[] }>(`/api/incidents?size=${size}`);
}

export async function getIncident(id: string): Promise<Incident> {
  return fetchJson<Incident>(`/api/incidents/${encodeURIComponent(id)}`);
}

export async function getIncidentGraph(id: string, size = 30): Promise<NetworkGraphData> {
  return fetchJson<NetworkGraphData>(`/api/incidents/${encodeURIComponent(id)}/graph?size=${size}`);
}

export async function getIncidentLogs(id: string, size = 20): Promise<ElasticsearchData> {
  return fetchJson<ElasticsearchData>(`/api/incidents/${encodeURIComponent(id)}/logs?size=${size}`);
}

/* ─── ES-native analytics ─────────────────── */

/**
 * Severity breakdown computed entirely server-side using ES runtime fields
 * and Painless scripts — zero Python iteration.
 */
export async function getSeverityBreakdown(): Promise<SeverityBreakdownResponse> {
  return fetchJson<SeverityBreakdownResponse>("/api/severity-breakdown");
}

/**
 * Query flows by runtime-computed severity level (critical/high/medium/low).
 * The severity is computed at query time by ES, not stored in the document.
 */
export async function getFlowsBySeverity(
  severity: string = "critical",
  size = 20,
): Promise<BackendFlowsResponse & { severity: string }> {
  return fetchJson(`/api/flows/severity?severity=${severity}&size=${size}`);
}

/**
 * Paginated flow search using ES search_after + point-in-time.
 * Unlike offset-based pagination, this doesn't degrade with depth.
 */
export async function searchFlowsPaginated(params?: {
  query?: Record<string, unknown>;
  size?: number;
  search_after?: unknown[];
  pit_id?: string;
}): Promise<PaginatedFlowsResponse> {
  return fetchJson<PaginatedFlowsResponse>("/api/flows/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: params?.query ?? null,
      size: params?.size ?? 20,
      search_after: params?.search_after ?? null,
      pit_id: params?.pit_id ?? null,
    }),
  });
}

/**
 * Full-text search over counterfactual explanation narratives.
 * Supports natural-language queries like "high packet count from source".
 */
export async function searchCounterfactuals(
  q: string,
  size = 10,
): Promise<CounterfactualSearchResponse> {
  return fetchJson<CounterfactualSearchResponse>(
    `/api/counterfactuals/search?q=${encodeURIComponent(q)}&size=${size}`,
  );
}

/**
 * Composite aggregation on any keyword field.
 * Handles unbounded cardinality (many unique IPs) without truncation.
 */
export async function getAggregation(
  field: string,
  size = 100,
): Promise<AggregationResponse> {
  return fetchJson<AggregationResponse>(
    `/api/aggregate/${encodeURIComponent(field)}?size=${size}`,
  );
}

/**
 * ES ML anomaly detection records.  Each record includes influencers —
 * the field values that drove the anomaly.
 */
export async function getMLAnomalies(params?: {
  job_id?: string;
  min_score?: number;
  size?: number;
}): Promise<MLAnomaliesResponse> {
  const qs = new URLSearchParams();
  if (params?.job_id) qs.set("job_id", params.job_id);
  if (params?.min_score !== undefined) qs.set("min_score", String(params.min_score));
  if (params?.size !== undefined) qs.set("size", String(params.size));
  const query = qs.toString();
  return fetchJson<MLAnomaliesResponse>(`/api/ml/anomalies${query ? `?${query}` : ""}`);
}

/**
 * ES ML influencer results — which src_ip, dst_ip, or protocol values
 * contributed most to detected anomalies.
 */
export async function getMLInfluencers(params?: {
  job_id?: string;
  min_score?: number;
  size?: number;
}): Promise<MLInfluencersResponse> {
  const qs = new URLSearchParams();
  if (params?.job_id) qs.set("job_id", params.job_id);
  if (params?.min_score !== undefined) qs.set("min_score", String(params.min_score));
  if (params?.size !== undefined) qs.set("size", String(params.size));
  const query = qs.toString();
  return fetchJson<MLInfluencersResponse>(`/api/ml/influencers${query ? `?${query}` : ""}`);
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
    let event: InvestigationEvent;
    try {
      event = JSON.parse(ev.data);
    } catch {
      error = new Error("Malformed JSON from server");
      resolve?.();
      return;
    }
    if (event.type === "done") {
      done = true;
      ws.close();
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
      // Restore runtime error handler after successful connection
      ws.onerror = () => {
        error = new Error("WebSocket error");
        resolve?.();
      };
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
