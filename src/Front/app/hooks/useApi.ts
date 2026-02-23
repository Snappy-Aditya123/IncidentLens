/**
 * React hooks for fetching data from the IncidentLens backend.
 *
 * Each hook:
 *  - Tries the live API first.
 *  - Falls back to mock data if the backend is unreachable.
 *  - Exposes `data`, `loading`, `error`, and a `refetch()` handle.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import type {
  Incident,
  NetworkGraphData,
  ElasticsearchData,
  CounterfactualExplanation,
  BackendFlow,
  BackendCounterfactualResponse,
  BackendSeverityResponse,
  InvestigationEvent,
} from "../types";
import * as api from "../services/api";
import {
  mockIncidents,
  mockElasticsearchResults,
  mockNetworkGraph,
  mockCounterfactuals,
} from "../data/mockData";

/* ──────────────────────────────────────────────
 * Generic async-data hook
 * ────────────────────────────────────────────── */

interface UseAsyncResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []): UseAsyncResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const tick = useRef(0);

  const refetch = useCallback(() => {
    tick.current += 1;
    setLoading(true);
    setError(null);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fn()
      .then((d) => {
        if (!cancelled) {
          setData(d);
          setError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick.current, ...deps]);

  return { data, loading, error, refetch };
}

/* ──────────────────────────────────────────────
 * Backend health check
 * ────────────────────────────────────────────── */

export function useBackendHealth() {
  return useAsync(() => api.checkHealth(), []);
}

/* ──────────────────────────────────────────────
 * Incidents list  (detect → map to Incident[])
 * ────────────────────────────────────────────── */

function flowToIncident(flow: BackendFlow, index: number): Incident {
  const score = flow.prediction_score ?? (flow.label === 1 ? 0.85 : 0.2);
  const severity: Incident["severity"] =
    score > 0.9 ? "critical" : score > 0.7 ? "high" : score > 0.5 ? "medium" : "low";

  return {
    id: flow._id,
    title: `Anomalous Flow: ${flow.src_ip} → ${flow.dst_ip}`,
    severity,
    status: "investigating",
    timestamp: new Date().toISOString(),
    affectedSystems: [flow.src_ip, flow.dst_ip].filter(Boolean) as string[],
    description: `Detected anomalous traffic — ${flow.packet_count ?? "?"} packets, ${flow.total_bytes ?? "?"} bytes. Prediction score: ${(score * 100).toFixed(0)}%.`,
    anomalyScore: score,
  };
}

export function useIncidents(): UseAsyncResult<Incident[]> {
  return useAsync(async () => {
    try {
      const res = await api.detectAnomalies({ size: 50 });
      if (res.flows.length > 0) {
        return res.flows.map(flowToIncident);
      }
    } catch {
      // backend unavailable — fall through to mock
    }
    return mockIncidents;
  }, []);
}

/* ──────────────────────────────────────────────
 * Single incident detail
 * ────────────────────────────────────────────── */

export function useIncident(incidentId: string | undefined): UseAsyncResult<Incident | null> {
  return useAsync(async () => {
    if (!incidentId) return null;
    try {
      // Try fetching all incidents and finding the one
      const res = await api.detectAnomalies({ size: 100 });
      const flow = res.flows.find((f) => f._id === incidentId);
      if (flow) return flowToIncident(flow, 0);
    } catch {
      // fall through
    }
    return mockIncidents.find((i) => i.id === incidentId) ?? null;
  }, [incidentId]);
}

/* ──────────────────────────────────────────────
 * Elasticsearch logs for an incident
 * ────────────────────────────────────────────── */

export function useElasticsearchData(incidentId: string | undefined): UseAsyncResult<ElasticsearchData | null> {
  return useAsync(async () => {
    if (!incidentId) return null;
    try {
      // Try to get flow detail + build an ES-like log view
      const flowsRes = await api.listFlows({ size: 1 });
      if (flowsRes.flows.length > 0) {
        // Search for flows related to this incident's IPs
        const allFlows = await api.listFlows({ size: 20 });
        const logs = allFlows.flows.slice(0, 8).map((f) => ({
          timestamp: new Date().toISOString(),
          source: f.src_ip || "unknown",
          message: `Flow ${f.src_ip} → ${f.dst_ip}: ${f.packet_count ?? 0} packets, ${f.total_bytes ?? 0} bytes${f.label === 1 ? " [ANOMALOUS]" : ""}`,
          level: f.label === 1 ? "CRITICAL" : "INFO",
        }));
        return {
          totalHits: allFlows.count,
          logs,
          query: {
            bool: {
              must: [
                { term: { label: 1 } },
                { range: { "@timestamp": { gte: "now-1h" } } },
              ],
            },
          },
        };
      }
    } catch {
      // fall through
    }
    const mock = mockElasticsearchResults[incidentId as keyof typeof mockElasticsearchResults];
    return mock ?? null;
  }, [incidentId]);
}

/* ──────────────────────────────────────────────
 * GNN network graph for an incident
 * ────────────────────────────────────────────── */

export function useNetworkGraph(incidentId: string | undefined): UseAsyncResult<NetworkGraphData | null> {
  return useAsync(async () => {
    if (!incidentId) return null;
    try {
      // Build graph from anomalous flows
      const res = await api.detectAnomalies({ size: 30 });
      if (res.flows.length > 0) {
        return flowsToGraph(res.flows);
      }
    } catch {
      // fall through
    }
    const mock = mockNetworkGraph[incidentId as keyof typeof mockNetworkGraph];
    return mock ?? null;
  }, [incidentId]);
}

/**
 * Transform a set of backend flows into a force-graph-compatible
 * nodes + edges structure for the D3 visualisation.
 */
function flowsToGraph(flows: BackendFlow[]): NetworkGraphData {
  const nodeMap = new Map<string, { risk: number; label: number }>();

  for (const f of flows) {
    if (f.src_ip) {
      const prev = nodeMap.get(f.src_ip);
      const score = f.prediction_score ?? (f.label === 1 ? 0.8 : 0.15);
      nodeMap.set(f.src_ip, {
        risk: Math.max(prev?.risk ?? 0, score),
        label: Math.max(prev?.label ?? 0, f.label),
      });
    }
    if (f.dst_ip) {
      const prev = nodeMap.get(f.dst_ip);
      const score = f.prediction_score ?? (f.label === 1 ? 0.8 : 0.15);
      nodeMap.set(f.dst_ip, {
        risk: Math.max(prev?.risk ?? 0, score),
        label: Math.max(prev?.label ?? 0, f.label),
      });
    }
  }

  const nodes = Array.from(nodeMap.entries()).map(([ip, info]) => ({
    id: ip,
    label: ip,
    type: "server" as const,
    status: info.risk > 0.8 ? "compromised" as const : info.risk > 0.5 ? "suspicious" as const : "normal" as const,
    risk: info.risk,
  }));

  const edges = flows
    .filter((f) => f.src_ip && f.dst_ip)
    .map((f) => ({
      source: f.src_ip,
      target: f.dst_ip,
      type: "data_flow" as const,
      weight: Math.min((f.packet_count ?? 1) / 100, 10),
      anomalous: f.label === 1,
    }));

  return { nodes, edges };
}

/* ──────────────────────────────────────────────
 * Counterfactual explanation for an incident
 * ────────────────────────────────────────────── */

export function useCounterfactual(incidentId: string | undefined): UseAsyncResult<CounterfactualExplanation | null> {
  return useAsync(async () => {
    if (!incidentId) return null;
    try {
      const cfRes = await api.getFlowCounterfactual(incidentId);
      return backendCfToFrontend(cfRes);
    } catch {
      // fall through
    }
    const mock = mockCounterfactuals[incidentId as keyof typeof mockCounterfactuals];
    return mock ?? null;
  }, [incidentId]);
}

function backendCfToFrontend(cf: BackendCounterfactualResponse): CounterfactualExplanation {
  const diffs = cf.diffs ?? [];
  const maxDiff = Math.max(...diffs.map((d) => d.abs_diff), 1);

  return {
    original: `Anomalous flow ${cf.flow_id} — deviates from normal traffic baseline`,
    counterfactual: `Nearest normal flow — baseline traffic pattern`,
    changes: diffs.slice(0, 6).map((d) => ({
      parameter: d.feature,
      original: String(d.anomalous_value?.toFixed?.(2) ?? d.anomalous_value),
      modified: String(d.normal_value?.toFixed?.(2) ?? d.normal_value),
      impact: Math.min(d.abs_diff / maxDiff, 1),
    })),
    prediction: {
      original: "Anomalous",
      counterfactual: "Normal",
    },
  };
}

/* ──────────────────────────────────────────────
 * Severity assessment
 * ────────────────────────────────────────────── */

export function useSeverity(flowId: string | undefined): UseAsyncResult<BackendSeverityResponse | null> {
  return useAsync(async () => {
    if (!flowId) return null;
    try {
      return await api.getFlowSeverity(flowId);
    } catch {
      return null;
    }
  }, [flowId]);
}

/* ──────────────────────────────────────────────
 * WebSocket investigation stream
 * ────────────────────────────────────────────── */

export function useInvestigationStream() {
  const [events, setEvents] = useState<InvestigationEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const start = useCallback(async (query: string) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setEvents([]);
    setRunning(true);
    setError(null);

    try {
      for await (const event of api.investigateStream(query, controller.signal)) {
        if (controller.signal.aborted) break;
        setEvents((prev) => [...prev, event]);
      }
    } catch (e) {
      if (!controller.signal.aborted) {
        setError(String(e));
      }
    } finally {
      setRunning(false);
    }
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setRunning(false);
  }, []);

  return { events, running, error, start, stop };
}
