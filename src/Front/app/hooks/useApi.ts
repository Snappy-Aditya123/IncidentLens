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
  BackendCounterfactualResponse,
  BackendSeverityResponse,
  InvestigationEvent,
  SeverityBreakdownResponse,
  MLAnomaliesResponse,
  MLInfluencersResponse,
  CounterfactualSearchResponse,
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
  const [tick, setTick] = useState(0);

  const refetch = useCallback(() => {
    setTick((t) => t + 1);
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
  }, [tick, ...deps]);

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

export function useIncidents(): UseAsyncResult<Incident[]> {
  return useAsync(async () => {
    try {
      const res = await api.listIncidents(50);
      if (res.incidents.length > 0) {
        return res.incidents;
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
      // Direct single-incident endpoint — no fetch-all needed
      return await api.getIncident(incidentId);
    } catch {
      // fall through to mock
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
      return await api.getIncidentLogs(incidentId, 20);
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
      const data = await api.getIncidentGraph(incidentId, 30);
      if (data.nodes?.length > 0) return data;
    } catch {
      // fall through
    }
    const mock = mockNetworkGraph[incidentId as keyof typeof mockNetworkGraph];
    return mock ?? null;
  }, [incidentId]);
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
      original: String(d.anomalous_value?.toFixed?.(2) ?? d.anomalous_value ?? "N/A"),
      modified: String(d.normal_value?.toFixed?.(2) ?? d.normal_value ?? "N/A"),
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

  // Cleanup on unmount — abort any in-progress stream
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

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

/* ──────────────────────────────────────────────
 * ES-NATIVE ANALYTICS HOOKS
 * These leverage Elasticsearch capabilities beyond basic CRUD:
 * runtime fields, ML anomaly detection, full-text search, composite aggs.
 * ────────────────────────────────────────────── */

/**
 * Severity distribution computed entirely server-side using ES runtime fields
 * and Painless scripts — zero client-side iteration.
 */
export function useSeverityBreakdown(): UseAsyncResult<SeverityBreakdownResponse | null> {
  return useAsync(async () => {
    try {
      return await api.getSeverityBreakdown();
    } catch {
      return null;
    }
  }, []);
}

/**
 * ES ML anomaly detection records above a score threshold.
 * Each record includes influencers — direct feature attribution from ES.
 */
export function useMLAnomalies(
  jobId?: string,
  minScore = 75,
): UseAsyncResult<MLAnomaliesResponse | null> {
  return useAsync(async () => {
    try {
      return await api.getMLAnomalies({
        job_id: jobId,
        min_score: minScore,
      });
    } catch {
      return null;
    }
  }, [jobId, minScore]);
}

/**
 * ES ML influencer results — which field values contributed most to anomalies.
 */
export function useMLInfluencers(
  jobId?: string,
  minScore = 50,
): UseAsyncResult<MLInfluencersResponse | null> {
  return useAsync(async () => {
    try {
      return await api.getMLInfluencers({
        job_id: jobId,
        min_score: minScore,
      });
    } catch {
      return null;
    }
  }, [jobId, minScore]);
}

/**
 * Full-text search over counterfactual explanation narratives.
 * Supports natural-language queries.
 */
export function useCounterfactualSearch(
  query: string | undefined,
): UseAsyncResult<CounterfactualSearchResponse | null> {
  return useAsync(async () => {
    if (!query?.trim()) return null;
    try {
      return await api.searchCounterfactuals(query);
    } catch {
      return null;
    }
  }, [query]);
}
