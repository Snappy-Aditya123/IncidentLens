/**
 * Shared TypeScript types for IncidentLens.
 *
 * The "Incident" / "NetworkNode" / etc. types used by the UI are kept
 * alongside the raw backend shapes so components can work with both
 * live API data *and* the mock dataset interchangeably.
 */

/* ──────────────────────────────────────────────
 * Frontend UI types (matches mockData.ts)
 * ────────────────────────────────────────────── */

export interface Incident {
  id: string;
  title: string;
  severity: "critical" | "high" | "medium" | "low";
  status: "investigating" | "resolved" | "escalated";
  timestamp: string;
  affectedSystems: string[];
  description: string;
  anomalyScore: number;
}

export interface NetworkNode {
  id: string;
  label: string;
  type: "server" | "service" | "database" | "endpoint" | "firewall";
  status: "normal" | "suspicious" | "compromised";
  risk: number;
}

export interface NetworkEdge {
  source: string;
  target: string;
  type: "connection" | "data_flow" | "dependency";
  weight: number;
  anomalous: boolean;
}

export interface ElasticsearchData {
  totalHits: number;
  logs: Array<{
    timestamp: string;
    source: string;
    message: string;
    level: string;
  }>;
  query: unknown;
}

export interface NetworkGraphData {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
}

export interface CounterfactualChange {
  parameter: string;
  original: string;
  modified: string;
  impact: number;
}

export interface CounterfactualExplanation {
  original: string;
  counterfactual: string;
  changes: CounterfactualChange[];
  prediction: {
    original: string;
    counterfactual: string;
  };
}

/* ──────────────────────────────────────────────
 * Backend raw response shapes
 * ────────────────────────────────────────────── */

/** A single flow document returned by the backend. */
export interface BackendFlow {
  _id: string;
  src_ip: string;
  dst_ip: string;
  label: number;
  packet_count?: number;
  total_bytes?: number;
  mean_payload?: number;
  mean_iat?: number;
  std_iat?: number;
  prediction?: number;
  prediction_score?: number;
  [key: string]: unknown;
}

export interface BackendDetectResponse {
  method: string;
  count: number;
  flows: BackendFlow[];
  threshold?: number;
}

export interface BackendFlowsResponse {
  count: number;
  flows: BackendFlow[];
}

export interface BackendCounterfactualResponse {
  flow_id: string;
  anomalous_flow: Record<string, unknown>;
  nearest_normal: Record<string, unknown>;
  diffs: Array<{
    feature: string;
    anomalous_value: number;
    normal_value: number;
    abs_diff: number;
    direction: string;
  }>;
}

export interface BackendSeverityResponse {
  flow_id: string;
  severity: string;
  z_scores: Record<string, number>;
  max_z: number;
}

export interface BackendHealthResponse {
  server: string;
  elasticsearch: string;
  indices?: Record<string, boolean>;
  error?: string;
}

/* ──────────────────────────────────────────────
 * WebSocket event types (from ws/investigate)
 * ────────────────────────────────────────────── */

export type InvestigationEventType =
  | "thinking"
  | "tool_call"
  | "tool_result"
  | "conclusion"
  | "error"
  | "done";

export interface InvestigationEvent {
  type: InvestigationEventType;
  content?: string;
  tool?: string;
  arguments?: Record<string, unknown>;
  result?: string;
}
