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
  max_z_score: number;
  feature_z_scores: Record<string, number>;
  flow?: BackendFlow;
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
  | "status"
  | "error"
  | "done";

export interface InvestigationEvent {
  type: InvestigationEventType;
  content?: string;
  tool?: string;
  arguments?: Record<string, unknown>;
  result?: string;
}

/* ──────────────────────────────────────────────
 * ES-native analytics response shapes
 * ────────────────────────────────────────────── */

/** Severity distribution computed by ES runtime fields. */
export interface SeverityBreakdownResponse {
  severity_levels: Record<string, number>;
  traffic_volume_categories: Record<string, number>;
  total_flows: number;
}

/** Paginated search result using search_after + PIT. */
export interface PaginatedFlowsResponse {
  hits: BackendFlow[];
  pit_id: string;
  search_after: unknown[] | null;
  total: number;
}

/** Full-text counterfactual search result. */
export interface CounterfactualSearchResult {
  _id: string;
  _score: number;
  flow_id: string;
  explanation_text: string;
  highlight?: Record<string, string[]>;
  [key: string]: unknown;
}

export interface CounterfactualSearchResponse {
  query: string;
  count: number;
  results: CounterfactualSearchResult[];
}

/** ES ML anomaly record. */
export interface MLAnomalyRecord {
  record_score: number;
  bucket_span: number;
  detector_index: number;
  is_interim: boolean;
  timestamp: number;
  function: string;
  field_name?: string;
  by_field_name?: string;
  by_field_value?: string;
  influencers: Array<{
    influencer_field_name: string;
    influencer_field_values: string[];
  }>;
  [key: string]: unknown;
}

export interface MLAnomaliesResponse {
  job_id: string;
  count: number;
  records: MLAnomalyRecord[];
}

/** ES ML influencer result. */
export interface MLInfluencer {
  influencer_field_name: string;
  influencer_field_value: string;
  influencer_score: number;
  initial_influencer_score: number;
  bucket_span: number;
  timestamp: number;
  [key: string]: unknown;
}

export interface MLInfluencersResponse {
  job_id: string;
  count: number;
  influencers: MLInfluencer[];
}

/** Composite aggregation buckets. */
export interface AggregationBucket {
  key: string;
  doc_count: number;
}

export interface AggregationResponse {
  field: string;
  buckets: AggregationBucket[];
  total_buckets: number;
}
