export interface Incident {
  id: string;
  title: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  status: 'investigating' | 'resolved' | 'escalated';
  timestamp: string;
  affectedSystems: string[];
  description: string;
  anomalyScore: number;
}

export interface NetworkNode {
  id: string;
  label: string;
  type: 'server' | 'service' | 'database' | 'endpoint' | 'firewall';
  status: 'normal' | 'suspicious' | 'compromised';
  risk: number;
}

export interface NetworkEdge {
  source: string;
  target: string;
  type: 'connection' | 'data_flow' | 'dependency';
  weight: number;
  anomalous: boolean;
}

export const mockIncidents: Incident[] = [
  {
    id: 'INC-2026-001',
    title: 'Unusual Database Access Pattern Detected',
    severity: 'critical',
    status: 'investigating',
    timestamp: '2026-02-23T09:15:00Z',
    affectedSystems: ['prod-db-01', 'auth-service', 'api-gateway'],
    description: 'Multiple unauthorized access attempts from internal network',
    anomalyScore: 0.94,
  },
  {
    id: 'INC-2026-002',
    title: 'Spike in Outbound Network Traffic',
    severity: 'high',
    status: 'investigating',
    timestamp: '2026-02-23T08:42:00Z',
    affectedSystems: ['web-server-03', 'cdn-node-12'],
    description: 'Abnormal data exfiltration pattern detected',
    anomalyScore: 0.87,
  },
  {
    id: 'INC-2026-003',
    title: 'Failed Authentication Attempts',
    severity: 'medium',
    status: 'resolved',
    timestamp: '2026-02-23T07:30:00Z',
    affectedSystems: ['auth-service', 'load-balancer-01'],
    description: 'Brute force attack attempt from external IP',
    anomalyScore: 0.72,
  },
  {
    id: 'INC-2026-004',
    title: 'Latency Spike in Service Mesh',
    severity: 'high',
    status: 'investigating',
    timestamp: '2026-02-23T06:15:00Z',
    affectedSystems: ['k8s-cluster-prod', 'istio-gateway'],
    description: 'Unusual service-to-service communication delays',
    anomalyScore: 0.81,
  },
  {
    id: 'INC-2026-005',
    title: 'Certificate Expiration Warning',
    severity: 'low',
    status: 'escalated',
    timestamp: '2026-02-22T23:45:00Z',
    affectedSystems: ['ssl-proxy-02'],
    description: 'SSL certificate approaching expiration',
    anomalyScore: 0.45,
  },
];

export const mockElasticsearchResults = {
  'INC-2026-001': {
    totalHits: 1247,
    logs: [
      {
        timestamp: '2026-02-23T09:15:23Z',
        source: 'prod-db-01',
        message: 'Authentication failed for user: admin from 10.0.2.45',
        level: 'ERROR',
      },
      {
        timestamp: '2026-02-23T09:15:45Z',
        source: 'prod-db-01',
        message: 'Multiple failed login attempts detected - Rate limit exceeded',
        level: 'CRITICAL',
      },
      {
        timestamp: '2026-02-23T09:16:02Z',
        source: 'auth-service',
        message: 'Token validation failed - Invalid signature',
        level: 'WARNING',
      },
      {
        timestamp: '2026-02-23T09:16:18Z',
        source: 'api-gateway',
        message: 'Request blocked - Suspicious activity pattern',
        level: 'INFO',
      },
      {
        timestamp: '2026-02-23T09:16:35Z',
        source: 'prod-db-01',
        message: 'Privilege escalation attempt detected',
        level: 'CRITICAL',
      },
    ],
    query: {
      bool: {
        must: [
          { match: { 'system.name': 'prod-db-01' } },
          { range: { '@timestamp': { gte: 'now-1h' } } },
        ],
        should: [
          { match: { level: 'ERROR' } },
          { match: { level: 'CRITICAL' } },
        ],
      },
    },
  },
};

export const mockNetworkGraph = {
  'INC-2026-001': {
    nodes: [
      { id: 'api-gateway', label: 'API Gateway', type: 'server', status: 'normal', risk: 0.3 },
      { id: 'auth-service', label: 'Auth Service', type: 'service', status: 'suspicious', risk: 0.75 },
      { id: 'prod-db-01', label: 'Production DB', type: 'database', status: 'compromised', risk: 0.95 },
      { id: 'web-app', label: 'Web Application', type: 'service', status: 'normal', risk: 0.2 },
      { id: 'internal-user', label: 'Internal Endpoint', type: 'endpoint', status: 'suspicious', risk: 0.8 },
      { id: 'firewall', label: 'Firewall', type: 'firewall', status: 'normal', risk: 0.1 },
    ] as NetworkNode[],
    edges: [
      { source: 'firewall', target: 'api-gateway', type: 'connection', weight: 5, anomalous: false },
      { source: 'api-gateway', target: 'auth-service', type: 'data_flow', weight: 8, anomalous: true },
      { source: 'auth-service', target: 'prod-db-01', type: 'data_flow', weight: 10, anomalous: true },
      { source: 'web-app', target: 'api-gateway', type: 'dependency', weight: 6, anomalous: false },
      { source: 'internal-user', target: 'prod-db-01', type: 'connection', weight: 9, anomalous: true },
    ] as NetworkEdge[],
  },
};

export interface CounterfactualExplanation {
  original: string;
  counterfactual: string;
  changes: Array<{
    parameter: string;
    original: string;
    modified: string;
    impact: number;
  }>;
  prediction: {
    original: string;
    counterfactual: string;
  };
}

export const mockCounterfactuals = {
  'INC-2026-001': {
    original: 'Current scenario: Unauthorized database access detected with 94% anomaly score',
    counterfactual: 'Alternative scenario: Normal database access with 12% anomaly score',
    changes: [
      {
        parameter: 'Request Rate',
        original: '450 requests/min',
        modified: '45 requests/min',
        impact: 0.85,
      },
      {
        parameter: 'Authentication Method',
        original: 'Direct DB connection',
        modified: 'OAuth via Auth Service',
        impact: 0.92,
      },
      {
        parameter: 'Source IP Pattern',
        original: 'Multiple rotating IPs',
        modified: 'Single known IP',
        impact: 0.78,
      },
      {
        parameter: 'Time of Access',
        original: '03:00 AM (off-hours)',
        modified: '10:00 AM (business hours)',
        impact: 0.45,
      },
    ],
    prediction: {
      original: 'Malicious Activity (94% confidence)',
      counterfactual: 'Normal Activity (88% confidence)',
    },
  } as CounterfactualExplanation,
};
