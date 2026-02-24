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

export const mockElasticsearchResults: Record<string, {
  totalHits: number;
  logs: Array<{ timestamp: string; source: string; message: string; level: string }>;
  query: unknown;
}> = {
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
  'INC-2026-002': {
    totalHits: 892,
    logs: [
      { timestamp: '2026-02-23T08:42:10Z', source: 'web-server-03', message: 'Outbound traffic spike detected — 450 MB/s to external IP 203.0.113.42', level: 'CRITICAL' },
      { timestamp: '2026-02-23T08:42:30Z', source: 'cdn-node-12', message: 'DNS query volume exceeded threshold', level: 'WARNING' },
      { timestamp: '2026-02-23T08:43:00Z', source: 'web-server-03', message: 'Large payload transfer in progress', level: 'ERROR' },
    ],
    query: { bool: { must: [{ match: { 'system.name': 'web-server-03' } }] } },
  },
  'INC-2026-003': {
    totalHits: 523,
    logs: [
      { timestamp: '2026-02-23T07:30:05Z', source: 'auth-service', message: 'Brute force attempt — 200 failed logins from 198.51.100.0/24', level: 'ERROR' },
      { timestamp: '2026-02-23T07:31:00Z', source: 'load-balancer-01', message: 'Rate limiter triggered for source subnet', level: 'WARNING' },
    ],
    query: { bool: { must: [{ match: { 'system.name': 'auth-service' } }] } },
  },
  'INC-2026-004': {
    totalHits: 346,
    logs: [
      { timestamp: '2026-02-23T06:15:12Z', source: 'k8s-cluster-prod', message: 'Inter-pod latency exceeds 500ms on service mesh', level: 'WARNING' },
      { timestamp: '2026-02-23T06:16:00Z', source: 'istio-gateway', message: 'Circuit breaker tripped for payment-service', level: 'ERROR' },
    ],
    query: { bool: { must: [{ match: { 'system.name': 'k8s-cluster-prod' } }] } },
  },
  'INC-2026-005': {
    totalHits: 12,
    logs: [
      { timestamp: '2026-02-22T23:45:00Z', source: 'ssl-proxy-02', message: 'TLS certificate expires in 7 days', level: 'WARNING' },
    ],
    query: { bool: { must: [{ match: { 'system.name': 'ssl-proxy-02' } }] } },
  },
};

export const mockNetworkGraph: Record<string, { nodes: NetworkNode[]; edges: NetworkEdge[] }> = {
  'INC-2026-001': {
    nodes: [
      { id: 'api-gateway', label: 'API Gateway', type: 'server', status: 'normal', risk: 0.3 },
      { id: 'auth-service', label: 'Auth Service', type: 'service', status: 'suspicious', risk: 0.75 },
      { id: 'prod-db-01', label: 'Production DB', type: 'database', status: 'compromised', risk: 0.95 },
      { id: 'web-app', label: 'Web Application', type: 'service', status: 'normal', risk: 0.2 },
      { id: 'internal-user', label: 'Internal Endpoint', type: 'endpoint', status: 'suspicious', risk: 0.8 },
      { id: 'firewall', label: 'Firewall', type: 'firewall', status: 'normal', risk: 0.1 },
    ],
    edges: [
      { source: 'firewall', target: 'api-gateway', type: 'connection', weight: 5, anomalous: false },
      { source: 'api-gateway', target: 'auth-service', type: 'data_flow', weight: 8, anomalous: true },
      { source: 'auth-service', target: 'prod-db-01', type: 'data_flow', weight: 10, anomalous: true },
      { source: 'web-app', target: 'api-gateway', type: 'dependency', weight: 6, anomalous: false },
      { source: 'internal-user', target: 'prod-db-01', type: 'connection', weight: 9, anomalous: true },
    ],
  },
  'INC-2026-002': {
    nodes: [
      { id: 'web-server-03', label: 'Web Server 03', type: 'server', status: 'compromised', risk: 0.9 },
      { id: 'cdn-node-12', label: 'CDN Node 12', type: 'service', status: 'suspicious', risk: 0.6 },
      { id: 'ext-ip', label: 'External IP', type: 'endpoint', status: 'compromised', risk: 0.95 },
      { id: 'firewall', label: 'Firewall', type: 'firewall', status: 'normal', risk: 0.2 },
    ],
    edges: [
      { source: 'web-server-03', target: 'ext-ip', type: 'data_flow', weight: 10, anomalous: true },
      { source: 'web-server-03', target: 'cdn-node-12', type: 'connection', weight: 6, anomalous: false },
      { source: 'cdn-node-12', target: 'firewall', type: 'dependency', weight: 4, anomalous: false },
    ],
  },
  'INC-2026-003': {
    nodes: [
      { id: 'auth-service', label: 'Auth Service', type: 'service', status: 'suspicious', risk: 0.7 },
      { id: 'load-balancer', label: 'Load Balancer', type: 'server', status: 'normal', risk: 0.3 },
      { id: 'attacker', label: 'External Attacker', type: 'endpoint', status: 'compromised', risk: 0.85 },
    ],
    edges: [
      { source: 'attacker', target: 'load-balancer', type: 'connection', weight: 9, anomalous: true },
      { source: 'load-balancer', target: 'auth-service', type: 'data_flow', weight: 8, anomalous: true },
    ],
  },
  'INC-2026-004': {
    nodes: [
      { id: 'k8s-cluster', label: 'K8s Cluster', type: 'server', status: 'suspicious', risk: 0.65 },
      { id: 'istio-gateway', label: 'Istio Gateway', type: 'service', status: 'suspicious', risk: 0.7 },
      { id: 'payment-svc', label: 'Payment Service', type: 'service', status: 'normal', risk: 0.4 },
    ],
    edges: [
      { source: 'istio-gateway', target: 'k8s-cluster', type: 'dependency', weight: 7, anomalous: true },
      { source: 'k8s-cluster', target: 'payment-svc', type: 'data_flow', weight: 5, anomalous: false },
    ],
  },
  'INC-2026-005': {
    nodes: [
      { id: 'ssl-proxy', label: 'SSL Proxy 02', type: 'server', status: 'normal', risk: 0.35 },
      { id: 'cert-mgr', label: 'Cert Manager', type: 'service', status: 'normal', risk: 0.2 },
    ],
    edges: [
      { source: 'cert-mgr', target: 'ssl-proxy', type: 'dependency', weight: 3, anomalous: false },
    ],
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

export const mockCounterfactuals: Record<string, CounterfactualExplanation> = {
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
  },
  'INC-2026-002': {
    original: 'Current scenario: Data exfiltration detected via web server 03 with 89% anomaly score',
    counterfactual: 'Alternative scenario: Normal outbound traffic with 15% anomaly score',
    changes: [
      {
        parameter: 'Outbound Bytes',
        original: '2.4 GB/hr',
        modified: '120 MB/hr',
        impact: 0.91,
      },
      {
        parameter: 'Destination Reputation',
        original: 'Unknown external IP',
        modified: 'Known CDN endpoint',
        impact: 0.82,
      },
      {
        parameter: 'Transfer Protocol',
        original: 'Raw TCP stream',
        modified: 'HTTPS with valid certificate',
        impact: 0.68,
      },
    ],
    prediction: {
      original: 'Data Exfiltration (89% confidence)',
      counterfactual: 'Normal Transfer (82% confidence)',
    },
  },
  'INC-2026-003': {
    original: 'Current scenario: Brute-force attack on auth service with 91% anomaly score',
    counterfactual: 'Alternative scenario: Normal authentication traffic with 10% anomaly score',
    changes: [
      {
        parameter: 'Login Attempts',
        original: '1200 attempts/min',
        modified: '3 attempts/min',
        impact: 0.95,
      },
      {
        parameter: 'Failure Rate',
        original: '98% failed',
        modified: '12% failed',
        impact: 0.88,
      },
      {
        parameter: 'Source Diversity',
        original: 'Distributed botnet IPs',
        modified: 'Single corporate VPN',
        impact: 0.76,
      },
    ],
    prediction: {
      original: 'Brute-Force Attack (91% confidence)',
      counterfactual: 'Normal Auth Traffic (86% confidence)',
    },
  },
  'INC-2026-004': {
    original: 'Current scenario: Lateral movement in K8s cluster with 78% anomaly score',
    counterfactual: 'Alternative scenario: Normal inter-service communication with 18% anomaly score',
    changes: [
      {
        parameter: 'Pod-to-Pod Connections',
        original: '340 unique pairs',
        modified: '28 unique pairs',
        impact: 0.83,
      },
      {
        parameter: 'Service Account Usage',
        original: 'Default SA with cluster-admin',
        modified: 'Scoped SA with least privilege',
        impact: 0.77,
      },
    ],
    prediction: {
      original: 'Lateral Movement (78% confidence)',
      counterfactual: 'Normal Traffic (74% confidence)',
    },
  },
  'INC-2026-005': {
    original: 'Current scenario: TLS certificate anomaly detected with 62% anomaly score',
    counterfactual: 'Alternative scenario: Valid certificate rotation with 8% anomaly score',
    changes: [
      {
        parameter: 'Certificate Validity',
        original: 'Self-signed, expired',
        modified: 'CA-signed, valid 90 days',
        impact: 0.72,
      },
      {
        parameter: 'Rotation Frequency',
        original: '14 rotations/day',
        modified: '1 rotation/90 days',
        impact: 0.58,
      },
    ],
    prediction: {
      original: 'Certificate Anomaly (62% confidence)',
      counterfactual: 'Normal Operation (90% confidence)',
    },
  },
};
