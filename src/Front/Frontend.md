# IncidentLens — Frontend

Interactive investigation dashboard built with React 19, TypeScript, Vite 6, Tailwind CSS v4, shadcn/ui, and D3.js — fully integrated with the FastAPI backend via a typed API service layer and React hooks.

---

## Overview

The frontend provides a **guided 4-step investigation workflow** that walks analysts through an incident from detection to explanation:

```
┌──────────────┐    ┌────────────────┐    ┌───────────────┐    ┌──────────────────┐
│   Overview   │ →  │  Log Analysis  │ →  │ Network Graph │ →  │ Explainability   │
│              │    │ (Elasticsearch)│    │   (GNN)       │    │ (Counterfactual) │
└──────────────┘    └────────────────┘    └───────────────┘    └──────────────────┘
```

Each step surfaces progressively deeper insight — from raw logs to graph-level anomaly scores to *"what would need to change for this to be normal."*

Data is fetched from the live FastAPI backend via typed hooks. If the backend is unreachable (e.g., during UI-only development), every hook falls back to mock data automatically — no code changes needed.

---

## Stack

| Layer | Technology |
|:------|:-----------|
| **Framework** | React 19 + TypeScript |
| **Build** | Vite 6 with `@vitejs/plugin-react` + `@tailwindcss/vite` |
| **Routing** | React Router v7 (nested routes, `<Outlet>`) |
| **Styling** | Tailwind CSS v4 (oklch design tokens, `@source` directive) |
| **Components** | shadcn/ui (Card, Badge, Button, Tabs, Progress, etc.) |
| **Visualization** | D3.js force-directed graph (interactive, drag-enabled) |
| **API layer** | Typed fetch client (`services/api.ts`) + WebSocket async-generator |
| **State** | React hooks (`hooks/useApi.ts`) — live backend with mock fallback |
| **Icons** | lucide-react |
| **Date formatting** | date-fns |
| **Animations** | tw-animate-css |

---

## Quick Start

```bash
cd src/Front
npm install
npm run dev          # → http://localhost:5173
```

The Vite dev server proxies `/api` and `/ws` to `http://localhost:8000` (the FastAPI backend). Both servers can run simultaneously.

| Script | Command | Description |
|:-------|:--------|:------------|
| dev | `npm run dev` | Start Vite dev server with HMR on `:5173` |
| build | `npm run build` | Type-check + production build → `dist/` |
| preview | `npm run preview` | Preview the production build locally |
| lint | `npm run lint` | Run ESLint |

---

## Routes

| Path | Component | Description |
|:-----|:----------|:------------|
| `/` | `Dashboard` | Main landing page — incident list with search, stats overview |
| `/investigation/:incidentId` | `Investigation` | 4-step guided investigation wizard |
| `*` | `NotFound` | 404 fallback |

---

## Pages

### Dashboard (`/`)

The entry point analysts see. Uses `useIncidents()` to fetch anomalous flows from the backend. Displays:

- **Stats grid** — 4 cards: Active Incidents, Critical Alerts, Avg Anomaly Score, Total Incidents (computed from incident data)
- **Search bar** — real-time client-side filtering by title or ID
- **Loading state** — `<Skeleton>` placeholders while data loads
- **Error banner** — `<AlertTriangle>` error display with a retry button
- **Incident cards** — each shows severity badge, status, description, affected systems, anomaly score percentage, and an **"Investigate"** CTA that routes to the investigation wizard
- **Refresh button** — `<RefreshCw>` spinner for manual data re-fetch

### Investigation (`/investigation/:incidentId`)

A 4-step wizard with a progress bar and clickable step navigation. Uses 4 hooks (`useIncident`, `useElasticsearchData`, `useNetworkGraph`, `useCounterfactual`) to fetch live data per step with loading spinners and empty-state handling:

| Step | Component | What It Shows |
|:-----|:----------|:--------------|
| **1. Overview** | Inline | Incident summary — description, affected systems, anomaly score, status, timestamp. "Start Investigation" CTA. |
| **2. Log Analysis** | `ElasticsearchStep` | ES query DSL display, hit count, log entries with color-coded severity levels (CRITICAL/ERROR/WARNING/INFO) |
| **3. Network Graph** | `GNNStep` | Interactive D3 force-directed graph — nodes colored by status (compromised/suspicious/normal), edges flagged as anomalous, risk scores. Risk assessment table for top 5 nodes. |
| **4. Explainability** | `CounterfactualStep` | Side-by-side scenario comparison (current vs counterfactual), per-parameter impact analysis with progress bars, AI-generated insights, recommended actions |

---

## Components

### Core Components

| Component | File | Purpose |
|:----------|:-----|:--------|
| `Root` | `Root.tsx` | Layout shell — dark theme wrapper with `<Outlet>` |
| `Dashboard` | `Dashboard.tsx` | Incident list page with stats and search |
| `Investigation` | `Investigation.tsx` | 4-step wizard orchestrator with step state management |
| `NotFound` | `NotFound.tsx` | 404 page |

### Investigation Step Components

| Component | File | Props | Key Feature |
|:----------|:-----|:------|:------------|
| `ElasticsearchStep` | `investigation/ElasticsearchStep.tsx` | `data` (logs + query), `onNext` | ES query DSL viewer + log entry browser |
| `GNNStep` | `investigation/GNNStep.tsx` | `data` (nodes + edges), `onNext` | D3 force-directed graph with drag interaction, risk assessment table |
| `CounterfactualStep` | `investigation/CounterfactualStep.tsx` | `data` (counterfactual) | Scenario comparison, parameter impact analysis, recommendations |

### Utility Components

| Component | File | Purpose |
|:----------|:-----|:--------|
| `ImageWithFallback` | `figma/ImageWithFallback.tsx` | `<img>` with graceful error state (gray placeholder SVG) |
| 48 shadcn/ui primitives | `ui/*.tsx` | Accordion, Badge, Button, Card, Dialog, Tabs, Table, Tooltip, etc. |

---

## Data Types

Shared types are defined in `app/types.ts` and split into two groups:

### Frontend UI Types

```typescript
interface Incident {
  id: string;
  title: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  status: 'investigating' | 'resolved' | 'escalated';
  description: string;
  timestamp: string;
  affectedSystems: string[];
  anomalyScore: number;
}

interface NetworkNode {
  id: string;
  label: string;
  type: 'server' | 'service' | 'database' | 'endpoint' | 'firewall';
  status: 'compromised' | 'suspicious' | 'normal';
  risk: number;
}

interface NetworkEdge {
  source: string;
  target: string;
  type: 'connection' | 'data_flow' | 'dependency';
  weight: number;
  anomalous: boolean;
}

interface CounterfactualExplanation {
  original: string;
  counterfactual: string;
  changes: Array<{
    parameter: string;
    original: string;
    modified: string;
    impact: number;
  }>;
  prediction: { original: string; counterfactual: string };
}
```

### Backend Response Types

```typescript
interface BackendFlow {
  _id: string;
  src_ip: string;   dst_ip: string;
  label: number;
  packet_count?: number;  total_bytes?: number;
  mean_payload?: number;  mean_iat?: number;  std_iat?: number;
  prediction?: number;    prediction_score?: number;
}

interface BackendDetectResponse { method: string; count: number; flows: BackendFlow[]; }
interface BackendFlowsResponse  { count: number; flows: BackendFlow[]; }
interface BackendCounterfactualResponse { flow_id: string; diffs: Array<{feature: string; anomalous_value: number; normal_value: number; abs_diff: number; direction: string}>; }
interface BackendSeverityResponse { flow_id: string; severity: string; z_scores: Record<string, number>; max_z: number; }
interface BackendHealthResponse  { server: string; elasticsearch: string; }

// WebSocket events
type InvestigationEventType = 'thinking' | 'tool_call' | 'tool_result' | 'conclusion' | 'error' | 'done';
interface InvestigationEvent { type: InvestigationEventType; content?: string; tool?: string; arguments?: Record<string, unknown>; result?: string; }
```

---

## API Service Layer

`app/services/api.ts` — A typed fetch client with every backend endpoint:

| Function | Endpoint | Return Type |
|:---------|:---------|:------------|
| `checkHealth()` | `GET /health` | `BackendHealthResponse` |
| `listFlows(params?)` | `GET /api/flows` | `BackendFlowsResponse` |
| `detectAnomalies(params?)` | `POST /api/detect` | `BackendDetectResponse` |
| `getFlowSeverity(flowId)` | `GET /api/severity/:id` | `BackendSeverityResponse` |
| `getFlowCounterfactual(flowId)` | `POST /api/counterfactual` | `BackendCounterfactualResponse` |
| `getSimilarIncidents(flowId, k?)` | `GET /api/similar/:id` | `{flow_id, similar}` |
| `getFeatureStats()` | `GET /api/stats` | `Record<string, unknown>` |
| `investigate(query)` | `POST /api/investigate` | `{events: InvestigationEvent[]}` |
| `investigateStream(query, signal?)` | `WS /ws/investigate` | `AsyncGenerator<InvestigationEvent>` |

`investigateStream()` is a WebSocket **async generator** — use it with `for await...of`:

```typescript
for await (const event of investigateStream("Why is 10.0.2.45 anomalous?")) {
  console.log(event.type, event.content);
}
```

---

## React Hooks

`app/hooks/useApi.ts` — Built on a generic `useAsync<T>` hook. Each returns `{ data, loading, error, refetch }`.

| Hook | Data Source | Fallback | Returns |
|:-----|:-----------|:---------|:--------|
| `useBackendHealth()` | `GET /health` | — | `BackendHealthResponse` |
| `useIncidents()` | `POST /api/detect` → map to `Incident[]` | `mockIncidents` | `Incident[]` |
| `useIncident(id)` | `POST /api/detect` → find by `_id` | `mockIncidents` lookup | `Incident \| null` |
| `useElasticsearchData(id)` | `GET /api/flows` → build log view | `mockElasticsearchResults` | `ElasticsearchData \| null` |
| `useNetworkGraph(id)` | `POST /api/detect` → `flowsToGraph()` | `mockNetworkGraph` | `NetworkGraphData \| null` |
| `useCounterfactual(id)` | `POST /api/counterfactual` → `backendCfToFrontend()` | `mockCounterfactuals` | `CounterfactualExplanation \| null` |
| `useSeverity(flowId)` | `GET /api/severity/:id` | `null` | `BackendSeverityResponse \| null` |
| `useInvestigationStream()` | `WS /ws/investigate` | — | `{events, running, error, start, stop}` |

**Mock fallback pattern:** Every data hook wraps its API call in `try/catch`. If the backend is unreachable, it falls back to the mock data in `data/mockData.ts`. This enables full offline UI development.

---

## Styling Architecture

```
styles/
├── fonts.css       # @font-face declarations (placeholder)
├── tailwind.css    # Tailwind v4 setup + @source directive
├── theme.css       # oklch design tokens (light + dark mode)
└── index.css       # Entry point — imports above in order
```

- **Dark theme by default** — `bg-slate-950` root, oklch color tokens for dark mode
- **shadcn/ui theming** — CSS custom properties (`--background`, `--primary`, `--destructive`, etc.) consumed by Tailwind utilities
- **Tailwind v4** — uses `@import 'tailwindcss' source(none)` with explicit `@source` scanning

---

## File Structure

```
src/Front/
├── Frontend.md                          # This file
├── package.json                         # Deps + scripts (dev, build, preview, lint)
├── vite.config.ts                       # Vite 6 — React + Tailwind plugins, proxy config
├── tsconfig.json                        # Strict TS, @/ → app/ alias
├── index.html                           # HTML entry point (dark class on root)
├── vite-env.d.ts                        # Vite type references
├── __init__.py
├── app/
│   ├── main.tsx                         # React 19 createRoot mount
│   ├── App.tsx                          # RouterProvider + Toaster
│   ├── routes.tsx                       # Route definitions
│   ├── types.ts                         # Shared UI + backend response types
│   ├── services/
│   │   └── api.ts                       # Typed fetch client + WebSocket stream
│   ├── hooks/
│   │   └── useApi.ts                    # 7 hooks — live API with mock fallback
│   ├── components/
│   │   ├── Root.tsx                     # Layout shell (dark bg + Outlet)
│   │   ├── Dashboard.tsx                # Incident list + stats (useIncidents)
│   │   ├── Investigation.tsx            # 4-step wizard (4 hooks, per-step loading)
│   │   ├── NotFound.tsx                 # 404 page
│   │   ├── investigation/
│   │   │   ├── ElasticsearchStep.tsx    # ES log analysis step
│   │   │   ├── GNNStep.tsx             # D3 network graph step
│   │   │   └── CounterfactualStep.tsx   # Counterfactual explanation step
│   │   ├── figma/
│   │   │   └── ImageWithFallback.tsx    # Image with error fallback
│   │   └── ui/                          # 48 shadcn/ui primitives
│   └── data/
│       └── mockData.ts                  # Mock data for offline fallback
└── styles/
    ├── index.css                        # Entry point
    ├── fonts.css                        # Font declarations
    ├── tailwind.css                     # Tailwind v4 config
    └── theme.css                        # oklch design tokens
```

---

## Build & Proxy Architecture

```
Browser (:5173)                    Vite Dev Server                    FastAPI (:8000)
    │                                   │                                  │
    │  GET /api/incidents  ──────────▶  │  proxy /api/* ─────────────────▶ │
    │                                   │                                  │
    │  WS /ws/investigate  ──────────▶  │  proxy /ws/* (ws: true) ──────▶ │
    │                                   │                                  │
    │  GET /health  ─────────────────▶  │  proxy /health ───────────────▶ │
    │                                   │                                  │
    │  GET / (React app)  ───────────▶  │  serves index.html + HMR        │
```

**Development:** Run both `npm run dev` (frontend on `:5173`) and `python src/Backend/main.py serve` (backend on `:8000`). The Vite proxy forwards all API requests transparently.

**Production:** `npm run build` outputs a static `dist/` directory. Serve it from the FastAPI server or any static host; update API URLs accordingly.

---

## Connecting to the Backend

The frontend is **already connected** to the backend. All data flows through the hooks in `hooks/useApi.ts`:

1. **Start the backend:** `python src/Backend/main.py serve --port 8000`
2. **Start the frontend:** `cd src/Front && npm run dev`
3. Dashboard calls `useIncidents()` → `POST /api/detect` → maps flows to `Incident[]`
4. Investigation uses `useIncident`, `useElasticsearchData`, `useNetworkGraph`, `useCounterfactual`
5. WebSocket streaming: `useInvestigationStream()` → `WS /ws/investigate`

If the backend is offline, all hooks silently fall back to mock data — no errors in the UI.
