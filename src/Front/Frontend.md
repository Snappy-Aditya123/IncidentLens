# IncidentLens — Frontend

Interactive investigation dashboard built with React, Tailwind CSS v4, shadcn/ui, and D3.js.

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

---

## Stack

| Layer | Technology |
|:------|:-----------|
| **Framework** | React 19 + TypeScript |
| **Routing** | React Router v7 (nested routes, `<Outlet>`) |
| **Styling** | Tailwind CSS v4 (oklch design tokens, `@source` directive) |
| **Components** | shadcn/ui (Card, Badge, Button, Tabs, Progress, etc.) |
| **Visualization** | D3.js force-directed graph (interactive, drag-enabled) |
| **Icons** | lucide-react |
| **Date formatting** | date-fns |
| **Animations** | tw-animate-css |

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

The entry point analysts see. Displays:

- **Stats grid** — 4 cards: Active Incidents, Critical Alerts, Avg Anomaly Score, Total Incidents (computed from incident data)
- **Search bar** — real-time client-side filtering by title or ID
- **Incident cards** — each shows severity badge, status, description, affected systems, anomaly score percentage, and an **"Investigate"** CTA that routes to the investigation wizard

### Investigation (`/investigation/:incidentId`)

A 4-step wizard with a progress bar and clickable step navigation:

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

Defined in `app/data/mockData.ts`:

```typescript
interface Incident {
  id: string;
  title: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  status: 'active' | 'investigating' | 'resolved' | 'monitoring';
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
  weight: number;
  anomalous: boolean;
}

interface CounterfactualExplanation {
  original: { prediction: string; confidence: number };
  counterfactual: { prediction: string; confidence: number };
  changes: Array<{
    parameter: string;
    original: string;
    modified: string;
    impact: number;
  }>;
}
```

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
├── __init__.py
├── app/
│   ├── App.tsx                          # Root — RouterProvider + Toaster
│   ├── routes.tsx                       # Route definitions
│   ├── components/
│   │   ├── Root.tsx                     # Layout shell (dark bg + Outlet)
│   │   ├── Dashboard.tsx                # Incident list + stats
│   │   ├── Investigation.tsx            # 4-step wizard orchestrator
│   │   ├── NotFound.tsx                 # 404 page
│   │   ├── investigation/
│   │   │   ├── ElasticsearchStep.tsx    # ES log analysis step
│   │   │   ├── GNNStep.tsx             # D3 network graph step
│   │   │   └── CounterfactualStep.tsx   # Counterfactual explanation step
│   │   ├── figma/
│   │   │   └── ImageWithFallback.tsx    # Image with error fallback
│   │   └── ui/                          # 48 shadcn/ui primitives
│   └── data/
│       └── mockData.ts                  # Type definitions + mock data
└── styles/
    ├── index.css                        # Entry point
    ├── fonts.css                        # Font declarations
    ├── tailwind.css                     # Tailwind v4 config
    └── theme.css                        # oklch design tokens
```

---

## Connecting to the Backend

The frontend currently uses static mock data. To connect to the live backend API:

1. **Start the backend:** `python src/Backend/main.py serve --port 8000`
2. **Replace mock data** in Dashboard/Investigation with `fetch()` calls to:
   - `GET /api/flows` — populate incident list
   - `POST /api/investigate` — trigger investigation
   - `WS /ws/investigate` — stream real-time investigation events
   - `GET /api/stats` — feature statistics
   - `POST /api/counterfactual` — counterfactual analysis
3. **WebSocket integration** — connect to `ws://localhost:8000/ws/investigate` to receive streaming `thinking`, `tool_call`, `tool_result`, and `conclusion` events to drive the step progression in real time
