<div align="center">

# IncidentLens

### AI-Powered Network Incident Investigation with Explainable Graph Intelligence

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Elasticsearch 8.12](https://img.shields.io/badge/Elasticsearch-8.12-005571?logo=elasticsearch&logoColor=white)](https://elastic.co)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.6-EE4C2C?logo=pytorch&logoColor=white)](https://pyg.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**IncidentLens** doesn't just detect threats — it *explains* them. An autonomous AI agent that investigates network anomalies, uncovers root causes through counterfactual reasoning, and delivers actionable intelligence in real time.

[Getting Started](#-getting-started) · [How It Works](#-how-it-works) · [API Reference](#-api-reference) · [Architecture](#-architecture)

</div>

---

## The Problem

Security teams drown in alerts. Traditional tools flag anomalies but can't answer the critical question: **"Why is this suspicious, and what would need to change for it to be normal?"**

Analysts spend hours manually correlating logs, comparing traffic patterns, and trying to articulate *what specifically makes a flow anomalous* — work that is repetitive, slow, and error-prone.

## Our Solution

IncidentLens is an **autonomous investigation agent** that combines three powerful capabilities:

| Capability | What It Does |
|:---|:---|
| **Graph Neural Networks** | Models network traffic as temporal graphs — nodes are IPs, edges are flows — to capture structural patterns invisible to flat feature analysis |
| **Counterfactual Explainability** | For every anomaly, finds the *nearest normal flow* and tells you exactly which features (packet count, byte volume, inter-arrival time) would need to change and by how much |
| **Elasticsearch-Powered Retrieval** | kNN vector search over flow embeddings, significant-terms aggregation, and feature distribution analysis — all powering a multi-step LLM reasoning loop |

The agent doesn't just say "this is malicious" — it says *"this flow has 47x the normal packet count, 12x the byte volume, and the nearest non-attack flow with similar structure is X; reducing packet_count from 450 to 9 would flip the classification."*

---

## Key Features

- **Autonomous Multi-Step Investigation** — An LLM agent with 15 specialized tools iterates through detection → analysis → explanation → severity assessment → recommendation, streaming each reasoning step in real time
- **Temporal Graph Construction** — Sliding-window graph builder converts raw packets into PyG `Data` objects with node features (degree, traffic volume) and edge features (packet count, bytes, payload, inter-arrival time)
- **Dual GNN Architecture** — EdgeGNN (GraphSAGE + Edge MLP) for static classification and EvolveGCN-O (LSTM-evolved weights) for capturing temporal attack patterns
- **Counterfactual Analysis** — Feature-level diffs ("what would need to change?") and graph-level edge perturbation ("which connections drive the anomaly?")
- **Real-Time WebSocket Streaming** — Every thinking step, tool call, and conclusion is streamed to the frontend as it happens
- **Interactive Investigation Dashboard** — React 19 + Vite 6 frontend with a 4-step guided wizard (Overview → ES Logs → Network Graph → Counterfactual Explainability), shadcn/ui components, Tailwind v4 dark theme, and typed API hooks that fall back to mock data when the backend is offline
- **Kitsune Dataset Validated** — Tested on 4M+ real SSDP flood attack packets with ground-truth labels

---

## How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│  React Frontend (Dashboard + 4-Step Investigation Wizard)        │
│  ┌────────────┐  ┌──────────────┐  ┌──────────┐  ┌───────────┐  │
│  │  Overview   │→ │ ES Log View  │→ │ D3 Graph │→ │Counterfact│  │
│  └────────────┘  └──────────────┘  └──────────┘  └───────────┘  │
└───────────────────────┬──────────────────────────────────────────┘
                        │ REST + WebSocket
                        ▼
              ┌─────────────────────┐
              │  FastAPI Server      │
              │  (REST + WS stream)  │
              └──────┬───────────────┘
                     │
              ┌──────▼──────────────┐
              │     LLM Agent       │ ◄── OpenAI / Azure / Ollama
              │  (Multi-Step Loop)  │
              └──────┬──────────────┘
                     │ tool calls
        ┌────────────┼────────────────┐
        ▼            ▼                ▼
  ┌───────────┐ ┌──────────┐  ┌──────────────┐
  │  Detect   │ │ Analyze  │  │   Explain    │
  │ Anomalies │ │ Features │  │Counterfactual│
  └─────┬─────┘ └────┬─────┘  └──────┬───────┘
        │             │               │
        └─────────────┼───────────────┘
                      ▼
          ┌───────────────────────┐
          │   Elasticsearch 8.12  │
          │                       │
          │  ▸ incidentlens-flows │  ← aggregated flow features
          │  ▸ ...-embeddings     │  ← kNN vector search (cosine)
          │  ▸ ...-counterfactuals│  ← feature diffs + explanations
          │  ▸ ...-packets        │  ← raw packet records
          └───────────────────────┘
```

**Investigation loop:** The frontend walks analysts through a 4-step guided wizard. The backend agent detects anomalies → retrieves flow details → runs counterfactual analysis (kNN nearest-normal + feature diff) → assesses severity → finds similar incidents → streams each reasoning step back to the UI in real time via WebSocket.

---

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+ and npm (for the React frontend)
- Docker (for Elasticsearch + Kibana)
- An OpenAI-compatible API key (OpenAI, Azure OpenAI, or local Ollama)

### 1. Clone & Install

```bash
git clone https://github.com/Snappy-Aditya123/IncidentLens.git
cd IncidentLens

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2. Start Elasticsearch

```bash
docker compose -f src/docker-compose.yml up -d
```

Elasticsearch: http://localhost:9200 · Kibana: http://localhost:5601

### 3. Verify Connectivity

```bash
python src/Backend/main.py health
```

### 4. Prepare Data

```bash
# Convert CSV → NDJSON (first time only)
python src/Backend/main.py convert \
  --packets path/to/ssdp_packets_rich.csv \
  --labels path/to/SSDP_Flood_labels.csv

# Run full ingestion + analysis pipeline
python src/Backend/main.py ingest
```

### 5. Investigate

```bash
export OPENAI_API_KEY=sk-...

# Auto-detect and investigate anomalies
python src/Backend/main.py investigate

# Ask a specific question
python src/Backend/main.py investigate "why is 192.168.100.5 anomalous?"
```

### 6. Start the API Server

```bash
python src/Backend/main.py serve --port 8000
```

### 7. Start the Frontend

```bash
cd src/Front
npm install
npm run dev          # → http://localhost:5173
```

The Vite dev server proxies `/api` and `/ws` requests to the backend at `localhost:8000`, so both servers can run simultaneously in development. In production, build the frontend with `npm run build` and serve the resulting `dist/` directory from any static host (e.g., Nginx, Vercel, or a CDN). The FastAPI server does **not** serve static files — you need a separate static host or a reverse proxy.

> **Note:** `.gitignore` excludes `*.csv` files, so raw datasets must be distributed separately.

---

## Project Structure

```
IncidentLens/
├── src/
│   ├── Backend/                        # Python backend
│   │   ├── main.py                     # CLI shim → imports from tests/testingentry.py
│   │   ├── agent.py                    # LLM agent — multi-step reasoning loop
│   │   ├── agent_tools.py              # 15 tools with OpenAI function-calling schemas
│   │   ├── server.py                   # FastAPI server (REST + WebSocket + Incident API)
│   │   ├── wrappers.py                 # ES client, indexing, kNN, counterfactuals
│   │   ├── graph_data_wrapper.py       # Vectorised sliding-window graph builder
│   │   ├── graph.py                    # Core graph data structures
│   │   ├── train.py                    # EdgeGNN (GraphSAGE) training pipeline
│   │   ├── temporal_gnn.py             # EvolveGCN-O semi-temporal model
│   │   ├── gnn_interface.py            # Abstract GNN encoder contract
│   │   ├── ingest_pipeline.py          # 8-step data ingestion pipeline
│   │   ├── csv_to_json.py              # CSV → NDJSON converter
│   │   ├── GNN.py                      # (Deprecated) standalone EdgeGNN ref
│   │   ├── __init__.py
│   │   ├── backup/                     # Earlier model versions kept for reference
│   │   │   ├── temporal_gnn_v1_backup.py
│   │   │   └── __init__.py
│   │   └── tests/                      # 166 tests (100% pass)
│   │       ├── testingentry.py         # Actual CLI implementation (5 commands)
│   │       ├── test_gnn_edge_cases.py
│   │       ├── test_temporal_gnn_full.py
│   │       ├── test_temporal_gnn_meticulous.py
│   │       ├── run_all.py              # Test suite runner
│   │       └── __init__.py
│   ├── Front/                          # React frontend
│   │   ├── package.json                 # Dependencies + scripts (dev, build, preview)
│   │   ├── vite.config.ts               # Vite 6 + proxy (/api → :8000, /ws → :8000)
│   │   ├── tsconfig.json                # Strict TS config + @/ alias
│   │   ├── index.html                   # HTML entry point
│   │   ├── app/
│   │   │   ├── main.tsx                 # React 19 root mount
│   │   │   ├── App.tsx                  # RouterProvider + Toaster
│   │   │   ├── routes.tsx               # Route definitions
│   │   │   ├── types.ts                 # Shared UI + backend response types
│   │   │   ├── services/api.ts          # Typed fetch client + WebSocket stream
│   │   │   ├── hooks/useApi.ts          # 8 hooks — live API with mock fallback
│   │   │   ├── components/
│   │   │   │   ├── Dashboard.tsx         # Incident list + stats (live data)
│   │   │   │   ├── Investigation.tsx     # 4-step wizard (live data)
│   │   │   │   ├── investigation/
│   │   │   │   │   ├── ElasticsearchStep.tsx  # ES log analysis
│   │   │   │   │   ├── GNNStep.tsx            # D3 network graph
│   │   │   │   │   └── CounterfactualStep.tsx # Explainability
│   │   │   │   └── ui/                  # 46 shadcn/ui components + utilities
│   │   │   └── data/mockData.ts         # Mock data for offline fallback
│   │   ├── styles/                      # Tailwind v4 + oklch theme tokens
│   │   ├── vite-env.d.ts                # Vite type references
│   │   └── __init__.py
│   └── docker-compose.yml               # ES 8.12 + Kibana 8.12
├── .gitignore                           # Ignores .venv, __pycache__, dist/, node_modules/, *.csv
├── data/                                # NDJSON data files
├── EDA/                                 # Exploratory data analysis (Jupyter notebooks)
│   └── EDA.ipynb                        # Dataset profiling and visualization
├── requirements.txt                     # Python dependencies (pip install -r)
└── LICENSE                              # MIT
```

---

## API Reference

### REST Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/health` | Server + Elasticsearch health check |
| `POST` | `/api/investigate` | Run a full investigation (returns all events as JSON) |
| `POST` | `/api/detect` | Quick anomaly detection without the LLM agent |
| `GET` | `/api/flows` | List flows with optional filters (label, src_ip, dst_ip) |
| `GET` | `/api/stats` | Feature statistics grouped by label |
| `POST` | `/api/counterfactual` | Run counterfactual analysis for a specific flow |
| `GET` | `/api/severity/{flow_id}` | Assess anomaly severity (low / medium / high) |
| `GET` | `/api/similar/{flow_id}` | Find similar historical incidents via kNN |
| `GET` | `/api/tools` | List all available agent tools |
| `GET` | `/api/incidents` | Anomalous flows as frontend-ready Incident objects |
| `GET` | `/api/incidents/{id}` | Single incident detail by flow ID |
| `GET` | `/api/incidents/{id}/graph` | Network graph (nodes + edges) scoped to incident IPs for D3 visualization |
| `GET` | `/api/incidents/{id}/logs` | ES-style log entries scoped to incident IPs for log viewer |

### WebSocket

| Endpoint | Description |
|:---------|:------------|
| `WS /ws/investigate` | Stream investigation events in real time |

**Protocol:** Connect → send `{"query": "..."}` → receive streaming events → receive `{"type": "done"}`.

Event types: `thinking`, `tool_call`, `tool_result`, `conclusion`, `error`, `status`, `done`.

---

## Agent Tools (15)

| Tool | Purpose |
|:-----|:--------|
| `es_health_check` | Cluster health and connectivity |
| `search_flows` | Query flows with filters (label, IP, thresholds) |
| `get_flow` | Retrieve a single flow document by ID |
| `detect_anomalies` | Detect anomalies via label, model score, or statistics |
| `feature_stats` | Extended statistics per feature grouped by label |
| `feature_percentiles` | Percentile distributions for any feature |
| `significant_terms` | IPs/protocols overrepresented in attack traffic |
| `counterfactual_analysis` | Find nearest normal flow + compute per-feature diffs |
| `counterfactual_narrative` | Human-readable counterfactual explanation |
| `explain_flow` | ES `_explain` API — why a flow matched a query |
| `search_raw_packets` | Search individual raw packet records |
| `find_similar_incidents` | kNN embedding search for historical matches |
| `assess_severity` | Z-score severity assessment (low / medium / high) |
| `graph_edge_counterfactual` | Identify which edges (connections) drive the anomaly |
| `graph_window_comparison` | Compare normal vs anomalous time windows structurally |

---

## Elasticsearch Indices

| Index | Contents |
|:------|:---------|
| `incidentlens-flows` | Aggregated flow features — packet_count, total_bytes, mean_payload, mean_iat, std_iat, label, predictions |
| `incidentlens-embeddings` | Per-flow embedding vectors (dense_vector with cosine similarity, kNN-indexed) |
| `incidentlens-counterfactuals` | Counterfactual diffs — per-feature original vs CF value, direction, percent change |
| `incidentlens-packets` | Raw individual packet records from the dataset |

> The 4th index (`incidentlens-packets`) is created and populated by `ingest_pipeline.py`. The mapping includes: `packet_index`, `timestamp`, `inter_arrival_time`, `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`, `ttl`, `ip_header_len`, `tcp_flags`, `udp_length`, `payload_length`, `packet_length`, `label`.

---

## Technical Highlights

- **Vectorized graph construction** — Pure numpy sliding-window builder with composite key packing, broadcast window assignment, and pre-built neighbor lists. Zero Python for-loops over packets.
- **Dual GNN models** — EdgeGNN (GraphSAGE + Edge MLP) for static edge classification; EvolveGCN-O (LSTM-evolved GCN weights) for temporal pattern detection across graph sequences.
- **Pre-processed GNN bottleneck removal** — Self-loops and degree normalization cached at data-prep time; LSTM weight evolution flattened from O(hidden_dim) batches to O(1).
- **Singleton ES client** with retry logic, bulk indexing with batch MD5 flow-ID generation, and pre-converted numpy→Python type coercion for minimal per-document overhead.
- **166 tests** covering graph construction, GNN forward/backward passes, temporal sequences, normalization, and edge-case handling — all passing.
- **Full-stack integration** — Typed API service layer (`services/api.ts`) with 9 typed fetch functions, WebSocket async-generator streaming client, and 8 React hooks (`useBackendHealth`, `useIncidents`, `useIncident`, `useElasticsearchData`, `useNetworkGraph`, `useCounterfactual`, `useSeverity`, `useInvestigationStream`) that try the live backend first and fall back to mock data for offline development. Incident-scoped hooks (`useElasticsearchData`, `useNetworkGraph`) call the `/api/incidents/{id}/logs` and `/api/incidents/{id}/graph` endpoints directly for per-incident data.
- **Zero-config dev proxy** — Vite dev server on `:5173` proxies `/api` and `/ws` to the FastAPI backend on `:8000`, so frontend and backend can be developed and run simultaneously with no CORS issues.

---

## CLI Reference

| Command | Description |
|:--------|:------------|
| `python src/Backend/main.py health` | Check ES connectivity and index document counts |
| `python src/Backend/main.py ingest` | Run full pipeline: graphs → flows → embeddings → counterfactuals |
| `python src/Backend/main.py investigate [query]` | Run the LLM agent investigation |
| `python src/Backend/main.py serve` | Start the REST + WebSocket API server |
| `python src/Backend/main.py convert` | Convert raw CSV data to NDJSON |

---

## Team

Built by **Aditya, Anuska, and Aayush** for the Elasticsearch Agent Builder Hackathon.

## License

[MIT](LICENSE)

