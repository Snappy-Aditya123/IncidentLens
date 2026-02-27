<div align="center">

# IncidentLens

### AI-Powered Network Incident Investigation with Explainable Graph Intelligence

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Elasticsearch 8.12](https://img.shields.io/badge/Elasticsearch-8.12-005571?logo=elasticsearch&logoColor=white)](https://elastic.co)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.6-EE4C2C?logo=pytorch&logoColor=white)](https://pyg.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?logo=openai&logoColor=white)](https://platform.openai.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**IncidentLens** doesn't just detect threats — it *explains* them. A real-time streaming engine that ingests network packets, builds temporal graphs, scores anomalies with GNNs, and autonomously investigates elevated threats using an LLM reasoner with tool-calling — delivering actionable intelligence and email alerts as incidents unfold.

[Getting Started](#-getting-started) · [How It Works](#-how-it-works) · [API Reference](#-api-reference) · [Architecture](#-architecture)

</div>

---

## The Problem

Security teams drown in alerts. Traditional tools flag anomalies but can't answer the critical question: **"Why is this suspicious, and what would need to change for it to be normal?"**

Analysts spend hours manually correlating logs, comparing traffic patterns, and trying to articulate *what specifically makes a flow anomalous* — work that is repetitive, slow, and error-prone.

## Our Solution

IncidentLens is an **end-to-end real-time investigation system** that combines four powerful capabilities:

| Capability | What It Does |
|:---|:---|
| **Real-Time Streaming Engine** | Ingests raw packets via `StreamSimulator`, aggregates them into time-windowed flows, and processes each window through the full graph + GNN + ES + LLM pipeline as it closes |
| **Graph Neural Networks** | Models network traffic as temporal graphs — nodes are IPs, edges are flows — to capture structural patterns invisible to flat feature analysis |
| **LLM Autonomous Reasoner** | An OpenAI-powered agent with 5 investigation tools (flow lookup, graph summary, severity breakdown, recent windows, email alerts) that autonomously investigates any window with an anomaly score >= 0.5 |
| **Elasticsearch-Powered Retrieval** | kNN vector search over flow embeddings, significant-terms aggregation, severity breakdowns, and graph summary indexing — powering both the LLM reasoner and the interactive dashboard |

The system doesn't just say "this is malicious" — it streams real-time analysis like *"Window 12 has anomaly score 0.78; 3 flows from 192.168.100.5 show 47x normal packet count targeting SSDP ports; recommend blocking source IP."* and sends email alerts for high/critical findings.

---

## Key Features

- **Real-Time Streaming Pipeline** — `StreamSimulator` replays packets at configurable rates, aggregates them into time windows, and feeds each window through graph construction → ES indexing → GNN scoring → LLM investigation. No batch preprocessing required.
- **Autonomous LLM Reasoner** — An `AsyncOpenAI` agent (gpt-4o-mini by default) with 5 specialized tools iterates through investigation steps autonomously via tool-calling, stores insights to Elasticsearch, and sends SMTP email alerts for high/critical-risk findings
- **Autonomous Multi-Step Investigation** — A separate LLM agent with 19 specialized tools for the interactive dashboard iterates through detection → analysis → explanation → severity assessment → recommendation, streaming each reasoning step via WebSocket
- **Temporal Graph Construction** — Sliding-window graph builder converts raw packets into PyG `Data` objects with node features (degree, traffic volume) and edge features (packet count, bytes, payload, inter-arrival time)
- **Dual GNN Architecture** — EvolveGCN-O (LSTM-evolved weights) for temporal patterns and Neural ODE variant (EvolvingGNN_ODE with RK4 solver) for continuous-time weight evolution. Trainable via CLI with CSV or NDJSON input.
- **Counterfactual Analysis** — Feature-level diffs ("what would need to change?") and graph-level edge perturbation ("which connections drive the anomaly?")
- **SMTP Email Alerts** — Automatic email notifications via Gmail SMTP for high/critical risk findings detected by the LLM reasoner
- **Real-Time WebSocket Streaming** — Every thinking step, tool call, and conclusion is streamed to the frontend as it happens
- **Interactive Investigation Dashboard** — React 19 + Vite 6 frontend with Error Boundary, a 4-step guided wizard (Overview → ES Logs → Network Graph → Counterfactual Explainability), shadcn/ui components, Tailwind v4 dark theme, 12 typed React hooks with mock fallback, and 20 typed API functions
- **Kitsune Dataset Validated** — Tested on 4M+ real SSDP flood attack packets with ground-truth labels

---

## How It Works

### Real-Time Streaming Architecture (Primary)

```
  NDJSON Packets
       │
       ▼
┌──────────────────┐
│  StreamSimulator  │  rate / realtime replay modes
│  (simulation.py)  │  configurable pps + window size
└───────┬──────────┘
        │ window closes -> flows emitted
        ▼
┌──────────────────────────────┐
│  RealTimeIncidentLens        │  (process_pipeline.py)
│                              │
│  1. Build temporal graphs    │  wrappers.build_and_index_graphs()
│  2. Generate embeddings      │  wrappers.generate_embeddings()
│  3. Index to Elasticsearch   │  wrappers.index_embeddings()
│  4. GNN anomaly scoring      │  graph.pred_scores (sigmoid)
│  5. Severity aggregation     │  wrappers.aggregate_severity_breakdown()
│  6. LLM trigger (>= 0.5)    │
│     └─> LLMReasoner          │  (llm_reasoner.py)
│         ├─ get_flow_details  │
│         ├─ get_graph_summary │
│         ├─ get_severity      │
│         ├─ get_recent_windows│
│         ├─ send_email_alert  │  SMTP (Gmail)
│         └─> store insight    │  -> incidentlens-llm-insights
└──────────────┬───────────────┘
               ▼
     ┌───────────────────────┐
     │   Elasticsearch 8.12  │
     │                       │
     │  incidentlens-flows   │  aggregated flow features
     │  ...-embeddings       │  kNN vector search (cosine)
     │  ...-counterfactuals  │  feature diffs + explanations
     │  ...-graph-summaries  │  per-window graph metadata
     │  ...-llm-insights     │  LLM reasoning results
     └───────────────────────┘
```

### Interactive Investigation Dashboard

```
┌──────────────────────────────────────────────────────────────────┐
│  React Frontend (Dashboard + 4-Step Investigation Wizard)        │
│  ┌────────────┐  ┌──────────────┐  ┌──────────┐  ┌───────────┐  │
│  │  Overview   │->│ ES Log View  │->│ D3 Graph │->│Counterfact│  │
│  └────────────┘  └──────────────┘  └──────────┘  └───────────┘  │
└───────────────────────┬──────────────────────────────────────────┘
                        │ REST + WebSocket
                        ▼
              ┌─────────────────────┐
              │  FastAPI Server      │  21 REST + 1 WS endpoint
              │  (server.py)         │
              └──────┬───────────────┘
                     │
              ┌──────▼──────────────┐
              │  Investigation Agent │ <-- OpenAI / Azure / Ollama
              │  (agent.py)          │     19 tools via agent_tools.py
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
            Elasticsearch 8.12
```

**Real-time loop:** Packets flow into `StreamSimulator` → aggregated into time windows → each closed window triggers `RealTimeIncidentLens.process_window()` which builds graphs, generates embeddings, scores with the GNN, and if `anomaly_score >= 0.5`, the `LLMReasoner` autonomously investigates using tool-calling and stores insights to Elasticsearch. For high/critical risk, it sends email alerts via SMTP.

**Interactive loop:** The frontend walks analysts through a 4-step guided wizard. The backend agent detects anomalies → retrieves flow details → runs counterfactual analysis (kNN nearest-normal + feature diff) → assesses severity → finds similar incidents → streams each reasoning step back to the UI in real time via WebSocket.

---

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+ and npm (for the React frontend)
- Docker (for Elasticsearch + Kibana)
- An OpenAI API key (for the LLM reasoner and investigation agent)

### Environment Variables

| Variable | Required | Description |
|:---------|:---------|:------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key (or set `OPENAI_BASE_URL` for Ollama/Azure) |
| `OPENAI_MODEL` | No | Model name (default: `gpt-4o-mini`) |
| `OPENAI_BASE_URL` | No | Custom API base URL for local models |
| `ALERT_EMAIL_SENDER` | No | Gmail address for SMTP alerts |
| `ALERT_EMAIL_PASSWORD` | No | Gmail App Password (not your regular password) |
| `ALERT_EMAIL_RECIPIENT` | No | Email address to receive alerts |
| `INCIDENTLENS_CORS_ORIGINS` | No | Comma-separated allowed origins (default: `http://localhost:5173,http://localhost:3000`) |

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
python -m src.Backend.cli health
```

### 4. Prepare Data

```bash
# Convert CSV -> NDJSON (first time only)
python -m src.Backend.cli convert \
  --packets path/to/ssdp_packets_rich.csv \
  --labels path/to/SSDP_Flood_labels.csv
```

### 5. Train the GNN (Optional but Recommended)

```bash
# From CSV files directly
python -m src.Backend.cli train \
  --packets path/to/ssdp_packets_rich.csv \
  --labels path/to/SSDP_Flood_labels.csv \
  --epochs 30 --hidden-dim 64

# Or from NDJSON directory
python -m src.Backend.cli train --data-dir data/ --epochs 30
```

The trained model is saved to `models/temporal_gnn.pt` and automatically loaded by the real-time engine.

### 6. Run Real-Time Streaming (Primary Mode)

```bash
set OPENAI_API_KEY=sk-...

# Stream packets through the full pipeline
python -m src.Backend.process_pipeline \
  --ndjson data/packets_0000.json \
  --rate 500 \
  --window-size 5.0 \
  --debug

# Or use the CLI simulate command (loads all data from --data-dir)
python -m src.Backend.cli simulate \
  --rate 200 --window-size 5.0 --mode rate
```

Each window is processed: graphs built → flows indexed → embeddings generated → GNN scored → LLM investigates (if score >= 0.5) → insights stored → email alerts sent (if high/critical).

### 7. Batch Ingestion (Alternative)

```bash
# Run full batch ingestion + analysis pipeline
python -m src.Backend.cli ingest
```

### 8. Interactive Investigation

```bash
set OPENAI_API_KEY=sk-...

# Auto-detect and investigate anomalies
python -m src.Backend.cli investigate

# Ask a specific question
python -m src.Backend.cli investigate "why is 192.168.100.5 anomalous?"
```

### 9. Start the API Server

```bash
python -m src.Backend.cli serve --port 8000
```

### 10. Start the Frontend

```bash
cd src/Front
npm install
npm run dev          # -> http://localhost:5173
```

The Vite dev server proxies `/api` and `/ws` requests to the backend at `localhost:8000`, so both servers can run simultaneously in development. In production, build the frontend with `npm run build` and serve the resulting `dist/` directory from any static host (e.g., Nginx, Vercel, or a CDN). The FastAPI server does **not** serve static files — you need a separate static host or a reverse proxy.

> **Note:** `.gitignore` excludes `*.csv` files, so raw datasets must be distributed separately.

---

## Project Structure

```
IncidentLens/
├── src/
│   ├── Backend/
│   │   ├── process_pipeline.py         # Real-time streaming engine (RealTimeIncidentLens + CLI)
│   │   ├── llm_reasoner.py             # Autonomous LLM agent with 5 tools + SMTP email alerts
│   │   ├── simulation.py               # StreamSimulator -- rate/realtime packet windowing
│   │   ├── cli.py                      # Unified CLI (7 commands: health, ingest, investigate, serve, convert, train, simulate)
│   │   ├── main.py                     # CLI shim -> cli.py
│   │   ├── agent.py                    # LLM agent — multi-step reasoning loop
│   │   ├── agent_tools.py              # 19 tools with OpenAI function-calling schemas
│   │   ├── server.py                   # FastAPI server (21 REST + 1 WS endpoint)
│   │   ├── wrappers.py                 # ES client -- 55+ functions: indexing, kNN, ILM, graph summaries, retrieval helpers
│   │   ├── graph_data_wrapper.py       # Vectorised sliding-window graph builder (numpy)
│   │   ├── graph.py                    # Core graph data structures (numpy-vectorised)
│   │   ├── temporal_gnn.py             # EvolveGCN-O + Neural ODE temporal models + training
│   │   ├── gnn_interface.py            # Abstract GNN encoder contract
│   │   ├── backfill.py                 # NDJSON loader + backfill utilities
│   │   ├── csv_to_json.py             # CSV -> NDJSON converter
│   │   ├── __init__.py
│   │   ├── backup/                     # Earlier model versions kept for reference
│   │   └── tests/                      # Test suites
│   │       ├── test_gnn_edge_cases.py
│   │       ├── test_temporal_gnn_full.py
│   │       ├── test_temporal_gnn_meticulous.py
│   │       ├── test_csv_to_json.py
│   │       ├── test_agent_tools.py
│   │       ├── test_e2e_pipeline.py
│   │       ├── run_all.py              # Unified test runner (unittest + pytest)
│   │       └── __init__.py
│   ├── Front/                          # React frontend
│   │   ├── package.json                 # Dependencies + scripts (dev, build, preview)
│   │   ├── vite.config.ts               # Vite 6 + proxy (/api → :8000, /ws → :8000)
│   │   ├── tsconfig.json                # Strict TS config + @/ alias
│   │   ├── index.html                   # HTML entry point
│   │   ├── app/
│   │   │   ├── main.tsx                 # React 19 root mount
│   │   │   ├── App.tsx                  # ErrorBoundary + RouterProvider + Toaster
│   │   │   ├── routes.tsx               # Route definitions
│   │   │   ├── types.ts                 # Shared UI + backend response types
│   │   │   ├── services/api.ts          # Typed fetch client + WebSocket stream
│   │   │   ├── hooks/useApi.ts          # 12 hooks -- live API with mock fallback
│   │   │   ├── components/
│   │   │   │   ├── Dashboard.tsx         # Incident list + stats (live data)
│   │   │   │   ├── Investigation.tsx     # 4-step wizard (live data)
│   │   │   │   ├── investigation/
│   │   │   │   │   ├── ElasticsearchStep.tsx  # ES log analysis
│   │   │   │   │   ├── GNNStep.tsx            # D3 network graph
│   │   │   │   │   └── CounterfactualStep.tsx # Explainability
│   │   │   │   └── ui/                  # shadcn/ui components
│   │   │   └── data/mockData.ts
│   │   └── styles/                      # Tailwind v4 + oklch theme tokens
│   └── docker-compose.yml               # ES 8.12 + Kibana 8.12 + backend + frontend
├── Dockerfile.backend                   # Python 3.12-slim + torch 2.6.0 CPU + healthcheck
├── Dockerfile.frontend                  # Node 20-alpine build -> nginx:alpine serve
├── data/                                # NDJSON data files (packets_0000..0040.json)
├── models/                              # GNN checkpoints (temporal_gnn.pt) -- created by train
├── EDA/
│   └── EDA.ipynb                        # Dataset profiling and visualization
├── requirements.txt
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
| `GET` | `/api/severity-breakdown` | Runtime-field severity distribution across all flows |
| `GET` | `/api/flows/severity` | Query flows filtered by runtime severity level |
| `POST` | `/api/flows/search` | Paginated flow search (search_after + PIT) |
| `GET` | `/api/counterfactuals/search` | Full-text search over counterfactual narratives |
| `GET` | `/api/aggregate/{field}` | Composite aggregation with cursor-based pagination |
| `GET` | `/api/ml/anomalies` | ES ML anomaly detection records |
| `GET` | `/api/ml/influencers` | ES ML top influencer results |

### WebSocket

| Endpoint | Description |
|:---------|:------------|
| `WS /ws/investigate` | Stream investigation events in real time |

**Protocol:** Connect -> send `{"query": "..."}` -> receive streaming events -> receive `{"type": "done"}`.

Event types: `thinking`, `tool_call`, `tool_result`, `conclusion`, `error`, `status`, `done`.

---

## Investigation Agent Tools (19)

The interactive investigation agent (`agent.py`) uses 19 tools exposed via `agent_tools.py` for the dashboard and CLI:

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
| `graph_edge_counterfactual` | Identify which edges (connections) drive the anomaly |
| `graph_window_comparison` | Compare normal vs anomalous time windows structurally |
| `get_ml_anomaly_records` | Fetch records from ES ML anomaly detection jobs |
| `get_ml_influencers` | Get top influencers from ES ML anomaly jobs |
| `severity_breakdown` | Runtime-field severity distribution across all flows |
| `search_counterfactuals` | Full-text search over counterfactual narratives |
| `assess_severity` | Z-score severity assessment (low / medium / high) |

---

## LLM Reasoner Tools (5)

The real-time autonomous reasoner (`llm_reasoner.py`) uses 5 focused tools for per-window investigation:

| Tool | Purpose |
|:-----|:--------|
| `get_flow_details` | Look up a specific flow document from Elasticsearch by flow_id |
| `get_graph_summary` | Retrieve the graph summary for a specific window_id |
| `get_severity_breakdown` | Get severity distribution across all indexed flows |
| `get_recent_windows` | Retrieve recent graph summaries for trend analysis |
| `send_email_alert` | Send an SMTP email alert for high/critical findings (Gmail) |

---

## Elasticsearch Indices

| Index | Contents | Created By |
|:------|:---------|:-----------|
| `incidentlens-flows` | Aggregated flow features -- packet_count, total_bytes, mean_payload, mean_iat, std_iat, label, predictions | `wrappers.py` (both batch + real-time) |
| `incidentlens-embeddings` | Per-flow embedding vectors (dense_vector with cosine similarity, kNN-indexed) | `wrappers.py` |
| `incidentlens-counterfactuals` | Counterfactual diffs -- per-feature original vs CF value, direction, percent change | `wrappers.py` (batch mode) |
| `incidentlens-graph-summaries` | Per-window graph metadata -- node/edge counts, anomaly scores, IP mappings | `wrappers.py` (real-time) |
| `incidentlens-llm-insights` | LLM reasoning results -- risk level, summary, recommendations, tool calls | `llm_reasoner.py` (real-time) |

---

## Technical Highlights

- **Real-time streaming architecture** -- `StreamSimulator` replays packets at configurable rates (pps or wall-clock realtime), aggregates into windows, and triggers the full pipeline per-window: graph construction -> ES indexing -> GNN scoring -> LLM investigation -> email alerting. Async end-to-end via `asyncio`.
- **Autonomous LLM reasoner** -- `AsyncOpenAI` tool-calling loop (max 5 iterations) that investigates each anomalous window, selects tools dynamically, stores structured insights to Elasticsearch, and sends SMTP email alerts for high/critical risk.
- **Vectorized graph construction** -- Pure numpy sliding-window builder with composite key packing, broadcast window assignment, and pre-built neighbor lists. Zero Python for-loops over packets.
- **Dual GNN models** -- EvolveGCN-O (LSTM-evolved GCN weights) for temporal pattern detection; Neural ODE variant (RK4 default) for continuous-time weight evolution. Trainable via `python -m src.Backend.cli train`.
- **Pre-processed GNN bottleneck removal** -- Self-loops and degree normalization cached at data-prep time; LSTM weight evolution flattened from O(hidden_dim) batches to O(1).
- **ES-native analytics** -- Runtime severity fields (Painless scripts), ILM lifecycle policies, ingest pipelines for NaN cleanup, index templates, composite aggregations with cursor pagination, search_after + PIT pagination, and full-text counterfactual narrative search.
- **Singleton ES client** with thread-safe initialization (`threading.Lock`), retry logic, bulk indexing with batch MD5 flow-ID generation, and pre-converted numpy->Python type coercion for minimal per-document overhead. Checkpoint loading uses `weights_only=True` to prevent arbitrary code execution.
- **Docker hardened** -- `.dockerignore` excludes test/dev files, backend Dockerfile has `HEALTHCHECK`, `torch==2.6.0` pinned, frontend uses `npm ci` for deterministic builds, Compose uses `condition: service_healthy` for service ordering.
- **Full-stack integration** -- Typed API service layer (`services/api.ts`) with 20 typed functions (19 REST + 1 WebSocket async-generator), and 12 React hooks that try the live backend first and fall back to mock data for offline development.
- **Zero-config dev proxy** -- Vite dev server on `:5173` proxies `/api` and `/ws` to the FastAPI backend on `:8000`, so frontend and backend can be developed and run simultaneously with no CORS issues.

---

## CLI Reference

All commands can be run via `python -m src.Backend.cli <command>` or `python src/Backend/main.py <command>`.

| Command | Description |
|:--------|:------------|
| `health` | Check ES connectivity and index document counts |
| `ingest` | Run full batch pipeline: graphs -> flows -> embeddings -> counterfactuals |
| `investigate [query]` | Run the interactive LLM agent investigation |
| `serve --port 8000` | Start the REST + WebSocket API server |
| `convert` | Convert raw CSV data to NDJSON |
| `train --packets ... --labels ...` | Train the Temporal GNN and save checkpoint to `models/temporal_gnn.pt` |
| `simulate --rate 200 --window-size 5` | Run real-time packet simulation through the full pipeline |

The real-time engine also has its own standalone CLI:

```bash
python -m src.Backend.process_pipeline --ndjson data/packets_0000.json --rate 500 --window-size 5.0 --debug
```

---

## Team

Built by **Aditya, Anuska, and Aayush** for the Elasticsearch Agent Builder Hackathon.

## License

[MIT](LICENSE)

