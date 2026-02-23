# IncidentLens

A multi-step AI agent for network incident investigation, combining Elasticsearch, Graph Neural Networks, and counterfactual explainability.

Built for the **Elasticsearch Agent Builder Hackathon**.

---

## Overview

IncidentLens analyzes network traffic to:

1. **Detect** anomalous flows using labels, model scores, or statistical deviation
2. **Investigate** root causes via feature analysis, significant terms, and counterfactual explanations
3. **Explain** what would need to change for an anomalous flow to be classified as normal
4. **Assess severity** (low/medium/high) based on z-score deviation from normal baselines
5. **Find similar historical incidents** via kNN embedding search
6. **Recommend actions** (block IP, rate-limit, investigate further)

### Stack

- **Elasticsearch 8.12** — scalable indexing, kNN vector search, aggregations, significant terms
- **Kitsune SSDP Flood Dataset** — 4M+ real network packets with ground-truth attack labels
- **PyTorch Geometric** — sliding-window temporal graph construction
- **OpenAI-compatible LLM** — multi-step reasoning agent with tool calling
- **FastAPI + WebSocket** — real-time streaming of investigation events

---

## Architecture

```text
User query
  |
  v
[LLM Agent] <-- tool-calling loop (OpenAI API)
  |  |  |
  v  v  v
[13 Agent Tools]  -- wrappers.py --> [Elasticsearch :9200]
                                         |
                                    [3 indices]
                                    - incidentlens-flows
                                    - incidentlens-embeddings
                                    - incidentlens-counterfactuals
```

The agent iteratively calls tools, reasons about results, and produces a structured investigation summary with severity, root cause, evidence, and recommendations.

---

## Quick Start

### 1. Start Elasticsearch + Kibana

```bash
docker compose up -d
```

Services:

- Elasticsearch: <http://localhost:9200>
- Kibana: <http://localhost:5601>

### 2. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 3. Verify connectivity

```bash
python main.py health
```

### 4. Convert CSV data to NDJSON (first time only)

```bash
python main.py convert --packets path/to/ssdp_packets_rich.csv --labels path/to/SSDP_Flood_labels.csv
```

### 5. Run ingestion + analysis pipeline

```bash
python main.py ingest                    # full pipeline
python main.py ingest --max-rows 10000   # quick test
```

This builds temporal graphs, indexes flows + embeddings, and runs counterfactual analysis.

### 6. Run AI agent investigation

```bash
export OPENAI_API_KEY=sk-...                          # or use Ollama:
# export OPENAI_BASE_URL=http://localhost:11434/v1
# export OPENAI_API_KEY=ollama

python main.py investigate                            # auto-detect anomalies
python main.py investigate "why is 192.168.100.5 anomalous?"
```

### 7. Start API server (optional, for frontend)

```bash
python main.py serve --port 8000
```

---

## CLI Reference

| Command | Description |
| ------- | ----------- |
| `python main.py health` | Check ES connectivity and index doc counts |
| `python main.py ingest` | Run full data pipeline (graphs, flows, embeddings, counterfactuals) |
| `python main.py investigate [query]` | Run LLM agent investigation |
| `python main.py serve` | Start REST + WebSocket API server |
| `python main.py convert` | Convert raw CSV to NDJSON |

---

## Agent Tools (13 total)

| Tool | Description |
| ---- | ----------- |
| `es_health_check` | Cluster health and connectivity |
| `search_flows` | Query flows index with filters (label, IP, packet count) |
| `get_flow` | Retrieve a single flow by ID |
| `detect_anomalies` | Detect anomalies via label, model score, or statistical method |
| `feature_stats` | Extended statistics per feature grouped by label |
| `feature_percentiles` | Percentile distribution for a feature |
| `significant_terms` | IPs/protocols overrepresented in attack traffic |
| `counterfactual_analysis` | Find nearest normal flow + compute feature diffs |
| `counterfactual_narrative` | Human-readable counterfactual explanation |
| `explain_flow` | ES _explain API — why a flow matched a query |
| `search_raw_packets` | Search individual packet records |
| `find_similar_incidents` | kNN embedding search for historical matches |
| `assess_severity` | Z-score severity assessment (low/medium/high) |

---

## API Endpoints (when running `main.py serve`)

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| GET | `/health` | Server + ES health |
| POST | `/api/investigate` | Run full investigation (JSON response) |
| WS | `/ws/investigate` | Stream investigation events in real time |
| POST | `/api/detect` | Quick anomaly detection (no LLM) |
| GET | `/api/flows` | List flows with filters |
| GET | `/api/stats` | Feature statistics by label |
| POST | `/api/counterfactual` | Run counterfactual for a flow |
| GET | `/api/severity/{flow_id}` | Assess severity |
| GET | `/api/similar/{flow_id}` | Find similar incidents |
| GET | `/api/tools` | List available agent tools |

---

## Project Structure

```text
IncidentLens/
  main.py                # Unified CLI entry point
  wrappers.py            # ES client + all index operations + graph building
  agent_tools.py         # 13 tools with OpenAI function-calling schemas
  agent.py               # LLM agent orchestrator (multi-step reasoning)
  server.py              # FastAPI server (REST + WebSocket)
  graph_data_wrapper.py  # Vectorised sliding-window graph builder
  graph.py               # Core graph data structures
  ingest_pipeline.py     # 8-step data ingestion + analysis pipeline
  csv_to_json.py         # CSV to NDJSON converter
  docker-compose.yml     # ES 8.12 + Kibana 8.12
  requirements.txt       # Python dependencies
  data/                  # NDJSON files (generated by csv_to_json.py)
  EDA/                   # Exploratory data analysis notebooks
```

---

## ES Indices

| Index | Contents |
| ----- | -------- |
| `incidentlens-flows` | Aggregated flow features (packet_count, total_bytes, mean_payload, mean_iat, std_iat, label) |
| `incidentlens-embeddings` | Flow embedding vectors (dense_vector, cosine similarity, kNN-indexed) |
| `incidentlens-counterfactuals` | Counterfactual diffs (per-feature original vs CF value, direction, pct_change) |
| `incidentlens-packets` | Raw individual packet records |

---

## License

See [LICENSE](LICENSE).

