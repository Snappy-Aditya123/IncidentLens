# IncidentLens — Backend

Developer reference for the Python backend powering IncidentLens.

---

## Module Map

| Module | Responsibility |
|:-------|:---------------|
| **main.py** | Unified CLI — `health`, `ingest`, `investigate`, `serve`, `convert` |
| **agent.py** | LLM agent orchestrator: multi-step reasoning loop with OpenAI tool-calling |
| **agent_tools.py** | 15 tools exposed to the agent (ES queries, counterfactuals, severity, graph analysis) |
| **server.py** | FastAPI server — REST endpoints + WebSocket streaming for real-time investigation |
| **wrappers.py** | Singleton ES client, index management, bulk flow/embedding ingestion, kNN search, counterfactual diff, ML jobs |
| **graph_data_wrapper.py** | Vectorised sliding-window graph builder — pure numpy, zero Python for-loops over packets |
| **graph.py** | Core graph data structures (`node`, `network`), snapshot dataset builder |
| **train.py** | EdgeGNN (GraphSAGE + Edge MLP) training pipeline with class-imbalance handling |
| **temporal_gnn.py** | EvolveGCN-O — semi-temporal GNN with LSTM-evolved weights for sequence-level detection |
| **gnn_interface.py** | `BaseGNNEncoder` abstract class — the contract any GNN must satisfy to plug into the pipeline |
| **ingest_pipeline.py** | 8-step data pipeline: load NDJSON → build graphs → index flows → embeddings → counterfactuals |
| **csv_to_json.py** | Converts raw CSV datasets to chunked NDJSON for the ingest pipeline |

---

## Data Flow

```
Raw CSV (Kitsune SSDP Flood)
    │
    ▼  csv_to_json.py
NDJSON chunks (data/packets_*.json)
    │
    ▼  ingest_pipeline.load_ndjson_files()
pandas DataFrame
    │
    ▼  graph_data_wrapper.build_sliding_window_graphs()
list[PyG Data]  ─────────────────────────────────────────┐
    │                                                     │
    ├─▶ wrappers.index_graphs_bulk()                      │
    │       → ES: incidentlens-flows                      │
    │                                                     │
    ├─▶ wrappers.generate_embeddings()                    │
    │       → ES: incidentlens-embeddings (kNN-indexed)   │
    │                                                     │
    ├─▶ wrappers.build_and_index_counterfactual()         │
    │       → ES: incidentlens-counterfactuals            │
    │                                                     │
    ├─▶ train.train_edge_gnn()                            │
    │       → EdgeGNN model                               │
    │                                                     │
    └─▶ temporal_gnn.prepare_temporal_dataset()            │
            → Temporal sequences → EvolveGCN-O             
```

---

## Graph Schema

Each PyG `Data` object produced by `graph_data_wrapper` contains:

| Attribute | Shape | Description |
|:----------|:------|:------------|
| `edge_index` | `(2, E)` long | Directed edges: src → dst |
| `x` | `(N, 6)` float | Node features: bytes_sent, bytes_recv, pkts_sent, pkts_recv, out_degree, in_degree |
| `edge_attr` | `(E, 5)` float | Edge features: packet_count, total_bytes, mean_payload, mean_iat, std_iat |
| `y` | `(E,)` long | Ground-truth edge labels: 0 = normal, 1 = malicious |
| `num_nodes` | int | Number of unique IPs in the window |
| `window_start` | float | Timestamp of window start |
| `window_id` | int | Sequential window identifier |
| `network` | object | `graph.network` with node/edge metadata |

---

## Elasticsearch Indices

### incidentlens-flows

Aggregated flow features per edge per time window.

| Field | Type | Description |
|:------|:-----|:------------|
| `flow_id` | keyword | Deterministic MD5 hash ID |
| `window_id` | integer | Time window index |
| `window_start` | float | Window start timestamp |
| `src_ip` / `dst_ip` | ip | Source and destination addresses |
| `packet_count` | float | Number of packets in the flow |
| `total_bytes` | float | Sum of packet sizes |
| `mean_payload` / `mean_iat` / `std_iat` | float | Payload and inter-arrival-time stats |
| `label` | integer | Ground truth (0/1) |
| `prediction` / `prediction_score` | integer/float | GNN model output |

### incidentlens-embeddings

Per-flow embedding vectors for kNN counterfactual retrieval.

| Field | Type | Description |
|:------|:-----|:------------|
| `flow_id` | keyword | Links to flows index |
| `label` | integer | 0 = normal, 1 = malicious |
| `embedding` | dense_vector (cosine, kNN-indexed) | GNN or projection embedding |

### incidentlens-counterfactuals

Feature-level diffs explaining each anomaly.

| Field | Type | Description |
|:------|:-----|:------------|
| `flow_id` | keyword | The anomalous flow |
| `nearest_normal_id` | keyword | Closest normal flow via kNN |
| `similarity_score` | float | Cosine similarity score |
| `feature_diffs` | nested | Per-feature: original_value, cf_value, abs_diff, pct_change, direction |
| `edges_removed` | nested | Graph-level: which edges were removed in the perturbation |

---

## Agent Tool Dispatch

All 15 tools follow the OpenAI function-calling schema format. The LLM agent calls them via `agent_tools.dispatch(tool_name, args)`, which routes to the appropriate `wrappers.*` function.

**Detection:** `es_health_check`, `detect_anomalies`, `search_flows`, `get_flow`, `search_raw_packets`

**Analysis:** `feature_stats`, `feature_percentiles`, `significant_terms`, `find_similar_incidents`

**Explainability:** `counterfactual_analysis`, `counterfactual_narrative`, `explain_flow`, `graph_edge_counterfactual`, `graph_window_comparison`

**Assessment:** `assess_severity`

---

## Server Endpoints

Start with `python src/Backend/main.py serve --port 8000`.

### REST

| Method | Path | Handler |
|:-------|:-----|:--------|
| `GET` | `/health` | Server + ES health |
| `POST` | `/api/investigate` | Full investigation (JSON) |
| `POST` | `/api/detect` | Quick anomaly detection |
| `GET` | `/api/flows` | List flows (optional label/IP filters) |
| `GET` | `/api/stats` | Feature stats by label |
| `POST` | `/api/counterfactual` | Counterfactual for a flow |
| `GET` | `/api/severity/{flow_id}` | Severity assessment |
| `GET` | `/api/similar/{flow_id}` | kNN similar incidents |
| `GET` | `/api/tools` | List agent tools |

### WebSocket

**`WS /ws/investigate`** — Connect and send `{"query": "..."}`. Server streams JSON events:

```json
{"type": "thinking",    "content": "Analyzing flow patterns..."}
{"type": "tool_call",   "tool": "detect_anomalies", "arguments": {...}}
{"type": "tool_result", "tool": "detect_anomalies", "result": "..."}
{"type": "conclusion",  "content": "## Investigation Summary\n..."}
{"type": "done"}
```

---

## Running Tests

```bash
python -m pytest src/Backend/tests/ -v
```

166 tests covering graph construction, GNN forward/backward passes, temporal sequences, normalization, collation, and edge cases.

---

## Environment Variables

| Variable | Default | Description |
|:---------|:--------|:------------|
| `OPENAI_API_KEY` | (none) | API key for the LLM agent |
| `OPENAI_MODEL` | `gpt-4o` | Model to use for the agent |
| `OPENAI_BASE_URL` | (none) | Custom endpoint (e.g., `http://localhost:11434/v1` for Ollama) |
| `PORT` | `8000` | Server port |

---

## Key Design Decisions

1. **Edge-level embeddings** — GNNs produce node embeddings; we concatenate `[node_emb[src], node_emb[dst]]` to get per-flow embeddings for kNN counterfactual retrieval.
2. **Singleton ES client** — Reused across all requests; matches Elasticsearch SDK best practices.
3. **Pre-processed GNN inputs** — Self-loops, degree normalization, and NaN sanitization happen *once* at data-prep time, not in every forward pass.
4. **Numpy-first graph building** — `graph_data_wrapper` uses composite key packing, `np.add.at`, and `searchsorted` for window assignment — no Python loops over packets.

