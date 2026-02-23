# IncidentLens — Comprehensive Documentation Audit

**Scope:** Every gap between the actual codebase and the three documentation files: `README.md`, `src/Backend/Backend.md`, `src/Front/Frontend.md`.

---

## Table of Contents

1. [Critical Inaccuracies](#1-critical-inaccuracies)
2. [Undocumented Backend Items](#2-undocumented-backend-items)
3. [Undocumented Frontend Items](#3-undocumented-frontend-items)
4. [Elasticsearch Index Schema Gaps](#4-elasticsearch-index-schema-gaps)
5. [Environment Variables](#5-environment-variables)
6. [Event / WebSocket Protocol Gaps](#6-event--websocket-protocol-gaps)
7. [Project Structure Discrepancies](#7-project-structure-discrepancies)
8. [Type Mismatches & Duplicate Definitions](#8-type-mismatches--duplicate-definitions)
9. [Dependency Gaps](#9-dependency-gaps)
10. [CORS / Server Configuration](#10-cors--server-configuration)
11. [Minor / Cosmetic Gaps](#11-minor--cosmetic-gaps)
12. [Items That Are Correct](#12-items-that-are-correct)

---

## 1. Critical Inaccuracies

### 1.1 `main.py` import path is wrong in documentation

| Detail | Value |
|:-------|:------|
| **Affected docs** | README.md, Backend.md |
| **What docs say** | `main.py` "delegates to `testingentry.py`" |
| **What code does** | `main.py` (line 49) imports `from src.Backend.tests.testingentry import main` |
| **Impact** | The actual CLI entry point is the **tests/** copy of `testingentry.py`, not the Backend root one. |

### 1.2 `testingentry.py` exists in two locations (undocumented)

Both `src/Backend/testingentry.py` (220 lines) and `src/Backend/tests/testingentry.py` (220 lines) exist. They appear to be identical copies. Only the Backend-root copy is mentioned in the documentation. `main.py` uses the **tests/** copy.

### 1.3 Hook count is wrong — 8, not 7

| Detail | Value |
|:-------|:------|
| **Affected docs** | README.md (line ~311), Frontend.md (file structure, hooks section header) |
| **What docs say** | "7 hooks" |
| **Actual hooks (8)** | `useBackendHealth`, `useIncidents`, `useIncident`, `useElasticsearchData`, `useNetworkGraph`, `useCounterfactual`, `useSeverity`, `useInvestigationStream` |

### 1.4 `BackendCounterfactualResponse` mismatch between types.ts and Frontend.md

| Field | `types.ts` (code) | Frontend.md |
|:------|:-------------------|:------------|
| `anomalous_flow` | `Record<string, unknown>` | **missing** |
| `nearest_normal` | `Record<string, unknown>` | **missing** |
| `diffs[].anomalous_value` | ✓ | ✓ |
| `diffs[].normal_value` | ✓ | ✓ |
| `diffs[].pct_change` | **missing** | **missing** |

Frontend.md omits the `anomalous_flow` and `nearest_normal` top-level fields that actually exist in types.ts.

---

## 2. Undocumented Backend Items

### 2.1 Undocumented functions in `wrappers.py`

`wrappers.py` is 1 484 lines and contains 44 functions. Backend.md describes it as one row in the module map. The following public functions have **no individual documentation** in any of the three markdown files:

| Function | Line | Purpose |
|:---------|:-----|:--------|
| `set_gnn_encoder()` | 61 | Register a GNN model into the global registry |
| `get_gnn_encoder()` | 73 | Retrieve the registered GNN model |
| `ping()` | 114 | Simple ES connectivity check |
| `create_index()` | 224 | Generic index creation with optional delete |
| `setup_all_indices()` | 244 | Create all three indices at once |
| `delete_all_indices()` | 262 | Delete all three indices |
| `_flow_id()` / `_flow_ids_batch()` | 275/282 | Deterministic MD5 flow ID generation |
| `index_pyg_graph()` | 301 | Index a single PyG Data object |
| `build_and_index_graphs()` | 577 | Convenience: build graphs from CSV + index into ES |
| `build_index_and_embed()` | 704 | Convenience: build + index + embed in one call |
| `index_embeddings()` | 739 | Bulk-index embedding vectors |
| `knn_search_nearest_normal()` | 780 | kNN search restricted to label=0 |
| `knn_search()` | 815 | Generic kNN embedding search |
| `search_anomalous_flows()` | 1291 | Search flows with label=1 |
| `get_counterfactuals_for_flow()` | 1312 | Retrieve all CFs for a flow |
| `format_counterfactual_narrative()` | 1326 | Generate human-readable CF explanation |
| `_self_test()` | 1356 | Self-test / smoke-test function |

### 2.2 ML anomaly detection job functions (completely undocumented)

`wrappers.py` includes four Elasticsearch ML anomaly-detection job management functions that are not mentioned in **any** documentation:

| Function | Line | Purpose |
|:---------|:-----|:--------|
| `create_anomaly_detection_job()` | 1005 | Create an ES ML anomaly detection job |
| `create_anomaly_datafeed()` | 1060 | Create an ES ML datafeed |
| `get_anomaly_records()` | 1083 | Retrieve anomaly records |
| `get_influencers()` | 1107 | Retrieve ML influencers |

### 2.3 Undocumented constants in `wrappers.py`

| Constant | Value | Notes |
|:---------|:------|:------|
| `FLOWS_INDEX` | `"incidentlens-flows"` | Named but schema details incomplete — see §4 |
| `COUNTERFACTUALS_INDEX` | `"incidentlens-counterfactuals"` | Same |
| `EMBEDDINGS_INDEX` | `"incidentlens-embeddings"` | Same |
| `EMBEDDINGS_MAPPING_DIM` | `16` | Default embedding dimensionality — not in docs |
| `FEATURE_FIELDS` | *(list of metric field names)* | Used throughout, not documented |

### 2.4 `AgentConfig` fields not documented

Backend.md's env-var table documents `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`, `PORT`. The `AgentConfig` dataclass also contains:

| Field | Default | Documented? |
|:------|:--------|:------------|
| `max_steps` | `15` | No |
| `temperature` | `0.1` | No |
| `max_tokens` | `4096` | No |

### 2.5 Pydantic request models in `server.py` not documented

Three Pydantic models are defined in `server.py` but not listed in Backend.md:

- `InvestigateRequest` — `query: str = ""`
- `DetectRequest` — `method: str = "label"`, `threshold: float = 0.5`, `size: int = 50`
- `CounterfactualRequest` — `flow_id: str`

These define the **request body schemas** for `POST /api/investigate`, `POST /api/detect`, and `POST /api/counterfactual`.

### 2.6 `backup/` directory not documented

`src/Backend/backup/` contains:
- `temporal_gnn_v1_backup.py` — earlier version of the temporal GNN
- `__init__.py`

Not mentioned in any of the three markdown files or the README project structure.

### 2.7 `GNN.py` deprecation status

Backend.md correctly marks `GNN.py` as deprecated ✓, but neither README nor Backend.md explain **why** it's kept or whether it should be deleted.

### 2.8 `server.py` has an `if __name__ == "__main__"` block

Lines 430–441 allow running the server directly via `python server.py` (uses `server:app` import path). Not documented — only `python main.py serve` is documented.

### 2.9 `graph_data_wrapper.py` helper functions undocumented

The following internal helpers are exported/defined but not mentioned:

- `_assign_window_ids()`
- `_aggregate_flows_numpy()`
- `_compute_node_features_arrays()`
- `_build_network_fast()`

### 2.10 `graph.py` classes and functions not individually documented

Backend.md describes `graph.py` as "Core graph data structures (`node`, `network`), snapshot dataset builder." The following are not individually documented:

| Item | Type |
|:-----|:-----|
| `node` | class |
| `network` | class (with `out_degree`, `in_degree`, `add_node`, `add_edge`, `set_node_features`, `build_edge_index`, `build_sparse_adjacency`, `from_edge_list`, `to_pyg_data`) |
| `add_window_id()` | function |
| `build_node_map()` | function |
| `build_flow_table()` | function |
| `build_window_data()` | function |
| `build_snapshot_dataset()` | function |
| `build_sample_graph()` | function |

### 2.11 `train.py` functions not individually documented

| Function | Purpose |
|:---------|:--------|
| `tensor_make_finite_()` | In-place NaN/Inf replacement |
| `sanitize_graphs_inplace()` | Clean all graphs before training |
| `time_split()` | Train/val split by time |
| `normalize_edge_features()` | Z-score normalization |
| `label_stats()` | Print class distribution |
| `compute_pos_weight()` | Class imbalance weight |
| `safe_roc_auc()` | ROC-AUC with error handling |
| `find_best_threshold()` | Threshold optimization |

### 2.12 `temporal_gnn.py` functions not individually documented

| Function | Purpose |
|:---------|:--------|
| `preprocess_graph()` / `preprocess_graphs()` | Add self-loops, normalize |
| `sanitize_graph()` | NaN/Inf cleanup |
| `normalize_features_global()` / `apply_normalization()` | Global feature normalization |
| `recompute_node_features()` | Recalculate node features from edge data |
| `build_temporal_sequences()` | Create sliding windows of graph snapshots |
| `_postprocess_graphs()` | Post-processing after dataset load |
| `prepare_temporal_dataset()` / `prepare_temporal_dataset_from_csv()` | End-to-end dataset prep |
| `collate_temporal_batch()` | Custom PyG batch collation |

### 2.13 `gnn_interface.py` details not documented

Backend.md calls it "BaseGNNEncoder abstract class — the contract any GNN must satisfy." Missing:

| Item | Purpose |
|:-----|:--------|
| `BaseGNNEncoder.forward()` | Abstract forward pass |
| `BaseGNNEncoder.predict()` | Predict labels |
| `BaseGNNEncoder.encode()` | Generate embeddings |
| `BaseGNNEncoder.predict_labels()` | Predict with threshold |
| `BaseGNNEncoder.save()` / `load()` | Serialization |
| `create_dataloaders()` | Function: train/val/test DataLoader creation |
| `compute_class_weights()` | Function: class weight computation |

### 2.14 `ingest_pipeline.py` internals not individually documented

Backend.md calls it "8-step data pipeline." The 8 steps and key items:

| Item | Documented? |
|:-----|:------------|
| `RAW_PACKETS_INDEX = "incidentlens-packets"` | README mentions index name, Backend.md does not |
| `RAW_PACKETS_MAPPING` (mapping definition) | No |
| `setup_raw_packet_index()` | No |
| `index_raw_packets()` | No |
| `build_graphs_from_df()` | No |
| `index_all_graphs()` | No |
| `generate_feature_embeddings()` | No |
| `run_counterfactual_analysis()` | No |
| `print_counterfactual_report()` | No |
| `run_feature_analysis()` | No |
| `run_pipeline()` (the orchestrator) | Mentioned implicitly |
| `DATA_DIR` constant | No |

---

## 3. Undocumented Frontend Items

### 3.1 Helper functions exported from `useApi.ts`

Three non-hook functions are exported from `hooks/useApi.ts` but not mentioned in Frontend.md:

| Function | Purpose |
|:---------|:--------|
| `flowToIncident(flow)` | Convert a `BackendFlow` to an `Incident` |
| `flowsToGraph(flows)` | Convert `BackendFlow[]` to `NetworkGraphData` |
| `backendCfToFrontend(cf)` | Convert `BackendCounterfactualResponse` to `CounterfactualExplanation` |

### 3.2 Types in `types.ts` not shown in Frontend.md

Frontend.md's "Data Types" section omits the following interfaces that exist in `types.ts`:

| Interface | Line |
|:----------|:-----|
| `ElasticsearchData` | 40 |
| `NetworkGraphData` | 51 |
| `CounterfactualChange` | 56 |

Frontend.md inlines the `changes` array shape in `CounterfactualExplanation` instead of showing the separate `CounterfactualChange` interface.

### 3.3 `BackendHealthResponse` optional fields omitted from docs

`types.ts` defines:
```typescript
export interface BackendHealthResponse {
  server: string;
  elasticsearch: string;
  indices?: Record<string, boolean>;  // ← NOT in Frontend.md
  error?: string;                     // ← NOT in Frontend.md
}
```

Frontend.md shows only `server` and `elasticsearch`.

### 3.4 `BackendDetectResponse.threshold` field omitted

`types.ts` has `threshold?: number` on `BackendDetectResponse`. Frontend.md omits it.

### 3.5 `BackendFlow` index signature omitted

`types.ts` has `[key: string]: unknown` on `BackendFlow` for forward compatibility. Frontend.md omits this.

### 3.6 `CounterfactualStep.tsx` has hardcoded insights

The "AI-Generated Insights" and "Recommended Actions" sections in `CounterfactualStep.tsx` are **completely static text**, not driven by backend data. No documentation mentions this. An analyst might assume these are generated per-incident, but they are not.

### 3.7 `sonner` toast library not in Stack table

`App.tsx` renders `<Toaster />` from `sonner`. This dependency is not listed in Frontend.md's Stack table.

### 3.8 `useAsync<T>` generic hook not documented

`useApi.ts` defines a reusable `useAsync<T>` hook (the foundation for all 8 hooks). Frontend.md mentions the 8 exported hooks but not this internal building block.

---

## 4. Elasticsearch Index Schema Gaps

### 4.1 `incidentlens-flows` — missing fields in docs

| Field | Type in code | In Backend.md? | In README? |
|:------|:-------------|:---------------|:-----------|
| `protocol` | keyword | **No** | **No** |
| `timestamp` | date (epoch_millis) | **No** | **No** |

Backend.md lists 11 fields. The actual mapping has 13.

### 4.2 `incidentlens-counterfactuals` — missing fields in docs

| Field | Type in code | In Backend.md? |
|:------|:-------------|:---------------|
| `cf_id` | keyword | **No** |
| `prediction` | keyword | **No** |
| `cf_prediction` | keyword | **No** |
| `explanation_text` | text | **No** |
| `timestamp` | date (epoch_millis) | **No** |

Backend.md lists 5 logical groups. The actual mapping has 10+ fields.

### 4.3 `incidentlens-embeddings` — missing fields in docs

| Field | Type in code | In Backend.md? |
|:------|:-------------|:---------------|
| `prediction` | integer | **No** |
| `window_id` | integer | **No** |
| `src_ip` | ip | **No** |
| `dst_ip` | ip | **No** |

Backend.md lists 3 fields. The actual mapping has 7.

### 4.4 `incidentlens-packets` raw index — mapping not fully documented

`ingest_pipeline.py` defines `RAW_PACKETS_INDEX = "incidentlens-packets"` with `RAW_PACKETS_MAPPING`. README mentions this index in the ES indices table, but **neither README nor Backend.md document the actual mapping fields** for this index.

---

## 5. Environment Variables

### 5.1 Missing from Backend.md env-var table

| Variable | Source File | Default | Purpose |
|:---------|:-----------|:--------|:--------|
| `INCIDENTLENS_DATA_ROOT` | `csv_to_json.py` | `.` | Root directory for data output |
| `INCIDENTLENS_PACKETS_CSV` | `csv_to_json.py` | *(none)* | Path to `ssdp_packets_rich.csv` |
| `INCIDENTLENS_LABELS_CSV` | `csv_to_json.py` | *(none)* | Path to `SSDP_Flood_labels.csv` |

### 5.2 `ES_URL` or Elasticsearch connection env var

The `get_client()` function in `wrappers.py` accepts a `url` parameter but there is no documented env var for overriding the default `http://localhost:9200`. The code hardcodes it. If there is an env var (e.g., `ES_URL` or `ELASTICSEARCH_URL`) it's undocumented; if there isn't one, that limitation is also undocumented.

---

## 6. Event / WebSocket Protocol Gaps

### 6.1 `"status"` event type is missing from all docs

`agent.py` defines and emits a `status_event()` on line 117:

```python
def status_event(message: str) -> dict:
    return _event("status", content=message)
```

This event type is **not listed in any of the three docs**:
- README says: `thinking`, `tool_call`, `tool_result`, `conclusion`, `error`, `done`
- Backend.md says the same 6
- Frontend.md `InvestigationEventType` type literal says the same 6
- **`types.ts`** also lists only 6 (missing `"status"`)

The frontend will **silently ignore** status events since the TypeScript type doesn't include it.

### 6.2 `timestamp` field in all events is undocumented

Every event produced by `agent.py` includes a `timestamp` field (Unix epoch float) via the `_event()` helper. None of the three markdown files mention this field. The `InvestigationEvent` interface in `types.ts` does not declare a `timestamp` property.

---

## 7. Project Structure Discrepancies

### 7.1 README project structure

| Issue | Detail |
|:------|:-------|
| `tests/` file count | README shows 4 items; actual directory has 6 (`+ testingentry.py`, `+ __init__.py`) |
| `__init__.py` files | 3 exist (`Backend/`, `Backend/backup/`, `Front/`) — none shown in README tree |
| `backup/` directory | `src/Backend/backup/` exists (2 files) — not shown |
| `.gitignore` | Exists at project root — not shown |
| `vite-env.d.ts` | Exists in `src/Front/` — not shown |
| `Front/__init__.py` | Exists — not shown |
| `LICENSE` | Exists at root — not shown in tree (though linked at bottom) |

### 7.2 Frontend.md file structure

Frontend.md's file tree is more complete than README's and includes `__init__.py` and `vite-env.d.ts`. However:

| Issue | Detail |
|:------|:-------|
| Hook count label | Tree comment says "7 hooks" — should be 8 |

### 7.3 Backend.md has no project structure section

Backend.md has no file tree. It relies on the module-map table, which is correct for the file list but omits `backup/`, `__init__.py`, and the `tests/testingentry.py` duplicate.

---

## 8. Type Mismatches & Duplicate Definitions

### 8.1 `mockData.ts` redeclares interfaces from `types.ts`

`data/mockData.ts` re-declares four interfaces that already exist in `types.ts`:
- `Incident`
- `NetworkNode`
- `NetworkEdge`
- `CounterfactualExplanation`

**Two step components import from `mockData.ts` instead of `types.ts`:**
- `GNNStep.tsx` line 8: `import { NetworkNode, NetworkEdge } from '../../data/mockData'`
- `CounterfactualStep.tsx` line 7: `import { CounterfactualExplanation } from '../../data/mockData'`

This means type definitions can **silently diverge** between components. Neither documentation file mentions this duplication or the incorrect import sources.

### 8.2 `BackendCounterfactualResponse` type vs actual server response

The `types.ts` interface expects:
```typescript
{
  flow_id: string;
  anomalous_flow: Record<string, unknown>;
  nearest_normal: Record<string, unknown>;
  diffs: Array<{ feature; anomalous_value; normal_value; abs_diff; direction }>;
}
```

The actual `wrappers.py` `compute_counterfactual_diff()` returns data matching the ES `COUNTERFACTUALS_MAPPING`, which uses:
- `feature_diffs` (not `diffs`)
- `original_value` / `cf_value` (not `anomalous_value` / `normal_value`)

The `agent_tools.dispatch("counterfactual_analysis", ...)` wrapper may reshape this, but the potential field-name mismatch is not documented.

---

## 9. Dependency Gaps

### 9.1 `package.json` dependencies not in Frontend.md Stack table

The following `package.json` dependencies are installed but not mentioned in Frontend.md's Stack section:

| Package | Category |
|:--------|:---------|
| `recharts` | Charting library |
| `embla-carousel-react` | Carousel component |
| `react-hook-form` | Form management |
| `react-resizable-panels` | Resizable panel layout |
| `vaul` | Drawer component |
| `cmdk` | Command palette |
| `input-otp` | OTP input component |
| `react-day-picker` | Date picker |
| `sonner` | Toast notifications |

Most are likely transitive shadcn/ui dependencies, but they're direct `dependencies` in `package.json`.

### 9.2 `requirements.txt` vs. documented dependencies

README badges mention Python 3.12+, Elasticsearch 8.12, PyTorch Geometric 2.6, FastAPI 0.115. The `requirements.txt` is the canonical source and is not reproduced in any MD.

---

## 10. CORS / Server Configuration

### 10.1 CORS middleware not documented

`server.py` lines 55–59:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # lock down in production
    allow_methods=["*"],
    allow_headers=["*"],
)
```

No documentation mentions this configuration. README only says "Zero-config dev proxy … no CORS issues" — which is true for dev, but in production the wide-open `allow_origins=["*"]` is a security concern. Backend.md should document this.

### 10.2 No static file serving documented or implemented

README says "In production, build the frontend with `npm run build` and serve the `dist/` output alongside the API." Frontend.md says "Serve it from the FastAPI server or any static host." But the server code has **zero** static file serving — no `StaticFiles` mount, no fallback to `index.html`. This is misleading.

### 10.3 FastAPI app metadata not documented

```python
app = FastAPI(
    title="IncidentLens API",
    description="AI-powered network incident investigation agent",
    version="0.1.0",
)
```

The `version="0.1.0"` is not mentioned in any doc.

---

## 11. Minor / Cosmetic Gaps

| # | Issue | Affected Doc |
|:--|:------|:-------------|
| 11.1 | README says "10+ typed fetch functions" — actual count is 9 exports + 1 private `fetchJson` | README.md |
| 11.2 | `server.py` module docstring lists only 7 of 15 endpoints (internal code doc) | server.py (not an MD issue, but worth noting) |
| 11.3 | `EDA/` directory mentioned in README tree but no description beyond "Exploratory analysis notebooks" | README.md |
| 11.4 | `.gitignore` ignores `*.csv` — meaning raw data files can't be committed; this workflow caveat is undocumented | README.md |
| 11.5 | The `_GRAPH_CACHE` and `_STATS_CACHE` (with 30s TTL) in `agent_tools.py` — caching behavior is undocumented | Backend.md |
| 11.6 | `agent_tools.py` exports `set_graph_cache()` — undocumented | Backend.md |
| 11.7 | `agent_tools.py` exports `_sanitize_for_json()` — undocumented | Backend.md |

---

## 12. Items That Are Correct

For completeness, the following claims in the documentation **are verified accurate**:

| Claim | Status |
|:------|:-------|
| 15 agent tools | ✅ Matches `agent_tools.py` — 15 tools registered |
| Tool names and purposes (all 15) | ✅ Match `agent_tools.py` |
| 14 REST endpoints + 1 WebSocket | ✅ `server.py` has exactly 14 `@app.*` REST handlers + 1 `@app.websocket` |
| REST endpoint paths and methods | ✅ All 14 REST paths match |
| 5 CLI commands (health, ingest, investigate, serve, convert) | ✅ Match `testingentry.py` |
| `ingest` pipeline CLI flags | ✅ All flags listed in `build_parser()` |
| `_flow_to_incident()` helper documented in Backend.md | ✅ Matches `server.py:227` |
| EdgeGNN model architecture (GraphSAGE + Edge MLP) | ✅ `train.py:EdgeGNN` |
| EvolveGCN-O with LSTM weight evolution | ✅ `temporal_gnn.py:EvolvingGNN` |
| Dual GNN architecture | ✅ Both models exist |
| Docker Compose: ES 8.12.0 + Kibana 8.12.0 | ✅ `docker-compose.yml` |
| Vite proxy: `/api`, `/ws`, `/health` | ✅ `vite.config.ts` has all 3 proxy rules |
| Route table: `/`, `/investigation/:incidentId`, `*` | ✅ `routes.tsx` |
| Frontend Stack table (except sonner omission) | ✅ |
| Mock fallback pattern description | ✅ All hooks use try/catch with mock fallback |
| `cn()` utility in `ui/utils.ts` | ✅ |
| Agent singleton pattern in server | ✅ `_get_agent()` |
| `investigate_auto()` method on IncidentAgent | ✅ |

---

## Summary Statistics

| Category | Count |
|:---------|:------|
| Critical inaccuracies | 4 |
| Undocumented backend functions | 40+ |
| Undocumented frontend items | 8 |
| Missing ES index fields | 14 fields across 3 indices |
| Missing environment variables | 3 |
| Event/protocol gaps | 2 |
| Project structure errors | 7 |
| Type mismatches / duplicates | 2 |
| Undocumented dependencies | 9 npm packages |
| CORS / server config gaps | 3 |
| Minor issues | 7 |
| **Total gaps identified** | **~90** |

---

*Generated by auditing every source file against README.md, Backend.md, and Frontend.md.*
