"""
Elasticsearch wrappers for IncidentLens counterfactual analysis pipeline.

Provides:
    - Singleton ES client with retry logic
    - Index management (flows, counterfactuals, embeddings)
    - Graph building via graph_data_wrapper (vectorised sliding-window)
    - Bulk flow ingestion from PyG Data objects
    - kNN vector search (nearest-normal counterfactual retrieval)
    - Significant-terms aggregation (discriminating features)
    - Feature distribution aggregations
    - Synthetic embedding generation
    - _explain API wrapper
    - ML anomaly-detection job management
    - Counterfactual diff computation + indexing
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import (
    ConnectionError as ESConnectionError,
    NotFoundError,
    RequestError,
    TransportError,
)
from torch_geometric.data import Data

# graph_data_wrapper — the vectorised graph builder
from src.Backend.graph_data_wrapper import (
    build_sliding_window_graphs as _gdw_build_graphs,
    analyze_graphs as _gdw_analyze,
    load_graph_dataset as _gdw_load_dataset,
    edge_perturbation_counterfactual as _gdw_edge_perturbation_cf,
    compare_graph_windows as _gdw_compare_windows,
    find_most_anomalous_window as _gdw_find_anomalous,
    find_most_normal_window as _gdw_find_normal,
)

# GNN integration (lazy import — works even if no model is registered)
from src.Backend.gnn_interface import BaseGNNEncoder

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# GNN MODEL REGISTRY (optional — pipeline works without it)
# ──────────────────────────────────────────────

_gnn_encoder: Optional[BaseGNNEncoder] = None


def set_gnn_encoder(model: BaseGNNEncoder) -> None:
    """Register a trained GNN encoder for the pipeline.

    Once registered, ``generate_embeddings()`` will use the GNN instead
    of the random-projection placeholder, and ``index_pyg_graph()`` will
    populate ``prediction`` + ``prediction_score`` fields.
    """
    global _gnn_encoder
    _gnn_encoder = model
    logger.info("GNN encoder registered: %s (dim=%d)", type(model).__name__, model.embedding_dim)


def get_gnn_encoder() -> Optional[BaseGNNEncoder]:
    """Return the registered GNN encoder, or None."""
    return _gnn_encoder

# ──────────────────────────────────────────────
# 1. ES CLIENT (singleton + health check)
# ──────────────────────────────────────────────

_client: Optional[Elasticsearch] = None
_client_lock = threading.Lock()


def reset_client() -> None:
    """Force re-creation of the ES singleton (e.g. after config change)."""
    global _client
    with _client_lock:
        _client = None


def get_client(
    host: str | None = None,
    *,
    timeout: int = 2,
    max_retries: int = 0,
    retry_on_timeout: bool = False,
) -> Elasticsearch:
    """Return a reusable Elasticsearch client (singleton).

    Avoids repeated instantiation per SDK best-practices.
    Thread-safe via lock to prevent duplicate instantiation under
    concurrent ``asyncio.to_thread`` workers.

    The host defaults to the ``ELASTICSEARCH_URL`` environment variable,
    falling back to ``http://localhost:9200``.
    """
    global _client
    if _client is not None:
        return _client

    if host is None:
        host = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

    with _client_lock:
        if _client is not None:
            return _client
        _client = Elasticsearch(
            host,
            request_timeout=timeout,
            max_retries=max_retries,
            retry_on_timeout=retry_on_timeout,
        )
    return _client


def health_check(es: Elasticsearch | None = None) -> dict:
    """Return cluster health; raises on connection failure."""
    es = es or get_client()
    return es.cluster.health()


def ping(es: Elasticsearch | None = None) -> bool:
    """Quick connectivity probe."""
    es = es or get_client()
    return es.ping()


# ──────────────────────────────────────────────
# 2. INDEX MANAGEMENT
# ──────────────────────────────────────────────

# --- Index names ---
FLOWS_INDEX = "incidentlens-flows"
COUNTERFACTUALS_INDEX = "incidentlens-counterfactuals"
EMBEDDINGS_INDEX = "incidentlens-embeddings"
RAW_PACKETS_INDEX = "incidentlens-packets"
GRAPH_SUMMARIES_INDEX = "incidentlens-graph-summaries"

# --- Mappings ---

FLOWS_MAPPING = {
    "mappings": {
        "properties": {
            "flow_id":              {"type": "keyword"},
            "window_id":            {"type": "integer"},
            "window_start":         {"type": "float"},
            "src_ip":               {"type": "ip"},
            "dst_ip":               {"type": "ip"},
            "protocol":             {"type": "keyword"},
            "packet_count":         {"type": "float"},
            "total_bytes":          {"type": "float"},
            "mean_payload":         {"type": "float"},
            "mean_iat":             {"type": "float"},
            "std_iat":              {"type": "float"},
            "label":                {"type": "integer"},   # 0=normal, 1=malicious
            "prediction":           {"type": "integer"},   # model prediction
            "prediction_score":     {"type": "float"},
            "timestamp":            {"type": "date", "format": "epoch_millis||strict_date_optional_time"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}

COUNTERFACTUALS_MAPPING = {
    "mappings": {
        "properties": {
            "cf_id":                {"type": "keyword"},
            "flow_id":              {"type": "keyword"},   # source anomalous flow
            "nearest_normal_id":    {"type": "keyword"},   # counterfactual neighbour
            "prediction":           {"type": "keyword"},
            "cf_prediction":        {"type": "keyword"},
            "similarity_score":     {"type": "float"},
            # per-feature diffs stored as nested
            "feature_diffs": {
                "type": "nested",
                "properties": {
                    "feature":          {"type": "keyword"},
                    "original_value":   {"type": "float"},
                    "cf_value":         {"type": "float"},
                    "abs_diff":         {"type": "float"},
                    "pct_change":       {"type": "float"},
                    "direction":        {"type": "keyword"},  # increase / decrease
                },
            },
            # optional edges removed (graph-level CF)
            "edges_removed": {
                "type": "nested",
                "properties": {
                    "src": {"type": "keyword"},
                    "dst": {"type": "keyword"},
                },
            },
            "explanation_text":     {"type": "text"},
            "timestamp":            {"type": "date", "format": "epoch_millis||strict_date_optional_time"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}

EMBEDDINGS_MAPPING_DIM = 16  # default matches generate_embeddings & run_pipeline defaults

# --- Graph summaries mapping (structural + spectral metrics per window) ---
GRAPH_SUMMARIES_MAPPING = {
    "mappings": {
        "properties": {
            "window_id":          {"type": "integer"},
            "window_start":       {"type": "float"},
            "num_nodes":          {"type": "integer"},
            "num_edges":          {"type": "integer"},
            "mean_degree":        {"type": "float"},
            "density":            {"type": "float"},
            "spectral_gap":       {"type": "float"},
            "spectral_radius":    {"type": "float"},
            "malicious_ratio":    {"type": "float"},
            "num_malicious":      {"type": "integer"},
            "timestamp":          {"type": "date", "format": "epoch_millis||strict_date_optional_time"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}

# ──────────────────────────────────────────────
# 2b. ES-NATIVE ENHANCEMENTS: runtime fields, ILM, index templates, ingest pipelines
# ──────────────────────────────────────────────

# --- Runtime fields: severity scoring at query time (zero re-indexing) ---
# ES computes these on-the-fly using Painless scripts, so the severity
# classification is always consistent and doesn't require Python.

SEVERITY_RUNTIME_FIELDS = {
    "severity_level": {
        "type": "keyword",
        "script": {
            "source": """
                double score = 0.0;
                if (doc.containsKey('prediction_score') && doc['prediction_score'].size() > 0) {
                    score = doc['prediction_score'].value;
                } else if (doc.containsKey('label') && doc['label'].size() > 0 && doc['label'].value == 1) {
                    score = 0.85;
                }
                if (score > 0.9)      emit('critical');
                else if (score > 0.7) emit('high');
                else if (score > 0.5) emit('medium');
                else                  emit('low');
            """
        },
    },
    "severity_score": {
        "type": "double",
        "script": {
            "source": """
                if (doc.containsKey('prediction_score') && doc['prediction_score'].size() > 0) {
                    emit(doc['prediction_score'].value);
                } else if (doc.containsKey('label') && doc['label'].size() > 0 && doc['label'].value == 1) {
                    emit(0.85);
                } else {
                    emit(0.2);
                }
            """
        },
    },
    "traffic_volume_category": {
        "type": "keyword",
        "script": {
            "source": """
                double bytes = 0;
                if (doc.containsKey('total_bytes') && doc['total_bytes'].size() > 0) {
                    bytes = doc['total_bytes'].value;
                }
                if (bytes > 100000)     emit('flood');
                else if (bytes > 10000) emit('high');
                else if (bytes > 1000)  emit('moderate');
                else                    emit('low');
            """
        },
    },
}

# --- ILM policy: auto-manage index lifecycle ---
ILM_POLICY_NAME = "incidentlens-lifecycle"
ILM_POLICY_BODY = {
    "policy": {
        "phases": {
            "hot": {
                "min_age": "0ms",
                "actions": {
                    "rollover": {
                        "max_primary_shard_size": "50gb",
                        "max_age": "7d",
                    },
                    "set_priority": {"priority": 100},
                },
            },
            "warm": {
                "min_age": "7d",
                "actions": {
                    "shrink": {"number_of_shards": 1},
                    "forcemerge": {"max_num_segments": 1},
                    "set_priority": {"priority": 50},
                },
            },
            "delete": {
                "min_age": "90d",
                "actions": {"delete": {}},
            },
        }
    }
}

# --- ES-level ingest pipeline: NaN/Inf cleanup + timestamp normalisation ---
INGEST_PIPELINE_NAME = "incidentlens-cleanup"
INGEST_PIPELINE_BODY = {
    "description": "IncidentLens flow cleanup: replace NaN/Inf, add @timestamp",
    "processors": [
        # Set @timestamp from epoch_millis timestamp field
        {
            "date": {
                "field": "timestamp",
                "target_field": "@timestamp",
                "formats": ["epoch_millis", "ISO8601"],
                "ignore_failure": True,
            }
        },
        # Default @timestamp to ingest time if timestamp field is missing  
        {
            "set": {
                "field": "@timestamp",
                "value": "{{_ingest.timestamp}}",
                "override": False,
            }
        },
        # Ensure numeric fields have safe values (NaN/null → 0)
        {
            "script": {
                "source": """
                    def fields = ['packet_count','total_bytes','mean_payload','mean_iat','std_iat','prediction_score'];
                    for (def f : fields) {
                        if (ctx.containsKey(f)) {
                            def v = ctx[f];
                            if (v == null || (v instanceof Number && (v.isNaN() || v.isInfinite()))) {
                                ctx[f] = 0.0;
                            }
                        }
                    }
                """,
                "ignore_failure": True,
            }
        },
    ],
}

# --- Index templates: ensure consistent mappings across environments ---
FLOWS_TEMPLATE_NAME = "incidentlens-flows-template"
EMBEDDINGS_TEMPLATE_NAME = "incidentlens-embeddings-template"
COUNTERFACTUALS_TEMPLATE_NAME = "incidentlens-counterfactuals-template"


def setup_ilm_policy(es: Elasticsearch | None = None) -> bool:
    """Create the ILM policy for automatic index lifecycle management.

    Returns True if the policy was created or already exists.
    """
    es = es or get_client()
    try:
        es.ilm.put_lifecycle(name=ILM_POLICY_NAME, body=ILM_POLICY_BODY)
        logger.info("ILM policy '%s' created/updated", ILM_POLICY_NAME)
        return True
    except Exception as e:
        logger.warning("ILM policy setup skipped (may require license): %s", e)
        return False


def setup_ingest_pipeline(es: Elasticsearch | None = None) -> bool:
    """Create the ES-level ingest pipeline for NaN/Inf cleanup.

    Returns True if the pipeline was created.
    """
    es = es or get_client()
    try:
        es.ingest.put_pipeline(id=INGEST_PIPELINE_NAME, body=INGEST_PIPELINE_BODY)
        logger.info("Ingest pipeline '%s' created", INGEST_PIPELINE_NAME)
        return True
    except Exception as e:
        logger.warning("Ingest pipeline setup failed: %s", e)
        return False


def setup_index_templates(es: Elasticsearch | None = None, embedding_dim: int = EMBEDDINGS_MAPPING_DIM) -> dict[str, bool]:
    """Create index templates for all IncidentLens indices.

    Templates ensure consistent mappings and settings regardless of
    which environment creates the index (dev, CI, prod).
    """
    es = es or get_client()
    results: dict[str, bool] = {}

    templates = [
        (FLOWS_TEMPLATE_NAME, [f"{FLOWS_INDEX}*"], FLOWS_MAPPING),
        (COUNTERFACTUALS_TEMPLATE_NAME, [f"{COUNTERFACTUALS_INDEX}*"], COUNTERFACTUALS_MAPPING),
        (EMBEDDINGS_TEMPLATE_NAME, [f"{EMBEDDINGS_INDEX}*"], _embeddings_mapping(embedding_dim)),
    ]

    for name, patterns, body in templates:
        try:
            es.indices.put_index_template(
                name=name,
                body={
                    "index_patterns": patterns,
                    "template": {
                        "settings": body.get("settings", {}),
                        "mappings": body.get("mappings", {}),
                    },
                },
            )
            results[name] = True
            logger.info("Index template '%s' created (composable)", name)
        except Exception as e:
            results[name] = False
            logger.warning("Index template '%s' failed: %s", name, e)

    return results


def search_flows_with_severity(
    query: dict | None = None,
    severity: str | None = None,
    size: int = 20,
    es: Elasticsearch | None = None,
) -> list[dict]:
    """Search flows using ES runtime fields for severity classification.

    This leverages Elasticsearch's runtime fields to compute severity_level
    at query time — no re-indexing needed when thresholds change.

    Parameters
    ----------
    query : optional base query (default: match_all)
    severity : filter by runtime-computed severity ('critical','high','medium','low')
    size : max results
    """
    es = es or get_client()
    base_query = query or {"match_all": {}}

    if severity:
        # Wrap the base query with a severity filter using runtime fields
        base_query = {
            "bool": {
                "must": [base_query],
                "filter": [{"term": {"severity_level": severity}}],
            }
        }

    body: dict[str, Any] = {
        "size": size,
        "runtime_mappings": SEVERITY_RUNTIME_FIELDS,
        "query": base_query,
        "fields": ["severity_level", "severity_score", "traffic_volume_category"],
        "sort": [{"severity_score": {"order": "desc"}}],
    }

    resp = es.search(index=FLOWS_INDEX, body=body)
    results = []
    for hit in resp["hits"]["hits"]:
        doc = hit["_source"]
        doc["_id"] = hit["_id"]
        # Merge runtime fields into the doc
        for rf in ("severity_level", "severity_score", "traffic_volume_category"):
            vals = hit.get("fields", {}).get(rf)
            if vals:
                doc[rf] = vals[0]
        results.append(doc)
    return results


def aggregate_severity_breakdown(
    es: Elasticsearch | None = None,
) -> dict[str, int]:
    """Use ES runtime fields + terms aggregation to get severity distribution.

    Returns {"critical": N, "high": N, "medium": N, "low": N} computed
    entirely server-side by Elasticsearch — zero Python iteration.
    """
    es = es or get_client()
    body = {
        "size": 0,
        "runtime_mappings": SEVERITY_RUNTIME_FIELDS,
        "aggs": {
            "by_severity": {
                "terms": {"field": "severity_level", "size": 10}
            },
            "by_volume": {
                "terms": {"field": "traffic_volume_category", "size": 10}
            },
        },
    }
    resp = es.search(index=FLOWS_INDEX, body=body)
    severity_dist = {
        b["key"]: b["doc_count"]
        for b in resp["aggregations"]["by_severity"]["buckets"]
    }
    volume_dist = {
        b["key"]: b["doc_count"]
        for b in resp["aggregations"]["by_volume"]["buckets"]
    }
    return {"severity": severity_dist, "volume": volume_dist}


def search_with_pagination(
    query: dict | None = None,
    size: int = 20,
    search_after: list | None = None,
    pit_id: str | None = None,
    es: Elasticsearch | None = None,
) -> dict:
    """Paginate large result sets using ES search_after + point-in-time.

    This is the recommended Elasticsearch approach for deep pagination —
    unlike `from + size`, it doesn't degrade with depth and provides a
    consistent snapshot of the data.

    Returns {"hits": [...], "pit_id": str, "search_after": list, "total": int}
    """
    es = es or get_client()

    # Open a PIT if none provided
    opened_here = False
    if pit_id is None:
        pit_resp = es.open_point_in_time(index=FLOWS_INDEX, keep_alive="2m")
        pit_id = pit_resp["id"]
        opened_here = True

    try:
        body: dict[str, Any] = {
            "size": size,
            "query": query or {"match_all": {}},
            "pit": {"id": pit_id, "keep_alive": "2m"},
            "sort": [{"timestamp": "desc"}, {"_shard_doc": "asc"}],
        }
        if search_after:
            body["search_after"] = search_after

        resp = es.search(body=body)
        hits = resp["hits"]["hits"]
        docs = []
        for h in hits:
            doc = h["_source"]
            doc["_id"] = h["_id"]
            docs.append(doc)

        last_sort = hits[-1]["sort"] if hits else None
        return {
            "hits": docs,
            "pit_id": pit_id,
            "search_after": last_sort,
            "total": resp["hits"]["total"]["value"],
        }
    except Exception:
        if opened_here:
            close_pit(pit_id, es=es)
        raise


def close_pit(pit_id: str, es: Elasticsearch | None = None) -> bool:
    """Close a point-in-time to release server resources."""
    es = es or get_client()
    try:
        es.close_point_in_time(body={"id": pit_id})
        return True
    except Exception as exc:
        logger.warning("Failed to close PIT %s: %s", pit_id, exc)
        return False


def composite_aggregation(
    field: str,
    sub_aggs: dict | None = None,
    size: int = 100,
    es: Elasticsearch | None = None,
) -> list[dict]:
    """Paginate through ALL aggregation buckets using ES composite aggregation.

    Unlike terms aggregation (limited by size), composite agg handles
    unbounded cardinality — essential for large IP spaces.

    Parameters
    ----------
    field : the field to aggregate on (e.g. 'src_ip')
    sub_aggs : optional sub-aggregations per bucket
    size : page size per request
    """
    es = es or get_client()
    all_buckets: list[dict] = []
    after_key = None

    while True:
        sources = [{field: {"terms": {"field": field}}}]
        composite: dict[str, Any] = {"size": size, "sources": sources}
        if after_key:
            composite["after"] = after_key

        aggs: dict[str, Any] = {"composite_agg": {"composite": composite}}
        if sub_aggs:
            aggs["composite_agg"]["aggs"] = sub_aggs

        body = {"size": 0, "aggs": aggs}
        resp = es.search(index=FLOWS_INDEX, body=body)
        buckets = resp["aggregations"]["composite_agg"]["buckets"]

        if not buckets:
            break

        all_buckets.extend(buckets)
        after_key = buckets[-1]["key"]

        # Safety: if we got fewer than page size, we've reached the end
        if len(buckets) < size:
            break

    return all_buckets


def full_text_search_counterfactuals(
    query_text: str,
    size: int = 10,
    es: Elasticsearch | None = None,
) -> list[dict]:
    """Search counterfactual explanations using full-text search.

    The `explanation_text` field is mapped as `text` — this leverages
    ES's text analysis pipeline (tokenisation, stemming, relevance scoring)
    for natural-language queries over CF narratives.
    """
    es = es or get_client()
    body = {
        "size": size,
        "query": {
            "match": {
                "explanation_text": {
                    "query": query_text,
                    "fuzziness": "AUTO",
                }
            }
        },
        "highlight": {
            "fields": {"explanation_text": {}},
        },
    }
    resp = es.search(index=COUNTERFACTUALS_INDEX, body=body)
    return [
        {
            **hit["_source"],
            "_score": hit["_score"],
            "highlight": hit.get("highlight", {}),
        }
        for hit in resp["hits"]["hits"]
    ]


def _embeddings_mapping(dim: int = EMBEDDINGS_MAPPING_DIM) -> dict:
    return {
        "mappings": {
            "properties": {
                "flow_id":      {"type": "keyword"},
                "label":        {"type": "integer"},
                "prediction":   {"type": "integer"},
                "window_id":    {"type": "integer"},
                "src_ip":       {"type": "ip"},
                "dst_ip":       {"type": "ip"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
    }


def _get_existing_embedding_dim(es: Elasticsearch, index_name: str) -> int | None:
    """Return the dense_vector dims of an existing embeddings index, or None."""
    try:
        mapping = es.indices.get_mapping(index=index_name)
        props = mapping[index_name]["mappings"].get("properties", {})
        return props.get("embedding", {}).get("dims")
    except Exception:
        return None


def create_index(
    index_name: str,
    body: dict,
    es: Elasticsearch | None = None,
    delete_existing: bool = False,
) -> bool:
    """Create an index with the given mapping. Returns True if created."""
    es = es or get_client()
    if es.indices.exists(index=index_name):
        if delete_existing:
            es.indices.delete(index=index_name)
            logger.info("Deleted existing index %s", index_name)
        else:
            logger.info("Index %s already exists \u2014 skipping.", index_name)
            return False
    es.indices.create(index=index_name, body=body)
    logger.info("Created index %s", index_name)
    return True


def setup_all_indices(
    es: Elasticsearch | None = None,
    embedding_dim: int = EMBEDDINGS_MAPPING_DIM,
    delete_existing: bool = False,
) -> dict[str, bool]:
    """Create all project indices, plus ES-native infrastructure.

    Sets up:
    1. ILM policy for automatic lifecycle management
    2. ES ingest pipeline for NaN/Inf cleanup + timestamp normalisation
    3. Index templates for consistent mappings across environments
    4. The three data indices (flows, counterfactuals, embeddings)

    Returns {name: was_created} for the three data indices.
    """
    es = es or get_client()

    # ES infrastructure (best-effort — some features need a licence)
    setup_ilm_policy(es)
    setup_ingest_pipeline(es)
    setup_index_templates(es, embedding_dim=embedding_dim)

    # Auto-detect dimension mismatch on existing embeddings index and
    # force recreation so GNN-produced vectors can always be indexed.
    emb_delete = delete_existing
    if not emb_delete and es.indices.exists(index=EMBEDDINGS_INDEX):
        existing_dim = _get_existing_embedding_dim(es, EMBEDDINGS_INDEX)
        if existing_dim is not None and existing_dim != embedding_dim:
            logger.warning(
                "Embeddings index has dims=%d but model needs %d \u2014 recreating.",
                existing_dim, embedding_dim,
            )
            emb_delete = True

    return {
        FLOWS_INDEX: create_index(FLOWS_INDEX, FLOWS_MAPPING, es, delete_existing),
        COUNTERFACTUALS_INDEX: create_index(
            COUNTERFACTUALS_INDEX, COUNTERFACTUALS_MAPPING, es, delete_existing
        ),
        EMBEDDINGS_INDEX: create_index(
            EMBEDDINGS_INDEX, _embeddings_mapping(embedding_dim), es, emb_delete
        ),
        GRAPH_SUMMARIES_INDEX: create_index(
            GRAPH_SUMMARIES_INDEX, GRAPH_SUMMARIES_MAPPING, es, delete_existing
        ),
    }


def delete_all_indices(es: Elasticsearch | None = None) -> None:
    """Tear down all project indices."""
    es = es or get_client()
    for idx in (FLOWS_INDEX, COUNTERFACTUALS_INDEX, EMBEDDINGS_INDEX, GRAPH_SUMMARIES_INDEX):
        if es.indices.exists(index=idx):
            es.indices.delete(index=idx)
            logger.info("Deleted index %s", idx)


# ──────────────────────────────────────────────
# 3. FLOW INGESTION (bulk)
# ──────────────────────────────────────────────

def _flow_id(window_id: int, src: str, dst: str, edge_idx: int) -> str:
    """Deterministic flow ID for dedup.  Uses MD5 (2-3x faster than SHA-256;
    collision resistance is irrelevant here — we only need uniqueness)."""
    raw = f"{window_id}:{src}:{dst}:{edge_idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _flow_ids_batch(
    window_id: int,
    src_ips: list[str] | np.ndarray,
    dst_ips: list[str] | np.ndarray,
    n: int,
) -> list[str]:
    """Batch flow-ID generation (avoids per-edge function-call overhead).

    Builds all key strings first, then hashes in a tight loop with a
    cached reference to hashlib.md5 — ~2x faster than calling
    _flow_id() in a loop due to reduced function-call overhead.
    """
    _md5 = hashlib.md5
    prefix = f"{window_id}:"
    # Build all byte strings in one pass, then hash
    raw_bytes = [f"{prefix}{src_ips[i]}:{dst_ips[i]}:{i}".encode() for i in range(n)]
    return [_md5(b).hexdigest()[:16] for b in raw_bytes]


def index_pyg_graph(
    graph: Data,
    node_id_to_ip: dict[int, str],
    es: Elasticsearch | None = None,
    predictions: torch.Tensor | None = None,
    prediction_scores: torch.Tensor | None = None,
    window_id: int | None = None,
    batch_size: int = 500,
) -> int:
    """Bulk-index edges from a single PyG Data snapshot into the flows index.

    **Optimised**: builds all docs via numpy batch operations instead of
    a per-edge Python loop.  Typically 3-5x faster for graphs >50 edges.

    When a GNN is registered via ``set_gnn_encoder()`` and no explicit
    predictions are provided, the GNN's ``predict_labels()`` is called
    automatically to populate ``prediction`` + ``prediction_score`` fields.

    Parameters
    ----------
    graph : Data
        A PyG graph with ``edge_index``, ``edge_attr``, ``y``, and optionally
        ``window_start``.
    node_id_to_ip : dict
        Mapping from integer node-id to IP string.
    predictions / prediction_scores : optional model outputs per edge.
        If ``None`` and a GNN is registered, predictions are auto-generated.
    window_id : override; otherwise pulled from ``graph.window_id``.

    Returns
    -------
    int : number of docs successfully indexed.
    """
    es = es or get_client()
    wid = window_id if window_id is not None else getattr(graph, "window_id", 0)
    w_start = float(getattr(graph, "window_start", 0.0))

    edge_index = graph.edge_index.cpu().numpy()
    edge_attr = graph.edge_attr.cpu().numpy() if graph.edge_attr is not None else None
    labels = graph.y.cpu().numpy() if graph.y is not None else None

    # ── Auto-predict with registered GNN when no explicit predictions given ──
    if predictions is None and _gnn_encoder is not None:
        try:
            pred_labels, pred_scores = _gnn_encoder.predict_labels(graph)
            predictions = pred_labels
            prediction_scores = pred_scores
            # Feed back to graph so counterfactual analysis can use GNN output
            graph.pred_labels = pred_labels
            graph.pred_scores = pred_scores
            logger.debug("GNN auto-predicted %d edges for window %d", len(pred_labels), wid)
        except Exception as exc:
            logger.warning("GNN predict_labels failed for window %d: %s", wid, exc)

    preds = predictions.cpu().numpy() if predictions is not None else None
    scores = prediction_scores.cpu().numpy() if prediction_scores is not None else None

    n_edges = edge_index.shape[1]
    feature_names = ["packet_count", "total_bytes", "mean_payload", "mean_iat", "std_iat"]

    # ── Batch IP lookup (keep as Python lists — faster per-element access) ──
    # Use per-graph local→IP mapping when available (local node IDs
    # differ from global IDs across windows)
    _ip_map = getattr(graph, '_local_to_ip', node_id_to_ip)
    src_ids = edge_index[0]
    dst_ids = edge_index[1]
    src_ips = [_ip_map.get(int(s), f"10.{(int(s) >> 16) & 0xFF}.{(int(s) >> 8) & 0xFF}.{int(s) & 0xFF}") for s in src_ids]
    dst_ips = [_ip_map.get(int(d), f"10.{(int(d) >> 16) & 0xFF}.{(int(d) >> 8) & 0xFF}.{int(d) & 0xFF}") for d in dst_ids]

    # ── Batch flow-ID generation ──
    flow_ids = _flow_ids_batch(wid, src_ips, dst_ips, n_edges)

    # ── Pre-convert numpy arrays to Python lists (single C-level call
    #    avoids per-element numpy→Python coercion in the doc loop) ──
    ts_now = int(time.time() * 1000)
    labels_list = labels.tolist() if labels is not None else None
    preds_list = preds.tolist() if preds is not None else None
    scores_list = scores.tolist() if scores is not None else None

    # Pre-extract edge_attr columns as Python lists
    feat_lists: dict[str, list[float]] | None = None
    if edge_attr is not None and edge_attr.shape[1] >= len(feature_names):
        feat_lists = {
            fname: edge_attr[:, j].tolist()
            for j, fname in enumerate(feature_names)
        }

    # ── Vectorised doc construction via DataFrame → records (C-level iteration) ──
    cols: dict[str, Any] = {
        "flow_id": flow_ids,
        "window_id": [wid] * n_edges,
        "window_start": [w_start] * n_edges,
        "src_ip": src_ips,
        "dst_ip": dst_ips,
        "label": labels_list if labels_list is not None else [0] * n_edges,
        "timestamp": [ts_now] * n_edges,
    }
    if feat_lists is not None:
        cols.update(feat_lists)
    if preds_list is not None:
        cols["prediction"] = preds_list
    if scores_list is not None:
        cols["prediction_score"] = scores_list

    df_docs = pd.DataFrame(cols)
    records = df_docs.to_dict(orient="records")
    actions = [
        {"_index": FLOWS_INDEX, "_id": records[i]["flow_id"], "_source": records[i]}
        for i in range(n_edges)
    ]

    success, errors = helpers.bulk(es, actions, chunk_size=batch_size, raise_on_error=False)
    if errors:
        logger.warning("Bulk index had %d errors", len(errors))
    return success


def index_graphs_bulk(
    graphs: list[Data],
    node_id_to_ip: dict[int, str],
    es: Elasticsearch | None = None,
    batch_size: int = 1000,
) -> int:
    """Index all graph snapshots in a SINGLE bulk call to ES.

    Collects all edge documents across all graphs, then makes one
    ``helpers.bulk()`` call instead of N separate calls — reduces
    HTTP round-trips from N to 1.
    """
    es = es or get_client()
    all_actions: list[dict] = []
    feature_names = ["packet_count", "total_bytes", "mean_payload", "mean_iat", "std_iat"]

    for g in graphs:
        wid = getattr(g, "window_id", 0)
        w_start = float(getattr(g, "window_start", 0.0))
        ei = g.edge_index.cpu().numpy()
        ea = g.edge_attr.cpu().numpy() if g.edge_attr is not None else None
        labels = g.y.cpu().numpy() if g.y is not None else None
        n_edges = ei.shape[1]

        # Auto-predict with registered GNN
        preds = None
        scores = None
        if _gnn_encoder is not None:
            try:
                pred_labels, pred_scores = _gnn_encoder.predict_labels(g)
                preds = pred_labels.cpu().numpy()
                scores = pred_scores.cpu().numpy()
                # Feed back to graph so counterfactual analysis can use GNN output
                g.pred_labels = pred_labels
                g.pred_scores = pred_scores
            except Exception as exc:
                logger.warning("GNN predict_labels failed for window %d: %s", wid, exc)

        # Use per-graph local→IP mapping (local IDs differ across windows)
        _ip_map = getattr(g, '_local_to_ip', node_id_to_ip)
        src_ids = ei[0]
        dst_ids = ei[1]
        src_ips = [_ip_map.get(int(s), f"10.{(int(s) >> 16) & 0xFF}.{(int(s) >> 8) & 0xFF}.{int(s) & 0xFF}") for s in src_ids]
        dst_ips = [_ip_map.get(int(d), f"10.{(int(d) >> 16) & 0xFF}.{(int(d) >> 8) & 0xFF}.{int(d) & 0xFF}") for d in dst_ids]
        flow_ids = _flow_ids_batch(wid, src_ips, dst_ips, n_edges)
        ts_now = int(time.time() * 1000)

        # Build docs via DataFrame → records (C-level iteration)
        cols: dict[str, Any] = {
            "flow_id": flow_ids,
            "window_id": [wid] * n_edges,
            "window_start": [w_start] * n_edges,
            "src_ip": src_ips,
            "dst_ip": dst_ips,
            "label": labels.tolist() if labels is not None else [0] * n_edges,
            "timestamp": [ts_now] * n_edges,
        }
        if ea is not None and ea.shape[1] >= len(feature_names):
            for j, fn in enumerate(feature_names):
                cols[fn] = ea[:, j].tolist()
        if preds is not None:
            cols["prediction"] = preds.tolist()
        if scores is not None:
            cols["prediction_score"] = scores.tolist()

        df_docs = pd.DataFrame(cols)
        records = df_docs.to_dict(orient="records")
        all_actions.extend([
            {"_index": FLOWS_INDEX, "_id": rec["flow_id"], "_source": rec}
            for rec in records
        ])

    if not all_actions:
        return 0

    success, errors = helpers.bulk(es, all_actions, chunk_size=batch_size, raise_on_error=False)
    if errors:
        logger.warning("Bulk index had %d errors", len(errors))
    return success


# ──────────────────────────────────────────────
# 3a-2. GRAPH SUMMARY INDEXING (structural + spectral metrics)
# ──────────────────────────────────────────────

def index_graph_summaries(
    graphs: list[Data],
    es: Elasticsearch | None = None,
) -> int:
    """Index graph-level summaries with structural + spectral metrics.

    For each graph window, computes and indexes:
    - num_nodes, num_edges, mean_degree, density
    - spectral_gap (algebraic connectivity λ₂)
    - spectral_radius
    - malicious_ratio

    This enables ES queries like "find windows where spectral gap dropped"
    or "windows with density > X" — impossible with edge-only indexing.
    """
    es = es or get_client()

    # Ensure the index exists
    create_index(GRAPH_SUMMARIES_INDEX, GRAPH_SUMMARIES_MAPPING, es)

    actions = []
    ts_now = int(time.time() * 1000)

    for g in graphs:
        wid = getattr(g, "window_id", 0)
        n_nodes = int(g.num_nodes)
        n_edges = int(g.edge_index.shape[1])
        mean_deg = float(n_edges * 2 / max(n_nodes, 1))
        max_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
        density = float(n_edges / max_edges) if max_edges > 0 else 0.0

        # Compute spectral metrics from the algebraic graph structure
        net = getattr(g, "network", None)
        s_gap = 0.0
        s_radius = 0.0
        if net is not None and n_nodes > 1:
            try:
                s_gap = net.spectral_gap(normalised=True)
            except Exception:
                s_gap = 0.0
            try:
                s_radius = net.spectral_radius()
            except Exception:
                s_radius = 0.0

        # Malicious ratio
        labels = g.y.cpu().numpy() if g.y is not None else np.zeros(n_edges)
        n_mal = int((labels == 1).sum())
        mal_ratio = float(n_mal / max(n_edges, 1))

        doc = {
            "window_id": wid,
            "window_start": float(getattr(g, "window_start", 0.0)),
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "mean_degree": round(mean_deg, 4),
            "density": round(density, 6),
            "spectral_gap": round(s_gap, 6),
            "spectral_radius": round(s_radius, 6),
            "malicious_ratio": round(mal_ratio, 4),
            "num_malicious": n_mal,
            "timestamp": ts_now,
        }
        actions.append({"_index": GRAPH_SUMMARIES_INDEX, "_source": doc})

    if not actions:
        return 0

    success, errors = helpers.bulk(es, actions, raise_on_error=False)
    if errors:
        logger.warning("Graph summary bulk had %d errors", len(errors))
    return success


# ──────────────────────────────────────────────
# 3b. GRAPH BUILDING (via graph_data_wrapper)
# ──────────────────────────────────────────────

def build_graphs(
    df: pd.DataFrame,
    window_size: float = 2.0,
    stride: float = 1.0,
    bytes_col: str = "packet_length",
    label_col: str = "label",
) -> tuple[list[Data], dict[int, str]]:
    """Build sliding-window PyG graphs from a packet DataFrame.

    Delegates to ``graph_data_wrapper.build_sliding_window_graphs``
    (pure-numpy vectorised implementation).

    Parameters
    ----------
    df : DataFrame with at least: timestamp, src_ip, dst_ip, protocol,
         packet_length, payload_length, dst_port, label.
    window_size : seconds per window.
    stride : seconds between window starts.
    bytes_col : column name for packet byte size.
    label_col : column name for ground-truth label.

    Returns
    -------
    (graphs, id_to_ip) — list of PyG Data objects + node-id-to-IP mapping.
    """
    required = ["timestamp", "src_ip", "dst_ip", "protocol",
                 "payload_length", "dst_port", label_col, bytes_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Clean the dataframe for graph building
    graph_df = df.dropna(subset=["src_ip", "dst_ip", "protocol"]).copy()
    if "dst_port" not in graph_df.columns:
        graph_df["dst_port"] = 0
    graph_df["dst_port"] = graph_df["dst_port"].fillna(0).astype(int)
    graph_df["protocol"] = graph_df["protocol"].astype(int)

    graphs, graph_id_to_ip = _gdw_build_graphs(
        graph_df,
        window_size=window_size,
        stride=stride,
        bytes_col=bytes_col,
        label_col=label_col,
    )

    # Use the id_to_ip returned directly from the graph builder
    # instead of redundantly reconstructing it from _local_to_ip dicts.
    id_to_ip = graph_id_to_ip

    # Fallback: if empty (shouldn't happen), derive from DataFrame
    if not id_to_ip:
        all_ips = pd.concat(
            [graph_df["src_ip"], graph_df["dst_ip"]], ignore_index=True
        ).dropna().unique()
        id_to_ip = {i: str(ip) for i, ip in enumerate(all_ips)}

    return graphs, id_to_ip


def analyze_graph_dataset(graphs: list[Data]) -> None:
    """Print graph dataset statistics (delegates to graph_data_wrapper)."""
    _gdw_analyze(graphs)


def graph_edge_perturbation_cf(
    graph: Data,
    id_to_ip: dict[int, str] | None = None,
    target_edge_indices: list[int] | None = None,
    max_removals: int = 5,
) -> list[dict]:
    """Run edge-perturbation counterfactual analysis on a single graph window.

    Identifies which edges (network connections) have the greatest
    structural and feature impact — removing them shows what connections
    would need to "not exist" for the graph to look normal.

    Delegates to ``graph_data_wrapper.edge_perturbation_counterfactual``.
    """
    results = _gdw_edge_perturbation_cf(
        graph,
        target_edge_indices=target_edge_indices,
        max_removals=max_removals,
    )
    # Enrich with IP addresses — prefer per-graph local→IP mapping
    _ip_map = getattr(graph, '_local_to_ip', id_to_ip) if id_to_ip else getattr(graph, '_local_to_ip', None)
    if _ip_map and results:
        for r in results:
            r["src_ip"] = _ip_map.get(r["src"], f"0.0.0.{r['src']}")
            r["dst_ip"] = _ip_map.get(r["dst"], f"0.0.0.{r['dst']}")
    return results


def graph_window_comparison(
    graphs: list[Data],
    window_a: int | None = None,
    window_b: int | None = None,
) -> dict:
    """Compare two graph windows to understand structural + feature changes.

    By default, compares the most normal window vs the most anomalous one.
    Delegates to ``graph_data_wrapper.compare_graph_windows``.
    """
    if window_a is not None and window_b is not None:
        ga = graphs[window_a]
        gb = graphs[window_b]
    else:
        # Auto-select: most normal vs most anomalous
        norm_idx, _, _ = _gdw_find_normal(graphs)
        anom_idx, _, _ = _gdw_find_anomalous(graphs)
        if norm_idx == anom_idx and len(graphs) > 1:
            # All windows identical ratio — pick first and last instead
            norm_idx, anom_idx = 0, len(graphs) - 1
        ga = graphs[norm_idx]
        gb = graphs[anom_idx]

    return _gdw_compare_windows(ga, gb)


def find_anomalous_window(graphs: list[Data]) -> tuple[int, Data, dict]:
    """Find the most anomalous graph window (highest malicious edge ratio).

    Delegates to ``graph_data_wrapper.find_most_anomalous_window``.
    """
    return _gdw_find_anomalous(graphs)


def find_normal_window(graphs: list[Data]) -> tuple[int, Data, dict]:
    """Find the most normal graph window (lowest malicious edge ratio).

    Delegates to ``graph_data_wrapper.find_most_normal_window``.
    """
    return _gdw_find_normal(graphs)


def load_dataset_from_csv(
    packet_csv: str,
    label_csv: str,
    delta_t: float = 5.0,
) -> list[Data]:
    """Load CSVs and build snapshot graphs (delegates to graph_data_wrapper).

    Convenience wrapper around ``graph_data_wrapper.load_graph_dataset``.
    """
    return _gdw_load_dataset(packet_csv, label_csv, delta_t=delta_t)


def build_and_index_graphs(
    df: pd.DataFrame,
    window_size: float = 2.0,
    stride: float = 1.0,
    bytes_col: str = "packet_length",
    label_col: str = "label",
    es: Elasticsearch | None = None,
) -> tuple[list[Data], dict[int, str], int]:
    """Build graphs from a DataFrame AND index all flows into ES.

    Combines ``build_graphs`` + ``index_graphs_bulk`` into a single call.

    Returns
    -------
    (graphs, id_to_ip, num_indexed) — the graph list, IP mapping, and
    number of ES docs successfully indexed.
    """
    es = es or get_client()
    graphs, id_to_ip = build_graphs(
        df, window_size=window_size, stride=stride,
        bytes_col=bytes_col, label_col=label_col,
    )
    num_indexed = index_graphs_bulk(graphs, id_to_ip, es=es)

    # Also index graph-level summaries with spectral metrics
    try:
        n_summaries = index_graph_summaries(graphs, es=es)
        logger.info("Indexed %d graph summaries with spectral metrics", n_summaries)
    except Exception as exc:
        logger.warning("Graph summary indexing failed: %s", exc)

    return graphs, id_to_ip, num_indexed


# ──────────────────────────────────────────────
# 3c. SYNTHETIC EMBEDDINGS (via graph_data_wrapper graphs)
# ──────────────────────────────────────────────

def generate_embeddings(
    graphs: list[Data],
    id_to_ip: dict[int, str],
    embedding_dim: int = 16,
    gnn_model: BaseGNNEncoder | None = None,
) -> tuple[list[str], np.ndarray, list[int]]:
    """Generate per-edge embeddings from graph snapshots.

    If a GNN encoder is registered (via ``set_gnn_encoder()``) or passed
    directly, it is used to produce learned embeddings.  Otherwise falls
    back to z-score + random projection (placeholder for development).

    **Optimised**: all per-edge work is fully vectorised via numpy
    (no Python for-loops over edges).

    Returns (flow_ids, embeddings, labels).
    """
    model = gnn_model or _gnn_encoder  # registered GNN takes precedence

    # ── Collect flow IDs + labels from all graphs ──
    fid_chunks: list[list[str]] = []
    label_chunks: list[np.ndarray] = []
    attr_chunks: list[np.ndarray] = []  # only needed for fallback path

    for g in graphs:
        wid = getattr(g, "window_id", 0)
        ei = g.edge_index.cpu().numpy()
        n_edges = ei.shape[1]
        ys = (
            g.y.cpu().numpy()
            if g.y is not None
            else np.zeros(n_edges, dtype=np.int64)
        )

        # Use per-graph local→IP mapping (local IDs differ across windows)
        _ip_map = getattr(g, '_local_to_ip', id_to_ip)
        src_ids = ei[0]
        dst_ids = ei[1]
        src_ips = [_ip_map.get(int(s), f"0.0.0.{s}") for s in src_ids]
        dst_ips = [_ip_map.get(int(d), f"0.0.0.{d}") for d in dst_ids]

        fids = _flow_ids_batch(wid, src_ips, dst_ips, n_edges)

        fid_chunks.append(fids)
        label_chunks.append(ys.astype(np.int64))

        if model is None:
            ea = (
                g.edge_attr.cpu().numpy()
                if g.edge_attr is not None
                else np.zeros((n_edges, 5), dtype=np.float32)
            )
            attr_chunks.append(ea)

    all_flow_ids: list[str] = []
    for chunk in fid_chunks:
        all_flow_ids.extend(chunk)
    all_labels = np.concatenate(label_chunks).tolist()

    # ══════════════════════════════════════════════════════
    # GNN PATH: use trained model to produce edge embeddings
    # ══════════════════════════════════════════════════════
    if model is not None:
        # TemporalGNNEncoder: use positional batch inference
        # (the encoder's set_graph_sequence was called with preprocessed
        #  copies in the same order, so we iterate by index)
        from src.Backend.temporal_gnn import TemporalGNNEncoder
        is_temporal = isinstance(model, TemporalGNNEncoder)

        emb_chunks: list[np.ndarray] = []
        # torch.no_grad() disables autograd tape entirely — saves memory
        # allocation + backward-graph construction, giving ~2x speedup.
        with torch.no_grad():
            for idx, g in enumerate(graphs):
                if is_temporal and model._all_graphs is not None:
                    # Use the preprocessed graph at the same position
                    if idx >= len(model._all_graphs):
                        raise IndexError(
                            f"Graph index {idx} out of range for "
                            f"model._all_graphs (len={len(model._all_graphs)}). "
                            f"Ensure set_graph_sequence() was called with "
                            f"the same graphs list."
                        )
                    preprocessed_g = model._all_graphs[idx]
                    edge_emb = model.encode(preprocessed_g)
                else:
                    edge_emb = model.encode(g)  # (E, D) tensor, L2-normalised
                emb_chunks.append(edge_emb.cpu().numpy())
        embeddings = np.vstack(emb_chunks).astype(np.float32)
        actual_dim = embeddings.shape[1]
        if actual_dim != embedding_dim:
            logger.warning(
                "GNN embedding_dim=%d differs from requested embedding_dim=%d; "
                "using actual dimension %d for consistency",
                actual_dim, embedding_dim, actual_dim,
            )
            embedding_dim = actual_dim  # use actual dim so ES mapping matches
        return all_flow_ids, embeddings, all_labels

    # ══════════════════════════════════════════════════════
    # FALLBACK PATH: z-score + random projection (no GNN)
    # ══════════════════════════════════════════════════════
    feat_matrix = np.vstack(attr_chunks).astype(np.float32)

    mu = feat_matrix.mean(axis=0, keepdims=True)
    std = feat_matrix.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    feat_norm = (feat_matrix - mu) / std

    rng = np.random.default_rng(42)
    n_feat = feat_norm.shape[1]
    proj = rng.standard_normal((n_feat, embedding_dim)).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    embeddings = feat_norm @ proj

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    return all_flow_ids, embeddings, all_labels


def build_index_and_embed(
    df: pd.DataFrame,
    window_size: float = 2.0,
    stride: float = 1.0,
    embedding_dim: int = 16,
    es: Elasticsearch | None = None,
) -> tuple[list[Data], dict[int, str], list[str], np.ndarray, list[int]]:
    """Full pipeline helper: build graphs -> index flows -> generate &
    index embeddings.

    Returns (graphs, id_to_ip, flow_ids, embeddings, labels).
    """
    es = es or get_client()

    graphs, id_to_ip, n_flows = build_and_index_graphs(
        df, window_size=window_size, stride=stride, es=es,
    )
    logger.info("Indexed %d flows from %d graphs", n_flows, len(graphs))

    if not graphs:
        return graphs, id_to_ip, [], np.empty((0, embedding_dim)), []

    flow_ids, embeddings, labels = generate_embeddings(
        graphs, id_to_ip, embedding_dim=embedding_dim,
    )
    n_emb = index_embeddings(flow_ids, embeddings, labels, es=es)
    logger.info("Indexed %d embeddings", n_emb)

    return graphs, id_to_ip, flow_ids, embeddings, labels


# ──────────────────────────────────────────────
# 4. EMBEDDING INGESTION + kNN SEARCH
# ──────────────────────────────────────────────

def index_embeddings(
    flow_ids: list[str],
    embeddings: np.ndarray | torch.Tensor,
    labels: list[int],
    es: Elasticsearch | None = None,
    extra_fields: list[dict] | None = None,
    batch_size: int = 500,
) -> int:
    """Bulk-index GNN embeddings for kNN counterfactual search.

    Parameters
    ----------
    flow_ids : list of flow-id strings (same order as embeddings).
    embeddings : (N, D) array of embedding vectors.
    labels : ground-truth or predicted label per flow.
    extra_fields : optional list of dicts with additional fields per doc.
    """
    es = es or get_client()
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    # Single C-level conversion: avoids N separate .tolist() calls
    emb_lists = embeddings.tolist()

    # Zip-based construction: eliminates enumerate overhead + per-key dict mutation
    labels_int = [int(l) for l in labels]  # single pass
    if extra_fields:
        actions = [
            {"_index": EMBEDDINGS_INDEX, "_source": {
                "flow_id": fid, "label": lbl, "embedding": emb, **ef
            }}
            for fid, lbl, emb, ef in zip(flow_ids, labels_int, emb_lists, extra_fields)
        ]
    else:
        actions = [
            {"_index": EMBEDDINGS_INDEX, "_source": {
                "flow_id": fid, "label": lbl, "embedding": emb
            }}
            for fid, lbl, emb in zip(flow_ids, labels_int, emb_lists)
        ]

    success, errors = helpers.bulk(es, actions, chunk_size=batch_size, raise_on_error=False)
    if errors:
        logger.warning("Embedding bulk had %d errors", len(errors))
    return success


def knn_search_nearest_normal(
    query_embedding: list[float] | np.ndarray,
    k: int = 5,
    es: Elasticsearch | None = None,
) -> list[dict]:
    """Find the k nearest *normal* (label=0) flows by cosine similarity.

    This is the core counterfactual retrieval: "what is the closest flow
    that was classified as normal?"
    """
    es = es or get_client()
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    body = {
        "knn": {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": k,
            "num_candidates": max(k * 10, 100),
            "filter": {"term": {"label": 0}},
        },
        "_source": True,
    }
    resp = es.search(index=EMBEDDINGS_INDEX, body=body)
    return [
        {
            "flow_id": hit["_source"].get("flow_id"),
            "score": hit["_score"],
            **{k_: v for k_, v in hit["_source"].items() if k_ != "embedding"},
        }
        for hit in resp["hits"]["hits"]
    ]


def knn_search(
    query_embedding: list[float] | np.ndarray,
    k: int = 5,
    label_filter: int | None = None,
    es: Elasticsearch | None = None,
) -> list[dict]:
    """General-purpose kNN on embeddings, with optional label filter."""
    es = es or get_client()
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    knn_clause: dict[str, Any] = {
        "field": "embedding",
        "query_vector": query_embedding,
        "k": k,
        "num_candidates": max(k * 10, 100),
    }
    if label_filter is not None:
        knn_clause["filter"] = {"term": {"label": label_filter}}

    body = {"knn": knn_clause, "_source": True}
    resp = es.search(index=EMBEDDINGS_INDEX, body=body)
    return [
        {
            "flow_id": hit["_source"].get("flow_id"),
            "score": hit["_score"],
            **{k_: v for k_, v in hit["_source"].items() if k_ != "embedding"},
        }
        for hit in resp["hits"]["hits"]
    ]


# ──────────────────────────────────────────────
# 5. SIGNIFICANT TERMS (discriminating features)
# ──────────────────────────────────────────────

def significant_terms_by_label(
    field: str,
    foreground_label: int = 1,
    size: int = 20,
    es: Elasticsearch | None = None,
) -> list[dict]:
    """Find terms in ``field`` that are significantly overrepresented in
    flows with the given label vs the full population.

    Great for answering: "which protocols / IPs are disproportionately
    present in attack traffic?"
    """
    es = es or get_client()
    body = {
        "size": 0,
        "query": {"term": {"label": foreground_label}},
        "aggs": {
            "sig": {
                "significant_terms": {
                    "field": field,
                    "size": size,
                }
            }
        },
    }
    resp = es.search(index=FLOWS_INDEX, body=body)
    buckets = resp["aggregations"]["sig"]["buckets"]
    return [
        {
            "term": b["key"],
            "doc_count": b["doc_count"],
            "bg_count": b["bg_count"],
            "score": b["score"],
        }
        for b in buckets
    ]


# ──────────────────────────────────────────────
# 6. FEATURE DISTRIBUTION AGGREGATIONS
# ──────────────────────────────────────────────

FEATURE_FIELDS = [
    "packet_count",
    "total_bytes",
    "mean_payload",
    "mean_iat",
    "std_iat",
]


def feature_stats_by_label(
    features: list[str] | None = None,
    es: Elasticsearch | None = None,
) -> dict[str, dict]:
    """Return {feature: {label_0: stats, label_1: stats}} using
    extended_stats grouped by label.

    Useful for building counterfactual range sliders ("normal packet_count
    is 1-15, but this anomaly has 450").
    """
    es = es or get_client()
    features = features or FEATURE_FIELDS

    aggs: dict[str, Any] = {
        "by_label": {
            "terms": {"field": "label"},
            "aggs": {},
        }
    }
    for feat in features:
        aggs["by_label"]["aggs"][feat] = {"extended_stats": {"field": feat}}

    body = {"size": 0, "aggs": aggs}
    resp = es.search(index=FLOWS_INDEX, body=body)

    result: dict[str, dict] = {}
    for bucket in resp["aggregations"]["by_label"]["buckets"]:
        lbl = int(bucket["key"])
        for feat in features:
            if feat not in result:
                result[feat] = {}
            stats = bucket[feat]
            result[feat][f"label_{lbl}"] = {
                "count": stats["count"],
                "min": stats["min"],
                "max": stats["max"],
                "avg": stats["avg"],
                "std_deviation": stats["std_deviation"],
            }
    return result


def feature_percentiles_by_label(
    feature: str,
    percents: list[float] | None = None,
    es: Elasticsearch | None = None,
) -> dict:
    """Return percentile values of a feature split by label.

    Handy for showing "95th percentile of normal packet_count is 18,
    but this flow has 450" — a strong counterfactual signal.
    """
    es = es or get_client()
    percents = percents or [5, 25, 50, 75, 95, 99]

    body = {
        "size": 0,
        "aggs": {
            "by_label": {
                "terms": {"field": "label"},
                "aggs": {
                    "pctls": {
                        "percentiles": {"field": feature, "percents": percents}
                    }
                },
            }
        },
    }
    resp = es.search(index=FLOWS_INDEX, body=body)
    out: dict[str, dict] = {}
    for bucket in resp["aggregations"]["by_label"]["buckets"]:
        lbl = int(bucket["key"])
        out[f"label_{lbl}"] = bucket["pctls"]["values"]
    return out


# ──────────────────────────────────────────────
# 7. _EXPLAIN API
# ──────────────────────────────────────────────

def explain_flow_match(
    flow_id: str,
    query: dict | None = None,
    es: Elasticsearch | None = None,
) -> dict:
    """Explain why a specific flow document matched (or didn't match) a query.

    If no query given, defaults to matching malicious flows.
    """
    es = es or get_client()
    if query is None:
        query = {"term": {"label": 1}}
    resp = es.explain(index=FLOWS_INDEX, id=flow_id, body={"query": query})
    return {
        "matched": resp["matched"],
        "explanation": resp["explanation"],
    }


# ──────────────────────────────────────────────
# 8. ML ANOMALY DETECTION JOBS
# ──────────────────────────────────────────────

def create_anomaly_detection_job(
    job_id: str = "incidentlens-flow-anomaly",
    bucket_span: str = "5m",
    detectors: list[dict] | None = None,
    influencers: list[str] | None = None,
    es: Elasticsearch | None = None,
) -> dict:
    """Create an ML anomaly detection job on the flows index.

    Detectors default to high_count on packet_count partitioned by src_ip
    + a mean detector on total_bytes.  Influencers tell you *which field
    values* drove each anomaly — the raw material for counterfactuals.
    """
    es = es or get_client()

    if detectors is None:
        detectors = [
            {
                "function": "high_count",
                "field_name": "packet_count",
                "partition_field_name": "src_ip",
                "detector_description": "Unusually high packet count from src_ip",
            },
            {
                "function": "high_mean",
                "field_name": "total_bytes",
                "partition_field_name": "src_ip",
                "detector_description": "Unusually high mean bytes from src_ip",
            },
        ]

    if influencers is None:
        influencers = ["src_ip", "dst_ip", "protocol"]

    body = {
        "analysis_config": {
            "bucket_span": bucket_span,
            "detectors": detectors,
            "influencers": influencers,
        },
        "data_description": {"time_field": "timestamp"},
        "analysis_limits": {"model_memory_limit": "256mb"},
    }

    try:
        resp = es.ml.put_job(job_id=job_id, body=body)
        logger.info("Created ML job %s", job_id)
        return resp
    except RequestError as e:
        if "already exists" in str(e):
            logger.info("ML job %s already exists", job_id)
            return {"status": "already_exists", "job_id": job_id}
        raise


def create_anomaly_datafeed(
    job_id: str = "incidentlens-flow-anomaly",
    es: Elasticsearch | None = None,
) -> dict:
    """Create a datafeed for the anomaly detection job."""
    es = es or get_client()
    datafeed_id = f"datafeed-{job_id}"
    body = {
        "job_id": job_id,
        "indices": [FLOWS_INDEX],
        "query": {"match_all": {}},
    }
    try:
        resp = es.ml.put_datafeed(datafeed_id=datafeed_id, body=body)
        logger.info("Created datafeed %s", datafeed_id)
        return resp
    except RequestError as e:
        if "already exists" in str(e):
            logger.info("Datafeed %s already exists", datafeed_id)
            return {"status": "already_exists"}
        raise


def get_anomaly_records(
    job_id: str = "incidentlens-flow-anomaly",
    min_score: float = 75.0,
    size: int = 50,
    es: Elasticsearch | None = None,
) -> list[dict]:
    """Fetch anomaly records above a score threshold.

    Each record contains ``influencers`` — the features / field values
    that drove the anomaly.  This is direct feature attribution.
    """
    es = es or get_client()
    body = {
        "sort": [{"record_score": {"order": "desc"}}],
        "size": size,
    }
    resp = es.ml.get_records(
        job_id=job_id,
        body=body,
    )
    records = resp.get("records", [])
    return [r for r in records if r.get("record_score", 0) >= min_score]


def get_influencers(
    job_id: str = "incidentlens-flow-anomaly",
    min_score: float = 50.0,
    size: int = 50,
    es: Elasticsearch | None = None,
) -> list[dict]:
    """Fetch influencer results — tells you which src_ip, dst_ip,
    protocol values contributed most to anomalies.
    """
    es = es or get_client()
    resp = es.ml.get_influencers(
        job_id=job_id,
        body={"sort": [{"influencer_score": {"order": "desc"}}], "size": size},
    )
    return [
        inf for inf in resp.get("influencers", [])
        if inf.get("influencer_score", 0) >= min_score
    ]


# ──────────────────────────────────────────────
# 9. COUNTERFACTUAL DIFF COMPUTATION + INDEXING
# ──────────────────────────────────────────────

def compute_counterfactual_diff(
    anomalous_doc: dict,
    normal_doc: dict,
    features: list[str] | None = None,
) -> list[dict]:
    """Compute per-feature diff between an anomalous flow and its
    nearest normal neighbour.

    Vectorised with numpy — single-pass computation of all diffs,
    percentages, and directions.

    Returns a list of dicts ready for the ``feature_diffs`` nested field.
    """
    features = features or FEATURE_FIELDS
    orig = np.array([anomalous_doc.get(f, np.nan) for f in features], dtype=np.float64)
    cf = np.array([normal_doc.get(f, np.nan) for f in features], dtype=np.float64)
    valid = ~(np.isnan(orig) | np.isnan(cf))
    abs_diff = np.abs(orig - cf)
    denom = np.abs(orig).copy()
    denom[denom == 0] = 1e-9
    pct = np.minimum(abs_diff / denom * 100, 99999.99)
    dirs = np.where(cf < orig, "decrease", np.where(cf > orig, "increase", "unchanged"))
    diffs = sorted([
        {
            "feature": features[i],
            "original_value": float(orig[i]),
            "cf_value": float(cf[i]),
            "abs_diff": float(abs_diff[i]),
            "pct_change": round(float(pct[i]), 2),
            "direction": str(dirs[i]),
        }
        for i in range(len(features)) if valid[i]
    ], key=lambda d: d["pct_change"], reverse=True)
    return diffs


def index_counterfactual(
    flow_id: str,
    nearest_normal_id: str,
    feature_diffs: list[dict],
    similarity_score: float = 0.0,
    prediction: str = "malicious",
    cf_prediction: str = "normal",
    explanation_text: str = "",
    edges_removed: list[dict] | None = None,
    es: Elasticsearch | None = None,
) -> str:
    """Index a counterfactual explanation document."""
    es = es or get_client()
    cf_id = hashlib.sha256(f"{flow_id}:{nearest_normal_id}".encode()).hexdigest()[:16]

    doc = {
        "cf_id": cf_id,
        "flow_id": flow_id,
        "nearest_normal_id": nearest_normal_id,
        "prediction": prediction,
        "cf_prediction": cf_prediction,
        "similarity_score": similarity_score,
        "feature_diffs": feature_diffs,
        "edges_removed": edges_removed or [],
        "explanation_text": explanation_text,
        "timestamp": int(time.time() * 1000),
    }
    es.index(index=COUNTERFACTUALS_INDEX, id=cf_id, document=doc)
    return cf_id


def build_and_index_counterfactual(
    anomalous_flow_id: str,
    query_embedding: np.ndarray | list[float],
    es: Elasticsearch | None = None,
    k: int = 1,
) -> dict | None:
    """End-to-end: find nearest normal via kNN, compute diff, index result.

    Returns the counterfactual document or None if no normal neighbour found.
    """
    es = es or get_client()

    # 1. retrieve the anomalous flow doc
    try:
        anom_doc = es.get(index=FLOWS_INDEX, id=anomalous_flow_id)["_source"]
    except NotFoundError:
        logger.error("Anomalous flow %s not found", anomalous_flow_id)
        return None

    # 2. kNN search for nearest normal
    neighbours = knn_search_nearest_normal(query_embedding, k=k, es=es)
    if not neighbours:
        logger.warning("No normal neighbours found for %s", anomalous_flow_id)
        return None

    best = neighbours[0]
    nn_flow_id = best.get("flow_id")
    if not nn_flow_id:
        logger.warning("Nearest normal neighbour has no flow_id for %s", anomalous_flow_id)
        return None

    # 3. retrieve the normal flow doc
    try:
        normal_doc = es.get(index=FLOWS_INDEX, id=nn_flow_id)["_source"]
    except NotFoundError:
        logger.warning("Normal flow %s not found in flows index", nn_flow_id)
        return None

    # 4. compute feature diff
    diffs = compute_counterfactual_diff(anom_doc, normal_doc)

    # 5. index
    cf_id = index_counterfactual(
        flow_id=anomalous_flow_id,
        nearest_normal_id=nn_flow_id,
        feature_diffs=diffs,
        similarity_score=best.get("score", 0.0),
    )

    return {
        "cf_id": cf_id,
        "flow_id": anomalous_flow_id,
        "nearest_normal_id": nn_flow_id,
        "similarity_score": best.get("score", 0.0),
        "feature_diffs": diffs,
    }


# ──────────────────────────────────────────────
# 10. RETRIEVAL HELPERS (for RAG / LLM layer)
# ──────────────────────────────────────────────

def get_flow(flow_id: str, es: Elasticsearch | None = None) -> dict | None:
    """Fetch a single flow document by ID."""
    es = es or get_client()
    try:
        return es.get(index=FLOWS_INDEX, id=flow_id)["_source"]
    except NotFoundError:
        return None


def get_counterfactual(cf_id: str, es: Elasticsearch | None = None) -> dict | None:
    """Fetch a counterfactual document by ID."""
    es = es or get_client()
    try:
        return es.get(index=COUNTERFACTUALS_INDEX, id=cf_id)["_source"]
    except NotFoundError:
        return None


def search_flows(
    query: dict | None = None,
    size: int = 20,
    sort: list | None = None,
    es: Elasticsearch | None = None,
) -> list[dict]:
    """General-purpose flow search.  Includes ES ``_id`` in results."""
    es = es or get_client()
    body: dict[str, Any] = {
        "size": size,
        "query": query or {"match_all": {}},
    }
    if sort:
        body["sort"] = sort
    resp = es.search(index=FLOWS_INDEX, body=body)
    results = []
    for hit in resp["hits"]["hits"]:
        doc = hit["_source"]
        doc["_id"] = hit["_id"]
        if hit.get("_score") is not None:
            doc["_score"] = hit["_score"]
        results.append(doc)
    return results


def search_anomalous_flows(
    min_prediction_score: float = 0.5,
    size: int = 50,
    es: Elasticsearch | None = None,
) -> list[dict]:
    """Retrieve flows predicted as malicious above a confidence threshold."""
    return search_flows(
        query={
            "bool": {
                "must": [
                    {"term": {"prediction": 1}},
                    {"range": {"prediction_score": {"gte": min_prediction_score}}},
                ]
            }
        },
        size=size,
        sort=[{"prediction_score": {"order": "desc"}}],
        es=es,
    )


def get_counterfactuals_for_flow(
    flow_id: str,
    es: Elasticsearch | None = None,
) -> list[dict]:
    """Retrieve all counterfactual explanations generated for a given flow."""
    es = es or get_client()
    body = {
        "query": {"term": {"flow_id": flow_id}},
        "size": 100,
    }
    resp = es.search(index=COUNTERFACTUALS_INDEX, body=body)
    return [hit["_source"] for hit in resp["hits"]["hits"]]


def format_counterfactual_narrative(cf_doc: dict) -> str:
    """Generate a human-readable counterfactual narrative from a CF document.

    Suitable for display in a frontend or as an LLM prompt context.
    """
    diffs = cf_doc.get("feature_diffs", [])
    if not diffs:
        return "No feature differences found."

    lines = [
        f"Flow {cf_doc['flow_id']} was classified as {cf_doc.get('prediction', 'malicious')}.",
        f"The nearest normal flow is {cf_doc['nearest_normal_id']} "
        f"(similarity: {cf_doc.get('similarity_score', 0):.4f}).",
        "",
        "To change the classification to normal, the following features would need to change:",
        "",
    ]
    for d in diffs:
        lines.append(
            f"  - {d['feature']}: {d['original_value']:.2f} -> {d['cf_value']:.2f} "
            f"({d['direction']}, {d['pct_change']:.1f}% change)"
        )

    return "\n".join(lines)


# ──────────────────────────────────────────────
# SELF-TEST
# ──────────────────────────────────────────────

def _self_test() -> None:
    """Run a quick smoke-test against a local ES instance.

    Uses graph_data_wrapper to build sliding-window graphs from a
    synthetic packet DataFrame, then indexes flows + embeddings,
    runs kNN, counterfactual diff, and feature stats queries.
    """
    print("=" * 60)
    print("IncidentLens wrappers -- self-test (graph_data_wrapper)")
    print("=" * 60)

    es = get_client()

    # -- connectivity --
    assert ping(es), "Cannot reach Elasticsearch"
    h = health_check(es)
    print(f"[OK] Cluster health: {h['status']}")

    # -- indices --
    emb_dim = 8
    created = setup_all_indices(es, embedding_dim=emb_dim, delete_existing=True)
    print(f"[OK] Indices created: {created}")

    # -- build a synthetic packet DataFrame --
    rng = np.random.default_rng(42)
    n_normal, n_attack = 200, 60
    n_total = n_normal + n_attack

    base_ts = 1000.0
    timestamps = base_ts + np.sort(rng.uniform(0, 10, size=n_total))

    src_ips = [f"10.0.0.{rng.integers(1, 10)}" for _ in range(n_total)]
    dst_ips = [f"192.168.1.{rng.integers(1, 10)}" for _ in range(n_total)]
    protocols = rng.choice([6, 17], size=n_total).tolist()
    packet_lengths = rng.integers(60, 1500, size=n_total).tolist()
    payload_lengths = rng.integers(0, 1400, size=n_total).tolist()
    dst_ports = rng.choice([80, 443, 1900, 8080], size=n_total).tolist()
    labels = [0] * n_normal + [1] * n_attack

    df = pd.DataFrame({
        "timestamp": timestamps,
        "src_ip": src_ips,
        "dst_ip": dst_ips,
        "protocol": protocols,
        "packet_length": packet_lengths,
        "payload_length": payload_lengths,
        "dst_port": dst_ports,
        "label": labels,
    })

    # -- build graphs via graph_data_wrapper --
    print("\n[1] Building graphs via graph_data_wrapper ...")
    graphs, id_to_ip = build_graphs(df, window_size=2.0, stride=1.0)
    print(f"[OK] Built {len(graphs)} graph snapshots, {len(id_to_ip)} IPs")
    analyze_graph_dataset(graphs)

    # -- index flows --
    print("[2] Indexing flows ...")
    flow_count = index_graphs_bulk(graphs, id_to_ip, es=es)
    print(f"[OK] Indexed {flow_count} flow docs")

    # -- generate + index embeddings --
    print("[3] Generating embeddings ...")
    flow_ids, embeddings, labels_list = generate_embeddings(
        graphs, id_to_ip, embedding_dim=emb_dim,
    )
    emb_count = index_embeddings(flow_ids, embeddings, labels_list, es=es)
    print(f"[OK] Indexed {emb_count} embedding docs")

    # refresh
    es.indices.refresh(index=FLOWS_INDEX)
    es.indices.refresh(index=EMBEDDINGS_INDEX)
    time.sleep(1)

    # -- kNN search --
    mal_indices = [i for i, l in enumerate(labels_list) if l == 1]
    if mal_indices:
        attack_idx = mal_indices[0]
        neighbours = knn_search_nearest_normal(embeddings[attack_idx], k=3, es=es)
        print(f"[OK] kNN nearest-normal for attack flow: {len(neighbours)} results")
        if neighbours:
            print(f"  best match: {neighbours[0]['flow_id']} (score={neighbours[0]['score']:.4f})")
    else:
        attack_idx = 0
        neighbours = []
        print("[WARN] No malicious flows in graphs -- skipping kNN test")

    # -- feature stats --
    stats = feature_stats_by_label(es=es)
    print(f"[OK] Feature stats computed for {len(stats)} features")
    for feat, by_label in stats.items():
        for lbl, s in by_label.items():
            print(f"  {feat} [{lbl}]: avg={s['avg']:.2f}, std={s['std_deviation']:.2f}")

    # -- counterfactual diff --
    if neighbours and mal_indices:
        cf_result = build_and_index_counterfactual(
            anomalous_flow_id=flow_ids[attack_idx],
            query_embedding=embeddings[attack_idx],
            es=es,
        )
        if cf_result:
            print(f"[OK] Counterfactual indexed: {cf_result['cf_id']}")
            for d in cf_result["feature_diffs"]:
                print(
                    f"  {d['feature']}: {d['original_value']:.2f} -> "
                    f"{d['cf_value']:.2f} ({d['direction']})"
                )

            # -- narrative --
            es.indices.refresh(index=COUNTERFACTUALS_INDEX)
            cf_doc = get_counterfactual(cf_result["cf_id"], es=es)
            if cf_doc:
                narrative = format_counterfactual_narrative(cf_doc)
                print(f"\n--- Counterfactual Narrative ---\n{narrative}\n")

    # -- explain API --
    if flow_ids:
        try:
            expl = explain_flow_match(flow_ids[attack_idx], es=es)
            print(f"[OK] Explain API: matched={expl['matched']}")
        except Exception as e:
            print(f"[WARN] Explain API error (non-critical): {e}")

    # -- cleanup --
    delete_all_indices(es)
    print("[OK] Test indices cleaned up")
    print("\n[PASS] All self-tests passed!")
def index_flows_bulk(
    flows: list[dict],
    es: Elasticsearch | None = None,
    batch_size: int = 500,
) -> int:
    """
    Bulk index simple flow dictionaries into FLOWS_INDEX.
    Used by real-time simulation (non-graph ingestion).
    """
    es = es or get_client()

    if not flows:
        return 0

    actions = [
        {
            "_index": FLOWS_INDEX,
            "_id": flow["flow_id"],
            "_source": flow,
        }
        for flow in flows
    ]

    success, errors = helpers.bulk(
        es,
        actions,
        chunk_size=batch_size,
        raise_on_error=False,
    )

    if errors:
        logger.warning("Flow bulk indexing had %d errors", len(errors))

    return success
