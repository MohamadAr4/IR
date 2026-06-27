"""Central configuration: dataset registry, on-disk paths and default params.

Everything that needs to know "where do the indexes live" or "what is the
ir_datasets id for this corpus" reads it from here, so the rest of the code
never hard-codes a path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Root under which every persisted index/artifact for every dataset lives.
INDEX_ROOT = os.environ.get("IR_INDEX_ROOT", os.path.join(PROJECT_ROOT, "indexes"))


@dataclass(frozen=True)
class DatasetSpec:
    """Describes one corpus: where its preprocessed docs are and how to get
    its evaluation queries/qrels from ir_datasets."""

    key: str                      # short id used in URLs / CLI / UI
    name: str                     # human friendly name
    processed_json: str           # path to the preprocessed docs JSON (array of records)
    ir_datasets_id: str           # id understood by ir_datasets (queries + qrels)
    # attribute names to try, in order, when pulling the query text from an
    # ir_datasets query object (different corpora expose different fields).
    query_text_attrs: tuple = ("text", "title", "description", "query")

    @property
    def index_dir(self) -> str:
        return os.path.join(INDEX_ROOT, self.key)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASETS: dict[str, DatasetSpec] = {
    "argsme": DatasetSpec(
        key="argsme",
        name="Args.me (Touché 2020 Task 1)",
        processed_json=os.path.join(PROJECT_ROOT, "argsme_processed.json"),
        ir_datasets_id="argsme/1.0/touche-2020-task-1/uncorrected",
        query_text_attrs=("title", "text", "description", "query"),
    ),
}

# A tiny sample corpus, handy for fast smoke tests / demos without ir_datasets.
SAMPLE = DatasetSpec(
    key="argsme_sample",
    name="Args.me sample (5 docs)",
    processed_json=os.path.join(PROJECT_ROOT, "argsme_sample_processed.json"),
    ir_datasets_id="argsme/1.0/touche-2020-task-1/uncorrected",
    query_text_attrs=("title", "text"),
)


def get_dataset(key: str) -> DatasetSpec:
    if key == SAMPLE.key:
        return SAMPLE
    try:
        return DATASETS[key]
    except KeyError:
        raise KeyError(f"Unknown dataset '{key}'. Known: {list(DATASETS)} (+ {SAMPLE.key})")


def list_datasets() -> list[dict]:
    specs = list(DATASETS.values()) + [SAMPLE]
    return [{"key": s.key, "name": s.name, "available": os.path.exists(s.processed_json)}
            for s in specs]


# ---------------------------------------------------------------------------
# Default model / retrieval parameters
# ---------------------------------------------------------------------------
@dataclass
class BM25Params:
    k1: float = 1.5
    b: float = 0.75


# sentence-transformers model used for the BERT representation.
BERT_MODEL_NAME = os.environ.get("IR_BERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Multilingual sentence encoder (~50 languages, shared vector space) for the
# cross-lingual retrieval model: a query in any supported language retrieves the
# (English) documents because both map into the same embedding space.
MULTILINGUAL_MODEL_NAME = os.environ.get(
    "IR_MULTILINGUAL_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Word2Vec (PyTorch skip-gram) hyper-parameters.
@dataclass
class Word2VecParams:
    dim: int = 100
    window: int = 5
    min_count: int = 2
    negatives: int = 5
    epochs: int = 3
    batch_size: int = 1024
    lr: float = 0.025
    max_vocab: int = 50000


# Weight applied to synonym-expansion terms in the lexical query vector,
# relative to the user's original query terms (1.0). Synonyms from WordNet are
# sense-blind and easily cause topic drift, so they contribute less than the
# real query terms. Tune in [0, 1]; 0 disables their effect, 1 = full weight.
SYNONYM_EXPANSION_WEIGHT = float(os.environ.get("IR_SYNONYM_EXPANSION_WEIGHT", "0.4"))

DEFAULT_TOP_K = 10

# Microservice endpoints (host:port). Overridable via env so the gateway can
# find each service in any deployment.
SERVICE_URLS = {
    "preprocessing": os.environ.get("IR_PREPROCESSING_URL", "http://127.0.0.1:8001"),
    "indexing":      os.environ.get("IR_INDEXING_URL",      "http://127.0.0.1:8002"),
    "retrieval":     os.environ.get("IR_RETRIEVAL_URL",     "http://127.0.0.1:8003"),
    "ranking_eval":  os.environ.get("IR_RANKING_EVAL_URL",  "http://127.0.0.1:8004"),
    "refinement":    os.environ.get("IR_REFINEMENT_URL",    "http://127.0.0.1:8005"),
    "gateway":       os.environ.get("IR_GATEWAY_URL",       "http://127.0.0.1:8000"),
}


def ensure_index_dir(dataset_key: str) -> str:
    d = get_dataset(dataset_key).index_dir
    os.makedirs(d, exist_ok=True)
    return d
