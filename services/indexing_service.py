"""Indexing Service (SOA component, port 8002).

Owns the inverted index and the vector stores: reports build status, exposes
term/vocabulary lookups, and can trigger a (capped) build. Heavy full-corpus
builds are normally done offline via scripts/build_indexes.py; the /build
endpoint here is meant for small/demo limits.

Run:  uvicorn services.indexing_service:app --port 8002
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ir_core.config import get_dataset, list_datasets
from ir_core.index.inverted_index import InvertedIndex
from ir_core.index.vector_store import VectorStore
from ir_core.representation.word2vec import Word2VecModel
from .common import make_app

app = make_app("indexing-service")


@app.get("/datasets")
def datasets():
    return {"datasets": list_datasets()}


@app.get("/status/{dataset_key}")
def status(dataset_key: str):
    spec = get_dataset(dataset_key)
    idx = InvertedIndex(spec)
    info = {"dataset": dataset_key, "inverted_index": idx.exists(),
            "bert": VectorStore(spec, "bert").exists(),
            "word2vec": VectorStore(spec, "word2vec").exists() and Word2VecModel.exists(spec)}
    if idx.exists():
        info["num_docs"] = idx.num_docs
        info["avgdl"] = round(idx.avgdl, 2)
    return info


@app.get("/vocab/{dataset_key}")
def vocab(dataset_key: str, prefix: str = "", limit: int = 20):
    spec = get_dataset(dataset_key)
    idx = InvertedIndex(spec)
    if not idx.exists():
        return {"dataset": dataset_key, "terms": [], "error": "index not built"}
    return {"dataset": dataset_key, "terms": idx.vocab_sample(prefix, limit=limit)}


class BuildRequest(BaseModel):
    dataset_key: str
    limit: Optional[int] = 5000   # cap for the online endpoint


@app.post("/build")
def build(req: BuildRequest):
    spec = get_dataset(req.dataset_key)
    meta = InvertedIndex(spec).build(limit=req.limit)
    return {"dataset": req.dataset_key, "built": True, **meta}
