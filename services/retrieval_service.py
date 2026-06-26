"""Retrieval Service (SOA component, port 8003).

Owns query matching & ranking (requirement #6): given a dataset, a model and a
query it returns the ranked documents. It delegates to the shared retrieval
:class:`~ir_core.engine.Engine`, which dispatches to TF-IDF / BM25 / BERT /
Word2Vec / hybrid. BM25's k1 and b are accepted per request (requirement #2
note 2), and hybrid options (parallel/serial, fusion, model set) are passed
through.

Run:  uvicorn services.retrieval_service:app --port 8003
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ir_core.engine import ENGINE
from .common import make_app

app = make_app("retrieval-service")


class SearchRequest(BaseModel):
    dataset: str
    model: str = "bm25"
    query: str
    top_k: int = 10
    bm25_k1: Optional[float] = None
    bm25_b: Optional[float] = None
    refine_opts: Optional[dict] = None
    history: Optional[list[str]] = None
    hybrid_opts: Optional[dict] = None


@app.get("/models/{dataset_key}")
def models(dataset_key: str):
    return {"dataset": dataset_key, "models": ENGINE.available_models(dataset_key)}


@app.post("/search")
def search(req: SearchRequest):
    return ENGINE.search(req.dataset, req.model, req.query, top_k=req.top_k,
                         bm25_k1=req.bm25_k1, bm25_b=req.bm25_b,
                         refine_opts=req.refine_opts, history=req.history,
                         hybrid_opts=req.hybrid_opts)
