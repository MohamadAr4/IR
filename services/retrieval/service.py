"""Retrieval business logic (framework-free).

Owns query matching & ranking (requirement #6): given a dataset, a model and a
query it returns the ranked documents. Delegates to the shared retrieval
:class:`~ir_core.engine.Engine`, which dispatches to TF-IDF / BM25 / BERT /
Word2Vec / hybrid. BM25's k1 and b are accepted per request (requirement #2
note 2), and hybrid options (parallel/serial, fusion, model set) pass through.
"""
from __future__ import annotations

from typing import Optional

from ir_core.engine import ENGINE


def models(dataset_key: str) -> dict:
    return {"dataset": dataset_key, "models": ENGINE.available_models(dataset_key)}


def search(dataset: str, model: str, query: str, *, top_k: int = 10,
           bm25_k1: Optional[float] = None, bm25_b: Optional[float] = None,
           refine_opts: Optional[dict] = None, history: Optional[list[str]] = None,
           hybrid_opts: Optional[dict] = None) -> dict:
    return ENGINE.search(dataset, model, query, top_k=top_k,
                         bm25_k1=bm25_k1, bm25_b=bm25_b,
                         refine_opts=refine_opts, history=history,
                         hybrid_opts=hybrid_opts)
