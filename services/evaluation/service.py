"""Ranking & evaluation business logic (framework-free).

Owns offline effectiveness measurement (requirement #8): MAP, Recall, P@10 and
nDCG@10 over a dataset's qrels, and the before/after-refinement comparison.
"""
from __future__ import annotations

from typing import Optional

from ir_core.eval.evaluate import compare_refinement, evaluate_model


def evaluate(dataset: str, model: str, *, num_queries: Optional[int] = 50,
             eval_depth: int = 100, bm25_k1: Optional[float] = None,
             bm25_b: Optional[float] = None, refine_opts: Optional[dict] = None,
             hybrid_opts: Optional[dict] = None) -> dict:
    return evaluate_model(dataset, model, num_queries=num_queries,
                          eval_depth=eval_depth, bm25_k1=bm25_k1,
                          bm25_b=bm25_b, refine_opts=refine_opts,
                          hybrid_opts=hybrid_opts)


def compare(dataset: str, model: str, *, num_queries: Optional[int] = 50,
            eval_depth: int = 100, bm25_k1: Optional[float] = None,
            bm25_b: Optional[float] = None, refine_opts: Optional[dict] = None,
            hybrid_opts: Optional[dict] = None) -> dict:
    """Evaluate the model with refinement OFF then ON (requirement #8)."""
    return compare_refinement(dataset, model, num_queries=num_queries,
                              eval_depth=eval_depth, bm25_k1=bm25_k1,
                              bm25_b=bm25_b, hybrid_opts=hybrid_opts,
                              refine_opts=refine_opts)
