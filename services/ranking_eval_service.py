"""Ranking & Evaluation Service (SOA component, port 8004).

Owns offline effectiveness measurement (requirement #8): MAP, Recall, P@10 and
nDCG@10 over a dataset's qrels, and the before/after-refinement comparison.

Run:  uvicorn services.ranking_eval_service:app --port 8004
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ir_core.eval.evaluate import compare_refinement, evaluate_model
from .common import make_app

app = make_app("ranking-eval-service")


class EvalRequest(BaseModel):
    dataset: str
    model: str = "bm25"
    num_queries: Optional[int] = 50
    eval_depth: int = 100
    bm25_k1: Optional[float] = None
    bm25_b: Optional[float] = None
    refine_opts: Optional[dict] = None
    hybrid_opts: Optional[dict] = None


@app.post("/evaluate")
def evaluate(req: EvalRequest):
    return evaluate_model(req.dataset, req.model, num_queries=req.num_queries,
                          eval_depth=req.eval_depth, bm25_k1=req.bm25_k1,
                          bm25_b=req.bm25_b, refine_opts=req.refine_opts,
                          hybrid_opts=req.hybrid_opts)


@app.post("/compare")
def compare(req: EvalRequest):
    """Evaluate the model with refinement OFF then ON (requirement #8)."""
    return compare_refinement(req.dataset, req.model, num_queries=req.num_queries,
                              eval_depth=req.eval_depth, bm25_k1=req.bm25_k1,
                              bm25_b=req.bm25_b, hybrid_opts=req.hybrid_opts,
                              refine_opts=req.refine_opts)
