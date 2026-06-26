"""HTTP layer for the ranking & evaluation service: schema + route wiring."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from . import service

router = APIRouter()


class EvalRequest(BaseModel):
    dataset: str
    model: str = "bm25"
    num_queries: Optional[int] = 50
    eval_depth: int = 100
    bm25_k1: Optional[float] = None
    bm25_b: Optional[float] = None
    refine_opts: Optional[dict] = None
    hybrid_opts: Optional[dict] = None


@router.post("/evaluate")
def evaluate(req: EvalRequest):
    return service.evaluate(req.dataset, req.model, num_queries=req.num_queries,
                            eval_depth=req.eval_depth, bm25_k1=req.bm25_k1,
                            bm25_b=req.bm25_b, refine_opts=req.refine_opts,
                            hybrid_opts=req.hybrid_opts)


@router.post("/compare")
def compare(req: EvalRequest):
    return service.compare(req.dataset, req.model, num_queries=req.num_queries,
                           eval_depth=req.eval_depth, bm25_k1=req.bm25_k1,
                           bm25_b=req.bm25_b, refine_opts=req.refine_opts,
                           hybrid_opts=req.hybrid_opts)
