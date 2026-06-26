"""HTTP layer for the retrieval service: request schema + route wiring."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from . import service

router = APIRouter()


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


@router.get("/models/{dataset_key}")
def models(dataset_key: str):
    return service.models(dataset_key)


@router.post("/search")
def search(req: SearchRequest):
    return service.search(req.dataset, req.model, req.query, top_k=req.top_k,
                          bm25_k1=req.bm25_k1, bm25_b=req.bm25_b,
                          refine_opts=req.refine_opts, history=req.history,
                          hybrid_opts=req.hybrid_opts)
