"""HTTP layer for the API gateway: request schemas + route wiring.

Mirrors the downstream request models so the gateway is the single public
contract clients code against.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from . import service

router = APIRouter()


# -- request models ----------------------------------------------------------
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


class RefineRequest(BaseModel):
    dataset: str
    query: str
    spell: bool = True
    expand: bool = True
    history_personalize: bool = True
    history: Optional[list[str]] = None


class EvalRequest(BaseModel):
    dataset: str
    model: str = "bm25"
    num_queries: Optional[int] = 50
    eval_depth: int = 100
    bm25_k1: Optional[float] = None
    bm25_b: Optional[float] = None
    refine_opts: Optional[dict] = None
    hybrid_opts: Optional[dict] = None


# -- discovery ---------------------------------------------------------------
@router.get("/services")
def services_health():
    return service.services_health()


@router.get("/datasets")
def datasets():
    return service.datasets()


@router.get("/status/{dataset_key}")
def status(dataset_key: str):
    return service.status(dataset_key)


@router.get("/models/{dataset_key}")
def models(dataset_key: str):
    return service.models(dataset_key)


# -- routed endpoints --------------------------------------------------------
@router.post("/search")
def search(req: SearchRequest):
    return service.search(req.model_dump())


@router.post("/refine")
def refine(req: RefineRequest):
    return service.refine(req.model_dump())


@router.post("/suggest")
def suggest(req: RefineRequest):
    return service.suggest(req.model_dump())


@router.post("/evaluate")
def evaluate(req: EvalRequest):
    return service.evaluate(req.model_dump())


@router.post("/compare")
def compare(req: EvalRequest):
    return service.compare(req.model_dump())


@router.post("/search_refined")
def search_refined(req: SearchRequest):
    return service.search_refined(req.model_dump())
