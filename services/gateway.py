"""API Gateway (SOA component, port 8000).

The single public entry point (Gateway / Facade pattern). It routes each call
to the responsible downstream service over REST and composes them when needed
(e.g. /search_refined = refinement-service then retrieval-service). Clients —
including the Streamlit UI — only ever talk to the gateway, so services stay
loosely coupled and independently deployable.

Run:  uvicorn services.gateway:app --port 8000
"""
from __future__ import annotations

from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

from ir_core.config import SERVICE_URLS
from .common import get_json, make_app, post_json

app = make_app("api-gateway")
S = SERVICE_URLS


def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:  # surface downstream failures as 502
        raise HTTPException(status_code=502, detail=f"downstream error: {e}")


# -- discovery ---------------------------------------------------------------
@app.get("/services")
def services_health():
    out = {}
    for name, url in S.items():
        if name == "gateway":
            continue
        try:
            out[name] = {"url": url, **get_json(f"{url}/health", timeout=3)}
        except Exception as e:
            out[name] = {"url": url, "status": "down", "error": str(e)}
    return out


@app.get("/datasets")
def datasets():
    return _safe(get_json, f"{S['indexing']}/datasets")


@app.get("/status/{dataset_key}")
def status(dataset_key: str):
    return _safe(get_json, f"{S['indexing']}/status/{dataset_key}")


@app.get("/models/{dataset_key}")
def models(dataset_key: str):
    return _safe(get_json, f"{S['retrieval']}/models/{dataset_key}")


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


# -- routed endpoints --------------------------------------------------------
@app.post("/search")
def search(req: SearchRequest):
    return _safe(post_json, f"{S['retrieval']}/search", req.model_dump())


@app.post("/refine")
def refine(req: RefineRequest):
    return _safe(post_json, f"{S['refinement']}/refine", req.model_dump())


@app.post("/suggest")
def suggest(req: RefineRequest):
    return _safe(post_json, f"{S['refinement']}/suggest", req.model_dump())


@app.post("/evaluate")
def evaluate(req: EvalRequest):
    return _safe(post_json, f"{S['ranking_eval']}/evaluate", req.model_dump())


@app.post("/compare")
def compare(req: EvalRequest):
    return _safe(post_json, f"{S['ranking_eval']}/compare", req.model_dump())


# -- composition: refine THEN retrieve --------------------------------------
@app.post("/search_refined")
def search_refined(req: SearchRequest):
    """Orchestrates two services: refine the query, then search with it. Shows
    the gateway composing capabilities the client would otherwise wire up."""
    refine_payload = {"dataset": req.dataset, "query": req.query,
                      "spell": True, "expand": True, "history_personalize": True,
                      "history": req.history}
    refinement = _safe(post_json, f"{S['refinement']}/refine", refine_payload)
    search_req = req.model_dump()
    search_req["query"] = refinement.get("refined_raw") or req.query
    result = _safe(post_json, f"{S['retrieval']}/search", search_req)
    result["refinement"] = refinement
    return result
