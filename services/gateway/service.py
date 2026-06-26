"""API Gateway business logic (framework-free).

Routes each call to the responsible downstream service over REST and composes
them when needed (e.g. :func:`search_refined` = refinement then retrieval).
Clients only ever talk to the gateway, so services stay loosely coupled and
independently deployable (Gateway / Facade pattern).
"""
from __future__ import annotations

from fastapi import HTTPException

from ir_core.config import SERVICE_URLS
from services.common import get_json, post_json

S = SERVICE_URLS


def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:  # surface downstream failures as 502
        raise HTTPException(status_code=502, detail=f"downstream error: {e}")


# -- discovery ---------------------------------------------------------------
def services_health() -> dict:
    out = {}
    for name, url in S.items():
        if name == "gateway":
            continue
        try:
            out[name] = {"url": url, **get_json(f"{url}/health", timeout=3)}
        except Exception as e:
            out[name] = {"url": url, "status": "down", "error": str(e)}
    return out


def datasets():
    return _safe(get_json, f"{S['indexing']}/datasets")


def status(dataset_key: str):
    return _safe(get_json, f"{S['indexing']}/status/{dataset_key}")


def models(dataset_key: str):
    return _safe(get_json, f"{S['retrieval']}/models/{dataset_key}")


# -- routed endpoints --------------------------------------------------------
def search(payload: dict):
    return _safe(post_json, f"{S['retrieval']}/search", payload)


def refine(payload: dict):
    return _safe(post_json, f"{S['refinement']}/refine", payload)


def suggest(payload: dict):
    return _safe(post_json, f"{S['refinement']}/suggest", payload)


def evaluate(payload: dict):
    return _safe(post_json, f"{S['ranking_eval']}/evaluate", payload)


def compare(payload: dict):
    return _safe(post_json, f"{S['ranking_eval']}/compare", payload)


# -- composition: refine THEN retrieve --------------------------------------
def search_refined(payload: dict):
    """Orchestrates two services: refine the query, then search with it. Shows
    the gateway composing capabilities the client would otherwise wire up."""
    refine_payload = {"dataset": payload["dataset"], "query": payload["query"],
                      "spell": True, "expand": True, "history_personalize": True,
                      "history": payload.get("history")}
    refinement = _safe(post_json, f"{S['refinement']}/refine", refine_payload)
    search_req = dict(payload)
    search_req["query"] = refinement.get("refined_raw") or payload["query"]
    result = _safe(post_json, f"{S['retrieval']}/search", search_req)
    result["refinement"] = refinement
    return result
