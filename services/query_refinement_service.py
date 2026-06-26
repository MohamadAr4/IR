"""Query Refinement Service (SOA component, port 8005).

Owns requirement #5: spelling correction, WordNet synonym expansion,
query suggestion and history-based personalisation.

Run:  uvicorn services.query_refinement_service:app --port 8005
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ir_core.engine import ENGINE
from .common import make_app

app = make_app("query-refinement-service")


class RefineRequest(BaseModel):
    dataset: str
    query: str
    spell: bool = True
    expand: bool = True
    history_personalize: bool = True
    history: Optional[list[str]] = None


class SuggestRequest(BaseModel):
    dataset: str
    query: str
    history: Optional[list[str]] = None


@app.post("/refine")
def refine(req: RefineRequest):
    r = ENGINE.refine(req.dataset, req.query, do_spell=req.spell, do_expand=req.expand,
                      use_history=req.history_personalize, history=req.history)
    return r.to_dict()


@app.post("/suggest")
def suggest(req: SuggestRequest):
    return {"query": req.query,
            "suggestions": ENGINE.suggest(req.dataset, req.query, history=req.history)}
