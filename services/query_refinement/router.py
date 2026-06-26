"""HTTP layer for the query refinement service: schemas + route wiring."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from . import service

router = APIRouter()


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


@router.post("/refine")
def refine(req: RefineRequest):
    return service.refine(req.dataset, req.query, spell=req.spell, expand=req.expand,
                          history_personalize=req.history_personalize,
                          history=req.history)


@router.post("/suggest")
def suggest(req: SuggestRequest):
    return service.suggest(req.dataset, req.query, history=req.history)
