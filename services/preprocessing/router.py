"""HTTP layer for the preprocessing service: request schema + route wiring."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from . import service

router = APIRouter()


class ProcessRequest(BaseModel):
    text: str
    do_stemming: bool = True
    do_lemmatize: bool = False
    remove_stopwords: bool = True


@router.post("/process")
def process_text(req: ProcessRequest):
    return service.process_text(req.text, do_stemming=req.do_stemming,
                                do_lemmatize=req.do_lemmatize,
                                remove_stopwords=req.remove_stopwords)
