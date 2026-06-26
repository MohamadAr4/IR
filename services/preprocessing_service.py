"""Preprocessing Service (SOA component, port 8001).

Single responsibility: turn raw text into normalised tokens using the *same*
pipeline applied to the corpus. Both documents and queries go through here,
guaranteeing the term-space agreement required by requirement #4.

Run:  uvicorn services.preprocessing_service:app --port 8001
"""
from __future__ import annotations

from pydantic import BaseModel

from ir_core.text.preprocessing import process
from .common import make_app

app = make_app("preprocessing-service")


class ProcessRequest(BaseModel):
    text: str
    do_stemming: bool = True
    do_lemmatize: bool = False
    remove_stopwords: bool = True


@app.post("/process")
def process_text(req: ProcessRequest):
    tokens = process(req.text, do_stemming=req.do_stemming,
                      do_lemmatize=req.do_lemmatize,
                      remove_stopwords=req.remove_stopwords)
    return {"text": req.text, "tokens": tokens, "num_tokens": len(tokens)}
