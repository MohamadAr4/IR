"""Preprocessing business logic (framework-free).

Single responsibility: turn raw text into normalised tokens using the *same*
pipeline applied to the corpus. Both documents and queries go through here,
guaranteeing the term-space agreement required by requirement #4. Kept free of
FastAPI/pydantic so it can be reused or unit-tested in isolation.
"""
from __future__ import annotations

from ir_core.text.preprocessing import process


def process_text(text: str,
                 *,
                 do_stemming: bool = True,
                 do_lemmatize: bool = False,
                 remove_stopwords: bool = True) -> dict:
    """Run the canonical preprocessing pipeline and return tokens + count."""
    tokens = process(text, do_stemming=do_stemming, do_lemmatize=do_lemmatize,
                     remove_stopwords=remove_stopwords)
    return {"text": text, "tokens": tokens, "num_tokens": len(tokens)}
