"""Query processing (requirement #4).

Queries are normalised with the *exact same* pipeline as the documents
(:func:`ir_core.text.preprocessing.process`) so query terms and index terms
live in one term space. This module just wraps that and packages the result
into a :class:`QueryContext` for the retrievers.
"""
from __future__ import annotations

from typing import Optional

from ..representation.base import QueryContext
from ..text.preprocessing import process


def build_context(raw_query: str,
                  bm25_k1: Optional[float] = None,
                  bm25_b: Optional[float] = None) -> QueryContext:
    return QueryContext(raw=raw_query or "",
                        tokens=process(raw_query or ""),
                        bm25_k1=bm25_k1, bm25_b=bm25_b)
