"""Uniform retriever interface so every model — lexical or dense — can be
driven the same way by the matching layer and the hybrid combiner.

A :class:`QueryContext` carries both the raw query string (needed by BERT) and
its processed tokens (needed by TF-IDF/BM25/Word2Vec). Each retriever pulls
whatever it needs, exposing:

    search(ctx, top_k)          -> ranked list[SearchResult]
    rescore(ctx, doc_ids)       -> {doc_id: score}     (for serial hybrid)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..index.inverted_index import InvertedIndex
from ..types import SearchResult
from .bm25 import BM25
from .tfidf import TfidfVSM
from .embeddings import EmbeddingRetriever


@dataclass
class QueryContext:
    raw: str
    tokens: list[str] = field(default_factory=list)
    bm25_k1: Optional[float] = None
    bm25_b: Optional[float] = None
    # Synonym-expansion terms (stemmed) and the weight they carry in the lexical
    # query vector relative to real query terms (1.0). Lets the lexical models
    # down-weight sense-blind WordNet synonyms so they don't cause topic drift.
    expansion_terms: set[str] = field(default_factory=set)
    expansion_weight: float = 1.0


class LexicalRetriever:
    """Wraps TF-IDF or BM25 (both consume tokens + the inverted index)."""

    def __init__(self, name: str, index: InvertedIndex):
        self.name = name
        self.index = index
        self.tfidf = TfidfVSM(index)
        self.bm25 = BM25(index)

    def search(self, ctx: QueryContext, top_k: int = 10) -> list[SearchResult]:
        if self.name == "tfidf":
            return self.tfidf.search(ctx.tokens, top_k=top_k,
                                     term_weights=self._term_weights(ctx))
        return self.bm25.search(ctx.tokens, top_k=top_k,
                                k1=ctx.bm25_k1, b=ctx.bm25_b,
                                term_weights=self._term_weights(ctx))

    def rescore(self, ctx: QueryContext, doc_ids: list[str]) -> dict[str, float]:
        rowids = self.index.rowids_for(doc_ids)
        if not rowids:
            return {}
        cand = set(rowids.values())
        tw = self._term_weights(ctx)
        if self.name == "tfidf":
            res = self.tfidf.search(ctx.tokens, top_k=len(cand), candidate_filter=cand,
                                    term_weights=tw)
        else:
            res = self.bm25.search(ctx.tokens, top_k=len(cand),
                                   k1=ctx.bm25_k1, b=ctx.bm25_b, candidate_filter=cand,
                                   term_weights=tw)
        return {r.doc_id: r.score for r in res}

    @staticmethod
    def _term_weights(ctx: QueryContext) -> Optional[dict[str, float]]:
        """Map each synonym-expansion term to its (reduced) weight; real query
        terms are left at the default 1.0. ``None`` when nothing is down-weighted."""
        if not ctx.expansion_terms or ctx.expansion_weight == 1.0:
            return None
        return {t: ctx.expansion_weight for t in ctx.expansion_terms}


class DenseRetriever:
    """Wraps an embedding model (BERT or Word2Vec)."""

    def __init__(self, name: str, retriever: EmbeddingRetriever):
        self.name = name
        self.retriever = retriever

    def search(self, ctx: QueryContext, top_k: int = 10) -> list[SearchResult]:
        return self.retriever.search(ctx.raw, top_k=top_k)

    def rescore(self, ctx: QueryContext, doc_ids: list[str]) -> dict[str, float]:
        return self.retriever.rescore(ctx.raw, doc_ids)
