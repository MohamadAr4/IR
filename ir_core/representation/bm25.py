"""BM25 / Okapi probabilistic model (requirement #2, model 3).

Scores over the same disk-backed inverted index as TF-IDF. The two free
parameters are exposed *per query* (requirement note #2): ``k1`` controls term
-frequency saturation and ``b`` controls length normalisation. The UI lets the
user change them before each search, and the report explains sensible defaults
(k1=1.5, b=0.75 — the values that perform robustly across TREC collections).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Optional

from ..config import BM25Params
from ..index.inverted_index import InvertedIndex
from ..types import SearchResult


class BM25:
    name = "bm25"

    def __init__(self, index: InvertedIndex, params: Optional[BM25Params] = None):
        self.index = index
        self.params = params or BM25Params()

    @staticmethod
    def _bm25_idf(N: int, df: int) -> float:
        # Robertson/Sparck-Jones idf with +1 to keep it non-negative.
        return math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: list[str], top_k: int = 10,
               k1: Optional[float] = None, b: Optional[float] = None,
               candidate_filter: Optional[set[int]] = None) -> list[SearchResult]:
        k1 = self.params.k1 if k1 is None else k1
        b = self.params.b if b is None else b
        N = self.index.num_docs
        avgdl = self.index.avgdl or 1.0

        scores: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            df = self.index.df(term)
            if df == 0:
                continue
            idf = self._bm25_idf(N, df)
            for p in self.index.postings(term):
                if candidate_filter is not None and p.doc_rowid not in candidate_filter:
                    continue
                denom = p.tf + k1 * (1.0 - b + b * (p.length / avgdl))
                scores[p.doc_rowid] += idf * (p.tf * (k1 + 1.0)) / denom

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        meta = self.index.get_docs([r for r, _ in ranked])
        out = []
        for rank, (rowid, score) in enumerate(ranked, 1):
            m = meta.get(rowid, {})
            out.append(SearchResult(doc_id=m.get("doc_id", str(rowid)), score=float(score),
                                    rank=rank, raw_text=m.get("raw_text", "")))
        return out
