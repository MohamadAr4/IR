"""VSM with TF-IDF weighting + cosine similarity (requirement #2, model 1).

Scoring follows the classic ``ltc`` scheme on both sides:
    weight(t) = (1 + ln tf) * idf(t)
and similarity = cosine(query_vector, doc_vector).

The document vectors are never materialised in full; we accumulate the dot
product term-by-term over the inverted-index postings, then divide by the
precomputed per-document L2 norm and the query norm.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Optional

from ..index.inverted_index import InvertedIndex
from ..types import SearchResult


class TfidfVSM:
    name = "tfidf"

    def __init__(self, index: InvertedIndex):
        self.index = index

    def _query_weights(self, query_tokens: list[str],
                       term_weights: Optional[dict[str, float]] = None) -> dict[str, float]:
        qtf = Counter(query_tokens)
        weights: dict[str, float] = {}
        for term, tf in qtf.items():
            idf = self.index.idf(term)
            if idf <= 0:
                continue
            mult = term_weights.get(term, 1.0) if term_weights else 1.0
            weights[term] = (1.0 + math.log(tf)) * idf * mult
        return weights

    def search(self, query_tokens: list[str], top_k: int = 10,
               candidate_filter: Optional[set[int]] = None,
               term_weights: Optional[dict[str, float]] = None) -> list[SearchResult]:
        qw = self._query_weights(query_tokens, term_weights)
        if not qw:
            return []
        q_norm = math.sqrt(sum(w * w for w in qw.values())) or 1.0

        scores: dict[int, float] = defaultdict(float)
        doc_norm: dict[int, float] = {}
        for term, wq in qw.items():
            idf = self.index.idf(term)
            for p in self.index.postings(term):
                if candidate_filter is not None and p.doc_rowid not in candidate_filter:
                    continue
                wd = (1.0 + math.log(p.tf)) * idf
                scores[p.doc_rowid] += wq * wd
                doc_norm[p.doc_rowid] = p.tfidf_norm  # carried on the posting

        # normalise by doc norm * query norm -> cosine
        result = []
        for rowid, dot in scores.items():
            dn = doc_norm.get(rowid, 0.0)
            cos = dot / (dn * q_norm) if dn else 0.0
            result.append((rowid, cos))

        result.sort(key=lambda x: x[1], reverse=True)
        top = result[:top_k]
        meta = self.index.get_docs([r for r, _ in top])
        out = []
        for rank, (rowid, score) in enumerate(top, 1):
            m = meta.get(rowid, {})
            out.append(SearchResult(doc_id=m.get("doc_id", str(rowid)), score=float(score),
                                    rank=rank, raw_text=m.get("raw_text", "")))
        return out
