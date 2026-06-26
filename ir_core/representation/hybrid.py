"""Hybrid representation (requirement #2, model 4) — implemented BOTH ways
(requirement note #3: "apply the hybrid twice, once parallel and once serial,
with a UI option").

Parallel
--------
Every base model scores the query independently, then the ranked lists are
combined with a **fusion method** (requirement note #1). Two are provided:

* ``weighted``  - min-max normalise each model's scores to [0,1], then take a
                  weighted sum (weights default to equal).
* ``rrf``       - Reciprocal Rank Fusion: score = Σ 1/(k + rank).

Serial
------
A cheap, high-recall lexical model (stage 1, e.g. BM25) produces a candidate
pool, then an expensive semantic model (stage 2, e.g. BERT) re-ranks only those
candidates. This is the classic retrieve-then-rerank cascade.

Both keep per-model component scores on each :class:`SearchResult` so the UI /
report can show exactly how a document's final score was assembled.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from ..types import SearchResult
from .base import QueryContext


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def parallel_hybrid(retrievers: dict, ctx: QueryContext, top_k: int = 10,
                    fusion: str = "weighted", weights: Optional[dict] = None,
                    pool: int = 100) -> list[SearchResult]:
    """Run each model independently and fuse. ``retrievers`` maps name->retriever."""
    weights = weights or {name: 1.0 for name in retrievers}
    per_model_results: dict[str, list[SearchResult]] = {}
    raw_text: dict[str, str] = {}
    for name, r in retrievers.items():
        res = r.search(ctx, top_k=pool)
        per_model_results[name] = res
        for item in res:
            if item.raw_text:
                raw_text.setdefault(item.doc_id, item.raw_text)

    fused: dict[str, float] = defaultdict(float)
    components: dict[str, dict] = defaultdict(dict)

    if fusion == "rrf":
        rrf_k = 60
        for name, res in per_model_results.items():
            w = weights.get(name, 1.0)
            for item in res:
                contrib = w * (1.0 / (rrf_k + item.rank))
                fused[item.doc_id] += contrib
                components[item.doc_id][f"{name}_rank"] = item.rank
    else:  # weighted sum of min-max normalised scores
        for name, res in per_model_results.items():
            w = weights.get(name, 1.0)
            norm = _min_max({item.doc_id: item.score for item in res})
            raw = {item.doc_id: item.score for item in res}
            for doc_id, ns in norm.items():
                fused[doc_id] += w * ns
                components[doc_id][name] = raw[doc_id]

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    out = []
    for rank, (doc_id, score) in enumerate(ranked, 1):
        out.append(SearchResult(doc_id=doc_id, score=float(score), rank=rank,
                                raw_text=raw_text.get(doc_id, ""),
                                components=dict(components[doc_id])))
    return out


def serial_hybrid(stage1, stage2, ctx: QueryContext, top_k: int = 10,
                  candidate_k: int = 100) -> list[SearchResult]:
    """Stage 1 retrieves ``candidate_k`` docs; stage 2 re-ranks them. Final order
    is by the stage-2 score (retrieve-then-rerank)."""
    candidates = stage1.search(ctx, top_k=candidate_k)
    if not candidates:
        return []
    cand_ids = [c.doc_id for c in candidates]
    stage1_score = {c.doc_id: c.score for c in candidates}
    raw_text = {c.doc_id: c.raw_text for c in candidates}

    stage2_score = stage2.rescore(ctx, cand_ids)
    # docs stage 2 can't score (e.g. missing vector) keep a tiny stage-1 tiebreak
    n1 = _min_max(stage1_score)
    n2 = _min_max(stage2_score) if stage2_score else {}

    final: dict[str, float] = {}
    components: dict[str, dict] = {}
    for doc_id in cand_ids:
        s2 = n2.get(doc_id)
        if s2 is None:
            s2 = 0.0
        final[doc_id] = s2 + 1e-3 * n1.get(doc_id, 0.0)
        components[doc_id] = {
            f"{stage1.name}_stage1": stage1_score.get(doc_id, 0.0),
            f"{stage2.name}_stage2": stage2_score.get(doc_id, 0.0),
        }

    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)[:top_k]
    out = []
    for rank, (doc_id, score) in enumerate(ranked, 1):
        out.append(SearchResult(doc_id=doc_id, score=float(score), rank=rank,
                                raw_text=raw_text.get(doc_id, ""),
                                components=components.get(doc_id, {})))
    return out
