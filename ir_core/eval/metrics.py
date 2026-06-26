"""Standard IR effectiveness metrics (requirement #8).

All functions take a ranked list of doc_ids (best first) and the query's
relevance judgements ``qrel = {doc_id: relevance}``. Binary metrics treat any
relevance > 0 as relevant; nDCG uses the graded relevance as the gain.
"""
from __future__ import annotations

import math


def precision_at_k(ranked: list[str], qrel: dict[str, int], k: int = 10) -> float:
    if k <= 0:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for d in topk if qrel.get(d, 0) > 0)
    return hits / k


def recall(ranked: list[str], qrel: dict[str, int], k: int | None = None) -> float:
    total_rel = sum(1 for r in qrel.values() if r > 0)
    if total_rel == 0:
        return 0.0
    pool = ranked[:k] if k else ranked
    hits = sum(1 for d in pool if qrel.get(d, 0) > 0)
    return hits / total_rel


def average_precision(ranked: list[str], qrel: dict[str, int], k: int | None = None) -> float:
    total_rel = sum(1 for r in qrel.values() if r > 0)
    if total_rel == 0:
        return 0.0
    pool = ranked[:k] if k else ranked
    hits = 0
    ap = 0.0
    for i, d in enumerate(pool, 1):
        if qrel.get(d, 0) > 0:
            hits += 1
            ap += hits / i
    return ap / total_rel


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, 1))


def ndcg_at_k(ranked: list[str], qrel: dict[str, int], k: int = 10) -> float:
    gains = [float(qrel.get(d, 0)) for d in ranked[:k]]
    ideal = sorted((float(v) for v in qrel.values() if v > 0), reverse=True)[:k]
    idcg = dcg(ideal)
    return (dcg(gains) / idcg) if idcg > 0 else 0.0


def evaluate_query(ranked: list[str], qrel: dict[str, int],
                   p_at: int = 10, ndcg_at: int = 10,
                   ap_depth: int | None = None,
                   recall_depth: int | None = None) -> dict[str, float]:
    return {
        "P@10": precision_at_k(ranked, qrel, p_at),
        "Recall": recall(ranked, qrel, recall_depth),
        "AP": average_precision(ranked, qrel, ap_depth),
        "nDCG@10": ndcg_at_k(ranked, qrel, ndcg_at),
    }


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Mean each metric across queries. ``AP`` averaged becomes ``MAP``."""
    if not per_query:
        return {"MAP": 0.0, "Recall": 0.0, "P@10": 0.0, "nDCG@10": 0.0, "num_queries": 0}
    n = len(per_query)
    return {
        "MAP": sum(q["AP"] for q in per_query) / n,
        "Recall": sum(q["Recall"] for q in per_query) / n,
        "P@10": sum(q["P@10"] for q in per_query) / n,
        "nDCG@10": sum(q["nDCG@10"] for q in per_query) / n,
        "num_queries": n,
    }
