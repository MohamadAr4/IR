"""Evaluation runner (requirement #8).

Runs a model over a dataset's queries, scores each against the qrels, and
aggregates to MAP / Recall / P@10 / nDCG@10. ``compare_refinement`` runs the
same model twice — without and with query refinement — so the report can show
the effect of the additional features (requirement #8: evaluate "before" and
"after" applying the extra features).
"""
from __future__ import annotations

from typing import Optional

from ..config import get_dataset
from ..data.corpus import load_qrels, load_queries
from ..engine import ENGINE
from . import metrics


def evaluate_model(dataset_key: str, model: str, *, num_queries: Optional[int] = 50,
                   eval_depth: int = 100, bm25_k1: Optional[float] = None,
                   bm25_b: Optional[float] = None, refine_opts: Optional[dict] = None,
                   hybrid_opts: Optional[dict] = None,
                   only_judged: bool = True, progress=None) -> dict:
    """Evaluate one model. Only queries that have qrels are scored."""
    spec = get_dataset(dataset_key)
    qrels = load_qrels(spec)
    queries = load_queries(spec)
    if only_judged:
        queries = [(qid, txt) for qid, txt in queries if qid in qrels]
    if num_queries:
        queries = queries[:num_queries]

    per_query = []
    details = []
    for i, (qid, qtext) in enumerate(queries, 1):
        res = ENGINE.search(dataset_key, model, qtext, top_k=eval_depth,
                            bm25_k1=bm25_k1, bm25_b=bm25_b,
                            refine_opts=refine_opts, hybrid_opts=hybrid_opts)
        ranked = [r["doc_id"] for r in res["results"]]
        m = metrics.evaluate_query(ranked, qrels.get(qid, {}))
        per_query.append(m)
        details.append({"query_id": qid, "query": qtext, **{k: round(v, 4) for k, v in m.items()}})
        if progress:
            progress(i, len(queries), qid)

    agg = metrics.aggregate(per_query)
    return {
        "dataset": dataset_key, "model": model,
        "refinement_enabled": bool(refine_opts and refine_opts.get("enabled")),
        "params": {"bm25_k1": bm25_k1, "bm25_b": bm25_b},
        "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in agg.items()},
        "per_query": details,
    }


def compare_refinement(dataset_key: str, model: str, *, num_queries: Optional[int] = 50,
                       eval_depth: int = 100, bm25_k1=None, bm25_b=None,
                       hybrid_opts=None, refine_opts: Optional[dict] = None,
                       progress=None) -> dict:
    """Run the model with refinement OFF then ON and report both + the delta.

    The "after" run uses ``refine_opts`` when an enabled config is supplied (so
    the UI's *selected* features are honoured — e.g. spelling only). When none
    is given (e.g. the report script), it defaults to full refinement
    (spelling + synonym expansion)."""
    after_opts = (refine_opts if (refine_opts and refine_opts.get("enabled"))
                  else {"enabled": True, "spell": True, "expand": True, "history": False})
    before = evaluate_model(dataset_key, model, num_queries=num_queries,
                            eval_depth=eval_depth, bm25_k1=bm25_k1, bm25_b=bm25_b,
                            refine_opts={"enabled": False}, hybrid_opts=hybrid_opts,
                            progress=progress)
    after = evaluate_model(dataset_key, model, num_queries=num_queries,
                           eval_depth=eval_depth, bm25_k1=bm25_k1, bm25_b=bm25_b,
                           refine_opts=after_opts, hybrid_opts=hybrid_opts,
                           progress=progress)
    delta = {k: round(after["metrics"][k] - before["metrics"][k], 4)
             for k in ("MAP", "Recall", "P@10", "nDCG@10")}
    return {"dataset": dataset_key, "model": model,
            "before": before["metrics"], "after": after["metrics"], "delta": delta,
            "after_opts": after_opts}
