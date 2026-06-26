"""Streaming access to the (potentially multi-GB) preprocessed corpora and to
the evaluation queries/qrels from ir_datasets.

The processed JSON files are pretty-printed arrays of
``{"doc_id", "tokens", "raw_text"}`` records and can be far larger than RAM
(the full args.me corpus is ~1.6 GB), so we never ``json.load`` them — we stream with ``ijson``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import ijson

from ..config import DatasetSpec


@dataclass
class Doc:
    doc_id: str
    tokens: list[str]
    raw_text: str


def iter_docs(spec: DatasetSpec, limit: Optional[int] = None) -> Iterator[Doc]:
    """Yield documents one at a time, in file order, without loading the whole
    file. ``limit`` caps the number of docs (None/0 = all)."""
    n = 0
    with open(spec.processed_json, "rb") as f:
        for rec in ijson.items(f, "item"):
            yield Doc(
                doc_id=rec.get("doc_id", str(n)),
                tokens=rec.get("tokens", []) or [],
                raw_text=rec.get("raw_text", "") or "",
            )
            n += 1
            if limit and n >= limit:
                return


def count_docs(spec: DatasetSpec, limit: Optional[int] = None) -> int:
    n = 0
    for _ in iter_docs(spec, limit=limit):
        n += 1
    return n


# ---------------------------------------------------------------------------
# Evaluation data (queries + qrels) via ir_datasets
# ---------------------------------------------------------------------------
def _query_text(query_obj, attrs: tuple) -> str:
    for a in attrs:
        v = getattr(query_obj, a, None)
        if v:
            return str(v)
    # last resort: ir_datasets exposes default_text on some types
    if hasattr(query_obj, "default_text"):
        try:
            return query_obj.default_text()
        except Exception:
            pass
    return ""


def load_queries(spec: DatasetSpec, limit: Optional[int] = None) -> list[tuple[str, str]]:
    """Return ``[(query_id, query_text), ...]`` for the dataset."""
    import ir_win_fix  # noqa: F401  apply Windows download patch
    import ir_datasets

    ds = ir_datasets.load(spec.ir_datasets_id)
    out: list[tuple[str, str]] = []
    for q in ds.queries_iter():
        out.append((q.query_id, _query_text(q, spec.query_text_attrs)))
        if limit and len(out) >= limit:
            break
    return out


def load_qrels(spec: DatasetSpec) -> dict[str, dict[str, int]]:
    """Return ``{query_id: {doc_id: relevance}}``."""
    import ir_win_fix  # noqa: F401
    import ir_datasets

    ds = ir_datasets.load(spec.ir_datasets_id)
    qrels: dict[str, dict[str, int]] = {}
    for qr in ds.qrels_iter():
        qrels.setdefault(qr.query_id, {})[qr.doc_id] = int(qr.relevance)
    return qrels
