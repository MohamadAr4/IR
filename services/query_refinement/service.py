"""Query refinement business logic (framework-free).

Owns requirement #5: spelling correction, WordNet synonym expansion, query
suggestion and history-based personalisation. Delegates to the shared engine.
"""
from __future__ import annotations

from typing import Optional

from ir_core.engine import ENGINE


def refine(dataset: str, query: str, *, spell: bool = True, expand: bool = True,
           history_personalize: bool = True,
           history: Optional[list[str]] = None) -> dict:
    r = ENGINE.refine(dataset, query, do_spell=spell, do_expand=expand,
                      use_history=history_personalize, history=history)
    return r.to_dict()


def suggest(dataset: str, query: str,
            history: Optional[list[str]] = None) -> dict:
    return {"query": query,
            "suggestions": ENGINE.suggest(dataset, query, history=history)}
