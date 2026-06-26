"""Small shared data types used across the retrieval layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchResult:
    doc_id: str
    score: float
    rank: int = 0
    raw_text: str = ""
    # per-model component scores (filled in by hybrid fusion for transparency)
    components: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "score": round(float(self.score), 6),
            "rank": self.rank,
            "raw_text": self.raw_text,
            "components": {k: round(float(v), 6) for k, v in self.components.items()},
        }
