"""Ranking & Evaluation Service (SOA component, port 8004).

Run:  uvicorn services.evaluation.app:app --port 8004
"""
from __future__ import annotations

from services.common import make_app
from .router import router

app = make_app("ranking-eval-service")
app.include_router(router)
