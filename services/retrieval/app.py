"""Retrieval Service (SOA component, port 8003).

Run:  uvicorn services.retrieval.app:app --port 8003
"""
from __future__ import annotations

from services.common import make_app
from .router import router

app = make_app("retrieval-service")
app.include_router(router)
