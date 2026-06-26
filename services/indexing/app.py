"""Indexing Service (SOA component, port 8002).

Run:  uvicorn services.indexing.app:app --port 8002
"""
from __future__ import annotations

from services.common import make_app
from .router import router

app = make_app("indexing-service")
app.include_router(router)
