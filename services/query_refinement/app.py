"""Query Refinement Service (SOA component, port 8005).

Run:  uvicorn services.query_refinement.app:app --port 8005
"""
from __future__ import annotations

from services.common import make_app
from .router import router

app = make_app("query-refinement-service")
app.include_router(router)
