"""Preprocessing Service (SOA component, port 8001).

ASGI entry point: builds the FastAPI app (with /health) and mounts the router.

Run:  uvicorn services.preprocessing.app:app --port 8001
"""
from __future__ import annotations

from services.common import make_app
from .router import router

app = make_app("preprocessing-service")
app.include_router(router)
