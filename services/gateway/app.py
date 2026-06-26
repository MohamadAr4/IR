"""API Gateway (SOA component, port 8000).

The single public entry point (Gateway / Facade pattern). Clients — including
the Streamlit UI — only ever talk to the gateway.

Run:  uvicorn services.gateway.app:app --port 8000
"""
from __future__ import annotations

from services.common import make_app
from .router import router

app = make_app("api-gateway")
app.include_router(router)
