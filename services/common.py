"""Shared helpers for the microservices: a FastAPI app factory and a tiny
stdlib HTTP client (urllib) used by the API Gateway to talk to downstream
services over REST — no extra dependency, loose coupling between services.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from fastapi import FastAPI


def make_app(title: str) -> FastAPI:
    app = FastAPI(title=title, version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": title}

    return app


def post_json(url: str, payload: dict, timeout: float = 600.0) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: float = 60.0) -> Any:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
