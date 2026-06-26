"""HTTP layer for the indexing service: request schema + route wiring."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from . import service

router = APIRouter()


@router.get("/datasets")
def datasets():
    return service.datasets()


@router.get("/status/{dataset_key}")
def status(dataset_key: str):
    return service.status(dataset_key)


@router.get("/vocab/{dataset_key}")
def vocab(dataset_key: str, prefix: str = "", limit: int = 20):
    return service.vocab(dataset_key, prefix=prefix, limit=limit)


class BuildRequest(BaseModel):
    dataset_key: str
    limit: Optional[int] = 5000   # cap for the online endpoint


@router.post("/build")
def build(req: BuildRequest):
    return service.build(req.dataset_key, limit=req.limit)
