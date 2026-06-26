"""Disk-backed dense-vector store for the embedding representations.

Vectors are streamed as raw little-endian float32 through gzip into
``<model>.vectors.f32.gz`` while building, so we never hold all embeddings in
RAM during the build and the on-disk artifact is **compressed**. At search time
the file is decompressed once into an in-memory array and scored with a batched
matrix-vector product (vectors are L2-normalised at build time, so a dot product
*is* the cosine similarity). The doc-id sidecar is gzip-compressed too; the tiny
meta file stays plain so we can read N/dim before inflating the vectors.

For corpora in the millions an ANN index (FAISS) would replace the brute-force
scan; the interface is kept small so that swap is localised here.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Iterator, Optional

import numpy as np

from ..config import DatasetSpec, ensure_index_dir
from ..types import SearchResult


def _paths(spec: DatasetSpec, model: str):
    base = os.path.join(spec.index_dir, f"emb_{model}")
    return base + ".vectors.f32.gz", base + ".meta.json", base + ".docids.json.gz"


class VectorStoreWriter:
    """Streams float32 rows to disk; call :meth:`add` per batch then :meth:`close`."""

    def __init__(self, spec: DatasetSpec, model: str, dim: int):
        ensure_index_dir(spec.key)
        self.vec_path, self.meta_path, self.ids_path = _paths(spec, model)
        self.model = model
        self.dim = dim
        self.n = 0
        self.doc_ids: list[str] = []
        self._fh = gzip.open(self.vec_path, "wb", compresslevel=6)

    def add(self, doc_ids: list[str], vectors: np.ndarray):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        # L2 normalise so dot product == cosine at query time
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        self._fh.write(vectors.tobytes())
        self.doc_ids.extend(doc_ids)
        self.n += len(doc_ids)

    def close(self):
        self._fh.close()
        with gzip.open(self.ids_path, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(self.doc_ids, f)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"model": self.model, "dim": self.dim, "n": self.n}, f)


class VectorStore:
    """Read-only mmap view used for cosine search."""

    def __init__(self, spec: DatasetSpec, model: str):
        self.spec = spec
        self.model = model
        self.vec_path, self.meta_path, self.ids_path = _paths(spec, model)
        self._vectors: Optional[np.ndarray] = None
        self._doc_ids: Optional[list[str]] = None
        self._meta: Optional[dict] = None

    def exists(self) -> bool:
        return all(os.path.exists(p) for p in (self.vec_path, self.meta_path, self.ids_path))

    @property
    def meta(self) -> dict:
        if self._meta is None:
            with open(self.meta_path, encoding="utf-8") as f:
                self._meta = json.load(f)
        return self._meta

    @property
    def doc_ids(self) -> list[str]:
        if self._doc_ids is None:
            with gzip.open(self.ids_path, "rt", encoding="utf-8") as f:
                self._doc_ids = json.load(f)
        return self._doc_ids

    @property
    def _id_to_row(self) -> dict[str, int]:
        if not hasattr(self, "_id_row_cache"):
            self._id_row_cache = {d: i for i, d in enumerate(self.doc_ids)}
        return self._id_row_cache

    def score_subset(self, query_vec: np.ndarray, doc_ids: list[str]) -> dict[str, float]:
        """Cosine of the query against a *specific* set of docs (for serial
        hybrid re-ranking of another model's candidates)."""
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        qn = np.linalg.norm(q)
        if qn == 0:
            return {}
        q = q / qn
        rows = [(d, self._id_to_row[d]) for d in doc_ids if d in self._id_to_row]
        if not rows:
            return {}
        idx = np.array([r for _, r in rows])
        sims = self.vectors[idx] @ q
        return {rows[i][0]: float(sims[i]) for i in range(len(rows))}

    @property
    def vectors(self) -> np.ndarray:
        """Inflate the gzip'd float32 blob once into an in-memory (n, dim)
        array. Decompression is a one-time cost on first access; scoring then
        runs on the resident array."""
        if self._vectors is None:
            n, dim = self.meta["n"], self.meta["dim"]
            with gzip.open(self.vec_path, "rb") as f:
                buf = f.read()
            self._vectors = np.frombuffer(buf, dtype=np.float32).reshape(n, dim)
        return self._vectors

    def search(self, query_vec: np.ndarray, top_k: int = 10,
               batch: int = 100_000) -> list[tuple[str, float]]:
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        q = q / qn
        vecs = self.vectors
        n = vecs.shape[0]
        # global top-k via a running heap-free argpartition over batches
        best_scores = np.empty(0, dtype=np.float32)
        best_idx = np.empty(0, dtype=np.int64)
        for start in range(0, n, batch):
            chunk = vecs[start:start + batch]
            sims = chunk @ q
            idx = np.argpartition(-sims, min(top_k, len(sims) - 1))[:top_k] \
                if len(sims) > top_k else np.arange(len(sims))
            best_scores = np.concatenate([best_scores, sims[idx]])
            best_idx = np.concatenate([best_idx, idx + start])
            # keep only the global top_k so far
            if len(best_scores) > top_k:
                keep = np.argpartition(-best_scores, top_k - 1)[:top_k]
                best_scores, best_idx = best_scores[keep], best_idx[keep]
        order = np.argsort(-best_scores)
        ids = self.doc_ids
        return [(ids[int(best_idx[i])], float(best_scores[i])) for i in order]
