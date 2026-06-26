"""Embedding representations (requirement #2, model 2): BERT + Word2Vec.

Both share the same on-disk :class:`VectorStore` and the same retrieval path
(encode the query -> cosine search). They differ only in the encoder:

* **BERT**  - sentence-transformers (all-MiniLM-L6-v2), encodes raw text.
* **Word2Vec** - our PyTorch skip-gram, mean of token vectors.

Using two embedding models satisfies the "more than one type of embedding"
note and lets the report compare a contextual model against a static one.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import BERT_MODEL_NAME, DatasetSpec
from ..index.inverted_index import InvertedIndex
from ..index.vector_store import VectorStore, VectorStoreWriter
from ..data.corpus import iter_docs
from ..text.preprocessing import process
from ..types import SearchResult
from .word2vec import Word2VecModel


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
class BertEncoder:
    """Lazy wrapper around a sentence-transformers model."""

    name = "bert"

    def __init__(self, model_name: str = BERT_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        get_dim = (getattr(self.model, "get_embedding_dimension", None)
                   or self.model.get_sentence_embedding_dimension)
        return get_dim()

    def encode_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=batch_size,
                                            show_progress_bar=False,
                                            convert_to_numpy=True), dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]


class Word2VecEncoder:
    """Adapts a trained Word2VecModel to the encoder interface."""

    name = "word2vec"

    def __init__(self, model: Word2VecModel):
        self.w2v = model

    @property
    def dim(self) -> int:
        return self.w2v.dim

    def encode_query(self, text: str) -> np.ndarray:
        return self.w2v.embed_tokens(process(text))


# ---------------------------------------------------------------------------
# Build: stream the corpus -> vector store on disk
# ---------------------------------------------------------------------------
def build_dense_index(spec: DatasetSpec, model_name: str, encoder,
                      limit: Optional[int] = None, batch_size: int = 256,
                      log_every: int = 2560) -> dict:
    """Stream the corpus through a text encoder into the model's vector store.
    Shared by the BERT and the multilingual indexes (only the encoder differs)."""
    dim = encoder.dim
    writer = VectorStoreWriter(spec, model_name, dim)
    buf_ids, buf_text, n = [], [], 0
    for doc in iter_docs(spec, limit=limit):
        buf_ids.append(doc.doc_id)
        buf_text.append(doc.raw_text or " ".join(doc.tokens))
        if len(buf_ids) >= batch_size:
            writer.add(buf_ids, encoder.encode_texts(buf_text, batch_size=batch_size))
            n += len(buf_ids)
            buf_ids, buf_text = [], []
            if log_every and n % log_every == 0:
                print(f"  [{model_name}] encoded {n} docs ...", flush=True)
    if buf_ids:
        writer.add(buf_ids, encoder.encode_texts(buf_text, batch_size=batch_size))
        n += len(buf_ids)
    writer.close()
    print(f"  [{model_name}] done: {n} doc vectors (dim={dim})", flush=True)
    return {"model": model_name, "n": n, "dim": dim}


def build_bert_index(spec: DatasetSpec, limit: Optional[int] = None,
                     batch_size: int = 256, log_every: int = 2560) -> dict:
    return build_dense_index(spec, "bert", BertEncoder(), limit, batch_size, log_every)


def build_multilingual_index(spec: DatasetSpec, limit: Optional[int] = None,
                             batch_size: int = 128, log_every: int = 2560) -> dict:
    """Cross-lingual index: encode docs with a multilingual sentence model so
    queries in any supported language land in the same space."""
    from ..config import MULTILINGUAL_MODEL_NAME
    enc = BertEncoder(MULTILINGUAL_MODEL_NAME)
    return build_dense_index(spec, "multilingual", enc, limit, batch_size, log_every)


def build_word2vec_index(spec: DatasetSpec, model: Word2VecModel,
                         limit: Optional[int] = None, batch_size: int = 4096,
                         log_every: int = 20000) -> dict:
    writer = VectorStoreWriter(spec, "word2vec", model.dim)
    buf_ids, buf_vecs, n = [], [], 0
    for doc in iter_docs(spec, limit=limit):
        buf_ids.append(doc.doc_id)
        buf_vecs.append(model.embed_tokens(doc.tokens))
        if len(buf_ids) >= batch_size:
            writer.add(buf_ids, np.vstack(buf_vecs))
            n += len(buf_ids)
            buf_ids, buf_vecs = [], []
            if log_every and n % log_every == 0:
                print(f"  [word2vec] embedded {n} docs ...", flush=True)
    if buf_ids:
        writer.add(buf_ids, np.vstack(buf_vecs))
        n += len(buf_ids)
    writer.close()
    print(f"  [word2vec] done: {n} doc vectors (dim={model.dim})", flush=True)
    return {"model": "word2vec", "n": n, "dim": model.dim}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class EmbeddingRetriever:
    """Cosine search over a stored embedding model. ``index`` (optional) is used
    only to attach display text to the returned hits."""

    def __init__(self, spec: DatasetSpec, model_name: str, encoder,
                 index: Optional[InvertedIndex] = None):
        self.name = model_name
        self.store = VectorStore(spec, model_name)
        self.encoder = encoder
        self.index = index

    def search(self, query_text: str, top_k: int = 10) -> list[SearchResult]:
        qvec = self.encoder.encode_query(query_text)
        if not np.any(qvec):
            return []
        hits = self.store.search(qvec, top_k=top_k)
        texts = self.index.get_docs_by_id([d for d, _ in hits]) if self.index else {}
        out = []
        for rank, (doc_id, score) in enumerate(hits, 1):
            out.append(SearchResult(doc_id=doc_id, score=float(score), rank=rank,
                                    raw_text=texts.get(doc_id, "")))
        return out

    def rescore(self, query_text: str, doc_ids: list[str]) -> dict[str, float]:
        qvec = self.encoder.encode_query(query_text)
        if not np.any(qvec):
            return {}
        return self.store.score_subset(qvec, doc_ids)
