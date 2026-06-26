"""Word2Vec (skip-gram + negative sampling) implemented in PyTorch.

gensim has no wheel for Python 3.14 and needs a C compiler, so we train a
genuine skip-gram model directly in torch. After training, a document/query is
embedded as the mean of its in-vocabulary word vectors — the standard way to
turn Word2Vec into a sentence representation for retrieval.

This is the *second* embedding model required alongside BERT (requirement #2
note: "more than one type of embedding model may be used").
"""
from __future__ import annotations

import gzip
import json
import os
import random
from collections import Counter
from typing import Iterable, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn

from ..config import DatasetSpec, Word2VecParams, ensure_index_dir


def _paths(spec: DatasetSpec):
    base = os.path.join(spec.index_dir, "word2vec")
    return base + ".vectors.npz", base + ".vocab.json.gz"


class Word2VecModel:
    """Holds the trained word matrix + vocab and turns token lists into vectors."""

    def __init__(self, vectors: np.ndarray, vocab: dict[str, int],
                 params: Optional[Word2VecParams] = None):
        self.vectors = vectors.astype(np.float32)          # (V, dim), L2-normalised
        self.vocab = vocab                                  # term -> row
        self.dim = vectors.shape[1]
        self.params = params or Word2VecParams(dim=self.dim)

    # -- inference --------------------------------------------------------
    def embed_tokens(self, tokens: Iterable[str]) -> np.ndarray:
        rows = [self.vocab[t] for t in tokens if t in self.vocab]
        if not rows:
            return np.zeros(self.dim, dtype=np.float32)
        return self.vectors[rows].mean(axis=0)

    def most_similar(self, term: str, topn: int = 10) -> list[tuple[str, float]]:
        if term not in self.vocab:
            return []
        v = self.vectors[self.vocab[term]]
        sims = self.vectors @ v
        idx = np.argsort(-sims)[:topn + 1]
        inv = {i: t for t, i in self.vocab.items()}
        return [(inv[int(i)], float(sims[int(i)])) for i in idx if inv[int(i)] != term][:topn]

    # -- persistence ------------------------------------------------------
    def save(self, spec: DatasetSpec):
        ensure_index_dir(spec.key)
        vpath, vocab_path = _paths(spec)
        # compressed .npz for the word matrix; gzip'd JSON for the vocab.
        np.savez_compressed(vpath, vectors=self.vectors)
        with gzip.open(vocab_path, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump({"vocab": self.vocab, "dim": self.dim}, f)

    @classmethod
    def load(cls, spec: DatasetSpec) -> "Word2VecModel":
        vpath, vocab_path = _paths(spec)
        with np.load(vpath) as npz:
            vectors = npz["vectors"]
        with gzip.open(vocab_path, "rt", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(vectors, meta["vocab"])

    @classmethod
    def exists(cls, spec: DatasetSpec) -> bool:
        return all(os.path.exists(p) for p in _paths(spec))


class _SkipGram(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.center = nn.Embedding(vocab_size, dim)
        self.context = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.center.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.context.weight)

    def forward(self, centers, contexts, negatives):
        c = self.center(centers)                         # (B, d)
        pos = self.context(contexts)                     # (B, d)
        neg = self.context(negatives)                    # (B, k, d)
        pos_score = torch.sum(c * pos, dim=1)            # (B,)
        neg_score = torch.bmm(neg, c.unsqueeze(2)).squeeze(2)  # (B, k)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)
        neg_loss = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)
        return -(pos_loss + neg_loss).mean()


def train_word2vec(token_stream_factory,
                   params: Optional[Word2VecParams] = None,
                   seed: int = 13,
                   log_every: int = 200) -> Word2VecModel:
    """Train skip-gram on a corpus.

    ``token_stream_factory`` is a *callable* returning a fresh iterator over
    documents' token lists each time it is called (so we can do multiple passes:
    one to build the vocab, then one per epoch).
    """
    params = params or Word2VecParams()
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # ---- pass 1: vocabulary ----
    freq: Counter = Counter()
    for tokens in token_stream_factory():
        freq.update(tokens)
    items = [(t, c) for t, c in freq.items() if c >= params.min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:params.max_vocab]
    vocab = {t: i for i, (t, _) in enumerate(items)}
    if not vocab:
        raise ValueError("Empty vocabulary - corpus too small or min_count too high.")
    V = len(vocab)

    # negative-sampling distribution ~ freq^0.75
    counts = np.array([c for _, c in items], dtype=np.float64) ** 0.75
    neg_prob = counts / counts.sum()
    neg_table = torch.tensor(neg_prob, dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _SkipGram(V, params.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params.lr)

    def gen_pairs() -> Iterator[tuple[int, int]]:
        for tokens in token_stream_factory():
            ids = [vocab[t] for t in tokens if t in vocab]
            for i, center in enumerate(ids):
                w = rng.randint(1, params.window)
                lo, hi = max(0, i - w), min(len(ids), i + w + 1)
                for j in range(lo, hi):
                    if j != i:
                        yield center, ids[j]

    print(f"  [word2vec] vocab={V}, dim={params.dim}, device={device}", flush=True)
    for epoch in range(params.epochs):
        centers_buf, contexts_buf = [], []
        step = 0
        total_loss = 0.0
        for center, ctx in gen_pairs():
            centers_buf.append(center)
            contexts_buf.append(ctx)
            if len(centers_buf) >= params.batch_size:
                loss = _step(model, opt, centers_buf, contexts_buf,
                             neg_table, params.negatives, V, device)
                total_loss += loss
                step += 1
                centers_buf, contexts_buf = [], []
                if log_every and step % log_every == 0:
                    print(f"  [word2vec] epoch {epoch+1} step {step} "
                          f"loss={total_loss/step:.4f}", flush=True)
        if centers_buf:
            _step(model, opt, centers_buf, contexts_buf,
                  neg_table, params.negatives, V, device)
        print(f"  [word2vec] epoch {epoch+1}/{params.epochs} done "
              f"(avg loss {total_loss/max(step,1):.4f})", flush=True)

    vectors = model.center.weight.detach().cpu().numpy()
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    return Word2VecModel(vectors, vocab, params)


def _step(model, opt, centers, contexts, neg_table, k, V, device) -> float:
    c = torch.tensor(centers, dtype=torch.long, device=device)
    ctx = torch.tensor(contexts, dtype=torch.long, device=device)
    negs = torch.multinomial(neg_table, len(centers) * k, replacement=True
                             ).view(len(centers), k).to(device)
    opt.zero_grad()
    loss = model(c, ctx, negs)
    loss.backward()
    opt.step()
    return float(loss.item())
