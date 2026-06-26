"""Build every representation/index for a dataset (requirement #3 + the index
side of requirement #2). Streams the preprocessed corpus, so it works on the
full multi-GB files without loading them into RAM.

Examples
--------
    # Inverted index (TF-IDF + BM25 share it) for the full args.me corpus:
    python scripts/build_indexes.py --dataset argsme --models inverted

    # Everything (inverted + word2vec + bert) on a 50k-doc cap, good for a demo:
    python scripts/build_indexes.py --dataset argsme --limit 50000

    # Full corpus, all models (slow; BERT on millions of docs is impractical):
    python scripts/build_indexes.py --dataset argsme --limit 0
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir_core.config import Word2VecParams, get_dataset
from ir_core.data.corpus import iter_docs
from ir_core.index.inverted_index import InvertedIndex
from ir_core.representation import embeddings as emb
from ir_core.representation.word2vec import train_word2vec


def build(dataset_key: str, models: list[str], limit, w2v_params: Word2VecParams,
          bert_batch: int, w2v_train_limit=None):
    spec = get_dataset(dataset_key)
    limit = None if not limit else limit
    # Word2Vec can be *trained* on a cheap sample but still *embed* the full
    # corpus (embedding is a fast vector lookup), giving full eval coverage
    # without paying to train on every document.
    w2v_train_limit = w2v_train_limit or limit
    print(f"=== Building [{models}] for '{spec.name}' (limit={limit or 'ALL'}) ===")

    if "inverted" in models:
        t = time.time()
        InvertedIndex(spec).build(limit=limit)
        print(f"  inverted index: {time.time()-t:.1f}s\n")

    if "word2vec" in models:
        t = time.time()
        factory = lambda: (d.tokens for d in iter_docs(spec, limit=w2v_train_limit))  # noqa: E731
        print(f"  [word2vec] training on {w2v_train_limit or 'ALL'} docs, "
              f"embedding {limit or 'ALL'} docs", flush=True)
        model = train_word2vec(factory, w2v_params)
        model.save(spec)
        emb.build_word2vec_index(spec, model, limit=limit)
        print(f"  word2vec: {time.time()-t:.1f}s\n")

    if "bert" in models:
        t = time.time()
        emb.build_bert_index(spec, limit=limit, batch_size=bert_batch)
        print(f"  bert: {time.time()-t:.1f}s\n")

    if "multilingual" in models:
        t = time.time()
        emb.build_multilingual_index(spec, limit=limit)
        print(f"  multilingual: {time.time()-t:.1f}s\n")

    print("=== done ===")


def main():
    ap = argparse.ArgumentParser(description="Build IR indexes for a dataset.")
    ap.add_argument("--dataset", required=True, help="argsme | argsme_sample")
    ap.add_argument("--models", default="inverted,word2vec,bert",
                    help="comma list: inverted,word2vec,bert,multilingual (default builds first three)")
    ap.add_argument("--limit", type=int, default=50000,
                    help="max docs to index/embed (0 = ALL). Default 50000.")
    ap.add_argument("--w2v-train-limit", type=int, default=None,
                    help="docs to TRAIN word2vec on (default = --limit). Lets you "
                         "train on a cheap sample but embed the full corpus.")
    ap.add_argument("--w2v-dim", type=int, default=100)
    ap.add_argument("--w2v-epochs", type=int, default=3)
    ap.add_argument("--w2v-min-count", type=int, default=2)
    ap.add_argument("--bert-batch", type=int, default=256)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    w2v = Word2VecParams(dim=args.w2v_dim, epochs=args.w2v_epochs,
                         min_count=args.w2v_min_count)
    train_limit = None if args.w2v_train_limit == 0 else args.w2v_train_limit
    build(args.dataset, models, args.limit, w2v, args.bert_batch,
          w2v_train_limit=train_limit)


if __name__ == "__main__":
    main()
