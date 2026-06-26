"""End-to-end smoke test on the tiny 5-doc sample (no ir_datasets needed).

Builds every representation for the sample corpus, then exercises each model,
both hybrids, and query refinement. Quick way to confirm an install works.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir_core.config import SAMPLE, Word2VecParams
from ir_core.data.corpus import iter_docs
from ir_core.engine import ENGINE
from ir_core.index.inverted_index import InvertedIndex
from ir_core.representation import embeddings as emb
from ir_core.representation.word2vec import train_word2vec


def main():
    print("Building sample indexes ...")
    InvertedIndex(SAMPLE).build(log_every=0)
    factory = lambda: (d.tokens for d in iter_docs(SAMPLE))  # noqa: E731
    w2v = train_word2vec(factory, Word2VecParams(dim=32, min_count=1, epochs=2,
                                                 batch_size=64), log_every=0)
    w2v.save(SAMPLE)
    emb.build_word2vec_index(SAMPLE, w2v, log_every=0)
    emb.build_bert_index(SAMPLE, batch_size=8, log_every=0)

    ENGINE._bundles.clear()
    q = "school condoms for students"
    print(f"\nQuery: {q!r}\nAvailable models:", ENGINE.available_models(SAMPLE.key))
    for model in ENGINE.available_models(SAMPLE.key):
        res = ENGINE.search(SAMPLE.key, model, q, top_k=2,
                            hybrid_opts={"fusion": "weighted"})
        top = res["results"][0] if res["results"] else None
        print(f"  {model:16s} -> "
              + (f"{top['doc_id'][:24]} (score {top['score']:.3f})" if top else "no hits"))

    r = ENGINE.refine(SAMPLE.key, "shcool condom", history=["teen pregnancy"])
    print("\nRefinement of 'shcool condom':")
    print("  corrected :", r.corrected)
    print("  expansions:", r.expansions)
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
