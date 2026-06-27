"""The retrieval engine: the single entry point that the microservices and the
Streamlit UI call. It lazily loads whichever indexes a dataset has, applies
optional query refinement, and dispatches to the requested representation —
including the two hybrids (parallel-with-fusion and serial-rerank).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import (MULTILINGUAL_MODEL_NAME, SYNONYM_EXPANSION_WEIGHT,
                     DatasetSpec, get_dataset)
from .index.inverted_index import InvertedIndex
from .query.processing import build_context
from .query.refinement import QueryRefiner, RefinementResult
from .representation.base import DenseRetriever, LexicalRetriever, QueryContext
from .representation.embeddings import (BertEncoder, EmbeddingRetriever,
                                        Word2VecEncoder)
from .representation.word2vec import Word2VecModel
from .representation import hybrid as hybrid_mod
from .index.vector_store import VectorStore
from .types import SearchResult

# Canonical model identifiers exposed to the API/UI.
SINGLE_MODELS = ["tfidf", "bm25", "bert", "word2vec", "multilingual"]
HYBRID_MODELS = ["hybrid_parallel", "hybrid_serial"]
ALL_MODELS = SINGLE_MODELS + HYBRID_MODELS


@dataclass
class _DatasetBundle:
    spec: DatasetSpec
    index: InvertedIndex
    retrievers: dict           # name -> Lexical/Dense retriever (only the available ones)
    refiner: QueryRefiner


class Engine:
    def __init__(self):
        self._bundles: dict[str, _DatasetBundle] = {}

    # -- loading ----------------------------------------------------------
    def _bundle(self, dataset_key: str) -> _DatasetBundle:
        if dataset_key in self._bundles:
            return self._bundles[dataset_key]
        spec = get_dataset(dataset_key)
        index = InvertedIndex(spec)
        if not index.exists():
            raise FileNotFoundError(
                f"Dataset '{dataset_key}' has no inverted index yet. "
                f"Run: python scripts/build_indexes.py --dataset {dataset_key}")

        retrievers: dict = {
            "tfidf": LexicalRetriever("tfidf", index),
            "bm25": LexicalRetriever("bm25", index),
        }
        # dense models are optional — only wire them up if their vectors exist
        if VectorStore(spec, "bert").exists():
            retrievers["bert"] = DenseRetriever(
                "bert", EmbeddingRetriever(spec, "bert", BertEncoder(), index=index))
        if VectorStore(spec, "word2vec").exists() and Word2VecModel.exists(spec):
            w2v = Word2VecModel.load(spec)
            retrievers["word2vec"] = DenseRetriever(
                "word2vec", EmbeddingRetriever(spec, "word2vec",
                                               Word2VecEncoder(w2v), index=index))
        # cross-lingual model: query in any language retrieves the English docs
        if VectorStore(spec, "multilingual").exists():
            retrievers["multilingual"] = DenseRetriever(
                "multilingual", EmbeddingRetriever(
                    spec, "multilingual", BertEncoder(MULTILINGUAL_MODEL_NAME), index=index))

        bundle = _DatasetBundle(spec=spec, index=index, retrievers=retrievers,
                                refiner=QueryRefiner(index))
        self._bundles[dataset_key] = bundle
        return bundle

    def available_models(self, dataset_key: str) -> list[str]:
        b = self._bundle(dataset_key)
        models = [m for m in SINGLE_MODELS if m in b.retrievers]
        # hybrids need >= 2 base models
        if len(b.retrievers) >= 2:
            models += HYBRID_MODELS
        return models

    # -- refinement -------------------------------------------------------
    def refine(self, dataset_key: str, query: str, *, do_spell=True, do_expand=True,
               use_history=True, history: Optional[list[str]] = None) -> RefinementResult:
        return self._bundle(dataset_key).refiner.refine(
            query, do_spell=do_spell, do_expand=do_expand,
            use_history=use_history, history=history)

    def suggest(self, dataset_key: str, query: str,
                history: Optional[list[str]] = None) -> list[str]:
        return self._bundle(dataset_key).refiner.suggest(query, history=history)

    # -- search -----------------------------------------------------------
    def search(self, dataset_key: str, model: str, query: str, *, top_k: int = 10,
               bm25_k1: Optional[float] = None, bm25_b: Optional[float] = None,
               refine_opts: Optional[dict] = None, history: Optional[list[str]] = None,
               hybrid_opts: Optional[dict] = None) -> dict:
        b = self._bundle(dataset_key)

        # ---- optional refinement -> produces the effective query ----
        refinement = None
        eff_raw, eff_tokens = query, None
        if refine_opts and refine_opts.get("enabled"):
            refinement = b.refiner.refine(
                query, do_spell=refine_opts.get("spell", True),
                do_expand=refine_opts.get("expand", True),
                use_history=refine_opts.get("history", True), history=history)
            eff_raw = refinement.refined_raw
            eff_tokens = refinement.refined_tokens

        ctx = build_context(eff_raw, bm25_k1=bm25_k1, bm25_b=bm25_b)
        if eff_tokens is not None:
            ctx.tokens = eff_tokens
        if refinement and refinement.expansion_terms:
            # Down-weight sense-blind synonyms in the lexical query vector.
            ctx.expansion_terms = set(refinement.expansion_terms)
            ctx.expansion_weight = SYNONYM_EXPANSION_WEIGHT

        # ---- dispatch ----
        if model in b.retrievers:
            results = b.retrievers[model].search(ctx, top_k=top_k)
        elif model == "hybrid_parallel":
            results = self._parallel(b, ctx, top_k, hybrid_opts or {})
        elif model == "hybrid_serial":
            results = self._serial(b, ctx, top_k, hybrid_opts or {})
        else:
            raise ValueError(f"Unknown or unavailable model '{model}' for '{dataset_key}'. "
                             f"Available: {self.available_models(dataset_key)}")

        for i, r in enumerate(results, 1):
            r.rank = i
        return {
            "dataset": dataset_key, "model": model, "query": query,
            "effective_query": eff_raw, "top_k": top_k,
            "params": {"bm25_k1": bm25_k1, "bm25_b": bm25_b},
            "refinement": refinement.to_dict() if refinement else None,
            "results": [r.to_dict() for r in results],
        }

    # -- hybrid helpers ---------------------------------------------------
    def _default_pair(self, b: _DatasetBundle) -> list[str]:
        """Pick a sensible default model set: prefer bm25 + a dense model."""
        order = ["bm25", "bert", "word2vec", "tfidf"]
        chosen = [m for m in order if m in b.retrievers]
        return chosen[:2] if len(chosen) >= 2 else list(b.retrievers)[:2]

    def _parallel(self, b, ctx, top_k, opts) -> list[SearchResult]:
        names = opts.get("models") or self._default_pair(b)
        names = [n for n in names if n in b.retrievers]
        retrievers = {n: b.retrievers[n] for n in names}
        if len(retrievers) < 2:
            raise ValueError("Parallel hybrid needs at least two available base models.")
        return hybrid_mod.parallel_hybrid(
            retrievers, ctx, top_k=top_k,
            fusion=opts.get("fusion", "weighted"),
            weights=opts.get("weights"),
            pool=opts.get("pool", max(100, top_k * 5)))

    def _serial(self, b, ctx, top_k, opts) -> list[SearchResult]:
        pair = self._default_pair(b)
        s1 = opts.get("stage1") or pair[0]
        s2 = opts.get("stage2") or (pair[1] if len(pair) > 1 else pair[0])
        if s1 not in b.retrievers or s2 not in b.retrievers:
            raise ValueError(f"Serial hybrid needs '{s1}' and '{s2}' to be available.")
        return hybrid_mod.serial_hybrid(
            b.retrievers[s1], b.retrievers[s2], ctx, top_k=top_k,
            candidate_k=opts.get("candidate_k", 100))


# a process-wide singleton is convenient for the services
ENGINE = Engine()
