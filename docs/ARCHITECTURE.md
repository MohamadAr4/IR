# Architecture

The system follows a **Service-Oriented Architecture** (requirement #7): each
capability is an independently deployable FastAPI service with one
responsibility, fronted by an **API Gateway**. All services share one reusable
library (`ir_core`) and one on-disk data layer (the indexes), which keeps the
services thin and the business logic in one tested place.

## Component diagram

```
                         ┌──────────────────────────┐
                         │     Streamlit UI (9)      │
                         │  ui/app.py                │
                         └─────────────┬─────────────┘
                                       │ REST (or in-process)
                                       ▼
                         ┌──────────────────────────┐
                         │     API Gateway  :8000    │   Gateway / Facade
                         │  routing + composition    │   (search_refined =
                         └───┬─────┬─────┬─────┬──────┘    refine ∘ retrieve)
            ┌────────────────┘     │     │     └─────────────────┐
            ▼                      ▼     ▼                        ▼
┌────────────────────┐ ┌───────────────────┐ ┌──────────────────┐ ┌───────────────────┐
│ Preprocessing :8001│ │  Indexing  :8002  │ │ Retrieval  :8003 │ │ Ranking&Eval :8004│
│ tokenise (3/4)     │ │ build/status (3)  │ │ match+rank (2/6) │ │ MAP/R/P@10/nDCG(8)│
└─────────┬──────────┘ └─────────┬─────────┘ └────────┬─────────┘ └─────────┬─────────┘
          │                      │                    │                     │
          │              ┌───────────────────┐        │                     │
          │              │ Query Refinement  │        │                     │
          │              │      :8005  (5)   │        │                     │
          │              └─────────┬─────────┘        │                     │
          └──────────────┬─────────┴─────────┬────────┴──────────┬──────────┘
                         ▼                    ▼                   ▼
                ┌─────────────────────────────────────────────────────┐
                │            ir_core  (shared engine library)          │
                │  text · data · index · representation · query · eval │
                │                    engine.Engine (Facade)            │
                └───────────────────────────┬─────────────────────────┘
                                             ▼
                ┌─────────────────────────────────────────────────────┐
                │     Data layer (per dataset, under indexes/<key>/)    │
                │  inverted_index.sqlite (full docs as zlib BLOBs) ·    │
                │  emb_*.vectors.f32.gz · *.docids.json.gz ·            │
                │  word2vec.vectors.npz · word2vec.vocab.json.gz        │
                └─────────────────────────────────────────────────────┘
```

## Services (contracts)

| Service | Port | Responsibility | Key endpoints |
|---------|------|----------------|---------------|
| Preprocessing | 8001 | normalise text (docs **and** queries, same pipeline) | `POST /process` |
| Indexing | 8002 | own the indexes; status, vocab, (capped) build | `GET /datasets`, `GET /status/{ds}`, `GET /vocab/{ds}`, `POST /build` |
| Retrieval | 8003 | query matching & ranking for every model | `GET /models/{ds}`, `POST /search` |
| Ranking & Evaluation | 8004 | offline metrics, before/after comparison | `POST /evaluate`, `POST /compare` |
| Query Refinement | 8005 | spell-correct, expand, suggest, personalise | `POST /refine`, `POST /suggest` |
| **API Gateway** | 8000 | single entry point, routing, composition, health | all of the above + `GET /services`, `POST /search_refined` |

Communication is **REST/JSON over HTTP**. The gateway client uses the standard
library (`urllib`) so there is no extra coupling, and each service exposes
`/health` for discovery. A message-queue transport (e.g. RabbitMQ) could be
dropped in for the async, long-running index builds without touching the
business logic — that boundary is already isolated in `services/common.py`.

## Design patterns

- **Facade** — `ir_core.engine.Engine` hides the loading/dispatch of six models
  behind one `search()` call; services and UI depend only on it.
- **Gateway / API Composition** — the gateway is the only public surface;
  `/search_refined` composes the refinement and retrieval services.
- **Strategy** — every representation implements the same
  `search(ctx, top_k)` / `rescore(ctx, doc_ids)` interface
  (`representation/base.py`), so models are interchangeable and the hybrids
  treat them uniformly.
- **Adapter** — `BertEncoder` / `Word2VecEncoder` adapt two very different
  embedding back-ends to one `encode_query` contract.
- **Builder + Repository** — `InvertedIndex.build()` constructs the index;
  the read accessors (`postings`, `df`, `idf`, `get_docs`) form a repository
  over the SQLite store.
- **Singleton** — a process-wide `ENGINE` caches loaded indexes per dataset.

## Quality attributes (requirement #7)

- **Clean architecture / separation of concerns** — UI → gateway → services →
  `ir_core` → data; dependencies point inward, the engine has no web/UI imports.
- **Loose coupling** — services share only the library and the on-disk index
  contract; each can be deployed/scaled/replaced on its own.
- **Scalability** — indexing streams the corpus (never loads it into RAM) and
  persists to disk, so the lexical index handles the full multi-GB datasets;
  the dense store is mmapped and scored in batches (swap in FAISS for ANN at
  millions of docs — the seam is `index/vector_store.py`).
- **Maintainability** — one preprocessing pipeline for docs and queries, one
  scoring substrate for TF-IDF and BM25, metrics isolated and unit-testable.
- **Reusability** — `ir_core` is a plain importable package; the UI uses it
  directly in "in-process" mode, proving the same code path serves both.

## Technology choices & rationale

- **SQLite inverted index** — zero-setup, on-disk, transactional, handles the
  full corpora without a server; `LOG`/`SQRT` registered as Python UDFs so the
  TF-IDF weighting is identical on the index and query sides.
- **sentence-transformers (all-MiniLM-L6-v2)** for BERT — strong quality at 384
  dims and small/fast on CPU.
- **PyTorch skip-gram** for Word2Vec — gensim has no Python 3.14 wheel and needs
  a C toolchain; training in torch keeps the dependency set buildable and gives
  a genuine second (static) embedding to contrast with BERT.
- **FastAPI + uvicorn** — typed request models, automatic OpenAPI docs per
  service, async-ready.
- **Streamlit** — fastest path to a controllable web UI that reuses the Python
  engine directly.
