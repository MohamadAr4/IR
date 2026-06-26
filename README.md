# Information Retrieval System

A complete, **service-oriented Information Retrieval (IR) system** over the **Args.me**
debate-argument corpus (Touché 2020 Task 1, **387,692 documents**). It implements every
representation, indexing, query-processing, refinement, ranking, evaluation, and UI requirement
of the project brief — plus **cross-lingual (multilingual) retrieval** as an advanced feature —
all built from scratch in Python with a clean, layered architecture.

One shared engine (`ir_core`) powers three front-ends: a **Streamlit web UI**, a
**Service-Oriented Architecture** of five FastAPI microservices behind an API gateway, and a set
of **command-line tools** for building indexes and running evaluations.

---

## Table of contents

- [Highlights](#highlights)
- [Requirements coverage](#requirements-coverage)
- [Retrieval models](#retrieval-models)
- [Architecture](#architecture)
- [Storage &amp; indexing](#storage--indexing)
- [Query processing &amp; refinement](#query-processing--refinement)
- [Evaluation](#evaluation)
- [The web UI](#the-web-ui)
- [Cross-lingual (multilingual) retrieval](#cross-lingual-multilingual-retrieval)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Building the indexes](#building-the-indexes)
- [Command-line tools](#command-line-tools)
- [REST API (gateway)](#rest-api-gateway)
- [Project layout](#project-layout)
- [Tech stack](#tech-stack)
- [Design notes](#design-notes)
- [Further documentation](#further-documentation)

---

## Highlights

- **7 retrieval models** behind one interface: TF-IDF (VSM), BM25, BERT embeddings, Word2Vec
  embeddings, **Multilingual** embeddings (cross-lingual), and two **Hybrids** (parallel fusion
  and serial re-rank).
- **Disk-backed SQLite inverted index** that streams the multi-GB corpus on build, stores the
  **full original document text** (zlib-compressed), and is shared by TF-IDF and BM25.
- **Compressed model/index artifacts** — gzip'd dense vectors, compressed Word2Vec matrix — and
  **no training at query time** (everything is pre-built and loaded).
- **Corpus-aware query refinement** — spell-correction, WordNet synonym expansion, query
  suggestion, and history personalization.
- **Full SOA**: five independent microservices + an API gateway (FastAPI/REST).
- **Standard evaluation**: MAP, Recall, Precision@10, nDCG@10, with a before/after query-refinement
  comparison.
- **Modern Streamlit UI** (dark theme) with live BM25 tuning, hybrid configuration, query-term
  highlighting, inline full-document reading, and an evaluation dashboard with charts.
- **Cross-lingual search**: ask in Arabic, French, Spanish, German, … and retrieve the English
  arguments via a shared multilingual embedding space.

---

## Requirements coverage

| # | Requirement | Where |
|---|-------------|-------|
| 2 | **Representations:** VSM TF-IDF, BM25, embeddings (BERT **and** Word2Vec), hybrid (serial **and** parallel with fusion) | `ir_core/representation/` |
| 3 | **Indexing:** disk-backed inverted index (SQLite), streamed build, compressed artifacts | `ir_core/index/inverted_index.py`, `ir_core/index/vector_store.py` |
| 4 | **Query processing:** identical preprocessing for documents &amp; queries | `ir_core/text/preprocessing.py`, `ir_core/query/processing.py` |
| 5 | **Query refinement:** spell-correction, WordNet expansion, suggestion, history | `ir_core/query/refinement.py` |
| 6 | **Matching &amp; ranking:** cosine / BM25 / dot product per model | `ir_core/representation/*`, `ir_core/engine.py` |
| 7 | **SOA:** 5 microservices + API gateway (FastAPI/REST) | `services/` |
| 8 | **Evaluation:** MAP, Recall, P@10, nDCG@10, before/after refinement | `ir_core/eval/` |
| 9 | **UI:** Streamlit web app | `ui/app.py` |
| 12* | **Multilingual retrieval** (advanced): cross-lingual embeddings | `ir_core/representation/embeddings.py`, `ir_core/engine.py` |

\* advanced/bonus feature.

---

## Retrieval models

All models share the same query→tokens pipeline and return a uniform ranked list, so the engine,
hybrids, and UI treat them identically.

| Model id | Name | Representation | Matching |
|----------|------|----------------|----------|
| `tfidf` | VSM · TF-IDF | sparse `ltc` term weights | cosine similarity |
| `bm25` | BM25 (probabilistic) | Okapi BM25, tunable `k1`/`b` | BM25 score |
| `bert` | Embedding · BERT | `all-MiniLM-L6-v2` sentence embeddings (384-d) | cosine |
| `word2vec` | Embedding · Word2Vec | PyTorch skip-gram, mean-pooled (100-d) | cosine |
| `multilingual` | Embedding · Multilingual 🌐 | `paraphrase-multilingual-MiniLM-L12-v2` (384-d, ~50 languages) | cosine (cross-lingual) |
| `hybrid_parallel` | Hybrid · Parallel | runs several models, fuses rankings | weighted-sum or RRF |
| `hybrid_serial` | Hybrid · Serial | one model retrieves, another re-ranks | stage-2 score |

- **TF-IDF (VSM):** classic `ltc` weighting — `weight(t) = (1 + ln tf) · idf(t)` — with cosine
  similarity. Document L2 norms are precomputed at index-build time.
- **BM25:** Okapi BM25 with Robertson/Spärck-Jones IDF; `k1` (term-frequency saturation) and `b`
  (length normalization) are tunable **per query** (defaults `k1=1.5`, `b=0.75`).
- **BERT / Word2Vec / Multilingual:** dense retrievers — encode the query, cosine-search a
  disk-backed vector store. Word2Vec is a genuine skip-gram model trained in PyTorch (no gensim);
  Multilingual enables cross-lingual queries.
- **Hybrid · Parallel:** each component model retrieves a candidate pool; scores are min-max
  normalized and fused by **weighted sum** or **Reciprocal Rank Fusion (RRF)**.
- **Hybrid · Serial:** a fast high-recall model (e.g. BM25) retrieves candidates, then a stronger
  model (e.g. BERT) re-ranks only those candidates.

---

## Architecture

The retrieval logic lives once in `ir_core`; every front-end reuses it.

```
        ┌───────────────┐          ┌──────────────────────────────┐
        │  Streamlit UI │          │   CLI tools (build / eval)    │
        └──────┬────────┘          └───────────────┬──────────────┘
               │ REST                              │ in-process
        ┌──────▼───────────────────────────────────────────────────┐
        │            API Gateway  (port 8000)                       │
        │   routes / composes calls to the 5 microservices          │
        └──────┬───────────────────────────────────────────────────┘
   ┌───────────┼───────────┬───────────────┬──────────────┐
   ▼           ▼           ▼               ▼              ▼
 prepro(8001) index(8002) retrieval(8003) rank-eval(8004) refine(8005)
   └───────────┴───────────┴───────┬───────┴──────────────┘
                                   ▼
        ┌───────────────────────────────────────────────┐
        │        ir_core  (shared engine library)        │
        │  text · data · index · representation ·        │
        │  query · eval · engine.Engine (Facade)         │
        └──────────────────────┬─────────────────────────┘
                               ▼
        ┌───────────────────────────────────────────────┐
        │   Data layer (per dataset, indexes/<key>/)     │
        │  inverted_index.sqlite (full docs, zlib BLOBs) │
        │  emb_{bert,word2vec,multilingual}.vectors.f32.gz│
        │  *.docids.json.gz · word2vec.vectors.npz       │
        └───────────────────────────────────────────────┘
```

**Microservices** (each FastAPI, with a `/health` route):

| Service | Port | Responsibility |
|---------|------|----------------|
| Preprocessing | 8001 | normalize text → tokens (same pipeline as the corpus) |
| Indexing | 8002 | index build status, vocabulary lookups |
| Retrieval | 8003 | matching &amp; ranking (delegates to `ir_core.engine.ENGINE`) |
| Ranking &amp; Evaluation | 8004 | MAP / Recall / P@10 / nDCG@10, before/after refinement |
| Query Refinement | 8005 | spell-correct, expand, suggest, personalize |
| **API Gateway** | **8000** | single public entry point; routes &amp; composes (`/search_refined`) |

**Design patterns:** the `Engine` is a **Facade** over all representations + the refiner; every
retriever follows the same `search`/`rescore` interface (**Strategy**); the gateway is a
**Gateway/Facade** holding no business logic; a process-wide `ENGINE` **singleton** keeps loaded
indexes warm.

---

## Storage &amp; indexing

Everything is **built once and persisted compressed**; nothing is trained or fitted at query time.

- **Inverted index** (`indexes/argsme/inverted_index.sqlite`): SQLite tables for `meta`, `docs`,
  `terms`, `postings`. Built by **streaming** the preprocessed JSON (so it scales past RAM). It
  precomputes document frequencies, IDF, and per-document TF-IDF L2 norms, and stores each
  document's **complete original text** as a **zlib-compressed BLOB** — so the UI can show the full,
  readable document straight from the database.
- **Dense vector stores** (`emb_<model>.vectors.f32.gz`): L2-normalized float32 vectors,
  **gzip-streamed** to disk; inflated once into memory at search time and scored with a batched
  cosine matrix-vector product. Doc-id sidecars are gzip'd JSON.
- **Word2Vec model**: matrix saved as a compressed `.npz`; vocabulary as gzip'd JSON.
- **No first-query training:** TF-IDF and BM25 score from the prebuilt SQLite index; dense models
  load prebuilt vectors. The first dense query inflates its compressed vector file into memory once,
  then queries are fast.

---

## Query processing &amp; refinement

- **Shared preprocessing** (`ir_core/text/preprocessing.py`): lowercase → punctuation removal →
  tokenization (alphabetic only) → stopword removal → Porter stemming. The **identical** pipeline
  is applied to documents at index time and to queries at search time, guaranteeing the same term
  space.
- **Refinement** (`ir_core/query/refinement.py`), all **corpus-aware** (only keeps terms that
  actually occur in the index, `df > 0`):
  - **Spell-correction** — Norvig edit-distance against the English dictionary, validated against
    the corpus vocabulary.
  - **WordNet synonym expansion** — adds up to N synonyms per query word.
  - **Query suggestion / auto-complete** — from history and the index vocabulary.
  - **History personalization** — biases retrieval toward terms from related past queries.

---

## Evaluation

Standard IR metrics over the dataset's relevance judgments (qrels), in `ir_core/eval/`:

- **MAP** (Mean Average Precision), **Recall**, **Precision@10**, **nDCG@10** (graded gains).
- Only **judged** queries are scored (Args.me has **49** judged Touché-2020 queries).
- **Before/after query-refinement** comparison to quantify the effect of the additional features.

Run from the UI's **Evaluation** tab (with bar charts and a before→after delta view) or from the
CLI (see below). The full analysis is in [`docs/REPORT.md`](docs/REPORT.md).

---

## The web UI

A single-page **Streamlit** app (`ui/app.py`) with a modern dark theme. It talks to the system
**through the SOA gateway** (service orientation is the architecture, not a user option), so start
the backend first (see [Quick start](#quick-start)).

**Sidebar:** gateway URL · dataset picker · index-status badges · representation-model picker (with
per-model help) · live **BM25 `k1`/`b`** sliders · hybrid configuration (models to fuse, fusion
method; serial stage-1/stage-2 + candidate pool) · **query-refinement** toggles (spelling, synonym
expansion, history) · top-k slider.

**🔍 Search tab:** query box + clickable example queries; live suggestions; ranked result cards
(rank, doc id, score bar, query-term **highlighting**, per-model component scores for hybrids); a
**"📄 Read full document"** expander that shows the complete original text inline; search history.

**📊 Evaluation tab:** run MAP/Recall/P@10/nDCG@10 for the selected model, and a
**before/after-refinement comparison** with grouped bar charts, a per-metric Δ chart, and a colored
delta table.

---

## Cross-lingual (multilingual) retrieval

The corpus is English, so "multilingual" here means **cross-lingual query support**: a query in
**Arabic, French, Spanish, German, …** retrieves the relevant **English** arguments — no translation
step. Both the query and the documents are encoded by the same **multilingual sentence model**
(`paraphrase-multilingual-MiniLM-L12-v2`, ~50 languages), so they live in one shared vector space.

- **Build it** (one-time, encodes all docs — a multi-hour CPU job for the full corpus):
  ```bash
  python scripts/build_indexes.py --dataset argsme --models multilingual --limit 0
  ```
- **Use it:** pick **"Embedding · Multilingual 🌐"** in the UI model dropdown, or call the API with
  `"model": "multilingual"`.
- **Quality note:** Latin-script languages (FR/ES/DE/IT/PT…) match very strongly; Arabic/CJK are
  supported but weaker with this model. For stronger coverage set
  `IR_MULTILINGUAL_MODEL=sentence-transformers/LaBSE` and rebuild (768-d, 109 languages, slower).

---

## Installation

```bash
pip install -r requirements.txt
```

- **Python 3.14** is supported.
- **`gensim` is intentionally not used** (no 3.14 wheel + needs a C compiler) — Word2Vec is trained
  directly in **PyTorch**.
- Sentence-transformer models download automatically on first use:
  `all-MiniLM-L6-v2` (~90 MB, BERT) and `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB,
  multilingual).
- The preprocessed corpus `argsme_processed.json` is expected at the project root.

---

## Quick start

```bash
# 1. (one-time) build the indexes you want — see "Building the indexes"
python scripts/build_indexes.py --dataset argsme --models inverted --limit 0      # TF-IDF + BM25
python scripts/build_indexes.py --dataset argsme --models word2vec,bert --limit 0 # dense models

# 2. start the SOA backend (5 services + gateway)
python -m services.run_all
#    gateway docs at http://127.0.0.1:8000/docs

# 3. in another terminal, start the UI (talks to the gateway)
streamlit run ui/app.py
```

A 5-document smoke test that needs no corpus download:

```bash
python scripts/smoke_test.py
```

---

## Building the indexes

`scripts/build_indexes.py` streams the preprocessed corpus and builds the requested
representations. `--limit 0` indexes the whole corpus; a positive limit caps it.

```bash
# lexical index (TF-IDF + BM25 share it), full corpus
python scripts/build_indexes.py --dataset argsme --models inverted --limit 0

# dense models
python scripts/build_indexes.py --dataset argsme --models word2vec,bert --limit 0
python scripts/build_indexes.py --dataset argsme --models multilingual   --limit 0

# everything at once (a 50k-doc cap is handy for a quick demo)
python scripts/build_indexes.py --dataset argsme --models inverted,word2vec,bert,multilingual --limit 50000
```

`--models` accepts any comma-list of `inverted,word2vec,bert,multilingual`. Useful flags:
`--w2v-dim`, `--w2v-epochs`, `--w2v-min-count`, `--bert-batch`, `--w2v-train-limit`.

> Encoding the full corpus with BERT/Multilingual on CPU takes a while (the multilingual full build
> is ~1–3 h). Use `--limit` for a faster demo.

---

## Command-line tools

| Script | Purpose |
|--------|---------|
| `scripts/build_indexes.py` | build inverted / word2vec / bert / multilingual indexes |
| `scripts/evaluate.py` | evaluate one model; `--compare` for before/after refinement |
| `scripts/run_full_eval.py` | evaluate **all** available models → Markdown table + JSON |
| `scripts/smoke_test.py` | end-to-end check on the built-in 5-doc sample |
| `process_docs.py` | preprocess an `ir_datasets` corpus → `*_processed.json` |
| `check_ready.py` / `download_dataset.py` | verify / pre-fetch the dataset via `ir_datasets` |

```bash
# single model, with the before/after-refinement comparison
python scripts/evaluate.py --dataset argsme --model bm25 --num-queries 49 --compare

# all models in one table (used to populate docs/REPORT.md)
python scripts/run_full_eval.py --dataset argsme --num-queries 49 --refine-compare
```

---

## REST API (gateway)

All requests go through the gateway at `http://127.0.0.1:8000` (interactive docs at `/docs`).

| Method &amp; path | Description |
|---------------|-------------|
| `GET /services` | health roll-up of all microservices |
| `GET /datasets` | available datasets |
| `GET /status/{dataset}` | index build status (docs, which models exist) |
| `GET /models/{dataset}` | retrieval models available for the dataset |
| `POST /search` | ranked retrieval (any model) |
| `POST /refine` · `POST /suggest` | query refinement / suggestions |
| `POST /evaluate` · `POST /compare` | metrics / before-after refinement |
| `POST /search_refined` | composition: refine the query, then search |

```bash
# search with a hybrid model
curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" \
  -d '{"dataset":"argsme","model":"hybrid_serial","query":"school uniforms","top_k":10}'

# cross-lingual search (French query → English results)
curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" \
  -d '{"dataset":"argsme","model":"multilingual","query":"les étudiants devraient-ils porter un uniforme","top_k":10}'
```

---

## Project layout

```
ir_core/                       # the IR engine (importable library)
  config.py                    #   dataset registry, paths, model names, service URLs
  types.py                     #   shared SearchResult type
  text/preprocessing.py        #   the shared doc/query normalization pipeline
  data/corpus.py               #   streaming corpus + qrels/queries (ir_datasets)
  index/
    inverted_index.py          #   SQLite inverted index (full docs as zlib BLOBs)
    vector_store.py            #   gzip'd dense vector store
  representation/
    tfidf.py · bm25.py         #   lexical scorers (share the inverted index)
    embeddings.py              #   BERT + Multilingual encoders, build helpers, retriever
    word2vec.py                #   PyTorch skip-gram model + persistence
    hybrid.py                  #   parallel fusion + serial re-rank
    base.py                    #   QueryContext + Lexical/Dense retriever wrappers
  query/processing.py          #   build QueryContext from a raw query
  query/refinement.py          #   spell / expand / suggest / history (corpus-aware)
  eval/metrics.py · evaluate.py#   MAP/Recall/P@10/nDCG@10 + runner
  engine.py                    #   orchestrator (Facade) used by services & UI
services/                      # SOA backend
  common.py                    #   FastAPI app factory + tiny HTTP client
  gateway.py                   #   API gateway (port 8000)
  preprocessing_service.py     #   8001
  indexing_service.py          #   8002
  retrieval_service.py         #   8003
  ranking_eval_service.py      #   8004
  query_refinement_service.py  #   8005
  run_all.py                   #   launches all services + gateway
ui/app.py                      # Streamlit UI
scripts/                       # build_indexes / evaluate / run_full_eval / smoke_test
docs/                          # ARCHITECTURE.md · REPORT.md · CODE_REFERENCE.{md,html,pdf}
ir_win_fix.py                  # Windows fix for an ir_datasets temp-file bug
preprocessing.py               # standalone preprocessing used to build the corpus JSON
indexes/<dataset>/             # built artifacts (SQLite + compressed vectors)
```

---

## Tech stack

- **Core:** Python 3.14, NumPy, SciPy, scikit-learn, NLTK, `ir_datasets`, `ijson` (streaming JSON).
- **Embeddings:** PyTorch (Word2Vec skip-gram), `sentence-transformers` + `transformers`
  (BERT &amp; multilingual).
- **Services &amp; UI:** FastAPI, Uvicorn, Pydantic, Streamlit, Pandas.
- **Storage:** SQLite (inverted index, with zlib-compressed document BLOBs), gzip/`.npz`
  (compressed dense artifacts).

---

## Design notes

- **Full documents live in the database** (compressed), and the top-k results are served from it —
  so the UI shows the complete, readable document without opening anything elsewhere.
- **Models &amp; indexes are stored compressed**, and nothing is trained at query time — the first
  run is "build once," and every search after loads prebuilt artifacts.
- **`ir_win_fix.py`** patches an `ir_datasets` temp-file rename bug on Windows; it's imported before
  any dataset load.
- The system is intentionally **from-scratch** — custom BM25, custom PyTorch Word2Vec, custom
  vector store — rather than leaning on a heavy IR framework.

---

## Further documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — design, diagram, patterns, service contracts.
- [`docs/REPORT.md`](docs/REPORT.md) — full evaluation analysis (per-model results, the effect of
  each representation, refinement, and parameter choices).
- [`docs/CODE_REFERENCE.md`](docs/CODE_REFERENCE.md) (also `.html` / `.pdf`) — every file and every
  function/class/method documented.
```
