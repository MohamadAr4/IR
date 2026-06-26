# Evaluation Report

This report covers the analysis required by requirement #8: per-model
effectiveness on the standard IR metrics, the effect of each representation on
retrieval quality, a comparison of the models, the contribution of the
additional features (query refinement), and the justification of model/parameter
choices.

## 1. Setup

- **Dataset:** Args.me (Touché 2020 Task 1, 387,692 docs). Queries and qrels
  come from `ir_datasets`.
- **Index coverage:** *all* models are built over the full corpus — the lexical
  inverted index and the dense (BERT + Word2Vec) indexes embed every document,
  so the numbers below have 100 % qrel coverage.
- **Metrics:** MAP, Recall, Precision@10, nDCG@10 (graded gains from qrels).
  Computed in `ir_core/eval/metrics.py`; unit-checked against a worked example.
- **Protocol:** only judged queries are scored; retrieval depth = 100 for
  MAP/Recall, cut-offs of 10 for P@10/nDCG@10. All 49 judged Touché-2020 queries
  are used. Each model is evaluated *before* and *after* query refinement
  (spelling correction + WordNet synonym expansion).
- **Reproduce:**
  ```bash
  python scripts/run_full_eval.py --dataset argsme --num-queries 49 --refine-compare
  ```

## 2. Models compared

| Model | Representation | Matching |
|-------|----------------|----------|
| TF-IDF (VSM) | sparse `ltc` term weights | cosine |
| BM25 | probabilistic, `k1=1.5`, `b=0.75` | Okapi BM25 score |
| Word2Vec | mean of trained skip-gram word vectors | cosine |
| BERT | all-MiniLM-L6-v2 sentence embedding | cosine |
| Hybrid (parallel) | BM25 + BERT, fused | weighted / RRF |
| Hybrid (serial) | BM25 recall → BERT rerank | stage-2 score |

## 3. Results

Measured on the **full Args.me corpus (387,692 documents)** over all **49 judged
Touché-2020 queries**, retrieval depth 100. Numbers are reproducible (retrieval
is deterministic for a fixed index and parameters) via `scripts/run_full_eval.py`.

### 3.1 Args.me — single models

| Model | MAP | Recall | P@10 | nDCG@10 |
|-------|-----|--------|------|---------|
| VSM · TF-IDF (cosine) | 0.0306 | 0.1613 | 0.1020 | 0.0412 |
| Embedding · Word2Vec (skip-gram, mean-pooled) | 0.0498 | 0.2023 | 0.1857 | 0.1062 |
| Embedding · BERT (all-MiniLM-L6-v2) | 0.0611 | 0.2252 | 0.2122 | 0.1254 |
| **BM25** (k1=1.5, b=0.75) | **0.2635** | **0.4387** | **0.6980** | **0.5600** |

All rows are measured on the **full 387,692-document corpus** with **100 % qrel
coverage** (BERT and Word2Vec embed every document). Note the clean ordering of
the dense models — **BERT > Word2Vec > TF-IDF** — i.e. contextual ≥ static ≥
none, exactly as theory predicts.

### 3.2 Args.me — hybrid models

| Hybrid | Components | Fusion | MAP | Recall | P@10 | nDCG@10 |
|--------|-----------|--------|-----|--------|------|---------|
| Parallel | BM25 + BERT | weighted | 0.1788 | 0.3894 | 0.5286 | 0.3827 |
| Parallel | BM25 + BERT | RRF | 0.1814 | 0.4021 | 0.4980 | 0.3672 |
| Serial | BM25 → BERT rerank | — | 0.1465 | 0.4387 | 0.3327 | 0.2175 |
| Parallel | BM25 + TF-IDF | weighted | 0.1312 | 0.3533 | 0.3980 | 0.2544 |
| Serial | BM25 → TF-IDF rerank | — | 0.1199 | 0.4387 | 0.2592 | 0.1330 |

### 3.3 Effect of query refinement (BM25, before → after)

| Configuration | MAP | Recall | P@10 | nDCG@10 |
|---------------|-----|--------|------|---------|
| Baseline (basic pipeline) | 0.2635 | 0.4387 | 0.6980 | 0.5600 |
| + spelling correction only | 0.2635 | 0.4387 | 0.6980 | 0.5600 |
| + synonym expansion only | 0.1535 | 0.3647 | 0.4265 | 0.3428 |
| + spell **and** expansion | 0.1535 | 0.3647 | 0.4265 | 0.3428 |
| **delta (full refinement)** | **−0.110** | **−0.074** | **−0.272** | **−0.217** |

## 4. Analysis

**BM25 vs TF-IDF — effect of the representation (measured).** BM25 hugely
outperforms TF-IDF cosine here (MAP 0.264 vs 0.031, P@10 0.70 vs 0.10). Both
match on the same exact (stemmed) terms, so the gap is entirely about
*weighting*. Args.me documents are long arguments (avg ≈ 147 tokens); cosine
similarity divides by the document's L2 norm, which over-penalises exactly these
long, relevant documents, while a 1–2 word query vector pulls short documents to
the top. BM25 instead uses **soft** length normalisation (`b`) plus
term-frequency **saturation** (`k1`), so a longer relevant document is no longer
unfairly demoted and repeated terms stop dominating. This is a textbook
illustration of why BM25 is the standard lexical baseline and why the brief
requires its parameters to be tunable (requirement #2 note 2).

**The dense models (measured).** Both embedding models sit between TF-IDF and
BM25, in the order theory predicts: **BERT (0.061 MAP) > Word2Vec (0.050) >
TF-IDF (0.031)**. Contextual sentence embeddings beat the static mean-of-word
vectors (which throw away word order), and both beat raw TF-IDF cosine because
they match on meaning, recovering some relevant documents that share no query
term. But on this dataset both are far below BM25 — see the next point for why.

**Why the hybrids underperform BM25 here (measured).** Every hybrid scores
*below* BM25 alone (best hybrid 0.181 MAP vs BM25 0.264). This is *not* a fusion
bug — it is a direct consequence of component quality. Fusion can only help when
the models are **complementary AND individually competent**. On Args.me the
dense signal is weak: `all-MiniLM-L6-v2` is a general-purpose encoder, and
*argument* retrieval (does this passage argue the claim?) is a hard semantic task
it was never tuned for, so BERT alone manages only 0.061 MAP. Fusing a strong
lexical ranker with a weak semantic one drags the strong one down. Two clear
sub-results: (1) the **BM25+BERT** hybrids (0.18 MAP, RRF ≈ weighted) clearly
beat the **BM25+TF-IDF** hybrids (0.13), because BERT is a better partner than
TF-IDF — *component choice matters*; and (2) the **serial** hybrid preserves
BM25's recall exactly (0.4387 — it only re-ranks BM25's own pool) but its
precision falls because the weak re-ranker reshuffles good results. The general
lesson is that *the best representation is dataset-dependent*: on a corpus where
the dense encoder is strong (e.g. short factual passages, the encoder's home
turf) the dense and hybrid rows would be expected to beat BM25 — which is exactly
why the system offers all representations behind one interface.

**The length-normalisation story.** The TF-IDF collapse is specifically a
length-normalisation effect. On **Args.me** (long documents, avgdl ≈ 147) TF-IDF
cosine drops to MAP 0.031, a ~9× gap below BM25 (0.264), because cosine's L2
normalisation over-penalises long relevant documents. The prediction is that on a
corpus of uniformly *short* documents the same TF-IDF code would be competitive,
since cosine's L2 normalisation only mis-ranks when document lengths vary a lot.
That contrast is the clearest evidence for *why* BM25's tunable length
normalisation (`b`) matters: it is the one knob that rescues long-document
collections like this one.

**Contribution of the additional features (measured).** The before/after table
isolates each feature. Spelling correction is **exactly neutral** on this query
set — the Touché queries are well-formed natural-language questions, so no token
falls outside the corpus vocabulary and no correction fires (it is a safety net
that only acts on malformed input). WordNet synonym expansion, by contrast,
**lowers every metric** (MAP −0.11, P@10 −0.27): expanding every query word with
up to two synonyms introduces non-topical terms that drift the query away from
the user's specific intent, and on already well-specified questions that costs
precision. Expansion is fundamentally a *recall* device for vocabulary-mismatch
or under-specified queries; on this dataset the queries are neither, so it
hurts. The corpus-aware filter (only add terms with df > 0) limits but does not
eliminate the drift. **Takeaway:** keep spell-correction always on (free safety),
but apply expansion selectively (short/zero-result queries) rather than globally
— the system exposes it as an independent toggle for exactly this reason.

**Justification of choices.** `k1=1.5`, `b=0.75` are robust across TREC
collections and are the defaults; the UI exposes them live so they can be tuned
per query/dataset (requirement #2 note 2). all-MiniLM-L6-v2 is chosen for its
strong quality-to-cost ratio on CPU. The default hybrid pairs BM25 with BERT
(lexical × semantic) because that is the genuinely complementary pairing — and
the measured BM25+BERT > BM25+TF-IDF gap confirms partner quality matters more
than simply adding models. RRF and weighted fusion land within noise of each
other here; RRF is the safer default because it is scale-free and needs no
score normalisation.

## 5. How to regenerate the tables

```bash
# all single + hybrid models in one table
python scripts/run_full_eval.py --dataset argsme --num-queries 49 --refine-compare

# or one model at a time
python scripts/evaluate.py --dataset argsme --model bm25 --num-queries 49
python scripts/evaluate.py --dataset argsme --model bm25 --num-queries 49 --compare
```

> **Note on scale.** Every index — lexical *and* dense (BERT + Word2Vec) — is
> built over the **full 387,692-document corpus**, so the numbers above have
> 100 % qrel coverage. Everything is streamed from disk, so the build scales to
> the full file without loading it into RAM.
