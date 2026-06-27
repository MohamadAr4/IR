# IR System — Evaluation Results

- **Dataset:** `argsme`
- **Judged queries scored:** 30
- **Retrieval depth:** 100
- **Metrics:** MAP · Recall · P@10 · nDCG@10 (only queries with qrels are scored)
- **Refinement ("after"):** spelling correction + synonym expansion (WordNet, top sense, 1/word; synonyms down-weighted to 0.4× on lexical models, withheld from dense models)

> *Before* = raw query (baseline). *After* = baseline + query refinement. Δ is after − before; positive means refinement helped.

## 1. Baseline retrieval quality (no refinement)

| Model | MAP | Recall | P@10 | nDCG@10 |
|-------|----:|-------:|-----:|--------:|
| VSM · TF-IDF (cosine) | 0.0294 | 0.1518 | 0.1033 | 0.0407 |
| BM25 (Okapi) | 0.2430 | 0.4224 | 0.6633 | 0.5214 |
| Embedding · BERT (MiniLM) | 0.0484 | 0.1997 | 0.1900 | 0.1136 |
| Embedding · Word2Vec (skip-gram) | 0.0471 | 0.1935 | 0.1867 | 0.0953 |
| Embedding · Multilingual (cross-lingual) | 0.0321 | 0.1629 | 0.1400 | 0.0714 |
| Hybrid · parallel (weighted fusion) | 0.1621 | 0.3680 | 0.4933 | 0.3487 |
| Hybrid · serial (rerank) | 0.1325 | 0.4224 | 0.3067 | 0.2023 |

## 2. After query refinement

| Model | MAP | Recall | P@10 | nDCG@10 |
|-------|----:|-------:|-----:|--------:|
| VSM · TF-IDF (cosine) | 0.0291 | 0.1540 | 0.1033 | 0.0394 |
| BM25 (Okapi) | 0.2315 | 0.4167 | 0.6267 | 0.4819 |
| Embedding · BERT (MiniLM) | 0.0484 | 0.1997 | 0.1900 | 0.1136 |
| Embedding · Word2Vec (skip-gram) | 0.0471 | 0.1935 | 0.1867 | 0.0953 |
| Embedding · Multilingual (cross-lingual) | 0.0321 | 0.1629 | 0.1400 | 0.0714 |
| Hybrid · parallel (weighted fusion) | 0.1587 | 0.3596 | 0.4833 | 0.3539 |
| Hybrid · serial (rerank) | 0.1323 | 0.4167 | 0.3033 | 0.2086 |

## 3. Effect of refinement (Δ = after − before)

| Model | ΔMAP | ΔRecall | ΔP@10 | ΔnDCG@10 |
|-------|-----:|--------:|------:|---------:|
| VSM · TF-IDF (cosine) | -0.0003 | +0.0022 | +0.0000 | -0.0013 |
| BM25 (Okapi) | -0.0115 | -0.0057 | -0.0366 | -0.0395 |
| Embedding · BERT (MiniLM) | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Embedding · Word2Vec (skip-gram) | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Embedding · Multilingual (cross-lingual) | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Hybrid · parallel (weighted fusion) | -0.0034 | -0.0084 | -0.0100 | +0.0052 |
| Hybrid · serial (rerank) | -0.0002 | -0.0057 | -0.0034 | +0.0063 |

---
*Generated in 362.8s.*
