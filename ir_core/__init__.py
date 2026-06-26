"""ir_core - Information Retrieval engine.

A modular IR library implementing multiple document representations
(TF-IDF VSM, BM25, BERT & Word2Vec embeddings, and Serial/Parallel hybrids),
disk-backed indexing, query processing/refinement, ranking and evaluation.

The microservices in ``services/`` and the Streamlit UI in ``ui/`` are thin
layers over this library.
"""

__version__ = "1.0.0"
