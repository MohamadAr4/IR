"""Document-representation / scoring models.

Each model exposes ``search(query_tokens, top_k, ...) -> list[SearchResult]``
so the retrieval layer and the hybrid combiner can treat them uniformly.
"""
