"""Cross-scoring and contextual re-ranker for retrieved candidate chunks."""
import re
from typing import List
from src.models import RetrievedChunk


class ReRanker:
    """
    Reranks candidate chunks using cross-context scoring:
    - Query term density in title vs body
    - Exact query sub-phrase alignment
    - Contextual term proximity
    - Length normalization penalty
    """

    def __init__(self, title_weight: float = 1.8, exact_phrase_weight: float = 2.0):
        self.title_weight = title_weight
        self.exact_phrase_weight = exact_phrase_weight

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int = 3) -> List[RetrievedChunk]:
        if not candidates:
            return []

        query_tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
        query_lower = query.lower().strip()

        scored_candidates = []
        for item in candidates:
            chunk = item.chunk
            base_score = item.score

            body_lower = chunk.content.lower()
            title_lower = f"{chunk.doc_title} {chunk.section_title}".lower()

            # 1. Title keyword overlap bonus
            title_hits = sum(1 for token in query_tokens if token in title_lower)
            title_boost = (title_hits / max(len(query_tokens), 1)) * 0.25 * self.title_weight

            # 2. Body token coverage
            body_hits = sum(1 for token in query_tokens if token in body_lower)
            coverage_boost = (body_hits / max(len(query_tokens), 1)) * 0.35

            # 3. Exact phrase matching bonus
            phrase_boost = 0.0
            if len(query_tokens) > 1 and query_lower in body_lower:
                phrase_boost = 0.30 * self.exact_phrase_weight
            elif any(query_tokens[i] + " " + query_tokens[i + 1] in body_lower for i in range(len(query_tokens) - 1)):
                phrase_boost = 0.15

            # 4. Synthesize final reranked score
            rerank_score = (0.50 * base_score) + title_boost + coverage_boost + phrase_boost

            scored_candidates.append((chunk, round(rerank_score, 4)))

        # Sort descending by re-ranked score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        reranked_chunks: List[RetrievedChunk] = []
        for rank, (chunk, score) in enumerate(scored_candidates[:top_k], start=1):
            reranked_chunks.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    retrieval_method="reranked",
                    rank=rank,
                )
            )

        return reranked_chunks
