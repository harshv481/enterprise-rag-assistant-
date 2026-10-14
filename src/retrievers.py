"""Retrieval engine supporting Keyword (TF-IDF), Semantic (Embeddings), and Parallel Hybrid Retrieval."""
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models import DocumentChunk, RetrievedChunk

logger = logging.getLogger("rag_retrieval")


class KeywordRetriever:
    """Keyword-based lexical search using TF-IDF."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrix = None
        self.chunks: List[DocumentChunk] = []

    def fit(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        if not chunks:
            self.tfidf_matrix = None
            return
        corpus = [c.content for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks or self.tfidf_matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:  # Only return non-zero matching candidates
                results.append((self.chunks[idx], score))
        return results


class SemanticRetriever:
    """Semantic vector search using SentenceTransformers or Fallback TF-IDF Dense Cosine."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[DocumentChunk] = []
        self.is_transformer = False
        self._init_encoder()

    def _init_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            self.is_transformer = True
        except Exception as e:
            logger.warning(
                f"SentenceTransformer unavailable ({e}). Using Fallback Dense Semantic Vectorizer."
            )
            self.model = None
            self.is_transformer = False
            self.fallback_vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True
            )

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.is_transformer and self.model:
            embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings
        else:
            # Fallback subword character-n-gram dense representation
            vecs = self.fallback_vectorizer.transform(texts).toarray()
            # Normalize vectors
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            return vecs / norms

    def fit(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        if not chunks:
            self.embeddings = None
            return

        corpus = [f"{c.doc_title} - {c.section_title}: {c.content}" for c in chunks]
        if not self.is_transformer:
            self.fallback_vectorizer.fit(corpus)

        self.embeddings = self.encode(corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks or self.embeddings is None:
            return []
        query_emb = self.encode([query])  # Shape: (1, dim)
        scores = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            results.append((self.chunks[idx], max(0.0, score)))
        return results


class ParallelHybridRetriever:
    """
    Executes Keyword and Semantic retrieval concurrently in parallel threads
    and merges scores via Reciprocal Rank Fusion & Convex Combination.
    """

    def __init__(self, keyword_retriever: KeywordRetriever, semantic_retriever: SemanticRetriever):
        self.keyword_retriever = keyword_retriever
        self.semantic_retriever = semantic_retriever
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag-retriever")

    def fit(self, chunks: List[DocumentChunk]):
        # Index chunks in parallel across both retrievers
        kw_future = self.executor.submit(self.keyword_retriever.fit, chunks)
        sem_future = self.executor.submit(self.semantic_retriever.fit, chunks)
        kw_future.result()
        sem_future.result()

    def retrieve(
        self, query: str, top_k: int = 5, alpha: float = 0.5
    ) -> Tuple[List[RetrievedChunk], Dict[str, float]]:
        """
        Executes parallel retrieval.
        alpha: weight for semantic score (0.0 to 1.0). 1 - alpha for keyword score.
        Returns: (merged_results, timing_metrics)
        """
        start_time = time.perf_counter()

        # Dispatch keyword and semantic retrieval concurrently
        t0 = time.perf_counter()
        kw_future = self.executor.submit(self.keyword_retriever.retrieve, query, top_k * 2)
        sem_future = self.executor.submit(self.semantic_retriever.retrieve, query, top_k * 2)

        kw_results = kw_future.result()
        kw_latency = (time.perf_counter() - t0) * 1000

        sem_results = sem_future.result()
        sem_latency = (time.perf_counter() - t0) * 1000

        total_parallel_latency = (time.perf_counter() - start_time) * 1000

        # Normalization and Fusion
        merged_scores: Dict[str, Dict] = {}

        # 1. Process keyword scores
        max_kw = max([s for _, s in kw_results], default=1.0)
        max_kw = max_kw if max_kw > 0 else 1.0
        for rank, (chunk, score) in enumerate(kw_results, start=1):
            norm_score = score / max_kw
            merged_scores[chunk.chunk_id] = {
                "chunk": chunk,
                "kw_score": norm_score,
                "sem_score": 0.0,
                "kw_rank": rank,
                "sem_rank": 999,
            }

        # 2. Process semantic scores
        max_sem = max([s for _, s in sem_results], default=1.0)
        max_sem = max_sem if max_sem > 0 else 1.0
        for rank, (chunk, score) in enumerate(sem_results, start=1):
            norm_score = score / max_sem
            if chunk.chunk_id not in merged_scores:
                merged_scores[chunk.chunk_id] = {
                    "chunk": chunk,
                    "kw_score": 0.0,
                    "sem_score": norm_score,
                    "kw_rank": 999,
                    "sem_rank": rank,
                }
            else:
                merged_scores[chunk.chunk_id]["sem_score"] = norm_score
                merged_scores[chunk.chunk_id]["sem_rank"] = rank

        # 3. Calculate Hybrid Score with Reciprocal Rank Fusion (RRF) smoothing
        # RRF formula: RRF_score = 1 / (60 + kw_rank) + 1 / (60 + sem_rank)
        hybrid_candidates = []
        for chunk_id, data in merged_scores.items():
            rrf = (1.0 / (60 + data["kw_rank"])) + (1.0 / (60 + data["sem_rank"]))
            weighted = (alpha * data["sem_score"]) + ((1.0 - alpha) * data["kw_score"])
            # Combined hybrid score (weighted dense + lexical + rank agreement)
            combined_score = (0.7 * weighted) + (0.3 * (rrf * 30.0))
            hybrid_candidates.append((data["chunk"], combined_score, data["kw_score"], data["sem_score"]))

        # Sort descending by combined score
        hybrid_candidates.sort(key=lambda x: x[1], reverse=True)

        final_chunks: List[RetrievedChunk] = []
        for rank, (chunk, score, kw, sem) in enumerate(hybrid_candidates[:top_k], start=1):
            final_chunks.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=round(score, 4),
                    retrieval_method="hybrid",
                    rank=rank,
                )
            )

        timing_breakdown = {
            "parallel_retrieval_ms": round(total_parallel_latency, 2),
            "keyword_latency_ms": round(kw_latency, 2),
            "semantic_latency_ms": round(sem_latency, 2),
            "keyword_hits": len(kw_results),
            "semantic_hits": len(sem_results),
            "hybrid_candidates": len(hybrid_candidates),
        }

        return final_chunks, timing_breakdown
