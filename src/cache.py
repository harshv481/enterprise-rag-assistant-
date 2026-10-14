"""In-memory role-aware Semantic Cache for low-latency retrieval."""
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from src.models import QueryResponse


class CacheEntry:
    def __init__(
        self,
        query: str,
        user_role: str,
        query_vector: np.ndarray,
        response_data: QueryResponse,
        created_at: float,
        ttl_seconds: float = 3600.0,
    ):
        self.query = query
        self.user_role = user_role
        self.query_vector = query_vector
        self.response_data = response_data
        self.created_at = created_at
        self.ttl_seconds = ttl_seconds
        self.hits = 0

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


class SemanticCache:
    """
    Simulates enterprise Redis-based Semantic Caching using vector similarity and exact match.
    Features:
    - Role isolation (prevents privilege leakage across RBAC boundaries)
    - Cosine similarity threshold for semantic matches (e.g. >= 0.88)
    - Latency savings tracking
    - TTL expiration support
    """

    def __init__(self, similarity_threshold: float = 0.88, max_entries: int = 500):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.entries: List[CacheEntry] = []
        self.stats = {
            "total_lookups": 0,
            "hits": 0,
            "misses": 0,
            "exact_hits": 0,
            "semantic_hits": 0,
            "total_latency_saved_ms": 0.0,
        }

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def get(
        self, query: str, user_role: str, query_vector: np.ndarray
    ) -> Tuple[Optional[QueryResponse], Optional[float]]:
        """Look up query in semantic cache matching user role."""
        self.stats["total_lookups"] += 1
        q_norm = self._normalize(query_vector.flatten())
        query_clean = query.strip().lower()

        # Clean expired entries
        self.entries = [e for e in self.entries if not e.is_expired()]

        best_entry: Optional[CacheEntry] = None
        best_sim = -1.0

        for entry in self.entries:
            # Enforce RBAC boundary in cache
            if entry.user_role.lower() != user_role.lower():
                continue

            # Check exact string match
            if entry.query.strip().lower() == query_clean:
                entry.hits += 1
                self.stats["hits"] += 1
                self.stats["exact_hits"] += 1
                # Copy and update cache hit metadata
                cached_resp = entry.response_data.model_copy(deep=True)
                cached_resp.cache_hit = True
                cached_resp.cache_similarity = 1.0
                return cached_resp, 1.0

            # Check semantic vector similarity
            sim = float(np.dot(q_norm, entry.query_vector.flatten()))
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        # Evaluate against semantic threshold
        if best_entry and best_sim >= self.similarity_threshold:
            best_entry.hits += 1
            self.stats["hits"] += 1
            self.stats["semantic_hits"] += 1
            cached_resp = best_entry.response_data.model_copy(deep=True)
            cached_resp.cache_hit = True
            cached_resp.cache_similarity = round(best_sim, 4)
            return cached_resp, best_sim

        self.stats["misses"] += 1
        return None, None

    def put(
        self,
        query: str,
        user_role: str,
        query_vector: np.ndarray,
        response_data: QueryResponse,
        ttl_seconds: float = 3600.0,
    ) -> None:
        """Store response in cache indexed by query vector and role."""
        if len(self.entries) >= self.max_entries:
            # Evict least accessed entry
            self.entries.sort(key=lambda e: e.hits)
            self.entries.pop(0)

        q_norm = self._normalize(query_vector.flatten())
        entry = CacheEntry(
            query=query,
            user_role=user_role,
            query_vector=q_norm,
            response_data=response_data,
            created_at=time.time(),
            ttl_seconds=ttl_seconds,
        )
        self.entries.append(entry)

    def record_saved_latency(self, baseline_ms: float):
        """Record latency avoided by cache hit."""
        self.stats["total_latency_saved_ms"] += baseline_ms

    def clear(self):
        """Reset cache and statistics."""
        self.entries.clear()
        self.stats = {
            "total_lookups": 0,
            "hits": 0,
            "misses": 0,
            "exact_hits": 0,
            "semantic_hits": 0,
            "total_latency_saved_ms": 0.0,
        }

    def get_stats(self) -> Dict[str, Any]:
        hit_ratio = (
            (self.stats["hits"] / self.stats["total_lookups"] * 100)
            if self.stats["total_lookups"] > 0
            else 0.0
        )
        return {
            **self.stats,
            "hit_ratio_percent": round(hit_ratio, 2),
            "cached_entries_count": len(self.entries),
            "similarity_threshold": self.similarity_threshold,
        }
