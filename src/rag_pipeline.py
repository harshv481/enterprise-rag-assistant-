"""Unified Enterprise RAG Pipeline orchestrating RBAC, Caching, Parallel Retrieval, and Grounding."""
import time
import logging
from typing import Dict, Any, Optional

from src.models import QueryRequest, QueryResponse
from src.document_store import DocumentStore
from src.retrievers import KeywordRetriever, SemanticRetriever, ParallelHybridRetriever
from src.reranker import ReRanker
from src.cache import SemanticCache
from src.generator import GroundedGenerator

logger = logging.getLogger("rag_pipeline")


class EnterpriseRAGPipeline:
    def __init__(self, docs_dir: str = "docs"):
        self.doc_store = DocumentStore(docs_dir=docs_dir)
        self.keyword_retriever = KeywordRetriever()
        self.semantic_retriever = SemanticRetriever()
        self.parallel_retriever = ParallelHybridRetriever(
            keyword_retriever=self.keyword_retriever,
            semantic_retriever=self.semantic_retriever,
        )
        self.reranker = ReRanker()
        self.cache = SemanticCache(similarity_threshold=0.88)
        self.generator = GroundedGenerator()

        logger.info(
            f"Initialized Enterprise RAG Pipeline with {len(self.doc_store.chunks)} chunks "
            f"across {len(self.doc_store.doc_registry)} documents."
        )

    def query(self, request: QueryRequest) -> QueryResponse:
        start_time = time.perf_counter()
        user_role = request.user_role.strip().lower()
        query_str = request.query.strip()

        # Step 1: Compute query embedding for semantic cache & retrieval
        query_emb = self.semantic_retriever.encode([query_str])[0]

        # Step 2: Semantic Cache Lookup (Role-Isolated)
        if request.use_cache:
            cached_resp, sim = self.cache.get(query_str, user_role, query_emb)
            if cached_resp is not None:
                total_latency = (time.perf_counter() - start_time) * 1000
                cached_resp.latency_ms = round(total_latency, 2)
                self.cache.record_saved_latency(baseline_ms=25.0)  # estimated pipeline time
                return cached_resp

        # Step 3: RBAC Document Filtering
        role_chunks = self.doc_store.get_chunks_for_role(user_role)
        if not role_chunks:
            total_latency = (time.perf_counter() - start_time) * 1000
            answer, citations = self.generator.generate(query_str, user_role, [])
            return QueryResponse(
                query=query_str,
                user_role=user_role,
                answer=answer,
                citations=[],
                retrieved_chunks=[],
                cache_hit=False,
                cache_similarity=None,
                latency_ms=round(total_latency, 2),
                retrieval_breakdown={"rbac_status": "No documents permitted for role"},
            )

        # Step 4: Index role-permitted chunks
        # In multi-tenant enterprise settings, indices are partitioned or pre-filtered
        self.parallel_retriever.fit(role_chunks)

        # Step 5: Parallel Hybrid Retrieval (Keyword + Semantic)
        hybrid_candidates, timing_metrics = self.parallel_retriever.retrieve(
            query=query_str,
            top_k=request.top_k * 2,
            alpha=request.alpha,
        )

        # Step 6: Contextual Re-ranking
        rerank_start = time.perf_counter()
        reranked_chunks = self.reranker.rerank(
            query=query_str,
            candidates=hybrid_candidates,
            top_k=request.top_k,
        )
        rerank_latency = (time.perf_counter() - rerank_start) * 1000
        timing_metrics["reranker_latency_ms"] = round(rerank_latency, 2)

        # Step 7: Source-Grounded Response Synthesis
        gen_start = time.perf_counter()
        answer, citations = self.generator.generate(
            query=query_str,
            user_role=user_role,
            retrieved_chunks=reranked_chunks,
        )
        gen_latency = (time.perf_counter() - gen_start) * 1000
        timing_metrics["generation_latency_ms"] = round(gen_latency, 2)

        total_latency = (time.perf_counter() - start_time) * 1000

        response = QueryResponse(
            query=query_str,
            user_role=user_role,
            answer=answer,
            citations=citations,
            retrieved_chunks=reranked_chunks,
            cache_hit=False,
            cache_similarity=None,
            latency_ms=round(total_latency, 2),
            retrieval_breakdown=timing_metrics,
        )

        # Step 8: Populate Semantic Cache
        if request.use_cache and citations:
            self.cache.put(
                query=query_str,
                user_role=user_role,
                query_vector=query_emb,
                response_data=response,
            )

        return response
