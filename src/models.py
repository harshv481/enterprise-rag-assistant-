"""Pydantic data models for the Enterprise RAG Assistant."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    section_title: str
    content: str
    allowed_roles: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    score: float
    retrieval_method: str  # "keyword", "semantic", "hybrid", "reranked"
    rank: int


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query string")
    user_role: str = Field(..., description="Role of the querying user (e.g., employee, hr, it_admin, executive)")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of source chunks to retrieve")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Hybrid weight: 1.0=semantic, 0.0=keyword")
    use_cache: bool = Field(default=True, description="Whether to check semantic cache")


class Citation(BaseModel):
    doc_id: str
    doc_title: str
    section_title: str
    chunk_id: str
    relevance_score: float


class QueryResponse(BaseModel):
    query: str
    user_role: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
    cache_hit: bool
    cache_similarity: Optional[float] = None
    latency_ms: float
    retrieval_breakdown: Dict[str, Any] = Field(default_factory=dict)
