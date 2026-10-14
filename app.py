"""FastAPI Web Server for Enterprise RAG Assistant."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.models import QueryRequest, QueryResponse
from src.rag_pipeline import EnterpriseRAGPipeline

app = FastAPI(
    title="Enterprise RAG Assistant API",
    description="Role-aware question-answering with low-latency parallel hybrid retrieval and semantic caching.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = EnterpriseRAGPipeline(docs_dir="docs")


@app.get("/")
def root():
    return {
        "service": "Enterprise RAG Assistant",
        "status": "online",
        "endpoints": {
            "query": "POST /query",
            "documents": "GET /documents",
            "cache_stats": "GET /cache/stats",
            "clear_cache": "DELETE /cache",
            "health": "GET /health",
        },
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Execute role-aware query with parallel hybrid retrieval, reranking,
    source-grounding, and semantic caching.
    """
    try:
        response = pipeline.query(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents():
    """List all ingested documents and their role permissions."""
    return pipeline.doc_store.get_summary()


@app.get("/cache/stats")
def get_cache_stats():
    """Retrieve semantic cache hit/miss rates and latency savings."""
    return pipeline.cache.get_stats()


@app.delete("/cache")
def clear_cache():
    """Clear semantic cache entries."""
    pipeline.cache.clear()
    return {"status": "Cache cleared successfully", "stats": pipeline.cache.get_stats()}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "semantic_encoder": "SentenceTransformer" if pipeline.semantic_retriever.is_transformer else "Fallback TF-IDF Dense",
        "total_documents": len(pipeline.doc_store.doc_registry),
        "total_chunks": len(pipeline.doc_store.chunks),
    }
