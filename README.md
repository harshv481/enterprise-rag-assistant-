# Enterprise RAG Assistant with Low-Latency Retrieval

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![RAG Architecture](https://img.shields.io/badge/Architecture-Hybrid%20RAG-orange?style=flat-square)](https://github.com/harshv481/enterprise-rag-assistant-)

A high-performance, self-contained Python demonstration of an **Enterprise Question-Answering System** utilizing Retrieval-Augmented Generation (RAG) over internal documents with Role-Based Access Control (RBAC), Parallel Hybrid Retrieval, Cross-Scoring Re-ranking, and In-Memory Semantic Caching.

---

## 🚀 Key Architectural Features

1. **Document Ingestion & Multi-Section Chunking**:
   - Ingests Markdown/Text enterprise policies across HR, IT Security, Cloud Infrastructure, Corporate Finance, and Executive Strategy.
   - Preserves section hierarchies and metadata tags.

2. **Role-Based Access Control (RBAC)**:
   - Dynamic clearance verification (`employee`, `manager`, `hr`, `it_admin`, `finance`, `executive`).
   - Pre-retrieval authorization partitioning ensures sensitive data is never indexed into unauthorized retrieval search spaces.

3. **Low-Latency Parallel Hybrid Retrieval**:
   - Concurrent execution of **TF-IDF Keyword Retrieval** (lexical/BM25 proxy) and **Dense Semantic Vector Search** using `concurrent.futures.ThreadPoolExecutor`.
   - Embeddings powered by `sentence-transformers` (`all-MiniLM-L6-v2`) with graceful fallback to dense subword character n-gram vectors.
   - Normalized Convex Combination + Reciprocal Rank Fusion (RRF) for robust multi-modal score blending.

4. **Cross-Scoring Contextual Re-Ranker**:
   - Re-ranks candidate passages by evaluating query term density, exact subphrase matches, title context bonuses, and query coverage.

5. **Role-Aware Semantic Cache**:
   - In-memory simulation of Redis vector caching.
   - Caches responses indexed by normalized query vectors and user role.
   - Evaluates cosine similarity threshold (default `0.88`) for instant sub-millisecond retrieval on identical or semantically paraphrased queries.
   - Strictly enforces role isolation so cached executive answers are never leaked to regular staff.

6. **Source-Grounded Response Synthesis**:
   - Template-based response generation attributing every fact directly to document source IDs, section titles, and relevance scores.
   - Zero external paid LLM APIs or proprietary cloud services required.

7. **FastAPI Web Interface**:
   - REST endpoints for query processing, document inspection, and cache monitoring.

---

## 📁 Project Structure

```text
project4_rag_assistant/
├── docs/                                    # Sample enterprise documents
│   ├── hr_remote_work_policy.md             # Remote work guidelines (All roles)
│   ├── hr_compensation_and_benefits.md      # Equity & salary bands (HR, Executive)
│   ├── it_vpn_and_security_guidelines.md    # Security policies (All roles)
│   ├── it_admin_infrastructure_access.md    # Root SSH & PAM (IT Admin only)
│   ├── finance_travel_expense_policy.md     # Per diem & travel (Finance, Employees)
│   └── executive_strategic_roadmap_2026.md  # M&A and ARR targets (Executive only)
├── src/
│   ├── __init__.py
│   ├── models.py                            # Pydantic data schemas
│   ├── document_store.py                    # Ingestion & RBAC filtering
│   ├── retrievers.py                        # TF-IDF, Semantic & Parallel Hybrid retrievers
│   ├── reranker.py                          # Cross-scoring contextual reranker
│   ├── cache.py                             # Role-aware in-memory semantic cache
│   ├── generator.py                         # Source-grounded template synthesis
│   └── rag_pipeline.py                      # Orchestrator pipeline
├── app.py                                   # FastAPI application server
├── main.py                                  # Entrypoint (CLI, Demo, Server)
├── demo.py                                  # TestClient integration test suite
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## 🛠️ Setup & Installation

### 1. Requirements
- Python 3.9+ (Windows compatible)

### 2. Install Dependencies
```powershell
cd c:\Users\choud\OneDrive\Desktop\resume\project4_rag_assistant
pip install -r requirements.txt
```

---

## 🎯 Quickstart & Execution Modes

### Mode 1: Automated Verification Demo
Runs a complete test demonstrating RBAC filtering, parallel hybrid retrieval metrics, re-ranking comparison, and semantic cache hit benchmarks:
```powershell
python main.py
```

### Mode 2: Automated API Test Suite
Executes FastAPI endpoint validation (health checks, documents registry, cache statistics, and access control):
```powershell
python demo.py
```

### Mode 3: Interactive CLI
Ask questions interactively and switch roles on the fly:
```powershell
python main.py --interactive
```
*Example CLI usage:*
```text
[employee] Query: What is the home office stipend amount?
[employee] Query: role it_admin
[it_admin] Query: How are break-glass root credentials stored?
```

### Mode 4: FastAPI Web Server
Start the HTTP REST server:
```powershell
python main.py --serve
# or: uvicorn app:app --port 8000
```
API Documentation will be available at: `http://127.0.0.1:8000/docs`

---

## 📡 REST API Reference

### `POST /query`
Performs role-aware retrieval and grounded generation.

**Request Body:**
```json
{
  "query": "What is the per diem meal allowance limit?",
  "user_role": "employee",
  "top_k": 3,
  "alpha": 0.5,
  "use_cache": true
}
```

**Response Body:**
```json
{
  "query": "What is the per diem meal allowance limit?",
  "user_role": "employee",
  "answer": "### Verified Enterprise Summary\n...",
  "citations": [
    {
      "doc_id": "finance_travel_expense_policy.md",
      "doc_title": "FIN-POL-405: Corporate Travel, Meals & Business Expense Policy",
      "section_title": "2. Daily Meal Allowances (Per Diem)",
      "chunk_id": "finance_travel_expense_policy.md#sec_2",
      "relevance_score": 0.842
    }
  ],
  "cache_hit": false,
  "cache_similarity": null,
  "latency_ms": 14.82,
  "retrieval_breakdown": {
    "parallel_retrieval_ms": 8.12,
    "keyword_latency_ms": 2.45,
    "semantic_latency_ms": 7.89,
    "keyword_hits": 3,
    "semantic_hits": 6,
    "hybrid_candidates": 6,
    "reranker_latency_ms": 0.45,
    "generation_latency_ms": 0.38
  }
}
```

### `GET /documents`
Returns all ingested documents, total chunks, and allowed roles.

### `GET /cache/stats`
Returns cache statistics:
```json
{
  "total_lookups": 12,
  "hits": 8,
  "misses": 4,
  "exact_hits": 6,
  "semantic_hits": 2,
  "hit_ratio_percent": 66.67,
  "cached_entries_count": 4,
  "similarity_threshold": 0.88
}
```

### `DELETE /cache`
Purges cached query embeddings and resets metrics.

---

## 🔒 Role-Based Access Control (RBAC) Matrix

| Document Category | Document Filename | Allowed Roles |
|---|---|---|
| **Remote Work Policy** | `hr_remote_work_policy.md` | `employee`, `manager`, `hr`, `it_admin`, `finance`, `executive` |
| **VPN & Endpoint Security** | `it_vpn_and_security_guidelines.md` | `employee`, `manager`, `hr`, `it_admin`, `finance`, `executive` |
| **Travel & Expense Policy** | `finance_travel_expense_policy.md` | `employee`, `manager`, `finance`, `executive` |
| **Compensation & Equity** | `hr_compensation_and_benefits.md` | `hr`, `executive` |
| **IT Bastion & Secrets** | `it_admin_infrastructure_access.md` | `it_admin` |
| **Strategic M&A Roadmap** | `executive_strategic_roadmap_2026.md` | `executive` |
