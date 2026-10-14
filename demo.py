"""Automated verification and API testing script using FastAPI TestClient."""
import sys
import time
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_api_endpoints():
    print(">>> Running FastAPI TestClient API Suite...")

    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health_data = res.json()
    print(f" [PASS] /health: {health_data['status']} (Encoder: {health_data['semantic_encoder']})")

    # 2. Documents check
    res = client.get("/documents")
    assert res.status_code == 200, f"Documents check failed: {res.text}"
    doc_data = res.json()
    assert doc_data["total_documents"] >= 5, "Expected at least 5 documents"
    print(f" [PASS] /documents: {doc_data['total_documents']} documents indexed with {doc_data['total_chunks']} chunks.")

    # 3. Clear cache
    res = client.delete("/cache")
    assert res.status_code == 200

    # 4. RBAC positive test: Executive queries M&A roadmap
    req_body = {
        "query": "What are the target acquisitions and budgets for Q2-Q3 2026?",
        "user_role": "executive",
        "top_k": 2,
        "alpha": 0.5,
        "use_cache": True,
    }
    res = client.post("/query", json=req_body)
    assert res.status_code == 200
    data = res.json()
    assert len(data["citations"]) > 0, "Executive should retrieve strategic roadmap"
    assert "45M" in data["answer"] or "Boreas" in data["answer"] or "M&A" in data["answer"]
    print(f" [PASS] RBAC Permitted (Role: executive) -> Citations: {len(data['citations'])} | Latency: {data['latency_ms']}ms")

    # 5. RBAC negative test: Standard employee queries same secret M&A roadmap
    req_body_blocked = {
        "query": "What are the target acquisitions and budgets for Q2-Q3 2026?",
        "user_role": "employee",
        "top_k": 2,
        "alpha": 0.5,
        "use_cache": False,
    }
    res_blocked = client.post("/query", json=req_body_blocked)
    assert res_blocked.status_code == 200
    data_blocked = res_blocked.json()
    assert len(data_blocked["citations"]) == 0, "Employee should NOT have access to executive roadmap"
    assert "Access Denied" in data_blocked["answer"]
    print(f" [PASS] RBAC Blocked (Role: employee) -> Access Denied verified.")

    # 6. Semantic Cache verification
    query_cache_test = "What is the policy for VPN access and split tunneling?"
    cache_req = {
        "query": query_cache_test,
        "user_role": "employee",
        "top_k": 3,
        "use_cache": True,
    }
    # Cold query
    t0 = time.time()
    res1 = client.post("/query", json=cache_req)
    t1 = (time.time() - t0) * 1000
    d1 = res1.json()
    assert d1["cache_hit"] is False

    # Warm repeated query
    t0 = time.time()
    res2 = client.post("/query", json=cache_req)
    t2 = (time.time() - t0) * 1000
    d2 = res2.json()
    assert d2["cache_hit"] is True
    print(f" [PASS] Semantic Cache Hit: Cold={t1:.1f}ms vs Cached={t2:.1f}ms (Cache similarity: {d2['cache_similarity']})")

    # 7. Cache statistics
    res = client.get("/cache/stats")
    assert res.status_code == 200
    cstats = res.json()
    print(f" [PASS] /cache/stats: Hits={cstats['hits']}, Total Lookups={cstats['total_lookups']}, Hit Ratio={cstats['hit_ratio_percent']}%")

    print("\n>>> All API integration tests PASSED successfully!\n")


if __name__ == "__main__":
    test_api_endpoints()
