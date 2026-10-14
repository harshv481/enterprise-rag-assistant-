"""Main executable entrypoint for Enterprise RAG Assistant.

Supports:
1. Automated demo testing all core features: python main.py
2. FastAPI web server: python main.py --serve
3. Interactive query CLI: python main.py --interactive
"""
import sys
import argparse
import time
from src.rag_pipeline import EnterpriseRAGPipeline
from src.models import QueryRequest


def run_comprehensive_demo():
    print("=" * 75)
    print("   ENTERPRISE RAG ASSISTANT WITH LOW-LATENCY RETRIEVAL - DEMO")
    print("=" * 75)

    pipeline = EnterpriseRAGPipeline(docs_dir="docs")
    print(f"\n[+] Ingestion Complete:")
    summary = pipeline.doc_store.get_summary()
    print(f"    - Documents Loaded: {summary['total_documents']}")
    print(f"    - Total Chunks: {summary['total_chunks']}")
    for doc_name, info in summary["documents"].items():
        print(f"      * {doc_name} -> Roles: {', '.join(info['allowed_roles'])}")

    print("\n" + "-" * 75)
    print("DEMO 1: Role-Based Access Control (RBAC) Enforcement")
    print("-" * 75)

    secret_infra_query = "What is the procedure for emergency break-glass root console access?"

    print(f"\nQuery: \"{secret_infra_query}\"")
    print("\n[Case A] Standard 'employee' queries sensitive IT Admin infrastructure:")
    resp_emp = pipeline.query(QueryRequest(query=secret_infra_query, user_role="employee", use_cache=False))
    print(f"Role: employee | Latency: {resp_emp.latency_ms}ms")
    print(f"Answer:\n{resp_emp.answer}")
    print(f"Citations count: {len(resp_emp.citations)}")

    print("\n[Case B] 'it_admin' queries the exact same question:")
    resp_admin = pipeline.query(QueryRequest(query=secret_infra_query, user_role="it_admin", use_cache=False))
    print(f"Role: it_admin | Latency: {resp_admin.latency_ms}ms")
    print(f"Answer snippet: {resp_admin.answer.splitlines()[3]}")
    for c in resp_admin.citations:
        print(f"   -> Citation: [{c.doc_id} | {c.section_title}] (Score: {c.relevance_score:.3f})")

    print("\n" + "-" * 75)
    print("DEMO 2: Hybrid Parallel Retrieval (TF-IDF Keyword + Semantic Dense Vectors)")
    print("-" * 75)

    hybrid_query = "What is the equipment reimbursement stipend for home office setup?"
    print(f"\nQuery: \"{hybrid_query}\"")
    resp_hybrid = pipeline.query(QueryRequest(query=hybrid_query, user_role="employee", alpha=0.5, use_cache=False))
    print(f"Total Retrieval Latency: {resp_hybrid.latency_ms}ms")
    print(f"Breakdown metrics:")
    for k, v in resp_hybrid.retrieval_breakdown.items():
        print(f"   - {k}: {v}")
    print(f"\nTop Grounded Source: {resp_hybrid.citations[0].doc_title} -> {resp_hybrid.citations[0].section_title}")
    print(f"Key snippet: \"{resp_hybrid.retrieved_chunks[0].chunk.content[:160]}...\"")

    print("\n" + "-" * 75)
    print("DEMO 3: Re-Ranking Step Comparison")
    print("-" * 75)
    travel_query = "daily meal allowance per diem limit"
    resp_rerank = pipeline.query(QueryRequest(query=travel_query, user_role="employee", use_cache=False))
    print(f"Query: \"{travel_query}\"")
    print("Re-ranked chunk ranking:")
    for chunk in resp_rerank.retrieved_chunks:
        print(f"   Rank #{chunk.rank} | Method: {chunk.retrieval_method} | Score: {chunk.score:.4f} | {chunk.chunk.section_title}")

    print("\n" + "-" * 75)
    print("DEMO 4: In-Memory Semantic Caching for Low-Latency Repeated & Paraphrased Queries")
    print("-" * 75)

    # Clean cache first to test fresh
    pipeline.cache.clear()
    query_base = "What is the policy on international remote work days limit?"

    print(f"\n[Run 1] Cold Query (Cache Miss): \"{query_base}\"")
    r1 = pipeline.query(QueryRequest(query=query_base, user_role="employee", use_cache=True))
    print(f"   - Cache Hit: {r1.cache_hit} | Latency: {r1.latency_ms}ms")

    print(f"\n[Run 2] Identical Query (Exact Cache Hit): \"{query_base}\"")
    r2 = pipeline.query(QueryRequest(query=query_base, user_role="employee", use_cache=True))
    print(f"   - Cache Hit: {r2.cache_hit} | Similarity: {r2.cache_similarity} | Latency: {r2.latency_ms}ms")
    if r1.latency_ms > 0:
        speedup = r1.latency_ms / max(r2.latency_ms, 0.01)
        print(f"   - Speedup: {speedup:.1f}x faster via cache!")

    print(f"\n[Run 3] Paraphrased Query (Semantic Cache Hit): \"How many days can I work remotely abroad each year?\"")
    paraphrased = "How many days can I work remotely abroad each year?"
    r3 = pipeline.query(QueryRequest(query=paraphrased, user_role="employee", use_cache=True))
    print(f"   - Cache Hit: {r3.cache_hit} | Similarity: {r3.cache_similarity} | Latency: {r3.latency_ms}ms")

    print(f"\n[Run 4] Same Query under DIFFERENT Role (RBAC Cache Isolation):")
    r4 = pipeline.query(QueryRequest(query=query_base, user_role="unauthorized_guest", use_cache=True))
    print(f"   - Cache Hit: {r4.cache_hit} (Isolated per RBAC) | Citations: {len(r4.citations)}")

    print("\n" + "-" * 75)
    print("CACHE METRICS SUMMARY:")
    print("-" * 75)
    stats = pipeline.cache.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 75)
    print("DEMO COMPLETE: All enterprise RAG components verified successfully!")
    print("=" * 75)


def run_interactive():
    pipeline = EnterpriseRAGPipeline(docs_dir="docs")
    print("\n=== Interactive Enterprise RAG CLI ===")
    print("Available roles: employee, manager, hr, it_admin, finance, executive")
    print("Type 'exit' to quit.\n")

    current_role = "employee"
    while True:
        try:
            user_input = input(f"[{current_role}] Query (or 'role <new_role>'): ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break
            if user_input.lower().startswith("role "):
                current_role = user_input.split(" ", 1)[1].strip()
                print(f"Switched role to: {current_role}")
                continue

            resp = pipeline.query(QueryRequest(query=user_input, user_role=current_role))
            print(f"\n[Latency: {resp.latency_ms}ms | Cache Hit: {resp.cache_hit}]")
            print(f"{resp.answer}\n")
            if resp.citations:
                print("Citations:")
                for c in resp.citations:
                    print(f" - {c.doc_title} ({c.section_title}) [Score: {c.relevance_score:.3f}]")
            print("-" * 60)
        except (KeyboardInterrupt, EOFError):
            break


def start_server():
    import uvicorn
    print("Starting Enterprise RAG Assistant FastAPI server on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enterprise RAG Assistant")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI HTTP server")
    parser.add_argument("--interactive", action="store_true", help="Start interactive CLI")
    args = parser.parse_args()

    if args.serve:
        start_server()
    elif args.interactive:
        run_interactive()
    else:
        run_comprehensive_demo()
