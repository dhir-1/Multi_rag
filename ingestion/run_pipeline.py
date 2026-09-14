import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.fetch_arxiv import fetch_arxiv_papers
from ingestion.chunker import process_all_papers
from ingestion.indexer import HybridIndexer, build_index_from_file


def run_ingestion_pipeline(
    query: str = "cat:cs.CL AND (ti:RAG OR ti:Retrieval OR ti:Reasoning)",
    max_papers: int = 5
):
    """
    Runs the complete Step 1 Ingestion Pipeline:
    1. Fetch papers from arXiv API.
    2. Section-aware extraction and chunking.
    3. Hybrid indexing (ChromaDB dense + BM25 sparse).
    4. Verification query test.
    """
    print("=" * 60)
    print("STEP 1: DOCUMENT INGESTION & HYBRID INDEXING PIPELINE")
    print("=" * 60)

    # 1. Fetch Papers
    print("\n--- Phase 1: Fetching Papers from arXiv ---")
    papers = fetch_arxiv_papers(query=query, max_results=max_papers)
    if not papers:
        print("[-] Pipeline stopped: No papers were fetched.")
        return

    # 2. Section-Aware Chunking
    print("\n--- Phase 2: Section-Aware Parsing & Chunking ---")
    chunks = process_all_papers()
    if not chunks:
        print("[-] Pipeline stopped: No chunks were created.")
        return

    # 3. Hybrid Indexing (ChromaDB + BM25)
    print("\n--- Phase 3: Building Hybrid Index ---")
    indexer = HybridIndexer()
    indexer.index_chunks(chunks)

    # 4. Quick Retrieval Verification Test
    print("\n--- Phase 4: Hybrid Search Verification Test ---")
    test_query = "What is retrieval augmented generation and how does it reduce hallucinations?"
    print(f"[*] Testing query: '{test_query}'\n")
    
    results = indexer.hybrid_search(query=test_query, top_k=3)
    for i, res in enumerate(results, 1):
        meta = res["metadata"]
        print(f"[Result {i}] Score (RRF): {res['rrf_score']:.4f}")
        print(f"  - Paper: {meta.get('paper_title')}")
        print(f"  - Section: {meta.get('section')} (Page {meta.get('page_num')})")
        print(f"  - Snippet: {res['text'][:180]}...\n")

    print("=" * 60)
    print("[SUCCESS] Step 1 Ingestion & Indexing Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion_pipeline()
