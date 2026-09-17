"""
Full End-to-End Ingestion & Indexing Pipeline for Form 10-K Filings.

Executes:
1. Verification & downloading of Form 10-K filings for major tech companies.
2. Sentence-aware section parsing & situational chunking.
3. Hybrid indexing into ChromaDB (Dense Vector) and BM25 (Sparse Keyword).
4. Automated verification search across the newly built index.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.fetch_tech_10k import download_and_extract_10k, COMPANIES, TECH_10K_DIR
from ingestion.chunker import process_all_10k_filings
from ingestion.indexer import HybridIndexer


def run_ingestion_pipeline():
    """
    Runs the complete 10-K Ingestion & Hybrid Indexing Pipeline.
    """
    print("=" * 80)
    print("📊 ENTERPRISE 10-K INGESTION & HYBRID INDEXING PIPELINE")
    print("=" * 80)

    # 1. Verify / Fetch Filings
    print("\n--- Phase 1: Checking Form 10-K Filings ---")
    os.makedirs(TECH_10K_DIR, exist_ok=True)
    existing_files = [f for f in os.listdir(TECH_10K_DIR) if f.endswith(".txt")]
    print(f"[*] Found {len(existing_files)} filings in {TECH_10K_DIR}.")

    if len(existing_files) < len(COMPANIES):
        print(f"[*] Fetching missing Form 10-Ks from SEC EDGAR...")
        for comp in COMPANIES:
            download_and_extract_10k(comp)
    else:
        print("[+] All target Form 10-K filings are present locally.")

    # 2. Section-Aware Sentence Chunking with Situational Headers
    print("\n--- Phase 2: Sentence-Aware Section Parsing & Chunking ---")
    chunks = process_all_10k_filings()
    if not chunks:
        print("[-] Pipeline stopped: No chunks were created.")
        return

    # 3. Hybrid Indexing (ChromaDB + BM25)
    print("\n--- Phase 3: Building Hybrid Index (ChromaDB + BM25) ---")
    indexer = HybridIndexer()
    indexer.index_chunks(chunks)

    # 4. Quick Retrieval Verification Test
    print("\n--- Phase 4: Hybrid Search Verification Test ---")
    test_query = "What did Microsoft report as its total revenue in fiscal 2024?"
    print(f"[*] Testing query: '{test_query}'\n")

    results = indexer.hybrid_search(query=test_query, top_k=3)
    for i, res in enumerate(results, 1):
        meta = res.get("metadata", {})
        ticker = meta.get("ticker", "UNKNOWN")
        sec = meta.get("section", "General")
        print(f"[Result {i}] Score (RRF): {res['rrf_score']:.4f}")
        print(f"  - Company: {ticker} ({meta.get('company', 'Unknown')})")
        print(f"  - Section: {sec}")
        print(f"  - Snippet: {res['text'][:180]}...\n")

    print("=" * 80)
    print("✅ Ingestion & Hybrid Indexing Pipeline Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_ingestion_pipeline()
