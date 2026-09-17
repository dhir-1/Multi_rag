"""
Hybrid Indexer for Financial Form 10-K Filings.

Builds and manages:
1. Dense Vector Index (ChromaDB) using fast, native ONNX embeddings (384-d).
2. Sparse Keyword Index (BM25Okapi) for exact financial terms, tickers, and figures.
3. Hybrid Search with Reciprocal Rank Fusion (RRF) for optimal precision.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import CHROMA_PERSIST_DIR, BM25_INDEX_PATH, CHUNKS_JSON_PATH

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


class HybridIndexer:
    """
    Manages both dense vector storage (ChromaDB) and sparse keyword indexing (BM25)
    for section-aware Form 10-K financial chunks.
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        bm25_path: str = BM25_INDEX_PATH,
        collection_name: str = "tech_10k_sec_filings"
    ):
        self.persist_dir = persist_dir
        self.bm25_path = bm25_path
        self.collection_name = collection_name
        
        self.chroma_client = None
        self.collection = None
        self.embed_fn = None
        self.bm25 = None
        self.chunk_ids = []
        self.chunks_lookup = {}

    def initialize_chroma(self):
        """Initializes ChromaDB client with fast ONNX DefaultEmbeddingFunction."""
        if chromadb is None:
            raise ImportError("chromadb is not installed. Run: pip install chromadb")

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embed_fn,
            metadata={"description": "SEC Form 10-K FY2024 Tech Companies Hybrid Index"}
        )

    def get_embedding_function(self):
        if self.embed_fn is None:
            self.initialize_chroma()
        return self.embed_fn

    def index_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 200):
        """Indexes chunks into ChromaDB (dense) and BM25 (sparse)."""
        if not chunks:
            print("[-] No chunks provided for indexing.")
            return

        print("=" * 80)
        print(f"📊 HYBRID INDEXING: {len(chunks):,} Chunks into ChromaDB + BM25")
        print("=" * 80)

        # 1. Reset and initialize Chroma collection
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=self.persist_dir)
        try:
            client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.initialize_chroma()

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "ticker": c.get("ticker", "UNKNOWN"),
                "company": c.get("company", "Unknown"),
                "fiscal_year": str(c.get("fiscal_year", "2024")),
                "section": c.get("section", "General"),
                "source_file": c.get("source_file", "")
            }
            for c in chunks
        ]

        print(f"[*] Embedding & indexing {len(ids)} documents into ChromaDB...")
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self.collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end]
            )
            print(f"    - Upserted chunks {i+1} to {min(end, len(ids))} / {len(ids)}")

        print(f"[+] Dense vector indexing complete ({len(ids)} documents in ChromaDB).")

        # 2. Build BM25 index for sparse keyword matching
        if BM25Okapi is None:
            raise ImportError("rank-bm25 is not installed. Run: pip install rank-bm25")

        print("[*] Building BM25 sparse index...")
        tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.chunk_ids = ids
        self.chunks_lookup = {c["chunk_id"]: c for c in chunks}

        bm25_data = {
            "bm25": self.bm25,
            "chunk_ids": ids,
            "chunks_lookup": self.chunks_lookup
        }
        Path(self.bm25_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_path, "wb") as f:
            pickle.dump(bm25_data, f)

        print(f"[+] BM25 sparse index saved to: {self.bm25_path}")
        print("=" * 80)

    def load_indexes(self):
        """Loads ChromaDB collection and BM25 index from disk."""
        self.initialize_chroma()
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.chunks_lookup = data["chunks_lookup"]
                self.chunk_ids = data["chunk_ids"]
        else:
            print(f"[-] BM25 index not found at {self.bm25_path}. Run index_chunks first.")

    def dense_search(self, query: str, top_k: int = 10, ticker_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Performs vector similarity search via ChromaDB."""
        if self.collection is None:
            self.load_indexes()

        where_clause = {"ticker": ticker_filter} if ticker_filter else None
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause
        )

        dense_hits = []
        if results and results["ids"] and results["ids"][0]:
            for idx, doc_id in enumerate(results["ids"][0]):
                has_dist = (
                    "distances" in results
                    and results["distances"]
                    and len(results["distances"]) > 0
                    and len(results["distances"][0]) > idx
                )
                dense_hits.append({
                    "chunk_id": doc_id,
                    "text": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "distance": results["distances"][0][idx] if has_dist else 0.0,
                    "rank": idx + 1
                })
        return dense_hits

    def bm25_search(self, query: str, top_k: int = 10, ticker_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Performs sparse keyword search via BM25."""
        if self.bm25 is None:
            self.load_indexes()

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        bm25_hits = []
        rank = 1
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc_id = self.chunk_ids[idx]
            chunk_info = self.chunks_lookup[doc_id]

            if ticker_filter and chunk_info.get("ticker") != ticker_filter:
                continue

            bm25_hits.append({
                "chunk_id": doc_id,
                "text": chunk_info["text"],
                "metadata": {
                    "ticker": chunk_info.get("ticker", "UNKNOWN"),
                    "company": chunk_info.get("company", "Unknown"),
                    "fiscal_year": str(chunk_info.get("fiscal_year", "2024")),
                    "section": chunk_info.get("section", "General"),
                    "source_file": chunk_info.get("source_file", "")
                },
                "score": float(scores[idx]),
                "rank": rank
            })
            rank += 1
            if len(bm25_hits) >= top_k:
                break

        return bm25_hits

    def hybrid_search(self, query: str, top_k: int = 6, rrf_k: int = 60, ticker_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Combines Dense + Sparse search using Reciprocal Rank Fusion (RRF).
        RRF Score = 1 / (rrf_k + rank_dense) + 1 / (rrf_k + rank_bm25)
        """
        dense_results = self.dense_search(query, top_k=top_k * 2, ticker_filter=ticker_filter)
        bm25_results = self.bm25_search(query, top_k=top_k * 2, ticker_filter=ticker_filter)

        rrf_scores = {}
        chunks_store = {}

        for hit in dense_results:
            c_id = hit["chunk_id"]
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (rrf_k + hit["rank"]))
            chunks_store[c_id] = hit

        for hit in bm25_results:
            c_id = hit["chunk_id"]
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (rrf_k + hit["rank"]))
            if c_id not in chunks_store:
                chunks_store[c_id] = hit

        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        merged_results = []
        for rank, cid in enumerate(sorted_ids, 1):
            item = chunks_store[cid]
            merged_results.append({
                "chunk_id": cid,
                "text": item["text"],
                "metadata": item["metadata"],
                "rrf_score": rrf_scores[cid],
                "rank": rank
            })

        return merged_results


def build_index_from_file(chunks_json_path: str = CHUNKS_JSON_PATH):
    """Builds the ChromaDB and BM25 indexes from data/chunks.json."""
    if not os.path.exists(chunks_json_path):
        print(f"[-] Chunks file not found at: {chunks_json_path}")
        return

    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    indexer = HybridIndexer()
    indexer.index_chunks(chunks)


if __name__ == "__main__":
    build_index_from_file()
