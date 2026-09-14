import os
import json
import pickle
from pathlib import Path
from typing import Optional
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import CHROMA_PERSIST_DIR, BM25_INDEX_PATH, EMBEDDING_MODEL_NAME, CHUNKS_JSON_PATH

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None

try:
    from rank_bm25 import BM25Okapi
except (ImportError, Exception):
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer
except (ImportError, OSError, Exception):
    SentenceTransformer = None


class HybridIndexer:
    """
    Manages both dense vector storage (ChromaDB) and sparse keyword indexing (BM25)
    for section-aware research paper chunks.
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        bm25_path: str = BM25_INDEX_PATH,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        collection_name: str = "arxiv_research_papers"
    ):
        self.persist_dir = persist_dir
        self.bm25_path = bm25_path
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        
        self.chroma_client = None
        self.collection = None
        self.embed_fn = None
        self.bm25 = None
        self.chunk_ids = []
        self.chunks_lookup = {}  # chunk_id -> chunk dict

    def initialize_chroma(self):
        """Initializes ChromaDB client and collection with resilient embedding fallback."""
        if chromadb is None:
            raise ImportError("chromadb is not installed. Run: pip install chromadb")

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Try SentenceTransformers embedding function; fallback to Chroma Default (ONNX) if torch DLL issues occur
        if SentenceTransformer is not None:
            try:
                self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model_name
                )
            except Exception as e:
                print(f"[!] Note: Using lightweight ONNX Default Embedding due to: {e}")
                self.embed_fn = embedding_functions.DefaultEmbeddingFunction()
        else:
            self.embed_fn = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embed_fn,
            metadata={"description": "ArXiv ML Papers Section-Aware Chunks"}
        )

    def get_embedding_function(self):
        """Returns the exact embedding function instance used by ChromaDB."""
        if self.embed_fn is None:
            self.initialize_chroma()
        return self.embed_fn

    def index_chunks(self, chunks: list[dict]):
        """
        Indexes chunks into both ChromaDB (Dense) and BM25 (Sparse).
        """
        if not chunks:
            print("[-] No chunks to index.")
            return

        print(f"[*] Indexing {len(chunks)} chunks...")

        # 1. Reset old Chroma collection and populate dense vector index
        if self.chroma_client is None:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.initialize_chroma()

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "paper_id": c["paper_id"],
                "paper_title": c["paper_title"],
                "section": c["section"],
                "page_num": c["page_num"],
            }
            for c in chunks
        ]

        # Upsert in batches of 100 to avoid memory spikes
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self.collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end]
            )
        print(f"[+] Dense vector indexing complete ({len(ids)} documents in ChromaDB).")

        # 2. Build BM25 index for sparse keyword matching
        if BM25Okapi is None:
            raise ImportError("rank-bm25 is not installed. Run: pip install rank-bm25")

        tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.chunk_ids = ids
        self.chunks_lookup = {c["chunk_id"]: c for c in chunks}

        # Save BM25 and chunk lookup
        bm25_data = {
            "bm25": self.bm25,
            "chunk_ids": ids,
            "chunks_lookup": self.chunks_lookup
        }
        Path(self.bm25_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_path, "wb") as f:
            pickle.dump(bm25_data, f)

        print(f"[+] BM25 sparse indexing complete (Saved to: {self.bm25_path}).")

    def load_indexes(self):
        """Loads existing ChromaDB and BM25 index from disk."""
        self.initialize_chroma()
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.chunks_lookup = data["chunks_lookup"]
                self.chunk_ids = data["chunk_ids"]
            print(f"[+] Loaded existing BM25 index from {self.bm25_path}.")
        else:
            print(f"[-] BM25 index not found at {self.bm25_path}. Please run index_chunks first.")

    def dense_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Performs vector similarity search via ChromaDB."""
        if self.collection is None:
            self.load_indexes()

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        dense_hits = []
        if results and results["ids"] and results["ids"][0]:
            for idx, doc_id in enumerate(results["ids"][0]):
                dense_hits.append({
                    "chunk_id": doc_id,
                    "text": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "distance": results["distances"][0][idx] if "distances" in results and results["distances"] else 0.0,
                    "rank": idx + 1
                })
        return dense_hits

    def bm25_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Performs keyword search via BM25."""
        if self.bm25 is None:
            self.load_indexes()

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        bm25_hits = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] <= 0:
                continue
            doc_id = self.chunk_ids[idx]
            chunk_info = self.chunks_lookup[doc_id]
            bm25_hits.append({
                "chunk_id": doc_id,
                "text": chunk_info["text"],
                "metadata": {
                    "paper_id": chunk_info["paper_id"],
                    "paper_title": chunk_info["paper_title"],
                    "section": chunk_info["section"],
                    "page_num": chunk_info["page_num"],
                },
                "score": float(scores[idx]),
                "rank": rank
            })
        return bm25_hits

    def hybrid_search(self, query: str, top_k: int = 5, rrf_k: int = 60) -> list[dict]:
        """
        Combines Dense + Sparse search using Reciprocal Rank Fusion (RRF).
        RRF Score = 1 / (rrf_k + rank_dense) + 1 / (rrf_k + rank_bm25)
        """
        dense_results = self.dense_search(query, top_k=top_k * 2)
        bm25_results = self.bm25_search(query, top_k=top_k * 2)

        rrf_scores = {}
        chunks_store = {}

        # Accumulate RRF scores from Dense search
        for hit in dense_results:
            c_id = hit["chunk_id"]
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (rrf_k + hit["rank"]))
            chunks_store[c_id] = hit

        # Accumulate RRF scores from BM25 search
        for hit in bm25_results:
            c_id = hit["chunk_id"]
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (rrf_k + hit["rank"]))
            if c_id not in chunks_store:
                chunks_store[c_id] = hit

        # Sort by total RRF score
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

    def get_all_chunks_for_paper(self, paper_id: str) -> list[dict]:
        """
        Retrieves all chunks belonging to a specific paper_id, sorted by their
        original sequential order (by page_num, then chunk_id).
        """
        if not self.chunks_lookup:
            self.load_indexes()

        matching_chunks = []
        for c_id, chunk in self.chunks_lookup.items():
            c_paper_id = chunk.get("paper_id", "")
            if c_paper_id == paper_id:
                matching_chunks.append({
                    "chunk_id": c_id,
                    "text": chunk["text"],
                    "metadata": {
                        "paper_id": chunk["paper_id"],
                        "paper_title": chunk["paper_title"],
                        "section": chunk["section"],
                        "page_num": chunk["page_num"],
                    }
                })

        # Sort sequentially by page_num, then chunk_id
        matching_chunks.sort(key=lambda c: (c["metadata"].get("page_num", 0), c["chunk_id"]))
        return matching_chunks


def build_index_from_file(chunks_json_path: str = CHUNKS_JSON_PATH):
    """Convenience function to build the hybrid index from saved chunks.json."""
    if not os.path.exists(chunks_json_path):
        print(f"[-] Chunks file not found at: {chunks_json_path}")
        return

    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    indexer = HybridIndexer()
    indexer.index_chunks(chunks)


if __name__ == "__main__":
    build_index_from_file()
