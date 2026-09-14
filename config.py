"""
Centralized Configuration for Multi-Agent RAG Assistant.

Loads environment variables from .env and exposes standardized, typed settings across:
- Groq pricing & token accounting
- Reranker thresholds & models
- Retrieval & chunking sizing
- Routing & loop constraints
- File and storage paths
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent


def _get_float_env(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


def _get_int_env(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


# ==============================================================================
# 1. System & Model Configuration
# ==============================================================================
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5").strip()
RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "ms-marco-TinyBERT-L-2-v2").strip()


# ==============================================================================
# 2. Token Pricing & Cost Accounting (Per 1 Million Tokens)
# ==============================================================================
# Default Groq pricing for openai/gpt-oss-20b: $0.10 / 1M prompt, $0.20 / 1M completion
INPUT_COST_PER_M: float = _get_float_env("GROQ_INPUT_COST_PER_M", 0.10)
OUTPUT_COST_PER_M: float = _get_float_env("GROQ_OUTPUT_COST_PER_M", 0.20)


def compute_token_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Computes total USD cost from prompt and completion token counts."""
    cost = (prompt_tokens * (INPUT_COST_PER_M / 1_000_000.0)) + (
        completion_tokens * (OUTPUT_COST_PER_M / 1_000_000.0)
    )
    return round(cost, 6)


# ==============================================================================
# 3. Two-Stage Retrieval & Reranker Relevance Thresholds
# ==============================================================================
# Chunks scoring below this are flagged as irrelevant / out-of-scope (instant abstention)
ABSTENTION_THRESHOLD: float = _get_float_env("RERANKER_ABSTENTION_THRESHOLD", 0.25)

# Fast-path queries scoring above this safely bypass the Critic audit
FAST_PATH_THRESHOLD: float = _get_float_env("RERANKER_FAST_PATH_THRESHOLD", 0.60)

# Very high confidence threshold to reduce context to 1-2 chunks on fast path
HIGH_CONFIDENCE_RERANK_THRESHOLD: float = _get_float_env("RERANKER_HIGH_CONFIDENCE_THRESHOLD", 0.85)

# Number of golden evidence chunks passed to Synthesizer after reranking
RERANK_TOP_K: int = _get_int_env("RERANK_TOP_K", 4)
FAST_PATH_TOP_K: int = _get_int_env("FAST_PATH_TOP_K", 2)


# ==============================================================================
# 4. Chunking & Retrieval Sizing
# ==============================================================================
CHUNK_SIZE_WORDS: int = _get_int_env("CHUNK_SIZE_WORDS", 250)
CHUNK_OVERLAP_WORDS: int = _get_int_env("CHUNK_OVERLAP_WORDS", 40)

# Stage 1 hybrid retrieval parameters
TOP_K_PER_SUBQUERY: int = _get_int_env("RETRIEVAL_TOP_K_PER_SUBQUERY", 8)
MAX_TOTAL_CHUNKS: int = _get_int_env("RETRIEVAL_MAX_TOTAL_CHUNKS", 16)
RRF_K: int = _get_int_env("RRF_K", 60)


# ==============================================================================
# 5. Routing & Self-Correction Loop Controls
# ==============================================================================
# Maximum iterations for the Planner -> Retriever -> Critic self-correction cycle
MAX_ITERATIONS: int = _get_int_env("RAG_MAX_ITERATIONS", 2)

# Maximum word count for a query to qualify as simple_direct (raised to 18 for simple multi-clause queries)
ROUTER_MAX_WORDS_SIMPLE: int = _get_int_env("ROUTER_MAX_WORDS_SIMPLE", 18)



# ==============================================================================
# 6. File & Storage Paths
# ==============================================================================
DATA_DIR: str = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma_db"))
BM25_INDEX_PATH: str = os.getenv("BM25_INDEX_PATH", str(PROJECT_ROOT / "data" / "bm25_index.pkl"))
CHUNKS_JSON_PATH: str = os.getenv("CHUNKS_JSON_PATH", str(PROJECT_ROOT / "data" / "chunks.json"))
RAW_PAPERS_DIR: str = os.getenv("RAW_PAPERS_DIR", str(PROJECT_ROOT / "data" / "raw_papers"))
METADATA_CATALOG_PATH: str = os.getenv("METADATA_CATALOG_PATH", str(PROJECT_ROOT / "data" / "raw_papers" / "metadata.json"))
