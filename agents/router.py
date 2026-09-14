"""
Query Router Node for Multi-Agent RAG.

Detects whether an incoming query is a 'whole-document' question (e.g. summarization,
high-level overview, paper contributions) vs. a 'pinpoint' retrieval question.

Routing behaviors:
1. Whole-Document Queries: Resolves the target paper via fuzzy title matching, skips
   top-k hybrid search, and retrieves all ordered chunks for that paper.
2. Pinpoint Queries: Executes normal dense + sparse hybrid search with RRF ranking.
3. Fallback: If a whole-document query does not match any paper title, it gracefully
   falls back to the standard hybrid search path.
"""

import json
import os
import re
import sys
import difflib
from typing import Optional, Tuple, List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

from agents.state import AgentState
from config import CHUNKS_JSON_PATH, ROUTER_MAX_WORDS_SIMPLE, MAX_ITERATIONS


# Keywords/phrases indicating a whole-document or summarization intent
WHOLE_DOC_PATTERNS = [
    r"\bsummariz(e|ation)\b",
    r"\bsummaris(e|ation)\b",
    r"\bsummary\b",
    r"\boverview\b",
    r"\boverall\s+contribution\b",
    r"\bmain\s+contribution\b",
    r"\bmain\s+idea\b",
    r"\bkey\s+takeaway[s]?\b",
    r"\bin\s+summary\b",
    r"\bgive\s+an\s+overview\b",
    r"\bexplain\s+the\s+(whole\s+)?paper\b",
    r"\bwhat\s+is\s+the\s+paper\s+about\b",
    r"\bwhat\s+is\s+this\s+paper\s+about\b",
    r"\bentire\s+paper\b",
    r"\bfull\s+paper\b",
]


def is_whole_document_query(query: str) -> bool:
    """
    Determines if a user query is asking for a whole-document summary or overview.
    Uses pattern and keyword checks without requiring an expensive LLM call.
    """
    cleaned_query = query.strip().lower()
    for pattern in WHOLE_DOC_PATTERNS:
        if re.search(pattern, cleaned_query):
            return True
    return False


def resolve_paper_from_query(
    query: str,
    chunks_path: str = CHUNKS_JSON_PATH
) -> Optional[Tuple[str, str]]:
    """
    Attempts to match a query to a specific paper in the database.
    Checks:
    1. Direct paper_id match (e.g. '2608.28389v1' or '2608.28389')
    2. Substring match for distinctive title keywords (e.g. 'CamoDocs', 'FinExam', 'Twin Worlds')
    3. Fuzzy string similarity match against all paper titles in chunks.json

    Returns:
        Tuple of (paper_id, paper_title) if resolved, else None.
    """
    if not os.path.exists(chunks_path):
        return None

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Collect unique papers: {paper_id: paper_title}
    unique_papers = {}
    for c in chunks:
        pid = c.get("paper_id")
        title = c.get("paper_title")
        if pid and title and pid not in unique_papers:
            unique_papers[pid] = title

    q_lower = query.lower()

    # 1. Direct paper_id matching
    for pid, title in unique_papers.items():
        base_pid = pid.split("v")[0]  # e.g., '2608.28389'
        if pid.lower() in q_lower or base_pid.lower() in q_lower:
            return pid, title

    # 2. Distinctive keyword/phrase substring matching
    for pid, title in unique_papers.items():
        title_lower = title.lower()
        # Extract main distinctive words from title (e.g. "camodocs", "finexam", "twin worlds")
        main_terms = re.split(r"[:\-\–]", title_lower)[0].strip()
        if main_terms and main_terms in q_lower:
            return pid, title

    # 3. Fuzzy similarity matching across all titles
    best_match_pid = None
    best_match_title = None
    highest_score = 0.0

    for pid, title in unique_papers.items():
        # Check token set similarity
        title_words = set(re.findall(r"\w+", title.lower()))
        query_words = set(re.findall(r"\w+", q_lower))
        
        # Exclude generic words
        stop_words = {"the", "a", "an", "for", "in", "on", "of", "and", "with", "what", "how", "paper", "summarize", "overview"}
        meaningful_title_words = title_words - stop_words
        meaningful_query_words = query_words - stop_words

        if meaningful_title_words and meaningful_query_words:
            overlap = len(meaningful_title_words & meaningful_query_words) / len(meaningful_title_words)
            fuzzy_ratio = difflib.SequenceMatcher(None, title.lower(), q_lower).ratio()
            combined_score = max(overlap, fuzzy_ratio)

            if combined_score > highest_score:
                highest_score = combined_score
                best_match_pid = pid
                best_match_title = title

    # Threshold for confident fuzzy match
    if highest_score >= 0.35:
        return best_match_pid, best_match_title

    return None


# Comparison / reasoning keywords that require planner decomposition and critic audit
REASONING_PATTERNS = [
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bdifference[s]?\b",
    r"\bvs\b",
    r"\bversus\b",
    r"\bwhy\b",
    r"\bhow\s+does\b",
    r"\bhow\s+do\b",
    r"\bhow\s+can\b",
    r"\brelationship\s+between\b",
    r"\bcontrast\b",
    r"\btrade[\s\-]?off[s]?\b",
    r"\bevaluate\b",
    r"\banalyze\b",
    r"\banalyse\b",
]


def is_simple_direct_query(query: str) -> bool:
    """
    Classifies a query as 'simple_direct' (fast path) if:
    1. It is short (word count <= ROUTER_MAX_WORDS_SIMPLE, default 18).
    2. It does NOT contain reasoning/comparison keywords (compare, vs, why, how does, etc.).
    """
    words = query.strip().split()
    if len(words) > ROUTER_MAX_WORDS_SIMPLE:
        return False

    q_lower = query.lower()
    for pat in REASONING_PATTERNS:
        if re.search(pat, q_lower):
            return False

    return True




def route_query_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node wrapper for Router.
    Analyzes state['query'] and determines the routing path and paper resolution.
    """
    query = state.get("query", "")
    is_whole_doc = is_whole_document_query(query)

    if is_whole_doc:
        resolved = resolve_paper_from_query(query)
        if resolved:
            paper_id, paper_title = resolved
            print(f"\n[ROUTER] Path Selected: >>> WHOLE-DOCUMENT RETRIEVAL <<<")
            print(f"  * Query: '{query}'")
            print(f"  * Resolved Paper: '{paper_title}' (ID: {paper_id})")
            return {
                "route": "whole_document",
                "resolved_paper_id": paper_id,
                "resolved_paper_title": paper_title,
                "iteration_count": 0,
                "max_iterations": state.get("max_iterations", MAX_ITERATIONS)
            }
        else:
            print(f"\n[ROUTER] Path Selected: >>> PINPOINT HYBRID SEARCH (Fallback) <<<")
            print(f"  * Query: '{query}' (No specific paper identified)")
            return {
                "route": "pinpoint_retrieval_fallback",
                "resolved_paper_id": None,
                "resolved_paper_title": None,
                "iteration_count": 0,
                "max_iterations": state.get("max_iterations", MAX_ITERATIONS)
            }
    elif is_simple_direct_query(query):
        print(f"\n[ROUTER] Path Selected: >>> SIMPLE DIRECT (Fast Path) <<<")
        print(f"  * Query: '{query}'")
        print(f"  * Cost-Gating: Fast execution (Skipping Planner & Critic).")
        return {
            "route": "simple_direct",
            "resolved_paper_id": None,
            "resolved_paper_title": None,
            "iteration_count": 0,
            "max_iterations": state.get("max_iterations", MAX_ITERATIONS)
        }
    else:
        print(f"\n[ROUTER] Path Selected: >>> PINPOINT HYBRID SEARCH <<<")
        print(f"  * Query: '{query}'")
        return {
            "route": "pinpoint_retrieval",
            "resolved_paper_id": None,
            "resolved_paper_title": None,
            "iteration_count": 0,
            "max_iterations": state.get("max_iterations", MAX_ITERATIONS)
        }


