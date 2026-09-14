"""
Planner Agent Node for Multi-Agent RAG.

Decomposes complex research queries into 1–3 focused, atomic sub-queries for multi-hop
retrieval. When receiving feedback from the Critic in self-correction iterations, it generates
targeted sub-queries specifically to retrieve missing or hallucinated facts.
"""

import json
import re
import sys
import os
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.state import AgentState
from agents.llm import get_llm

PLANNER_PROMPT = """You are an expert Research Planning Agent for an Academic RAG system.
Analyze the user's research query about ML/NLP papers and decompose it into 1 to 3 focused, atomic search sub-queries.

Search Strategy:
1. Strip conversational filler (e.g., "according to the papers", "can you tell me", "what does the author say").
2. Expand abbreviations and synonyms to match academic paper vocabulary (e.g., "LLMs" -> "models / language models", "financial reasoning" -> "financial examination reasoning").
3. Target specific metrics, benchmarks, datasets, or methodologies.

Output strictly JSON format:
{
  "sub_queries": ["sub query 1", "sub query 2"]
}
"""

REPLANNER_PROMPT = """You are an expert Research Planning Agent performing query refinement.
A previous draft answer failed the Critic's factual grounding audit.
Examine the Critic's feedback and generate 1 to 2 targeted search queries to fetch the missing evidence.
Output strictly JSON format:
{
  "sub_queries": ["targeted query 1"]
}
"""


def _fallback_plan(query: str, feedback: str = "") -> List[str]:
    """Generates simple rule-based sub-queries when LLM is unavailable."""
    if feedback:
        return [query, f"{query} {feedback[:80]}"]

    parts = re.split(r"\b(?:and|vs|versus|compared to|while|as well as)\b", query, flags=re.IGNORECASE)
    cleaned = [p.strip() for p in parts if len(p.strip().split()) >= 3]
    return cleaned[:3] if len(cleaned) >= 2 else [query]


def plan_queries(state: AgentState) -> Dict[str, Any]:
    """
    Planner Node function for LangGraph:
    - Generates 1-3 sub-queries for hybrid search.
    - Adapts dynamically if Critic feedback is present.
    """
    query = state.get("query", "")
    route = state.get("route", "pinpoint_retrieval")
    feedback = state.get("critic_feedback", "")
    iteration = state.get("iteration_count", 0)

    print(f"\n[PLANNER] Analyzing query decomposition (Iteration {iteration})...")

    if route == "whole_document":
        return {"sub_queries": [query]}

    llm = get_llm(temperature=0.1)
    if llm is None:
        sub_queries = _fallback_plan(query, feedback)
        _print_subqueries(sub_queries, "(Rule-based)")
        return {"sub_queries": sub_queries}

    try:
        system_prompt = REPLANNER_PROMPT if feedback else PLANNER_PROMPT
        user_content = f"Original Query: {query}\nCritic Feedback: {feedback}" if feedback else f"Research Query: {query}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = llm.invoke(messages)
        usage = getattr(response, "usage_metadata", {}) or {}
        p_tok = state.get("prompt_tokens", 0) + usage.get("input_tokens", 0)
        c_tok = state.get("completion_tokens", 0) + usage.get("output_tokens", 0)
        t_tok = state.get("total_tokens", 0) + usage.get("total_tokens", 0)

        json_match = re.search(r"\{.*\}", response.content.strip(), re.DOTALL)
        if json_match:
            raw_match = json_match.group(0)
            try:
                parsed_plan = json.loads(raw_match)
            except json.JSONDecodeError:
                fixed_match = re.sub(r'\\(?![/"\\bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', raw_match)
                parsed_plan = json.loads(fixed_match)
            raw_sub_queries = parsed_plan.get("sub_queries", [])
            import unicodedata
            sub_queries = [
                unicodedata.normalize("NFKD", sq).encode("ascii", "ignore").decode("utf-8")
                for sq in raw_sub_queries if sq
            ]
            if sub_queries:
                _print_subqueries(sub_queries, "(LLM)")
                return {
                    "sub_queries": sub_queries,
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "total_tokens": t_tok
                }
    except Exception as e:
        if os.getenv("STRICT_BENCHMARK_MODE", "0") == "1":
            raise e
        print(f"  [!] Note: LLM planner error ({e}). Using rule-based fallback.")

    sub_queries = _fallback_plan(query, feedback)
    _print_subqueries(sub_queries, "(Fallback)")
    return {
        "sub_queries": sub_queries,
        "prompt_tokens": state.get("prompt_tokens", 0),
        "completion_tokens": state.get("completion_tokens", 0),
        "total_tokens": state.get("total_tokens", 0)
    }



def _print_subqueries(sub_queries: List[str], tag: str):
    print(f"  * Generated {len(sub_queries)} sub-queries {tag}:")
    for i, sq in enumerate(sub_queries, 1):
        try:
            print(f"    [{i}] {sq}")
        except UnicodeEncodeError:
            safe_sq = sq.encode("ascii", errors="replace").decode("ascii")
            print(f"    [{i}] {safe_sq}")
