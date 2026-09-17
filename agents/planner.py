"""
Planner Agent Node for Multi-Agent Financial RAG.

Decomposes complex financial queries into 1–3 focused, atomic sub-queries for multi-hop
or cross-company retrieval. When receiving feedback from the Critic in self-correction
iterations, it generates targeted sub-queries specifically to retrieve missing or ungrounded facts.
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
from agents.memory import get_episodic_memory

PLANNER_PROMPT = """You are an expert Financial Research Planning Agent for an Enterprise 10-K RAG system.
Analyze the user's question across corporate Form 10-K filings and formulate an optimal retrieval plan.

Available Programmatic FilingTools you can include in your JSON plan:
1. "get_financial_statement_note": args {"ticker": "TICKER", "note_number": "NUMBER"} (e.g. Note on Investments, Leases, Contingencies).
2. "get_section": args {"ticker": "TICKER", "section_name": "Item 1" | "Item 1A" | "Item 7" | "Item 8"}.
3. "search_keyword": args {"query": "search query", "ticker": "TICKER" or null}.

Search Strategy Guidelines:
1. Strip conversational filler and prompt meta-commentary.
2. Cross-Company Comparisons: When a query involves multiple companies, ALWAYS generate separate sub-queries or tool calls explicitly targeting each company independently (e.g. one for AMD, one for NVDA).
3. If an entity is described by role rather than ticker (e.g. an AI partner or major customer), formulate keyword searches targeting that description alongside likely company names.

CRITICAL FORMATTING REQUIREMENT:
Do NOT use native tool-calling tokens or functions. Output ONLY a valid JSON object formatted as follows:
```json
{
  "sub_queries": ["sub query 1", "sub query 2"],
  "tool_calls": [
    {"tool": "search_keyword", "args": {"query": "company partnership investment", "ticker": "TICKER"}}
  ]
}
```
"""

REPLANNER_PROMPT = """You are an expert Financial Research Planning Agent performing query refinement.
A previous draft answer failed the Critic's factual grounding audit.
Examine the Critic's feedback and generate 1 to 2 targeted search queries or specific tool calls (e.g. get_financial_statement_note, get_section) to fetch missing evidence.

Output strictly valid JSON format:
{
  "sub_queries": ["targeted query 1"],
  "tool_calls": []
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
    route = state.get("route", "multi_agent")
    feedback = state.get("critic_feedback", "")
    iteration = state.get("iteration_count", 0)

    print(f"\n[PLANNER] Analyzing query decomposition (Iteration {iteration})...")

    if route in ["fast_path", "simple_direct"]:
        return {"sub_queries": [query]}

    llm = get_llm(temperature=0.1)
    if llm is None:
        sub_queries = _fallback_plan(query, feedback)
        _print_subqueries(sub_queries, "(Rule-based)")
        return {"sub_queries": sub_queries}

    try:
        memory = get_episodic_memory()
        lessons = memory.get_relevant_lessons(query)
        lesson_text = ""
        if lessons:
            lesson_text = "\n\nCRITICAL AUDIT LESSONS LEARNED:\n" + "\n".join(f"- {l}" for l in lessons)

        base_prompt = REPLANNER_PROMPT if feedback else PLANNER_PROMPT
        system_prompt = base_prompt + lesson_text
        user_content = f"Original Query: {query}\nCritic Feedback: {feedback}" if feedback else f"Financial Query: {query}"
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
            tool_calls = parsed_plan.get("tool_calls", [])
            raw_sub_queries = parsed_plan.get("sub_queries", [])
            import unicodedata
            sub_queries = [
                unicodedata.normalize("NFKD", sq).encode("ascii", "ignore").decode("utf-8")
                for sq in raw_sub_queries if sq
            ]
            if sub_queries or tool_calls:
                _print_subqueries(sub_queries, "(LLM)")
                if tool_calls:
                    print(f"  * Programmatic Tool Calls Planned: {tool_calls}")
                return {
                    "sub_queries": sub_queries or [query],
                    "tool_calls": tool_calls,
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "total_tokens": t_tok
                }
    except Exception as e:
        err_str = str(e)
        if "failed_generation" in err_str:
            try:
                gen_match = re.search(r"['\"]failed_generation['\"]:\s*['\"](\{.*?\})['\"]", err_str)
                if gen_match:
                    raw_gen = gen_match.group(1).encode('utf-8').decode('unicode_escape')
                    gen_data = json.loads(raw_gen)
                    t_name = gen_data.get("name")
                    t_args = gen_data.get("arguments", {})
                    s_q = t_args.get("query") or query
                    t_calls = [{"tool": t_name, "args": t_args}]
                    print(f"  * Recovered tool call from failed_generation: {t_calls}")
                    return {
                        "sub_queries": [s_q],
                        "tool_calls": t_calls,
                        "prompt_tokens": state.get("prompt_tokens", 0),
                        "completion_tokens": state.get("completion_tokens", 0),
                        "total_tokens": state.get("total_tokens", 0)
                    }
            except Exception as parse_err:
                print(f"  [!] Failed to parse failed_generation: {parse_err}")

        if os.getenv("STRICT_BENCHMARK_MODE", "0") == "1":
            raise e
        print(f"  [!] Note: LLM planner error ({e}). Using rule-based fallback.")

    sub_queries = _fallback_plan(query, feedback)
    _print_subqueries(sub_queries, "(Fallback)")
    return {
        "sub_queries": sub_queries,
        "tool_calls": [],
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
