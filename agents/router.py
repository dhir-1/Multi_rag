"""
Cognitive Depth Router for Financial Multi-Agent RAG.

Classifies incoming queries into:
1. 'fast_path' (Baseline): Direct factoids, 2-number scalar comparisons ("which is higher"),
   isolated multi-entity lookups, single unreleased product checks.
2. 'multi_agent' (Deep Path): Qualitative strategy/risk synthesis (Item 1A), full multi-year
   structured table extractions, sequential multi-hop dependencies, internal financial reconciliations.
"""

import os
import sys
import time
import json
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.state import AgentState
from agents.llm import get_llm
from config import MAX_ITERATIONS

ROUTER_SYSTEM_PROMPT = """You are an expert Query Classifier for an Enterprise Financial RAG System.
Analyze the user's question and classify it into exactly one of two routes:

1. 'fast_path' (Baseline):
   - Direct factoids & metrics: specific numbers, percentages, dates, definitions, headcount, or standalone facts.
   - Simple scalar comparisons: "Which company had higher revenue/capex/margin: Company A or Company B?", "Did X or Y spend more on R&D?" (These only require checking two numbers, not complex synthesis).
   - Independent lookups across multiple entities: "What were Company A's and Company B's revenues?", "Who are the independent auditors for Company X and Company Y?"
   - Unreleased or speculative initiative checks: "What was Company X's revenue from an unreleased prototype project?" (Direct fact check).

2. 'multi_agent' (Deep Path):
   - Qualitative strategy & risk synthesis: comparing business strategies, contrasting qualitative risk factor disclosures (Item 1A), analyzing geopolitical exposure, global supply chain dependencies, or evaluating trade-offs across filings.
   - Full multi-row table extractions: complete structured tables covering multiple years, segments, or geographic regions (e.g., "Extract the complete table of revenue by segment for 2022-2024", "Provide the complete balance sheet table"). Full table fidelity requires multi-agent synthesis, even for a single company.
   - Sequential dependent multi-hop questions: queries where finding the second piece of information requires first discovering an unknown entity from the first document (e.g., "Find Company A's primary supplier, then check that supplier's 10-K for their revenue").
   - Internal financial reconciliation: reconciling conflicting disclosures across sections (e.g., MD&A operating margin expansion vs cash flow restructuring charges).

Respond strictly with valid JSON with these exact keys:
{
  "route": "fast_path" | "multi_agent",
  "reasoning": "<concise 10-word explanation>"
}"""


def classify_query(query: str, llm=None) -> Dict[str, Any]:
    """
    Classifies a query using the Cognitive Depth Router.
    Runs in ~500-900ms using Groq openai/gpt-oss-20b.
    """
    if llm is None:
        llm = get_llm(temperature=0.0)

    if llm is None:
        return {"route": "fast_path", "reasoning": "LLM client unavailable, default to fast_path", "latency_ms": 0.0}

    prompt = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Classify this question:\n\"{query}\""}
    ]
    t0 = time.time()
    resp = llm.invoke(prompt)
    latency_ms = round((time.time() - t0) * 1000, 1)

    content = resp.content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(content)
    except Exception:
        route = "multi_agent" if "multi_agent" in content.lower() else "fast_path"
        data = {"route": route, "reasoning": content[:80]}

    data["latency_ms"] = latency_ms
    return data


def route_query_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node wrapper for Router.
    Analyzes state['query'] and returns the routing decision.
    """
    query = state.get("query", "")
    decision = classify_query(query)
    route = decision.get("route", "fast_path")
    reasoning = decision.get("reasoning", "")
    latency_ms = decision.get("latency_ms", 0)

    print(f"\n[ROUTER] Path Selected: >>> {route.upper()} <<< ({latency_ms}ms)")
    print(f"  * Query    : '{query}'")
    print(f"  * Reasoning: {reasoning}")

    return {
        "route": route,
        "iteration_count": 0,
        "max_iterations": state.get("max_iterations", MAX_ITERATIONS)
    }
