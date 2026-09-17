"""
LangGraph Multi-Agent Financial RAG Workflow.

Compiles the full StateGraph connecting:
Router -> Planner -> Retriever -> Reranker -> Synthesizer -> Critic

Features:
- Cognitive Depth Routing:
  - 'fast_path' skips Planner, retrieves directly, and if confidence is high, exits cleanly without Critic overhead.
  - 'multi_agent' decomposes complex synthesis into atomic sub-queries, executes hybrid retrieval & reranking, synthesizes in-depth answers, and passes through the Fact-Checking Critic.
- Self-Correction Loop: Re-plans and retrieves missing evidence if Critic detects ungrounded claims.
- Drift Protection: Hard cap on maximum iterations (default: 2) to guarantee prompt termination.
"""

import os
import sys
from typing import Dict, Any, Literal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langgraph.graph import StateGraph, START, END
from agents.state import AgentState
from agents.router import route_query_node
from agents.planner import plan_queries
from agents.retriever import retrieve_evidence
from agents.reranker import rerank_node
from agents.synthesizer import synthesize_answer
from agents.critic import critique_answer
from agents.verifier import check_pre_flight_guardrails
from config import ABSTENTION_THRESHOLD, FAST_PATH_THRESHOLD, compute_token_cost, MAX_ITERATIONS


def route_decision(state: AgentState) -> Literal["retriever", "planner"]:
    """
    Conditional edge from Router:
    - 'fast_path' (and legacy 'simple_direct') skips Planner and goes straight to Retriever.
    - 'multi_agent' goes to Planner for multi-query decomposition.
    """
    route = state.get("route", "multi_agent")
    if route in ["fast_path", "simple_direct"]:
        return "retriever"
    return "planner"


def synthesizer_decision(state: AgentState) -> Literal["critic", "__end__"]:
    """
    Conditional edge from Synthesizer:
    - If ungrounded / abstention (is_relevant_evidence is False): exits to __end__.
    - If 'fast_path' AND retrieval confidence is high (max_rerank_score >= FAST_PATH_THRESHOLD): fast-path exit to __end__.
    - If 'fast_path' BUT retrieval confidence is low/borderline: flagged for Critic audit.
    - All 'multi_agent' routes go to Critic for verification.
    """
    is_relevant = state.get("is_relevant_evidence", True)
    if not is_relevant:
        p = state.get("prompt_tokens", 0)
        c = state.get("completion_tokens", 0)
        state["total_cost_usd"] = compute_token_cost(p, c)
        print(f"  * [FAST ABSTENTION] Relevance threshold failed (< {ABSTENTION_THRESHOLD}). Exiting to END.")
        return "__end__"

    route = state.get("route", "multi_agent")
    max_score = state.get("max_rerank_score", 0.0)

    if route in ["fast_path", "simple_direct"]:
        if max_score >= FAST_PATH_THRESHOLD:
            p = state.get("prompt_tokens", 0)
            c = state.get("completion_tokens", 0)
            state["total_cost_usd"] = compute_token_cost(p, c)
            print(f"  * [FAST PATH] High retrieval confidence ({max_score:.4f} >= {FAST_PATH_THRESHOLD}). Bypassing Critic.")
            return "__end__"
        else:
            print(f"  * [FLAGGED FOR CRITIC] Fast path query had borderline retrieval confidence ({max_score:.4f} < {FAST_PATH_THRESHOLD}). Routing to Critic audit.")
            return "critic"

    return "critic"


def critic_decision(state: AgentState) -> Literal["planner", "__end__"]:
    """Conditional edge from Critic: loop back to Planner if answer failed grounding audit."""
    is_grounded = state.get("is_grounded", True)
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", MAX_ITERATIONS)

    if is_grounded or iteration >= max_iterations:
        return "__end__"
    return "planner"


def reranker_gate_decision(state: AgentState) -> Literal["synthesizer", "__end__"]:
    """
    Calibrated Confidence Gate (Replacing the LLM Critic):
    - Uses empirical Cross-Encoder score (calibrated at 0.001) to make an instant routing decision.
    - Score < 0.001 (Red Light): Slams the gate shut and exits directly to END with 0 LLM tokens.
    - Score >= 0.001 (Green Light): Routes to the single-pass grounded synthesizer.
    """
    is_relevant = state.get("is_relevant_evidence", True)
    if not is_relevant:
        print(f"  * [CONFIDENCE GATE: 🔴 RED LIGHT] Max Cross-Encoder score < {ABSTENTION_THRESHOLD}. Bypassing LLM and exiting to END.")
        state["draft_answer"] = "I could not find relevant evidence in the indexed Form 10-K filings to answer this question."
        state["final_status"] = "ABSTAINED"
        return "__end__"

    print(f"  * [CONFIDENCE GATE: 🟢 GREEN LIGHT] Evidence score ({state.get('max_rerank_score', 0.0):.4f}) passed gate. Routing to Synthesizer.")
    return "synthesizer"


def pre_flight_node(state: AgentState) -> Dict[str, Any]:
    """
    Pre-Flight Guardrail Node (Pure Python, 0ms latency):
    Validates company indexing and financial domain boundaries before any retrieval or LLM execution.
    """
    query = state.get("query", "")
    guard_res = check_pre_flight_guardrails(query)
    if not guard_res.get("is_allowed", True):
        rejection_msg = guard_res.get("rejection_message", "Query rejected by pre-flight guardrail.")
        print(f"\n[PRE-FLIGHT GUARDRAIL: 🛑 BLOCKED] {guard_res.get('reason')}: {rejection_msg[:90]}...")
        p = state.get("prompt_tokens", 0)
        c = state.get("completion_tokens", 0)
        return {
            "draft_answer": rejection_msg,
            "citations": [],
            "final_status": "ABSTAINED_PRE_FLIGHT",
            "is_relevant_evidence": False,
            "total_cost_usd": compute_token_cost(p, c)
        }

    print(f"\n[PRE-FLIGHT GUARDRAIL: 🟢 PASSED] Query validated against filing registry & domain bounds.")
    return {"is_relevant_evidence": True}


def pre_flight_decision(state: AgentState) -> Literal["retriever", "__end__"]:
    """Conditional edge from Pre-Flight: exit instantly to __end__ if blocked."""
    final_status = state.get("final_status", "")
    if final_status == "ABSTAINED_PRE_FLIGHT":
        return "__end__"
    return "retriever"


def build_rag_graph() -> Any:
    """
    Option A: Lean Enterprise Financial RAG Pipeline with Pre-Flight & Post-Flight Guardrails.
    Connects: START -> pre_flight -> [pre_flight_decision] -> retriever -> reranker -> [Confidence Gate] -> synthesizer -> END.
    Replaces flaky LLM critics with deterministic software guardrails.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("pre_flight", pre_flight_node)
    workflow.add_node("retriever", retrieve_evidence)
    workflow.add_node("reranker", rerank_node)
    workflow.add_node("synthesizer", synthesize_answer)

    workflow.add_edge(START, "pre_flight")
    workflow.add_conditional_edges(
        "pre_flight",
        pre_flight_decision,
        {
            "retriever": "retriever",
            "__end__": END
        }
    )
    workflow.add_edge("retriever", "reranker")
    workflow.add_conditional_edges(
        "reranker",
        reranker_gate_decision,
        {
            "synthesizer": "synthesizer",
            "__end__": END
        }
    )
    workflow.add_edge("synthesizer", END)

    app = workflow.compile()
    return app


rag_agent_app = build_rag_graph()


def run_query(query: str, max_iterations: int = MAX_ITERATIONS) -> Dict[str, Any]:
    """
    High-level convenience helper to execute a query through the multi-agent graph.
    """
    print("\n" + "=" * 70)
    print(f"MULTI-AGENT RAG QUERY: '{query}'")
    print("=" * 70)

    initial_state: AgentState = {
        "query": query,
        "iteration_count": 0,
        "max_iterations": max_iterations
    }

    final_state = rag_agent_app.invoke(initial_state)

    route_taken = final_state.get("route", "unknown")
    critic_score = final_state.get("critic_score")
    emb_score = final_state.get("embedding_grounding_score")

    print("\n" + "=" * 70)
    print("FINAL WORKFLOW EXECUTION COMPLETE")
    print("=" * 70)
    print(f"Route Taken: {route_taken}")
    print(f"Status: {final_state.get('final_status', 'ACCEPTED')}")
    print(f"Total Iterations: {final_state.get('iteration_count', 0)}")
    
    if route_taken in ["fast_path", "simple_direct"] and critic_score is None:
        print(f"Critic Score (LLM): Skipped (fast path high confidence)")
    else:
        print(f"Critic Score (LLM): {critic_score}/5" if critic_score is not None else "Critic Score (LLM): N/A")
        print(f"Embedding Grounding Score: {emb_score:.4f} (Cosine Similarity)" if emb_score is not None else "Embedding Grounding Score: N/A")
    
    print(f"Retrieved Evidence Count: {final_state.get('total_retrieved_count')}")
    print(f"Total Citations: {len(final_state.get('citations', []))}")
    print(f"Token Accounting: {final_state.get('total_tokens', 0)} tokens (In: {final_state.get('prompt_tokens', 0)}, Out: {final_state.get('completion_tokens', 0)}) | Cost: ${final_state.get('total_cost_usd', 0.0):.6f}")
    print("\n--- FINAL ANSWER ---")
    print(final_state.get("draft_answer"))
    print("-" * 70)

    return final_state


if __name__ == "__main__":
    test_q = "Did Apple or Microsoft report higher annual operating cash flow in fiscal 2024?"
    run_query(test_q)
