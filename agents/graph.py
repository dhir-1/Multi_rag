"""
LangGraph Multi-Agent RAG Workflow.

Compiles the full StateGraph connecting:
Router -> Planner -> Retriever -> Synthesizer -> Critic

Features:
- Conditional Routing: Skips Planner for whole-document summarization.
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
    sys.stdout.reconfigure(encoding="utf-8")

from langgraph.graph import StateGraph, START, END
from agents.state import AgentState
from agents.router import route_query_node
from agents.planner import plan_queries
from agents.retriever import retrieve_evidence
from agents.reranker import rerank_node
from agents.synthesizer import synthesize_answer
from agents.critic import critique_answer
from config import ABSTENTION_THRESHOLD, FAST_PATH_THRESHOLD, compute_token_cost, MAX_ITERATIONS


def route_decision(state: AgentState) -> Literal["retriever", "planner"]:
    """
    Conditional edge from Router:
    - 'whole_document' and 'simple_direct' skip Planner and go straight to Retriever.
    - 'pinpoint_retrieval' goes to Planner for multi-query decomposition.
    """
    route = state.get("route", "pinpoint_retrieval")
    if route in ["whole_document", "simple_direct"]:
        return "retriever"
    return "planner"


def synthesizer_decision(state: AgentState) -> Literal["critic", "__end__"]:
    """
    Conditional edge from Synthesizer:
    - If ungrounded / abstention (is_relevant_evidence is False): exits to __end__ (clean refusal).
    - If 'simple_direct' AND retrieval confidence is high (max_rerank_score >= FAST_PATH_THRESHOLD): fast-path exit to __end__.
    - If 'simple_direct' BUT retrieval confidence is low/borderline (max_rerank_score < FAST_PATH_THRESHOLD): flagged for Critic audit!
    - All other routes (pinpoint, whole_document) go to Critic.
    """
    is_relevant = state.get("is_relevant_evidence", True)
    if not is_relevant:
        p = state.get("prompt_tokens", 0)
        c = state.get("completion_tokens", 0)
        state["total_cost_usd"] = compute_token_cost(p, c)
        print(f"  * [FAST ABSTENTION] Relevance threshold failed (< {ABSTENTION_THRESHOLD}). Exiting to END.")
        return "__end__"

    route = state.get("route", "pinpoint_retrieval")
    max_score = state.get("max_rerank_score", 0.0)

    # Simple direct only skips Critic if retrieval confidence is high (>= FAST_PATH_THRESHOLD)
    if route == "simple_direct":
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


def build_rag_graph() -> Any:
    """
    Constructs and compiles the complete LangGraph multi-agent RAG pipeline:
    Router -> Planner -> Retriever -> Reranker -> Synthesizer -> Critic
    """
    workflow = StateGraph(AgentState)

    # 1. Add all Agent Nodes
    workflow.add_node("router", route_query_node)
    workflow.add_node("planner", plan_queries)
    workflow.add_node("retriever", retrieve_evidence)
    workflow.add_node("reranker", rerank_node)
    workflow.add_node("synthesizer", synthesize_answer)
    workflow.add_node("critic", critique_answer)

    # 2. Add Edges & Conditional Branches
    workflow.add_edge(START, "router")

    # Branch from Router (whole-doc and simple-direct skip planner)
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "retriever": "retriever",
            "planner": "planner"
        }
    )

    # Linear workflow from Planner to Retriever
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "reranker")
    workflow.add_edge("reranker", "synthesizer")

    # Branch from Synthesizer (simple-direct and fast abstention skip critic)
    workflow.add_conditional_edges(
        "synthesizer",
        synthesizer_decision,
        {
            "critic": "critic",
            "__end__": END
        }
    )

    # Conditional loop from Critic
    workflow.add_conditional_edges(
        "critic",
        critic_decision,
        {
            "planner": "planner",
            "__end__": END
        }
    )

    app = workflow.compile()
    return app


# Pre-compiled application instance
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
    print(f"Status: {final_state.get('final_status', 'ACCEPTED (Fast Path)' if route_taken == 'simple_direct' else 'ACCEPTED')}")
    print(f"Total Iterations: {final_state.get('iteration_count', 0)}")
    
    if route_taken == "simple_direct":
        print(f"Critic Score (LLM): Skipped (simple_direct fast path)")
        print(f"Embedding Grounding Score: Skipped (simple_direct fast path)")
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
    # Test 1: Simple Direct Query (Cost-Gated Fast Path)
    print("\n>>> TEST 1: SIMPLE DIRECT QUERY (FAST PATH) <<<")
    run_query("What is the primary contribution of the FinExam benchmark?")

    # Test 2: Complex Pinpoint Query with Reasoning & Dual Grounding Scores
    print("\n\n>>> TEST 2: COMPLEX PINPOINT QUERY (FULL MULTI-AGENT LOOP) <<<")
    run_query("What is the Directional Impact metric and how does it help detect topical collapse in argument retrieval?")
