"""
Agent State Definition for Multi-Agent RAG.

Defines the centralized TypedDict schema passed between all LangGraph nodes:
Router -> Planner -> Retriever -> Synthesizer -> Critic
"""

from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    """
    Central state object representing the workflow context.
    """
    # 1. User Input & Routing
    query: str
    route: str  # 'whole_document', 'pinpoint_retrieval', 'pinpoint_retrieval_fallback', 'simple_direct'
    resolved_paper_id: Optional[str]
    resolved_paper_title: Optional[str]

    # 2. Planning & Decomposition
    sub_queries: List[str]

    # 3. Retrieval & Reranking Results
    retrieved_chunks: List[Dict[str, Any]]
    total_retrieved_count: int
    reranked_chunks: List[Dict[str, Any]]
    max_rerank_score: float
    is_relevant_evidence: bool

    # 4. Synthesizer Output
    draft_answer: str
    citations: List[Dict[str, Any]]

    # 5. Critic & Quality Evaluation
    critic_score: int  # 1 to 5 (LLM judge)
    embedding_grounding_score: float  # Cosine similarity grounding score (0.0 to 1.0)
    is_grounded: bool
    critic_feedback: str
    unsupported_claims: List[str]

    # 6. Self-Correction Loop Controls
    iteration_count: int
    max_iterations: int
    final_status: str  # 'ACCEPTED', 'ACCEPTED_WITH_WARNING', 'MAX_ITERATIONS_REACHED'

    # 7. Token Usage & Cost Accounting (from response.usage_metadata)
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost_usd: float
