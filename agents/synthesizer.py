"""
Synthesizer Agent Node for Multi-Agent RAG.

Synthesizes a comprehensive, evidence-grounded research answer from retrieved chunks.
Enforces strict inline source citations in the format [Paper_ID, Section, Page X]
and extracts structured citation references.
"""

import sys
import os
import re
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.state import AgentState
from agents.llm import get_llm
from config import compute_token_cost

SYNTHESIZER_SYSTEM_PROMPT = """You are an expert Academic Research Synthesizer.
Provide an accurate, in-depth, and completely evidence-grounded answer to the user's research query based ONLY on the provided context passages.

Strict Citation and Grounding Rules:
1. Every factual claim, metric, or technique MUST be backed by an inline citation: [Paper_ID, Section, Page X].
   Example: "CamoDocs optimizes dispersion tokens to evade clustering defenses [2608.28389v1, Method, Page 3]."
2. Do NOT hallucinate or assume unstated information.
3. If evidence is insufficient, explicitly state what is missing.
4. Format in clean, readable Markdown.
"""

WHOLE_DOC_SYSTEM_PROMPT = """You are an expert Academic Research Synthesizer providing a comprehensive whole-document synthesis.
Provide an accurate, structured, and completely evidence-grounded answer to the user's research query based ONLY on the provided context passages.

Strict Citation and Grounding Rules:
1. Every factual claim, metric, or technique MUST be backed by an inline citation: [Paper_ID, Section, Page X].
2. Do NOT hallucinate or assume unstated information.
3. If evidence is insufficient, explicitly state what is missing.
4. Format in clean, readable Markdown.
5. Table Conciseness: When providing multi-stage breakdowns or comparative defense tables, keep each table cell concise (1-2 sentences) to ensure all stages and defenses are fully represented without exceeding length limits.
"""

SIMPLE_DIRECT_SYSTEM_PROMPT = """You are an expert Academic Research Assistant.
Provide a direct, factual answer to the user's question in 1-2 concise sentences based ONLY on the provided context passages.

Strict Fast-Path Rules:
1. Answer directly and concisely in 1-2 sentences.
2. Include exactly ONE inline citation per distinct factual claim: [Paper_ID, Section, Page X].
3. Do NOT add unstated background context, speculative analysis, or lengthy introductory/concluding remarks.
4. Output clean Markdown.
"""


def _format_context_passages(chunks: List[Dict[str, Any]]) -> str:
    """Formats evidence chunks into a structured prompt block."""
    context_blocks = []
    for idx, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        pid = meta.get("paper_id", "Unknown")
        title = meta.get("paper_title", "Unknown")
        sec = meta.get("section", "General")
        page = meta.get("page_num", "?")
        text = c.get("text", "").strip()

        block = (
            f"--- [SOURCE {idx}] ---\n"
            f"Paper ID: {pid} | Title: {title}\n"
            f"Section: {sec} (Page {page})\n"
            f"Content: {text}\n"
        )
        context_blocks.append(block)

    return "\n".join(context_blocks)


def _extract_citations(text: str) -> List[Dict[str, str]]:
    """Extracts all [Paper_ID, Section, Page X] citations from generated text."""
    pattern = r"\[([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)[,\s]+([^,\]]+)[,\s]+(?:Page\s+)?([0-9]+)\]"
    matches = re.findall(pattern, text, re.IGNORECASE)

    citations = []
    seen = set()
    for m in matches:
        key = (m[0], m[1].strip(), m[2].strip())
        if key not in seen:
            seen.add(key)
            citations.append({
                "paper_id": m[0],
                "section": m[1].strip(),
                "page": m[2].strip(),
                "citation_text": f"[{m[0]}, {m[1].strip()}, Page {m[2].strip()}]"
            })
    return citations


def _generate_dry_run_answer(query: str, chunks: List[Dict[str, Any]], error_msg: str = None) -> str:
    """Generates a structured dry-run answer when GROQ_API_KEY is not configured or an API error occurs."""
    first_chunk = chunks[0] if chunks else {}
    meta = first_chunk.get("metadata", {})
    pid = meta.get("paper_id", "Unknown")
    sec = meta.get("section", "General")
    pg = meta.get("page_num", 1)
    snippet = first_chunk.get("text", "")[:250].strip() if first_chunk else "No evidence available"

    note = f"> *Note: {error_msg}*" if error_msg else "> *Note: Running in dry-run mode until GROQ_API_KEY is configured in .env.*"
    return (
        f"### Research Synthesis (Dry-Run Mode)\n\n"
        f"**User Query:** {query}\n\n"
        f"Based on the retrieved research evidence, {snippet} [{pid}, {sec}, Page {pg}].\n\n"
        f"{note}"
    )


def synthesize_answer(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizer Node function for LangGraph:
    - Prompts Groq LLM to synthesize answer with strict citations.
    - Extracts citations and attaches to state.
    """
    query = state.get("query", "")
    chunks = state.get("retrieved_chunks", [])
    iteration = state.get("iteration_count", 0)

    print(f"\n[SYNTHESIZER] Generating grounded answer from {len(chunks)} evidence chunks (Iteration {iteration})...")

    is_relevant = state.get("is_relevant_evidence", True)
    if not chunks or not is_relevant:
        print("  [-] Fast Abstention: No relevant evidence passed the Cross-Encoder relevance threshold.")
        p = state.get("prompt_tokens", 0)
        c = state.get("completion_tokens", 0)
        return {
            "draft_answer": "I could not find relevant evidence in the indexed research papers to answer this question.",
            "citations": [],
            "final_status": "ABSTAINED",
            "total_cost_usd": compute_token_cost(p, c)
        }

    route = state.get("route", "pinpoint_retrieval")
    if route == "whole_document":
        route_max_tokens = 8192
        system_prompt = WHOLE_DOC_SYSTEM_PROMPT
    elif route == "simple_direct":
        route_max_tokens = 1024
        system_prompt = SIMPLE_DIRECT_SYSTEM_PROMPT
    else:
        route_max_tokens = 4096
        system_prompt = SYNTHESIZER_SYSTEM_PROMPT

    syn_temp = 0.0 if (os.getenv("STRICT_BENCHMARK_MODE", "0") == "1" or os.getenv("BENCHMARK_RUN", "0") == "1") else 0.1
    llm = get_llm(temperature=syn_temp, max_tokens=route_max_tokens)
    if llm is None:
        if os.getenv("STRICT_BENCHMARK_MODE", "0") == "1" or os.getenv("BENCHMARK_RUN", "0") == "1":
            raise RuntimeError(
                "CRITICAL: Cannot synthesize answer in strict/benchmark mode: "
                "LLM client is None (GROQ_API_KEY is missing or invalid in .env). "
                "Silent dry-run fallbacks are prohibited during benchmark evaluation."
            )
        draft_answer = _generate_dry_run_answer(query, chunks)
        citations = _extract_citations(draft_answer)
        print(f"  * Synthesis complete in dry-run mode ({len(citations)} citations).")
        return {
            "draft_answer": draft_answer,
            "citations": citations,
            "is_dry_run": True,
            "final_status": "DRY_RUN_MOCK"
        }

    try:
        formatted_context = _format_context_passages(chunks)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"EVIDENCE CONTEXT:\n{formatted_context}\n\nQUESTION:\n{query}\n\nANSWER:"}
        ]
        response = llm.invoke(messages)
        draft_answer = response.content.strip()

        # Protection against reasoning token budget exhaustion:
        # If response content is empty but previous iteration already had an answer, preserve it.
        if not draft_answer:
            if state.get("draft_answer"):
                print("  [!] Warning: Synthesizer response was empty. Preserving non-empty answer from previous iteration.")
                draft_answer = state.get("draft_answer")
            elif hasattr(response, "additional_kwargs") and response.additional_kwargs.get("reasoning_content"):
                draft_answer = response.additional_kwargs.get("reasoning_content", "").strip()

        citations = _extract_citations(draft_answer)
        if not citations and state.get("citations"):
            citations = state.get("citations")

        usage = getattr(response, "usage_metadata", {}) or {}
        p_tok = state.get("prompt_tokens", 0) + usage.get("input_tokens", 0)
        c_tok = state.get("completion_tokens", 0) + usage.get("output_tokens", 0)
        t_tok = state.get("total_tokens", 0) + usage.get("total_tokens", 0)

        print(f"  * Synthesis complete ({len(draft_answer.split())} words, {len(citations)} citations, {usage.get('total_tokens', 0)} tokens).")
        cost = compute_token_cost(p_tok, c_tok)
        return {
            "draft_answer": draft_answer,
            "citations": citations,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": t_tok,
            "total_cost_usd": cost,
            "is_dry_run": False,
            "final_status": "ACCEPTED (Fast Path)" if state.get("route") == "simple_direct" else state.get("final_status", "PENDING_AUDIT")
        }
    except Exception as e:
        if os.getenv("STRICT_BENCHMARK_MODE", "0") == "1" or os.getenv("BENCHMARK_RUN", "0") == "1":
            raise RuntimeError(
                f"CRITICAL: LLM synthesis failed during benchmark/strict mode ({e}). "
                f"Dry-run fallback prohibited."
            ) from e
        err_str = str(e)
        if "429" in err_str or "rate_limit" in err_str.lower():
            display_err = "Groq API token rate limit reached. Please wait a moment for the quota window to reset."
        else:
            display_err = f"API error encountered: {err_str[:120]}"
        print(f"  [!] Note: LLM synthesis error ({e}). Using dry-run fallback.")
        draft_answer = _generate_dry_run_answer(query, chunks, error_msg=display_err)
        citations = _extract_citations(draft_answer)
        p_tok = state.get("prompt_tokens", 0)
        c_tok = state.get("completion_tokens", 0)
        t_tok = state.get("total_tokens", 0)
        return {
            "draft_answer": draft_answer,
            "citations": citations,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": t_tok,
            "total_cost_usd": compute_token_cost(p_tok, c_tok),
            "is_dry_run": True,
            "final_status": "DRY_RUN_MOCK"
        }

