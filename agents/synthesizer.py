"""
Synthesizer Agent Node for Multi-Agent Financial RAG.

Synthesizes a comprehensive, evidence-grounded financial answer from retrieved 10-K chunks.
Enforces strict inline source citations in the format [TICKER, Section]
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
from agents.memory import get_episodic_memory
from config import compute_token_cost

SYNTHESIZER_SYSTEM_PROMPT = """You are an expert Enterprise Financial Research Synthesizer.
Provide an accurate, in-depth, and completely evidence-grounded answer to the user's financial question based ONLY on the provided Form 10-K context passages.

Strict Grounding & Professional Presentation Rules:
1. Strict Citation Enum: Every factual claim, financial figure, percentage, or risk factor MUST be backed by an inline citation in the exact format: [TICKER, Section].
   Allowed TICKER values are strictly limited to the registered SEC reporting companies:
   - AAPL (Apple Inc.)
   - MSFT (Microsoft Corporation)
   - AMZN (Amazon.com, Inc.)
   - GOOGL (Alphabet Inc.)
   - META (Meta Platforms, Inc. / Facebook / Instagram)
   - NVDA (NVIDIA Corporation)
   - AMD (Advanced Micro Devices, Inc.)
   - TSLA (Tesla, Inc.)
   - NFLX (Netflix, Inc.)
   - CRM (Salesforce, Inc.)
   Examples: [META, Item 7], [META, Item 1A], [AAPL, Item 1], [MSFT, Item 8], [NVDA, Item 7], [NFLX, Item 15].
   Generic tags like [SOURCE 1], [Passage 1], or [Doc 1] are STRICTLY PROHIBITED and will be rejected.

2. Presentation Formatting Standards (No Forced Tables on Single Companies & No Wall-of-Text Blobs):
   - Single-Company Deep Dives (e.g., Netflix transaction, Salesforce RPO, Alphabet litigation, Meta metrics):
     Do NOT force a table. Instead, structure your response as an executive briefing using clean Markdown section headings (###) and structured bullet points with bold lead-ins.
     Example:
     ### Overview & Current Status
     ...
     ### Key Terms & Financial Commitments
     - **Cash Consideration:** ... [NFLX, Item 15]
     - **Financing Structure:** ... [NFLX, Item 15]
     ### Operational & Regulatory Conditions
     ...
   - Cross-Company Qualitative Comparisons (Risks, Geopolitics, Foundry Dependencies, Cloud Rivalry):
     Use structured narrative sections with clear sub-headings:
     ### [Company A]: [Key Disclosure Theme]
     ### [Company B]: [Key Disclosure Theme]
     ### Comparative Synthesis & Key Differences
   - Cross-Company Quantitative Comparisons (Numerical Line Items: CapEx, Revenues, Margins):
     Use a clean Markdown table with explicit columns:
     | Company | Metric / Topic | Disclosed 10-K Value | Citation |
     | :--- | :--- | :--- | :--- |
     followed by 2 to 3 analytical takeaways.
   - Non-Redundancy Rule: NEVER repeat the same facts twice. If a table is used for numbers, do NOT write a giant summary paragraph repeating the identical table text.
   - Readability: Keep paragraphs concise (2-4 sentences max). Use bold lead-in bullets to prevent dense, unreadable text walls.

3. Substantive Analytical Depth:
   Provide thorough, executive-grade financial analysis (typically 200–350 words). Unpack management commentary, operational nuances, contract terms, and specific risk mechanisms stated in the 10-K excerpts. Do NOT arbitrarily truncate answers to shallow 1-sentence bullets.

4. Closed-World Assumption & Non-Disclosure: Do NOT hallucinate, infer, or extrapolate beyond the provided text. If the provided excerpts do not explicitly confirm a transaction, marketing dollar figure, or specific metric requested by the user, state clearly and affirmatively: "The provided Form 10-K excerpts do not disclose [requested topic]." Never fabricate or estimate a figure not directly reported in the text.

5. Exact Entity Grounding: When naming external corporate entities, suppliers, foundries, or commercial partners, use exact verbatim naming as written in the filing excerpts. If an external entity is not explicitly named in the text (e.g. designated anonymously as "Customer A" or "third-party foundry"), state explicitly that the entity is not identified by name in the filing.

Keep internal reasoning concise and output the final grounded answer immediately.
"""

FAST_PATH_SYSTEM_PROMPT = """You are an expert Financial Research Assistant.
Provide a direct, factual answer to the user's question in 1-2 concise sentences based ONLY on the provided 10-K passages.

Strict Fast-Path Schema Rules:
1. Strict Citation Enum: Include inline citations for each distinct fact/number in the exact format: [TICKER, Section]. TICKER must strictly be one of the registered companies: AAPL, MSFT, AMZN, GOOGL, META, NVDA, AMD, TSLA, NFLX, CRM (e.g. [META, Item 7], [AAPL, Item 1]). Never use [SOURCE 1].
2. Directness: Answer directly and concisely in 1-2 sentences.
3. Strict Non-Disclosure Rule: If the provided excerpts do not disclose the requested figure or transaction, state clearly that the filing does not disclose it rather than guessing.
4. Output clean Markdown.
"""


def _format_context_passages(chunks: List[Dict[str, Any]]) -> str:
    """Formats evidence chunks into a structured prompt block."""
    context_blocks = []
    for idx, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        ticker = meta.get("ticker", meta.get("paper_id", "Unknown"))
        company = meta.get("company", "Unknown")
        sec = meta.get("section", "General")
        sec_short = sec.split("–")[0].split("-")[0].strip()
        text = c.get("text", "").strip()

        block = (
            f"--- [PASSAGE {idx}] ---\n"
            f"Company: {company} ({ticker})\n"
            f"Section: {sec}\n"
            f"Citation Tag: [{ticker}, {sec_short}]\n"
            f"Content:\n{text}\n"
        )
        context_blocks.append(block)

    return "\n".join(context_blocks)


def _extract_citations(text: str) -> List[Dict[str, str]]:
    """Extracts all [TICKER, Section] citations from generated text."""
    pattern = r"\[([A-Z]{1,5}|[A-Za-z\s]+),\s*([^\]]+)\]"
    matches = re.findall(pattern, text)

    citations = []
    seen = set()
    for m in matches:
        ticker = m[0].strip()
        sec = m[1].strip()
        key = (ticker, sec)
        if key not in seen:
            seen.add(key)
            citations.append({
                "ticker": ticker,
                "section": sec,
                "citation_text": f"[{ticker}, {sec}]"
            })
    return citations


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
            "draft_answer": "I could not find relevant evidence in the indexed Form 10-K filings to answer this question.",
            "citations": [],
            "final_status": "ABSTAINED",
            "total_cost_usd": compute_token_cost(p, c)
        }

    route = state.get("route", "multi_agent")
    if route in ["fast_path", "simple_direct"]:
        route_max_tokens = 1024
        system_prompt = FAST_PATH_SYSTEM_PROMPT
    else:
        route_max_tokens = 6000
        system_prompt = SYNTHESIZER_SYSTEM_PROMPT

    memory = get_episodic_memory()
    lessons = memory.get_relevant_lessons(query)
    if lessons:
        lesson_block = "\n\nCRITICAL AUDIT RULES / LESSONS LEARNED:\n" + "\n".join(f"- {l}" for l in lessons)
        system_prompt += lesson_block

    llm = get_llm(temperature=0.0, max_tokens=route_max_tokens)
    if llm is None:
        raise RuntimeError("LLM client is None (GROQ_API_KEY missing or invalid).")

    formatted_context = _format_context_passages(chunks)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"EVIDENCE CONTEXT:\n{formatted_context}\n\nQUESTION:\n{query}\n\nANSWER:"}
    ]
    response = llm.invoke(messages)
    draft_answer = response.content.strip() if response.content else ""

    if not draft_answer:
        additional = getattr(response, "additional_kwargs", {}) or {}
        reasoning = additional.get("reasoning_content", "")
        if reasoning:
            print(f"  [!] Note: Reasoning content captured ({len(reasoning.split())} words).")
        if state.get("draft_answer"):
            draft_answer = state.get("draft_answer")
        else:
            draft_answer = "The provided Form 10-K excerpts do not disclose information regarding the requested transaction or metric."

    # Apply deterministic citation sanitization, hallucination scrubbing & post-flight factual audit
    from agents.verifier import (
        sanitize_and_validate_citations,
        scrub_unsupported_hallucinations,
        audit_claims_deterministically
    )
    draft_answer, verified_citations, citation_violations = sanitize_and_validate_citations(draft_answer, chunks)
    draft_answer = scrub_unsupported_hallucinations(draft_answer, chunks)
    audit_res = audit_claims_deterministically(draft_answer, chunks)

    unsupported = list(audit_res.get("unsupported_claims", []))
    if unsupported:
        print(f"  [!] Factual Audit Warning: {len(unsupported)} claims/citations not evidenced in 10-K text: {unsupported}")
        # Enforce guardrail: surface audit caveat directly in the response so users/API see it
        warning_bullets = "\n".join(f"- {c}" for c in unsupported)
        draft_answer += f"\n\n> ⚠️ **Factual Audit Caveat:** The following figures/citations could not be confirmed in the provided 10-K passages:\n{warning_bullets}"

    citations = verified_citations if verified_citations else _extract_citations(draft_answer)
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
        "is_grounded": audit_res.get("is_grounded", True),
        "audit_warnings": unsupported,
        "final_status": "FLAGGED_UNGROUNDED" if unsupported else "ACCEPTED"
    }
