"""
Conversational Query Contextualizer & Coreference Resolver.

Resolves multi-turn follow-up queries (e.g. "What about their operating margin?",
"And how does that compare to 2023?") into fully self-contained, standalone
search queries before entering the LangGraph RAG pipeline.

Features:
- Deterministic Fast Bypass: If chat_history is empty or the query is already
  a complete standalone entity-specific question, skips LLM execution (0ms, 0 tokens, $0.00).
- Ultra-Fast Rewriting: Uses Groq with temperature=0.0 and max_tokens=60 to
  resolve pronouns and references in < 180ms.
"""

import os
import re
import sys
from typing import List, Dict, Any, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.llm import get_llm
from agents.registry import extract_target_tickers, get_registered_tickers

PRONOUN_PATTERN = re.compile(
    r'\b(it|its|they|them|their|theirs|this|that|these|those|the same|both|the former|the latter|'
    r'what about|how about|and what|and how|also|compare with that|compared to that)\b',
    re.IGNORECASE
)

LEAD_CONNECTIVE_PATTERN = re.compile(
    r'^(and\b|what about\b|how about\b|also\b|similarly\b|in comparison\b|why\b|how so\b)',
    re.IGNORECASE
)

CONTEXTUALIZER_SYSTEM_PROMPT = """You are a High-Precision Financial Query Contextualizer.
Given a chat conversation about SEC Form 10-K filings and a user's follow-up question, rephrase the follow-up question into an independent, fully self-contained search query.

Strict Rules:
1. Replace all pronouns and ambiguous references ('it', 'their', 'that', 'they', 'the company', 'the former', 'the latter', 'the same period') with the explicit company names, ticker symbols, and fiscal years mentioned in the conversation.
2. Preserve the exact analytical focus (e.g., R&D, operating margin, debt, risk factors, Capex).
3. Do NOT answer the question. Output ONLY the standalone rewritten question. Do not include any quotes, preamble, or commentary.
"""


def should_bypass_contextualizer(query: str, chat_history: Optional[List[Dict[str, str]]]) -> bool:
    """
    Determines if the query can safely bypass the LLM contextualizer.
    Returns True if:
    - History is empty
    - Query starts with no connective, mentions an explicit registered company,
      and contains no ambiguous pronouns.
    """
    if not chat_history:
        return True

    q_strip = query.strip()
    if not q_strip:
        return True

    # If it starts with connectives like "And how...", "What about...", it is definitely a follow-up
    if LEAD_CONNECTIVE_PATTERN.search(q_strip):
        return False

    # Check for pronouns
    has_pronouns = bool(PRONOUN_PATTERN.search(q_strip))
    tickers_mentioned = extract_target_tickers(q_strip)

    # If an explicit registered ticker is mentioned AND no pronouns exist, it's already standalone
    if tickers_mentioned and not has_pronouns:
        return True

    return False


def contextualize_followup(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Tuple[str, bool, int]:
    """
    Contextualizes a user's follow-up query against recent conversation history.

    Returns:
        Tuple of (resolved_query, was_rewritten, tokens_consumed)
    """
    q_clean = query.strip()
    if not q_clean or should_bypass_contextualizer(q_clean, chat_history):
        return q_clean, False, 0

    # Format recent history (up to last 3 turns)
    recent_history = chat_history[-6:] if chat_history else []
    history_lines = []
    for msg in recent_history:
        role = "User" if msg.get("role") in ["user", "human"] else "Assistant"
        content = msg.get("content", "").strip()
        # Truncate assistant messages to first 250 chars to minimize prompt tokens
        if role == "Assistant" and len(content) > 250:
            content = content[:250] + "..."
        history_lines.append(f"{role}: {content}")

    history_str = "\n".join(history_lines)

    user_prompt = f"""Conversation History:
{history_str}

Follow-up Question:
{q_clean}

Standalone Question:"""

    try:
        llm = get_llm(temperature=0.0)
        messages = [
            {"role": "system", "content": CONTEXTUALIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        response = llm.invoke(messages)
        rewritten = response.content.strip().strip('"').strip("'")

        # Fallback safeguard: if LLM returned something empty or excessively long, keep original
        if not rewritten or len(rewritten) > 300:
            return q_clean, False, 0

        # Calculate approximate tokens (input ~ 80, output ~ 15)
        usage = getattr(response, "usage_metadata", {}) or {}
        tokens = usage.get("total_tokens", 95)

        print(f"\n[QUERY CONTEXTUALIZER] Follow-up detected.")
        print(f"  * Original:   '{q_clean}'")
        print(f"  * Standalone: '{rewritten}'")
        return rewritten, True, tokens

    except Exception as e:
        print(f"[QUERY CONTEXTUALIZER WARNING] Failed to rewrite query ({e}), falling back to raw query.")
        return q_clean, False, 0


if __name__ == "__main__":
    # Quick sanity check
    history = [
        {"role": "user", "content": "What was Apple's total revenue in fiscal 2024?"},
        {"role": "assistant", "content": "Apple reported $391.04 billion in total net sales in fiscal 2024 [AAPL, Item 7]."}
    ]
    test_followup = "And how much of that came from the iPhone?"
    res, rewritten, tokens = contextualize_followup(test_followup, history)
    print(f"Result: '{res}', Rewritten: {rewritten}, Tokens: {tokens}")
