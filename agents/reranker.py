"""
Cross-Encoder Reranker Node for Multi-Agent Financial RAG.

Executes Stage 2 high-precision reranking over candidate chunks fetched in Stage 1:
1. Scores query-chunk pairs using cross-attention relevance.
2. Sorts candidates and filters out weak distractors.
3. Implements an explicit abstention threshold (default: 0.25): if all chunks score
   below this threshold, flags the query as out-of-scope to prevent hallucinations.
4. Uses resilient hybrid engine: FlashRank (ONNX) with seamless LLM-as-a-Reranker fallback.
"""

import json
import os
import re
import sys
from typing import Dict, Any, List, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.state import AgentState
from agents.llm import get_llm
from config import (
    ABSTENTION_THRESHOLD,
    RERANK_TOP_K,
    FAST_PATH_TOP_K,
    HIGH_CONFIDENCE_RERANK_THRESHOLD,
    RERANKER_MODEL_NAME
)

RERANKER_SYSTEM_PROMPT = """You are a High-Precision Financial Retrieval Reranker.
Evaluate the semantic relevance of each candidate 10-K passage to the user's financial query.
Assign each passage a normalized relevance score between 0.00 and 1.00:
- 0.80 - 1.00: Directly answers or provides essential, conclusive evidence/numbers for the query.
- 0.50 - 0.79: Moderately relevant; provides useful context, background, or partial financial evidence.
- 0.00 - 0.49: Distractor, boilerplate legal disclaimer, or completely irrelevant.

Output strictly valid JSON with the following structure:
{
  "rankings": [
    {"id": 1, "score": 0.95, "reason": "Contains exact revenue breakdown table for the company."},
    {"id": 2, "score": 0.15, "reason": "Standard forward-looking statement disclaimer."}
  ]
}
"""


class CrossEncoderReranker:
    """
    Two-Stage Cross-Encoder Reranker supporting local ONNX (FlashRank)
    with seamless LLM-based Cross-Attention fallback.
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model_name = model_name
        self.flashrank_client = None
        self._init_flashrank()

    def _init_flashrank(self):
        """Attempts to load local FlashRank ONNX cross-encoder."""
        try:
            from flashrank import Ranker
            self.flashrank_client = Ranker(model_name=self.model_name)
            print(f"[+] FlashRank Cross-Encoder loaded: {self.model_name}")
        except Exception:
            self.flashrank_client = None

    def _collect_queries(self, query: str, sub_queries: Optional[List[str]] = None) -> List[str]:
        queries = [query.strip()]
        if sub_queries:
            for sq in sub_queries:
                sq_clean = sq.strip()
                if sq_clean and sq_clean.lower() != query.strip().lower() and sq_clean not in queries:
                    queries.append(sq_clean)
        return queries

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        sub_queries: Optional[List[str]] = None,
        top_k: int = RERANK_TOP_K,
        threshold: float = ABSTENTION_THRESHOLD
    ) -> Tuple[List[Dict[str, Any]], float, bool, Dict[str, int]]:
        """
        Reranks a list of candidate passages for a given query and its variants.
        """
        if not chunks:
            return [], 0.0, False, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # 1. Try local FlashRank if initialized
        if self.flashrank_client is not None:
            try:
                from flashrank import RerankRequest
                from agents.retriever import extract_target_tickers
                all_q_text = query + " " + " ".join(sub_queries or [])
                target_tickers = extract_target_tickers(all_q_text)

                passages = []
                for idx, c in enumerate(chunks):
                    meta = c.get("metadata", {})
                    ticker = meta.get("ticker", meta.get("paper_id", ""))
                    company = meta.get("company", "")
                    sec = meta.get("section", "")
                    text_body = c.get("text", "")
                    if ticker and not text_body.startswith(f"[{ticker}"):
                        prefix = f"[{ticker} - {company} Form 10-K, {sec}]\n"
                    else:
                        prefix = ""
                    passages.append({
                        "id": idx,
                        "text": prefix + text_body,
                        "meta": meta
                    })

                query_variants = self._collect_queries(query, sub_queries)
                best_scores = {idx: 0.0 for idx in range(len(chunks))}

                for q_var in query_variants:
                    req = RerankRequest(query=q_var, passages=passages)
                    results = self.flashrank_client.rerank(req)
                    for res in results:
                        c_id = res["id"]
                        score = float(res.get("score", 0.0))
                        if score > best_scores[c_id]:
                            best_scores[c_id] = score

                ranked_chunks = []
                for idx, c in enumerate(chunks):
                    orig_chunk = c.copy()
                    c_ticker = orig_chunk.get("metadata", {}).get("ticker", "")
                    # If query specifically targets named companies, zero out chunks from other companies
                    if target_tickers and c_ticker and c_ticker not in target_tickers:
                        score = 0.0
                    else:
                        score = round(best_scores[idx], 4)
                    orig_chunk["rerank_score"] = score
                    ranked_chunks.append(orig_chunk)

                ranked_chunks.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
                max_score = ranked_chunks[0]["rerank_score"] if ranked_chunks else 0.0
                is_relevant = max_score >= threshold
                if len(target_tickers) >= 2:
                    quota_per_entity = max(1, top_k // len(target_tickers))
                    balanced_golden = []
                    for tkr in sorted(list(target_tickers)):
                        tkr_matches = [
                            c for c in ranked_chunks
                            if c.get("metadata", {}).get("ticker") == tkr
                            and c.get("rerank_score", 0.0) >= threshold
                        ][:quota_per_entity]
                        balanced_golden.extend(tkr_matches)
                    used_ids = {c.get("chunk_id") for c in balanced_golden if c.get("chunk_id")}
                    remaining_slots = top_k - len(balanced_golden)
                    if remaining_slots > 0:
                        other_matches = [
                            c for c in ranked_chunks
                            if c.get("chunk_id") not in used_ids
                            and c.get("rerank_score", 0.0) >= threshold
                        ][:remaining_slots]
                        balanced_golden.extend(other_matches)
                    golden_chunks = balanced_golden
                else:
                    golden_chunks = [c for c in ranked_chunks if c.get("rerank_score", 0.0) >= threshold][:top_k]

                return golden_chunks, max_score, is_relevant, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            except Exception as e:
                print(f"  [!] Note: FlashRank runtime fallback to LLM reranker ({e}).")

        # 2. Resilient High-Precision LLM Cross-Attention Reranker (Groq)
        return self._rerank_with_llm(query, chunks, top_k=top_k, threshold=threshold)

    def _rerank_with_llm(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = RERANK_TOP_K,
        threshold: float = ABSTENTION_THRESHOLD
    ) -> Tuple[List[Dict[str, Any]], float, bool, Dict[str, int]]:
        """LLM-based Cross-Attention Reranking via Groq with exact token accounting."""
        llm = get_llm(temperature=0.0)
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if llm is None:
            for c in chunks:
                c["rerank_score"] = c.get("rrf_score", 0.5)
            return chunks[:top_k], 0.5, True, token_usage

        candidate_blocks = []
        for idx, c in enumerate(chunks, 1):
            text_snippet = c.get("text", "").strip()[:400].replace("\n", " ")
            meta = c.get("metadata", {})
            ticker = meta.get("ticker", meta.get("paper_id", "UNKNOWN"))
            sec = meta.get("section", "")
            candidate_blocks.append(f"[{idx}] (Company: {ticker}, Section: {sec}) {text_snippet}")

        user_content = (
            f"User Query: {query}\n\n"
            f"Candidate Passages:\n" + "\n".join(candidate_blocks) + "\n\n"
            f"Output strictly valid JSON with score (0.00 to 1.00) for all {len(chunks)} candidates."
        )

        try:
            response = llm.invoke([
                {"role": "system", "content": RERANKER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ])

            usage = getattr(response, "usage_metadata", {}) or {}
            token_usage["prompt_tokens"] = usage.get("input_tokens", 0)
            token_usage["completion_tokens"] = usage.get("output_tokens", 0)
            token_usage["total_tokens"] = usage.get("total_tokens", 0)

            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)
            rankings = data.get("rankings", [])
            score_map = {item.get("id"): float(item.get("score", 0.0)) for item in rankings}

            ranked_chunks = []
            for idx, c in enumerate(chunks, 1):
                chunk_copy = c.copy()
                chunk_copy["rerank_score"] = round(score_map.get(idx, 0.0), 4)
                ranked_chunks.append(chunk_copy)

            ranked_chunks.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
            max_score = ranked_chunks[0]["rerank_score"] if ranked_chunks else 0.0
            is_relevant = max_score >= threshold
            golden_chunks = [c for c in ranked_chunks if c.get("rerank_score", 0.0) >= threshold][:top_k]

            return golden_chunks, max_score, is_relevant, token_usage
        except Exception as e:
            print(f"  [!] Note: LLM reranking parse error ({e}). Falling back to top RRF chunks.")
            for c in chunks:
                c["rerank_score"] = c.get("rrf_score", 0.5)
            return chunks[:top_k], 0.5, True, token_usage


_RERANKER_INSTANCE = None


def get_reranker() -> CrossEncoderReranker:
    """Returns singleton instance of CrossEncoderReranker."""
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        _RERANKER_INSTANCE = CrossEncoderReranker()
    return _RERANKER_INSTANCE


def rerank_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node wrapper for Cross-Encoder Reranker.
    Scores and filters Stage 1 candidate passages into the golden top 3-4 chunks.
    """
    query = state.get("query", "")
    chunks = state.get("retrieved_chunks", [])
    route = state.get("route", "multi_agent")

    print(f"\n[RERANKER] Stage 2: Cross-Encoder reranking over {len(chunks)} candidate chunks...")
    reranker = get_reranker()
    golden_chunks, max_score, is_relevant, tokens = reranker.rerank(
        query=query,
        chunks=chunks,
        sub_queries=state.get("sub_queries", []),
        top_k=RERANK_TOP_K,
        threshold=ABSTENTION_THRESHOLD
    )

    p_tok = state.get("prompt_tokens", 0) + tokens.get("prompt_tokens", 0)
    c_tok = state.get("completion_tokens", 0) + tokens.get("completion_tokens", 0)
    t_tok = state.get("total_tokens", 0) + tokens.get("total_tokens", 0)

    print(f"  * Top relevance score: {max_score:.4f} | Relevant evidence found: {is_relevant}")
    if is_relevant:
        if route in ["fast_path", "simple_direct"] and max_score >= HIGH_CONFIDENCE_RERANK_THRESHOLD:
            golden_chunks = golden_chunks[:FAST_PATH_TOP_K]
            print(f"  * [FAST PATH] High confidence ({max_score:.4f} >= {HIGH_CONFIDENCE_RERANK_THRESHOLD}). Context reduced to {len(golden_chunks)} golden chunks.")

        print(f"  * Filtered {len(chunks)} candidate chunks down to {len(golden_chunks)} golden evidence chunks:")
        for idx, gc in enumerate(golden_chunks, 1):
            meta = gc.get("metadata", {})
            sec = meta.get("section", "General")
            ticker = meta.get("ticker", meta.get("paper_id", "UNKNOWN"))
            print(f"    [{idx}] [{ticker}] {sec} | Cross-Encoder Score: {gc.get('rerank_score', 0.0):.4f}")
    else:
        print(f"  [!] Abstention Triggered: All candidate chunks scored below relevance threshold ({ABSTENTION_THRESHOLD}).")

    result = {
        "retrieved_chunks": golden_chunks,
        "reranked_chunks": golden_chunks,
        "max_rerank_score": max_score,
        "is_relevant_evidence": is_relevant,
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "total_tokens": t_tok
    }
    if not is_relevant:
        result["draft_answer"] = "I could not find relevant evidence in the indexed Form 10-K filings to answer this question."
        result["final_status"] = "ABSTAINED"
        result["total_cost_usd"] = 0.0

    return result
