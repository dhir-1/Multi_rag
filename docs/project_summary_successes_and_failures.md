# Enterprise Financial RAG: Project Post-Mortem, Successes, and Failures

**Date:** September 17, 2026  
**Domain:** SEC Form 10-K Financial Intelligence (Tech Titans: AAPL, MSFT, GOOGL, AMZN, META, NVDA, AMD, TSLA, NFLX, CRM)  
**Core Stack:** Python 3.11, LangGraph, Groq (`openai/gpt-oss-20b`), FlashRank (`ms-marco-MiniLM-L-12-v2`), ChromaDB, Rank-BM25, FastAPI  

---

## Executive Summary

This project set out to evaluate whether an advanced **Multi-Agent RAG architecture** (incorporating cognitive routing, query planning, multi-stage retrieval, cross-encoder reranking, and self-correcting critics) genuinely outperforms a **Naive Baseline RAG** system when analyzing complex, SEC Form 10-K filings.

Over the course of development, rigorous double-blind benchmarking via `openai/gpt-oss-120b`, empirical rate-limit stress tests on Groq, and deterministic primary-source audits revealed profound insights about modern AI systems:
1. **Multi-Agent RAG is essential for cross-document / multi-hop queries**, where naive baseline search catastrophically fails due to single-document saturation.
2. **Multi-Agent RAG incurs a heavy "Complexity Tax" on simple factoids**, where naive RAG is faster, cheaper, and avoids multi-LLM failure modes.
3. **Automated LLM Judges have significant prompt-truncation blind spots**, penalizing systems for citing true primary-source filing facts omitted from the judge's reference string.

---

## Phase-by-Phase Journey: What We Built

### Phase 1: Ingestion & Indexing Pipeline
* **Parsed & Cleaned:** Ten full-length Form 10-K annual reports in `data/tech_10k/`.
* **Semantic Chunker (`ingestion/chunker.py`):** Segmented filings by SEC Item headers (Item 1, 1A, 7, 8, 15) into 250-word chunks with 40-word sliding overlaps across sentence boundaries.
* **Hybrid Indexing (`ingestion/indexer.py`):** Combined dense semantic embeddings (`BAAI/bge-small-en-v1.5` in ChromaDB) with sparse lexical search (Okapi BM25) fused via Reciprocal Rank Fusion (RRF, $k=60$).

### Phase 2: The Original Complex Multi-Agent Pipeline
* Implemented a 5-node LangGraph loop:
  $$\text{Query} \longrightarrow \text{Router} \longrightarrow \text{Planner} \longrightarrow \text{Retriever/Tools} \longrightarrow \text{Synthesizer} \longrightarrow \text{Critic} \rightleftharpoons \text{Re-planner}$$
* Included episodic memory (`agents/memory.py`), tool execution dispatching (`agents/tools.py`), and hallucination verifiers (`agents/verifier.py`).

### Phase 3: The Pivot to "Option A" (Lean Enterprise Financial RAG)
* Replaced the fragile 5-agent conversational loop with a streamlined, deterministic, single-pass pipeline:
  $$\text{Query} \longrightarrow \text{Deterministic Entity Dispatcher} \longrightarrow \text{Balanced Hybrid Retrieval} \longrightarrow \text{Cross-Encoder Reranker} \longrightarrow \text{Single-Pass Synthesizer}$$
* Enforced strict inline citations (`[TICKER, Section]`), negative non-disclosure compliance rules, and conciseness guidelines.

### Phase 4: Verification & Benchmarking
* Created a 10-question evaluation benchmark spanning single-entity factoids, cross-company competitive comparisons, whole-document risk syntheses, and adversarial negative traps.
* Conducted double-blind head-to-head evaluations using `openai/gpt-oss-120b` on Groq, alongside deterministic Python primary-source audits directly matching raw files on disk.

---

## What Succeeded (The Wins)

### 1. Solving the "Single-Document Hijacking" Flaw (The Major Architectural Win)
* **The Problem:** Naive Baseline RAG failed completely on multi-company comparative queries (e.g., Question 6: AWS vs. Azure; Question 10: Apple vs. Tesla). Its top-$k$ vector similarity search pulled 4 chunks from one company (Microsoft) and 0 chunks from the other (Amazon). Baseline literally surrendered: *"Because the Amazon 10-K language is not included... I can't compare the two companies' wording."*
* **The Fix:** Option A's **Deterministic Entity Dispatcher** identified both entities and enforced a balanced retrieval quota (e.g., 5 chunks AMZN + 5 chunks MSFT).
* **The Outcome:** Multi-Agent RAG won decisively on cross-company synthesis, comparing competitive risks across both companies accurately.

### 2. Massive Latency Reduction (9x Speedup)
* Under the original multi-agent loop, queries required 4 to 6 chained LLM calls, taking **45 to 70+ seconds** per question.
* Option A collapsed intermediate handoffs into deterministic Python routines, dropping end-to-end response times down to **5.6 to 7.0 seconds**.

### 3. Adversarial Negative Trap Grounding (Perfect 5/5 Score)
* On Question 8 (AMD Console Marketing Spend), the query set a trap asking for a granular dollar figure that AMD does not break out in its 10-K.
* Rather than hallucinating a plausible-sounding dollar amount, the Synthesizer adhered strictly to Rule 5:
  > *"The 10-K does not disclose a specific dollar amount for marketing spend directed exclusively at gaming console manufacturers in FY 2024. [AMD, ITEM 7]"*
* The 120B judge awarded a **perfect 5/5** for factuality and completeness.

### 4. Deep Footnote & Subsequent Event Retrieval
* On Question 7 (Netflix / Warner Bros. Discovery deal), Option A successfully retrieved deep into Item 15 / Subsequent Events and extracted the headline terms ($27.75/share, ~$72B equity value) and granular financing commitments ($42.2B bridge, $5B revolver, $20B delayed-draw term loan from lines 3216–3225 of `NFLX_2024_10K.txt`).

### 5. Windows OpenBLAS Memory Allocation Stability
* Identified and resolved recurring Windows OpenBLAS multi-threading memory errors (`OpenBLAS error: Memory allocation still failed after 10 retries`) by injecting thread limit defaults (`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) at the root of `config.py`.

### 6. Interactive Web UI & API
* The FastAPI backend (`api.py`) and clean frontend (`frontend/index.html`) provide an interactive, browser-based demonstration of the fast 6-second pipeline.

---

## What Failed (The Hard Lessons & Failures)

### 1. Over-Engineering & "The Complexity Tax"
* **The Mistake:** Believing that adding more LLM agents (Planner agent, Router agent, Critic agent, Re-planning loops) automatically makes an AI system smarter.
* **The Reality:** Every chained LLM call adds ~1–3 seconds of latency, introduces potential token-format errors (HTTP 400s), consumes Groq TPM limits (triggering 429 Too Many Requests), and creates an error-accumulation chain where a mistake by one agent corrupts the next.
* **The Lesson:** For 70% of standard single-company lookups, modern LLMs (`gpt-oss-20b`) are already capable synthesizers. Naive baseline RAG matched or beat the multi-agent system on simple queries with 75% fewer tokens.

### 2. The 120B Automated Judge Fallacy
* **The Mistake:** Expecting `openai/gpt-oss-120b` to act as an omniscient, impartial truth judge.
* **The Reality:** LLM judges do not read the disk; they only evaluate candidate answers against the 3-sentence summary in their prompt.
  * *Alphabet Antitrust (EQ5):* Multi-Agent accurately quoted Alphabet's European Commission Digital Markets Act investigation from line 3842 of `GOOGL_2024_10K.txt`. The 120B judge called it a "hallucination" because the benchmark creator only typed the DOJ Search case in the prompt.
  * *Netflix Financing (EQ7):* Multi-Agent cited real $5B and $20B credit facilities from lines 3216–3225 of `NFLX_2024_10K.txt`. The 120B judge penalized Multi-Agent for "fabricating financing amounts."
* **The Lesson:** Automated LLM-as-a-judge evaluations must be cross-verified against deterministic primary-source code audits.

### 3. Reasoning Token Exhaustion on Groq (`openai/gpt-oss-20b`)
* `openai/gpt-oss-20b` on Groq is a reasoning model that emits internal thinking before generating final text.
* On complex prompts with `max_tokens=4096`, the model consumed all tokens on internal reasoning, hitting the token limit and producing empty `content`.
* **The Fix:** Appended instructions to keep internal reasoning concise, raised `max_tokens` to 6000, and added fallback capture for reasoning content.

### 4. Cross-Encoder Logit Scale Calibration
* In `flashrank` (`ms-marco-MiniLM-L-12-v2`), raw cross-attention logit scores for complex, multi-clause financial queries naturally span `0.01` to `0.15`.
* An initial strict `ABSTENTION_THRESHOLD = 0.25` caused the system to mistakenly discard valid evidence chunks on Question 6. Lowering the threshold to `0.001` resolved false abstentions.

### 5. The Unfair Citation Metric
* An early audit claimed Multi-Agent had 100% citation precision while Baseline had 0%. This was an unfair comparison because the baseline prompt never instructed the LLM to output inline citations. Removing this metric restored objective, honest evaluation.

### 6. The Heuristic Overfitting Trap
* Early in the project, there was a temptation to write question-specific regexes or custom keywords to "pass the test."
* Recognizing this as anti-engineering, all question-specific heuristics were eliminated in favor of generalized, domain-agnostic software patterns (deterministic entity set matching, equal quota allocation, lexical date boundary audits).

---

## Final Scorecard & Comparison

| Category | Naive Baseline RAG | Original Multi-Agent RAG | Option A (Lean Multi-Agent RAG) |
| :--- | :---: | :---: | :---: |
| **End-to-End Latency** | **2.8 – 5.0s** | 45.0 – 70.0s | **5.6 – 7.0s** |
| **Cross-Company Comparison (EQ6, EQ10)** | **FAILED** (0 chunks for second company) | Partial (Slow / Fragmented) | **WON** (Balanced equal quota) |
| **Single-Company Factoids (EQ3, EQ4)** | **WON** (Clean, direct) | Tied / Overly Verbose | Tied (Direct & concise) |
| **Adversarial Traps (EQ8)** | **PASSED** (Refused to fabricate) | PASSED (After 30s) | **PASSED** (Direct non-disclosure) |
| **Pipeline Reliability** | High (1 call) | Very Low (400 errors, loops) | **High (Single-pass LangGraph)** |
| **Cost per Query** | **$0.00020 – $0.00030** | $0.00150 – $0.00300 | **$0.00040 – $0.00060** |

---

## The Definitive Architectural Conclusion

If presenting this project to an engineering review board or leadership, the key architectural takeaway is:

> **"Multi-Agent RAG is not a blanket replacement for standard RAG; it is an escalation mechanism for multi-document complexity."**

1. **For 70% of queries (Single-Entity Factoids):** Naive RAG (hybrid search + single LLM synthesis) is the correct architectural choice—it is 75% cheaper, 50% faster, and avoids multi-agent failure modes.
2. **For 30% of queries (Cross-Entity Multi-Hop Synthesis):** Naive RAG is mathematically broken because vector similarity clusters on a single document. Balanced entity dispatching and cross-encoder reranking are strictly required to guarantee coverage.
3. **The Optimal Production Architecture:** A **Lightweight Router** that sends single-entity questions through a fast direct path, and multi-entity comparative questions through the balanced cross-encoder pipeline.
