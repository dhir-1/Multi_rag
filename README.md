# Enterprise Multi-Agent Financial Form 10-K RAG Assistant

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/LLM-Groq%20(gpt--oss--20b)-green.svg)](https://groq.com)
[![Retrieval](https://img.shields.io/badge/Retrieval-ChromaDB%20%2B%20BM25-purple.svg)](https://github.com/chroma-core/chroma)
[![Cost](https://img.shields.io/badge/Stack-100%25%20Free%20%2F%20Local-emerald.svg)](#100-free--local-technology-stack)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An institutional-grade, multi-agent Retrieval-Augmented Generation (RAG) system engineered for deep financial research across **SEC Form 10-K annual reports**. Built on **LangGraph**, the system replaces naive vector search and flaky LLM self-reflection with **4 modern enterprise RAG methods**: dynamic zero-hardcoding filing registration with balanced entity dispatch, CPU-native cross-encoder confidence gating, schema-bounded adaptive synthesis with strict SEC citation enums, and deterministic pre-flight/post-flight mathematical guardrails.

---

## Architecture Overview

```
                                      [ User Query ]
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ 1. Pre-Flight Guardrails  │  (Pure Python, < 5ms, $0.00)
                               │ (Filing Bounds & Domain)  │
                               └─────────────┬─────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     │ (Out-of-Scope: DIS, Boeing, Recipes)          │ (Valid 10-K Query)
                     ▼                                               ▼
            ┌───────────────────┐                         ┌───────────────────────┐
            │ Instant Rejection │                         │  2. Entity Dispatch   │
            │   (0ms, 0 Tokens) │                         │ (Balanced 50/50 Quota)│
            └───────────────────┘                         └───────────┬───────────┘
                                                                      │
                                              ┌───────────────────────┴───────────────────────┐
                                              │ (ChromaDB Dense)              (BM25 Sparse)   │
                                              └───────────────────────┬───────────────────────┘
                                                                      │ (Reciprocal Rank Fusion)
                                                                      ▼
                                                          ┌───────────────────────┐
                                                          │   Hybrid Retrieval    │
                                                          │  (Candidate Chunks)   │
                                                          └───────────┬───────────┘
                                                                      │
                                                                      ▼
                                                          ┌───────────────────────┐
                                                          │  FlashRank Reranker   │ (CPU-Native ONNX)
                                                          │ (Cross-Encoder MS-12) │
                                                          └───────────┬───────────┘
                                                                      │
                                              ┌───────────────────────┴───────────────────────┐
                                              │ (Score < 0.001)                               │ (Score >= 0.001)
                                              ▼                                               ▼
                                     ┌─────────────────┐                           ┌─────────────────────┐
                                     │ Confidence Gate │                           │ 3. Adaptive         │
                                     │ (Instant Exit)  │                           │    Synthesizer      │
                                     └─────────────────┘                           │ (Executive Briefing │
                                                                                   │  or Metric Table)   │
                                                                                   └──────────┬──────────┘
                                                                                              │
                                                                                              ▼
                                                                                   ┌─────────────────────┐
                                                                                   │ 4. Post-Flight      │
                                                                                   │    Verifier         │
                                                                                   │ (Citation Provenance│
                                                                                   │  & Numeric Regex)   │
                                                                                   └──────────┬──────────┘
                                                                                              │
                                                                                              ▼
                                                                                       [ Final Answer ]
```

---

## The 4 Modern Enterprise Methods

### Method 1: Dynamic Zero-Hardcoding Filing Registry & Entity Dispatch (`agents/registry.py`)
- **Auto-Discovery**: Scans indexed filings dynamically without static dictionaries. Registers companies, tickers, CIKs, and fiscal dates directly from 10-K metadata.
- **Alias Resolution**: Normalizes corporate name variants, tickers, and brand entities (e.g. `meta`, `facebook`, `instagram`, `whatsapp`, `threads` $\rightarrow$ `META`; `alphabet`, `google`, `youtube` $\rightarrow$ `GOOGL`).
- **50/50 Balanced Entity Quota**: In comparative queries (e.g., *"Compare AMD vs NVIDIA foundries"*), naive vector search often returns 100% of one entity and starves the other. Our dispatcher enforces strict $50/50$ quota partitioning across entities before merging.

### Method 2: Calibrated Cross-Encoder Confidence Gating (`agents/reranker.py`, `agents/graph.py`)
- **Eliminating Flaky LLM Critics**: Replaces multi-thousand-token LLM evaluator loops with local **FlashRank Cross-Encoder** (`ms-marco-MiniLM-L-12-v2`) running on CPU ($0 cost).
- **Logit Confidence Gate ($0.001$)**: 
  - If candidate evidence scores $< 0.001$: The gate closes instantly, exiting with **0 LLM tokens** and preventing hallucinations on unanswerable topics.
  - If candidate evidence scores $\ge 0.001$: Green-lights single-pass synthesis with high relevance assurance.

### Method 3: Schema-Bounded "Type-Safe" Output & Adaptive Presentation (`agents/synthesizer.py`)
- **Strict SEC Citation Enums**: Every factual assertion, revenue line, contract term, and risk factor must end with a valid SEC citation in the format `[TICKER, Section]` (e.g. `[NFLX, Item 15.]`, `[AMD, ITEM 1A.]`).
- **Sanitizing Generic Tags**: Generic tags like `[SOURCE 1]` or `[Passage 2]` are deterministically rewritten to their genuine SEC filing coordinates.
- **Adaptive Content-Aware Synthesis**:
  - **Single-Company Deep Dives & Qualitative Strategies**: Formatted as structured **Executive Briefings** (`###`) with bold semantic bullet leads (`Cash Consideration:`, `Financing Structure:`, `Closing Window:`).
  - **Multi-Company Metric Comparisons**: Reserved for clean Markdown comparison tables.
  - **Non-Redundancy Rule**: Eliminates repetitive summary walls of text below tables.

### Method 4: Deterministic Pre-Flight & Post-Flight Guardrails (`agents/verifier.py`)
- **Pre-Flight Guardrail (< 5ms, $0.00 Cost)**: 
  - Rejects unindexed corporate entities (e.g. Walt Disney $DIS, Boeing $BA, Walmart $WMT) and non-financial queries (e.g. recipes, coding) before vector search or LLM invocation.
  - Emits user-friendly guidance listing supported tickers.
- **Post-Flight Guardrail**:
  - Validates SEC filing dates and boundary constraints.
  - **Numeric & Financial Claim Provenance**: Audits all dollar amounts, percentages, and integer/decimal equivalents (`$72B` vs `$72.0B`) against retrieved 10-K text via regex.
  - Flags unsupported assertions with direct audit warnings.

---

## Empirical Benchmark: Baseline Naive RAG vs. Advanced Multi-Agent RAG

A comprehensive benchmark was executed across 5 canonical query categories comparing **Baseline Naive RAG** (standard top-6 dense retrieval without reranking or guardrails) against our **Advanced Multi-Agent RAG**.

Full report: [`evaluation/BENCHMARK_REPORT.md`](evaluation/BENCHMARK_REPORT.md) | Raw Data: [`evaluation/benchmark_results.json`](evaluation/benchmark_results.json)

### Aggregate Scorecard

| Evaluation Metric | ❌ Baseline Naive RAG | ✅ Advanced Multi-Agent RAG | Operational Impact |
| :--- | :---: | :---: | :--- |
| **Verified SEC Citations** | **0** (unverifiable text) | **7** verified `[TICKER, Section]` | **100% regulatory auditability** |
| **Out-of-Scope Interception Rate** | **0%** (burned tokens on all cases) | **100%** (instant abstention) | **Zero hallucination / Zero token bleed** |
| **Out-of-Scope Latency** | 11.68s average | **0.003s (3 milliseconds)** | **~3,800x faster response** |
| **Out-of-Scope Tokens & Cost** | 3,437 tokens ($0.000364) | **0 tokens ($0.000000)** | **100% cost & token savings** |
| **Cross-Company Balanced Recall** | Prone to single-entity bias | **100% Guaranteed 50/50 Quota** | Equal representation for all entities |
| **In-Scope Tokens (Q1–Q3)** | 9,068 tokens | 17,691 tokens | Deeper 8-chunk context & full debt analysis |
| **Total API Cost (All 5 Queries)** | $0.001533 | $0.002170 | +$0.000637 total difference |

### Key Query-Level Takeaways
1. **Factual Completeness (Netflix M&A)**: Baseline missed Note 15's credit agreements and literally stated: *"financing arrangements are not known"*. Multi-Agent retrieved the **$42.2B bridge facility**, **$5B revolver**, and **$20B delayed-draw term loan** with exact citations.
2. **Unindexed Entity Protection (Walt Disney)**: Baseline retrieved Netflix streaming chunks by mistake, burned **1,900 tokens**, and took **17.6 seconds** to say it didn't know. Multi-Agent intercepted in **0.002 seconds at 0 tokens ($0.00)**.
3. **Foundry Accuracy (AMD vs NVIDIA)**: Baseline claimed NVIDIA named no foundries. Multi-Agent checked Item 1 Business, discovered **TSMC & Samsung**, and detailed AMD's **GlobalFoundries** agreement with 4 verified citations.

---

## 100% Free & Local Technology Stack

| Layer | Component | Implementation Details | Cost |
| :--- | :--- | :--- | :---: |
| **Orchestration** | `LangGraph` + `LangChain` | StateGraph multi-agent workflow with conditional edges | **$0.00** |
| **Dense Embeddings** | ChromaDB ONNX Default | Native 384-dimensional local vector embeddings | **$0.00** |
| **Vector Storage** | `ChromaDB` (Persistent) | Local on-disk persistent vector index | **$0.00** |
| **Sparse Retrieval** | `rank-bm25` (Okapi) | Local tokenized keyword index for exact financial terms | **$0.00** |
| **Reranking & Gate** | `FlashRank` (`ms-marco-MiniLM-L-12-v2`) | Local CPU-native cross-encoder inference (ONNX) | **$0.00** |
| **Synthesis LLM** | Groq (`openai/gpt-oss-20b`) | Developer Free Tier with audited rate-limit backoff | **$0.00** |
| **Backend API** | `FastAPI` + `Uvicorn` | Asynchronous REST server with CORS and `/health` | **$0.00** |

---

## Indexed Filing Registry (SEC Form 10-K)

The system indexes official Form 10-K filings across 10 major enterprise technology companies:

| Company | Ticker | CIK | Recognized Aliases | Primary Disclosure Focus |
| :--- | :---: | :---: | :--- | :--- |
| **Apple Inc.** | `AAPL` | `0000320193` | `apple`, `iphone`, `macbook`, `tim cook` | Supply chain, China outsourcing, Section 232 |
| **Advanced Micro Devices** | `AMD` | `0000002488` | `amd`, `advanced micro devices`, `lisa su` | GlobalFoundries WSA, TSMC, wafer risks |
| **Amazon.com, Inc.** | `AMZN` | `0001018724` | `amazon`, `aws`, `prime`, `andy jassy` | AWS segment, retail fulfillment, cloud Capex |
| **Salesforce, Inc.** | `CRM` | `0001108524` | `salesforce`, `crm`, `marc benioff` | Remaining Performance Obligations (RPO), Agentforce |
| **Alphabet Inc.** | `GOOGL` | `0001652044` | `google`, `alphabet`, `youtube`, `sundar pichai` | Search ad revenue, Google Cloud, DOJ antitrust |
| **Meta Platforms, Inc.** | `META` | `0001326801` | `meta`, `facebook`, `instagram`, `whatsapp`, `threads` | Family of Apps, Reality Labs Capex, AI servers |
| **Microsoft Corporation** | `MSFT` | `0000789019` | `microsoft`, `azure`, `satya nadella`, `copilot` | Azure growth, commercial cloud, OpenAI investment |
| **Netflix, Inc.** | `NFLX` | `0001065280` | `netflix`, `nflx` | WBD M&A, bridge debt facilities, ARM agreement |
| **NVIDIA Corporation** | `NVDA` | `0001045810` | `nvidia`, `nvda`, `jensen huang` | Compute & Networking, TSMC/Samsung foundries |
| **Tesla, Inc.** | `TSLA` | `0001318605` | `tesla`, `tsla`, `elon musk` | Gigafactory Shanghai, automotive gross margin |

---

## Repository Structure

```
RAG/
├── agents/
│   ├── graph.py               # LangGraph StateGraph pipeline with calibrated logit gate
│   ├── llm.py                 # AuditedChatGroq factory with zero-substitution & backoff
│   ├── memory.py              # Episodic memory for persisting audit lessons
│   ├── registry.py            # Dynamic filing registry & 50/50 entity dispatcher
│   ├── reranker.py            # FlashRank cross-encoder reranker node
│   ├── retriever.py           # Hybrid RRF search with balanced entity quota
│   ├── synthesizer.py         # Schema-bounded adaptive synthesis node
│   ├── tools.py               # Calculator and financial verification tools
│   └── verifier.py            # Pre-flight (< 5ms) & post-flight deterministic guardrails
├── config.py                  # Centralized configuration, thresholds, and token accounting
├── api.py                     # FastAPI backend service (http://127.0.0.1:8000)
├── evaluation/
│   ├── benchmark_baseline_vs_multi_agent.py # Comparative benchmark runner
│   ├── BENCHMARK_REPORT.md    # Formal evaluation report with scorecard
│   └── benchmark_results.json # Full raw empirical test metrics
├── frontend/
│   └── index.html             # Interactive browser client with real-time token tracking
├── ingestion/
│   ├── chunker.py             # Section-aware 10-K financial chunking
│   ├── fetch_tech_10k.py      # SEC EDGAR downloader for 10-K reports
│   ├── indexer.py             # ChromaDB vector + BM25 sparse indexer
│   └── run_pipeline.py        # End-to-end ingestion runner
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Setup & Quickstart

### 1. Prerequisites
- **Python**: Version `3.10` or `3.11`
- **Groq API Key**: Free developer API key from [console.groq.com](https://console.groq.com)

### 2. Installation
```bash
git clone https://github.com/dhir-1/Multi_rag.git
cd Multi_rag

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the project root:
```ini
GROQ_API_KEY=your_free_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
GROQ_INPUT_COST_PER_M=0.10
GROQ_OUTPUT_COST_PER_M=0.20

# Relevance & Gating Thresholds
RERANKER_ABSTENTION_THRESHOLD=0.001
RERANK_TOP_K=8

# Storage Directories
CHROMA_PERSIST_DIR=./data/chroma_db
BM25_INDEX_PATH=./data/bm25_index.pkl
CHUNKS_JSON_PATH=./data/chunks.json
```

### 4. Running the Web UI & API
Launch the backend server:
```bash
python api.py
```
Open **`http://127.0.0.1:8000`** in your browser to access the interactive financial intelligence interface.

Health check endpoint:
```bash
curl http://127.0.0.1:8000/health
# {"status":"healthy","service":"multi-agent-rag"}
```

### 5. Running the Comparative Benchmark
Execute the automated test suite comparing Baseline Naive RAG against Advanced Multi-Agent RAG:
```bash
python evaluation/benchmark_baseline_vs_multi_agent.py
```
The test runs all 5 canonical queries, records tokens, cost, latency, and outputs a detailed Markdown report at `evaluation/BENCHMARK_REPORT.md`.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
