"""
FilingTools Module for Enterprise ReAct Financial Agent.

Provides dedicated programmatic query tools across SEC Form 10-K filings:
1. `search_keyword`: Hybrid dense + BM25 search with Reciprocal Rank Fusion (RRF).
2. `get_financial_statement_note`: Directly retrieves audited footnote blocks from Item 8
   (e.g., Note 14 Investments/OpenAI, Note 17 Segment Reporting, Note 10 Leases).
3. `get_section`: Directly retrieves chunks from primary 10-K sections (Item 1, Item 1A, Item 7, Item 8).
4. `get_filing_metadata`: Provides filing date, fiscal year, and official registrant name.
5. `list_available_companies`: Discovers all dynamically registered companies.

Enables multi-turn autonomous tool execution when a single search pass lacks necessary evidence.
"""

import os
import re
import sys
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.registry import get_filing_registry, get_filing_date
from ingestion.indexer import HybridIndexer

_INDEXER_INSTANCE = None


def _get_indexer() -> HybridIndexer:
    global _INDEXER_INSTANCE
    if _INDEXER_INSTANCE is None:
        _INDEXER_INSTANCE = HybridIndexer()
        _INDEXER_INSTANCE.load_indexes()
    return _INDEXER_INSTANCE


class FilingTools:
    """Enterprise Tool Interface for Autonomous Financial RAG."""

    @staticmethod
    def search_keyword(query: str, ticker: Optional[str] = None, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Executes hybrid dense vector + sparse keyword search with Reciprocal Rank Fusion.
        Optionally scoped to a specific company ticker (e.g., 'MSFT', 'NVDA').
        """
        indexer = _get_indexer()
        ticker_filter = ticker.upper().strip() if ticker else None
        return indexer.hybrid_search(query=query, top_k=top_k, ticker_filter=ticker_filter)

    @staticmethod
    def get_financial_statement_note(ticker: str, note_number: Any) -> List[Dict[str, Any]]:
        """
        Directly retrieves audited footnote chunks from Item 8 for a specific company
        and footnote number (e.g., Note 14, Note 18, Note 1).
        """
        tkr = ticker.upper().strip()
        num_str = str(note_number).strip()
        indexer = _get_indexer()

        # 1. First try exact metadata section match from chunks_lookup
        matched_chunks = []
        if indexer and indexer.chunks_lookup:
            for c in indexer.chunks_lookup.values():
                if c.get("ticker", "").upper() == tkr:
                    sec = c.get("section", "")
                    # Matches "Item 8 - Note 14" or "Note 14"
                    if re.search(r'\bnote\s+' + re.escape(num_str) + r'\b', sec, re.IGNORECASE):
                        matched_chunks.append({
                            "chunk_id": c["chunk_id"],
                            "text": c["text"],
                            "metadata": {
                                "ticker": tkr,
                                "company": c.get("company", ""),
                                "section": sec,
                                "fiscal_year": c.get("fiscal_year", "")
                            },
                            "rrf_score": 1.0
                        })

        if matched_chunks:
            return matched_chunks[:6]

        # 2. Fallback: targeted hybrid search for Note and Footnotes
        query = f"Note {num_str} to financial statements footnotes Item 8"
        return indexer.hybrid_search(query=query, top_k=4, ticker_filter=tkr)

    @staticmethod
    def get_section(ticker: str, section_name: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Directly retrieves chunks belonging to a major Form 10-K section:
        - Item 1: Business Overview & Products
        - Item 1A: Risk Factors
        - Item 7: Management's Discussion & Analysis (MD&A)
        - Item 8: Financial Statements and Supplementary Data
        """
        tkr = ticker.upper().strip()
        sec_clean = section_name.strip().upper()
        indexer = _get_indexer()

        matched_chunks = []
        if indexer and indexer.chunks_lookup:
            for c in indexer.chunks_lookup.values():
                if c.get("ticker", "").upper() == tkr:
                    sec = c.get("section", "").upper()
                    if sec_clean in sec or (sec_clean == "ITEM 1A" and "1A" in sec):
                        matched_chunks.append({
                            "chunk_id": c["chunk_id"],
                            "text": c["text"],
                            "metadata": {
                                "ticker": tkr,
                                "company": c.get("company", ""),
                                "section": c.get("section", ""),
                                "fiscal_year": c.get("fiscal_year", "")
                            },
                            "rrf_score": 0.95
                        })

        if matched_chunks:
            return matched_chunks[:top_k]

        return indexer.hybrid_search(query=section_name, top_k=top_k, ticker_filter=tkr)

    @staticmethod
    def get_filing_metadata(ticker: str) -> Dict[str, Any]:
        """Returns metadata for a company's filing: filing date, CIK, fiscal year."""
        reg = get_filing_registry()
        return reg.get(ticker.upper(), {"ticker": ticker, "company": "Unknown"})

    @staticmethod
    def list_available_companies() -> List[Dict[str, Any]]:
        """Lists all companies discovered in data/tech_10k/."""
        reg = get_filing_registry()
        return [
            {
                "ticker": tkr,
                "company": m.get("company", ""),
                "filing_date": m.get("filing_date", ""),
                "fiscal_year": m.get("fiscal_year", "")
            }
            for tkr, m in reg.items()
        ]
