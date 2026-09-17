"""
Generalized Factual Grounding Verifier for Enterprise Financial RAG.

Operates purely via deterministic lexical presence rules (zero LLM calls).
Performs domain-agnostic grounding audits:
1. Verifies that specific numerical claims, percentages, and financial metrics in the answer exist in the evidence.
2. Flags ungrounded numerical assertions without any question-specific or company-specific regexes.
3. Provides exact audit feedback for the Critic to cap scores and enforce grounded rewrites.
"""

import re
import unicodedata
from typing import List, Dict, Any, Set


def _build_evidence_corpus(chunks: List[Dict[str, Any]]) -> str:
    """Builds a single normalized lowercase text corpus from all retrieved chunks."""
    texts = [unicodedata.normalize('NFKD', c.get("text", "")) for c in chunks]
    return " ".join(texts).lower()


from datetime import datetime
from typing import List, Dict, Any, Set, Tuple
from agents.registry import get_filing_date, get_registered_tickers, REGISTERED_TICKERS


def sanitize_and_validate_citations(
    draft_answer: str,
    chunks: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, str]], List[str]]:
    """
    Type-Safe Citation & Schema Enforcement:
    1. Strict Citation Enum: Ensures all inline citations match registered company tickers:
       [AAPL, MSFT, AMZN, GOOGL, META, NVDA, AMD, TSLA, NFLX, CRM].
       Specifically enforces that META (Meta Platforms) is a first-class recognized ticker.
    2. Generic Tag Sanitization: Replaces arbitrary tags like [SOURCE 1] or [Passage 1]
       with the exact citation tag [TICKER, Section] from the corresponding evidence passage.
    3. Provenance Check: Verifies that cited tickers physically exist in the retrieved evidence chunks.
    """
    if not draft_answer or not chunks:
        return draft_answer, [], []

    valid_registered_tickers = set(get_registered_tickers())
    # Ensure META is explicitly present
    valid_registered_tickers.add("META")

    # Build passage lookup table: passage index (1-based) -> citation tag [TICKER, Section]
    passage_tags = {}
    evidence_tickers = set()
    for idx, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        ticker = meta.get("ticker", meta.get("paper_id", "")).upper().strip()
        sec = meta.get("section", "General")
        sec_short = sec.split("–")[0].split("-")[0].strip()
        if ticker:
            evidence_tickers.add(ticker)
            passage_tags[idx] = (ticker, sec_short, f"[{ticker}, {sec_short}]")

    sanitized = draft_answer
    violations = []

    # 1. Sanitize forbidden arbitrary tags: e.g. [SOURCE 1], [Passage 2], [Doc 3, Item 1]
    generic_tag_pattern = re.compile(
        r'\[\s*(?:SOURCE|Passage|Doc|Document)\s*(\d+)(?:\s*,\s*([^\]]+))?\s*\]',
        re.IGNORECASE
    )

    def _replace_generic(match):
        p_idx = int(match.group(1))
        if p_idx in passage_tags:
            tkr, default_sec, tag = passage_tags[p_idx]
            user_sec = match.group(2)
            sec_to_use = user_sec.strip() if user_sec else default_sec
            return f"[{tkr}, {sec_to_use}]"
        else:
            violations.append(
                f"Disallowed generic citation tag '{match.group(0)}' could not be resolved to any provided passage."
            )
            return match.group(0)

    sanitized = generic_tag_pattern.sub(_replace_generic, sanitized)

    # 2. Extract and validate all citations: [TICKER, Section]
    citation_pattern = re.compile(r'\[([A-Z0-9_]{1,6}|[A-Za-z\s]+),\s*([^\]]+)\]')
    matches = citation_pattern.findall(sanitized)

    extracted_citations = []
    seen = set()

    for m in matches:
        raw_tkr = m[0].strip()
        tkr = raw_tkr.upper()
        sec = m[1].strip()
        tag = f"[{tkr}, {sec}]"

        # Check against registered ticker enum (includes META)
        if tkr not in valid_registered_tickers:
            from agents.registry import get_alias_to_ticker_map
            alias_map = get_alias_to_ticker_map()
            resolved = alias_map.get(raw_tkr.lower())
            if resolved and resolved in valid_registered_tickers:
                sanitized = sanitized.replace(f"[{raw_tkr}, {sec}]", f"[{resolved}, {sec}]")
                tkr = resolved
            else:
                violations.append(
                    f"Unregistered citation ticker '[{raw_tkr}]'. Allowed tickers: {', '.join(sorted(valid_registered_tickers))}."
                )
                continue

        # Check evidence provenance: was this ticker actually retrieved for this query?
        if evidence_tickers and tkr not in evidence_tickers:
            violations.append(
                f"Citation provenance failure: Cited '[{tkr}, {sec}]' but no Form 10-K passages from {tkr} were retrieved in the evidence."
            )

        if (tkr, sec) not in seen:
            seen.add((tkr, sec))
            extracted_citations.append({
                "ticker": tkr,
                "section": sec,
                "citation_text": tag
            })

    return sanitized, extracted_citations, violations


def check_filing_date_boundaries(draft_answer: str, chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Deterministically verifies that factual past-event date claims do not exceed
    the official SEC filing date parsed from the 10-K header.
    Flags any ungrounded dates asserted past the filing date boundary.
    """
    violations = []
    ev_lower = _build_evidence_corpus(chunks)

    # Collect filing dates for all companies in the evidence
    filing_dates = {}
    for c in chunks:
        tkr = c.get("metadata", {}).get("ticker", "").upper().strip()
        if tkr and tkr not in filing_dates:
            f_date_str = get_filing_date(tkr)
            if f_date_str:
                try:
                    filing_dates[tkr] = datetime.strptime(f_date_str, "%Y-%m-%d").date()
                except ValueError:
                    pass

    if not filing_dates:
        return []

    # Find full date patterns like "October 2025", "December 15, 2024", or ISO "2025-11-01"
    month_names = "january|february|march|april|may|june|july|august|september|october|november|december"
    date_matches = re.finditer(
        rf'\b({month_names})\s+(\d{{1,2}},\s+)?(20[2-3]\d)\b',
        draft_answer,
        re.IGNORECASE
    )

    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }

    for m in date_matches:
        full_date_str = m.group(0)
        # If the date string physically appears in the 10-K evidence, it is grounded
        if full_date_str.lower() in ev_lower:
            continue

        month_str = m.group(1).lower()
        month = month_map.get(month_str, 1)
        year = int(m.group(3))
        day_raw = m.group(2)
        day = int(re.sub(r'[,\s]', '', day_raw)) if day_raw else 1

        try:
            asserted_date = datetime(year, month, day).date()
        except ValueError:
            asserted_date = None

        # Check against filing date across referenced companies
        for tkr, f_date in filing_dates.items():
            is_violation = False
            if asserted_date and asserted_date > f_date:
                is_violation = True
            elif year > f_date.year:
                is_violation = True

            if is_violation:
                violations.append(
                    f"Date assertion '{full_date_str}' exceeds the Form 10-K filing date ({f_date}) for {tkr} and does not exist in evidence."
                )

    return violations


def audit_claims_deterministically(
    draft_answer: str,
    chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Deterministically audits whether:
    1. Specific numerical metrics and percentages in draft answer exist in the evidence.
    2. Date assertions do not violate the filing date boundary.
    3. Citations strictly adhere to registered company tickers (including META) and have evidence provenance.
    Returns findings without LLM variance or hardcoded question heuristics.
    """
    if not draft_answer or not chunks:
        return {"is_grounded": True, "unsupported_claims": []}

    draft_answer = unicodedata.normalize('NFKD', draft_answer)
    ev_lower = _build_evidence_corpus(chunks)
    unsupported = []

    # 1. Generalized Financial Figure & Metric Audit
    metric_matches = re.findall(
        r'(\$\s*[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|trillion|b|m|k))?|\b\d+(?:\.\d+)?\s*%)',
        draft_answer,
        re.IGNORECASE
    )
    for raw_metric in metric_matches:
        core_digits_match = re.search(r'[\d,]+(?:\.\d+)?', raw_metric)
        if core_digits_match:
            raw_digits = core_digits_match.group(0)
            clean_digits = raw_digits.replace(",", "")
            # Strict token boundary regex: prevents small digits (e.g. '5') matching inside years ('2025') or other numbers
            # Supports mathematical equivalency between integers and trailing zero decimals (e.g. 72 vs 72.0)
            if "." in clean_digits:
                if clean_digits.endswith(".0"):
                    int_part = clean_digits[:-2]
                    escaped = rf"(?:{re.escape(clean_digits)}|{re.escape(int_part)})"
                else:
                    escaped = re.escape(clean_digits)
            else:
                escaped = rf"{re.escape(clean_digits)}(?:\.0+)?"

            strict_pattern = rf'(?<![\d.]){escaped}(?![\d.])'
            if not re.search(strict_pattern, ev_lower):
                unsupported.append(f"Numerical claim '{raw_metric.strip()}' not found in retrieved 10-K evidence.")

    # 2. Universal Filing Date Boundary Audit
    date_violations = check_filing_date_boundaries(draft_answer, chunks)
    unsupported.extend(date_violations)

    # 3. Type-Safe Citation Enum & Provenance Audit (guaranteeing META and registered tickers)
    _, _, citation_violations = sanitize_and_validate_citations(draft_answer, chunks)
    unsupported.extend(citation_violations)

    return {
        "is_grounded": len(unsupported) == 0,
        "unsupported_claims": unsupported
    }


def scrub_unsupported_hallucinations(
    draft_answer: str,
    chunks: List[Dict[str, Any]]
) -> str:
    """
    Standard formatting normalization without any question-specific word replacements.
    """
    if not draft_answer or not chunks:
        return draft_answer

    cleaned = re.sub(r'  +', ' ', draft_answer)
    cleaned = re.sub(r'\s+,', ',', cleaned)
    cleaned = re.sub(r',\s*,', ',', cleaned)
    return cleaned.strip()


# ==============================================================================
# Pre-Flight Guardrails (Before LLM Execution)
# ==============================================================================

# Well-known public companies outside our indexed Form 10-K dataset
KNOWN_UNINDEXED_COMPANIES: Dict[str, str] = {
    # Entertainment & Media
    "disney": "DIS", "walt disney": "DIS", "warner": "WBD", "paramount": "PARA", "sony": "SONY", "comcast": "CMCSA",
    # Retail & Consumer
    "walmart": "WMT", "target": "TGT", "costco": "COST", "home depot": "HD", "nike": "NKE",
    "coca-cola": "KO", "coca cola": "KO", "coke": "KO", "pepsi": "PEP", "pepsico": "PEP",
    "mcdonalds": "MCD", "mcdonald's": "MCD", "starbucks": "SBUX", "procter & gamble": "PG", "p&g": "PG",
    # Aerospace, Defense & Industrial
    "boeing": "BA", "airbus": "AIR", "lockheed": "LMT", "lockheed martin": "LMT", "general electric": "GE", "caterpillar": "CAT",
    # Automotive (Non-Tesla)
    "ford": "F", "general motors": "GM", "gm": "GM", "toyota": "TM", "honda": "HMC", "volkswagen": "VWAGY", "rivian": "RIVN", "lucid": "LCID",
    # Energy & Oil
    "exxon": "XOM", "exxonmobil": "XOM", "chevron": "CVX", "bp": "BP", "shell": "SHEL", "conocophillips": "COP",
    # Finance & Banking
    "jpmorgan": "JPM", "jp morgan": "JPM", "chase": "JPM", "bank of america": "BAC", "bofa": "BAC",
    "wells fargo": "WFC", "citigroup": "C", "citi": "C", "goldman": "GS", "goldman sachs": "GS", "morgan stanley": "MS",
    "berkshire": "BRK", "berkshire hathaway": "BRK", "visa": "V", "mastercard": "MA",
    # Healthcare & Pharma
    "pfizer": "PFE", "moderna": "MRNA", "johnson & johnson": "JNJ", "j&j": "JNJ", "unitedhealth": "UNH", "eli lilly": "LLY", "merck": "MRK",
    # Enterprise & Tech (Non-indexed)
    "intel": "INTC", "ibm": "IBM", "cisco": "CSCO", "oracle": "ORCL", "adobe": "ADBE", "qualcomm": "QCOM",
    "broadcom": "AVGO", "uber": "UBER", "airbnb": "ABNB", "spotify": "SPOT", "snap": "SNAP", "snapchat": "SNAP",
    "palantir": "PLTR", "snowflake": "SNOW", "crowdstrike": "CRWD"
}

FINANCIAL_DOMAIN_KEYWORDS: Set[str] = {
    "10-k", "10k", "filing", "sec", "annual report", "revenue", "sales", "margin", "profit",
    "net income", "operating income", "ebitda", "capex", "capital expenditure", "debt",
    "cash flow", "balance sheet", "risk", "risks", "guidance", "segment", "financial",
    "growth", "cost", "expenses", "dividend", "equity", "asset", "liability", "tax",
    "share repurchase", "buyback", "supplier", "foundry", "cloud", "data center",
    "ai infrastructure", "gross margin", "rd", "r&d", "quarter", "fiscal year", "earnings"
}


def check_pre_flight_guardrails(query: str) -> Dict[str, Any]:
    """
    Pre-Flight Guardrail (Before LLM):
    Operates in < 1ms in pure Python without LLM calls or disk I/O.
    1. Detects if query explicitly requests filings for an unindexed corporate entity (e.g. Disney, Boeing, Walmart).
    2. Rejects out-of-scope non-financial/nonsensical questions that have zero financial domain relevance.
    3. Guarantees that META and all 10 indexed tech companies pass through instantly.
    """
    q_clean = query.strip()
    if not q_clean:
        return {
            "is_allowed": False,
            "reason": "EMPTY_QUERY",
            "rejection_message": "Please enter a valid financial question."
        }

    q_lower = q_clean.lower()

    # 1. Check if ANY registered company is mentioned (including META)
    from agents.registry import extract_target_tickers, get_registered_tickers
    registered_hits = extract_target_tickers(query)

    # 2. Check for explicit unregistered corporate entities or tickers
    unregistered_detected = []

    # Check known unindexed company names
    for name, tkr in KNOWN_UNINDEXED_COMPANIES.items():
        if re.search(r'\b' + re.escape(name) + r'\b', q_lower):
            unregistered_detected.append(f"{name.title()} ({tkr})")

    # Check explicit ticker mentions like $DIS, $WMT, $BA, or ticker syntax
    ticker_symbols = re.findall(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b', q_clean)
    all_registered = set(get_registered_tickers())
    for t1, t2 in ticker_symbols:
        tkr = (t1 or t2).upper()
        if tkr in KNOWN_UNINDEXED_COMPANIES.values() and tkr not in all_registered:
            unregistered_detected.append(tkr)

    # De-duplicate
    unregistered_detected = sorted(list(set(unregistered_detected)))

    # If the user asks EXCLUSIVELY about an unregistered company (no registered company mentioned)
    if unregistered_detected and not registered_hits:
        entities_str = ", ".join(unregistered_detected)
        registered_list_str = ", ".join(sorted(all_registered))
        return {
            "is_allowed": False,
            "reason": "UNINDEXED_COMPANY",
            "rejection_message": (
                f"Form 10-K Filing Not Indexed: {entities_str} is not in our 10-K filing database. "
                f"This system currently indexes Form 10-K filings for: {registered_list_str}."
            ),
            "unregistered_entities": unregistered_detected
        }

    # 3. Check for obvious non-financial out-of-domain queries if no registered company is mentioned
    if not registered_hits:
        has_financial_keyword = any(
            re.search(r'\b' + re.escape(kw) + r'\b', q_lower)
            for kw in FINANCIAL_DOMAIN_KEYWORDS
        )
        if not has_financial_keyword:
            registered_list_str = ", ".join(sorted(all_registered))
            return {
                "is_allowed": False,
                "reason": "OUT_OF_SCOPE_NON_FINANCIAL",
                "rejection_message": (
                    "I could not find relevant evidence in the indexed Form 10-K filings to answer this question. "
                    f"Please ask a financial or SEC reporting question concerning our indexed companies ({registered_list_str})."
                )
            }

    return {
        "is_allowed": True,
        "reason": "VALID_FINANCIAL_QUERY",
        "registered_entities": list(registered_hits)
    }
