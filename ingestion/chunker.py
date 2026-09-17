"""
Section-Aware Financial 10-K Chunker.

Parses official Form 10-K filings for 10 major Tech Companies (FY2024/FY2025):
NVIDIA, Apple, Microsoft, Alphabet, Amazon, Tesla, Meta, Netflix, AMD, Salesforce.

Features:
- Sentence-aware & paragraph-aware chunk boundaries (never truncates mid-word or mid-sentence).
- Strict boilerplate & pagination filtering (removes running headers/footers, Table of Contents, page numbers).
- Drops non-substantive micro-chunks (< 25 words, e.g. empty [Reserved] sections or "None.").
- Preserves markdown financial tables and structured notes.
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, CHUNKS_JSON_PATH

TECH_10K_DIR = os.path.join(PROJECT_ROOT, "data", "tech_10k")
MIN_CHUNK_WORDS = 25

ITEM_HEADER_PATTERN = re.compile(
    r"^(?:item\s+(?:1a|1b|1c|1|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15)[\.:\s\|\-]+[^\n]{2,80})",
    re.IGNORECASE
)

NOTE_HEADER_PATTERN = re.compile(
    r"^(?:NOTE\s+(\d+)[\.:\s—–\|\-]+([^\n]{0,80}))",
    re.IGNORECASE
)

# XBRL taxonomy / tag line patterns to clean out
XBRL_PATTERN = re.compile(
    r"^(https?://|iso\d+:|xbrli?:|[a-z0-9_]+:[a-z0-9_]+|\d{10}|\d{4}-\d{2}-\d{2}|p\d+[ymd]|true|false|\d+)$",
    re.IGNORECASE
)

# Boilerplate pagination and running header/footer patterns
BOILERPLATE_PATTERNS = [
    # Running page header/footer: e.g. "Apple Inc. | 2025 Form 10-K | 20" or "NVIDIA CORPORATION | 2024 FORM 10-K | 45"
    re.compile(r"^(?:[A-Za-z0-9\.,\s&'\-]+)\s*\|\s*(?:\d{4}\s*)?Form\s+10-K(?:\s*\|\s*\d+)?\s*$", re.IGNORECASE),
    # Running header variant: "Form 10-K | 20" or "2025 Form 10-K"
    re.compile(r"^(?:\d{4}\s*)?Form\s+10-K(?:\s*\|\s*\d+)?\s*$", re.IGNORECASE),
    # Standalone table of contents line
    re.compile(r"^table of contents(?:\s*\|\s*[A-Za-z0-9\s]+)?$", re.IGNORECASE),
    # Standalone page numbers or index markers
    re.compile(r"^\d{1,4}$"),
    re.compile(r"^part\s+[ivxlcdm]+(?:\s*\|\s*\d+)?$", re.IGNORECASE)
]


def is_boilerplate_line(line: str) -> bool:
    """Checks if a line is a running header, page footer, TOC artifact, or pure page number."""
    line_clean = line.strip()
    if not line_clean:
        return True
    for pat in BOILERPLATE_PATTERNS:
        if pat.match(line_clean):
            return True
    return False


def parse_header_metadata(lines: List[str]) -> Dict[str, str]:
    """Extracts structured metadata from the 10-line header block."""
    meta = {
        "company": "Unknown",
        "ticker": "UNKNOWN",
        "cik": "",
        "year": "2024",
        "filing_date": "",
        "sec_url": ""
    }
    for line in lines[:15]:
        line = line.strip()
        if line.startswith("Company Name :"):
            meta["company"] = line.split(":", 1)[1].strip()
        elif line.startswith("Ticker Symbol:"):
            meta["ticker"] = line.split(":", 1)[1].strip()
        elif line.startswith("CIK Number   :"):
            meta["cik"] = line.split(":", 1)[1].strip()
        elif line.startswith("Fiscal Year  :"):
            meta["year"] = line.split(":", 1)[1].strip()
        elif line.startswith("Filing Date  :"):
            meta["filing_date"] = line.split(":", 1)[1].strip()
        elif line.startswith("SEC URL      :"):
            meta["sec_url"] = line.split(":", 1)[1].strip()
    return meta


def split_into_semantic_units(text: str) -> List[str]:
    """
    Splits text into coherent semantic units:
    - Markdown tables (consecutive lines containing '|') are preserved as single atomic units.
    - Prose lines are split into sentences.
    - Headings remain intact.
    """
    lines = text.split("\n")
    units = []
    table_buffer = []

    for line in lines:
        line_clean = line.strip()
        if not line_clean or is_boilerplate_line(line_clean):
            if table_buffer:
                units.append("\n".join(table_buffer))
                table_buffer = []
            continue

        if "|" in line_clean:
            table_buffer.append(line_clean)
        else:
            if table_buffer:
                units.append("\n".join(table_buffer))
                table_buffer = []

            # Split prose on sentence boundaries
            if len(line_clean) < 60 and not line_clean.endswith("."):
                units.append(line_clean)
            else:
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\"\'\(\[])', line_clean)
                for s in sentences:
                    s_strip = s.strip()
                    if s_strip and not is_boilerplate_line(s_strip):
                        units.append(s_strip)

    if table_buffer:
        units.append("\n".join(table_buffer))

    return units


def split_into_sentence_windows(
    text: str,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
    min_chunk_words: int = MIN_CHUNK_WORDS
) -> List[str]:
    """
    Splits text into sentence-aware overlapping windows with table preservation.
    Guarantees markdown tables are never split across chunk boundaries.
    """
    units = split_into_semantic_units(text)
    if not units:
        return []

    windows = []
    curr_units = []
    curr_words = 0

    for u in units:
        u_words = len(u.split())
        is_table = "|" in u and "\n" in u

        # If it's a multi-row table, treat as an atomic block
        if is_table:
            if curr_units:
                win_text = "\n\n".join(curr_units).strip()
                if curr_words >= min_chunk_words:
                    windows.append(win_text)
                curr_units = []
                curr_words = 0
            # Add table as its own complete chunk
            if u_words >= min_chunk_words:
                windows.append(u.strip())
            continue

        if curr_words + u_words > chunk_size_words and curr_units:
            win_text = "\n\n".join(curr_units).strip()
            if curr_words >= min_chunk_words:
                windows.append(win_text)

            # Compute sentence overlap without breaking tables
            overlap_units = []
            overlap_accum = 0
            for rev_u in reversed(curr_units):
                if "|" in rev_u and "\n" in rev_u:
                    break
                w_count = len(rev_u.split())
                if overlap_accum + w_count <= overlap_words:
                    overlap_units.insert(0, rev_u)
                    overlap_accum += w_count
                else:
                    break
            curr_units = overlap_units
            curr_words = sum(len(x.split()) for x in curr_units)

        curr_units.append(u)
        curr_words += u_words

    if curr_units:
        win_text = "\n\n".join(curr_units).strip()
        if curr_words >= min_chunk_words:
            windows.append(win_text)

    return windows


def chunk_10k_file(
    file_path: str,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS
) -> List[Dict[str, Any]]:
    """Chunks a single 10-K text file into section-aware, sentence-boundary chunks."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if not lines:
        return []

    meta = parse_header_metadata(lines)
    ticker = meta["ticker"]
    company = meta["company"]
    year = meta["year"]

    # 1. Locate start of actual 10-K report (after XBRL dump)
    start_line = 0
    for idx in range(11, len(lines)):
        l = lines[idx].strip().lower()
        if "table of contents" in l or "form 10-k" in l or "annual report pursuant to section 13" in l:
            start_line = idx
            break

    doc_lines = lines[start_line:]

    # 2. Group into semantic sections
    sections = []
    current_sec_name = "Cover & Overview"
    current_sec_lines = []

    for line in doc_lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Filter out standalone XBRL taxonomy lines
        if XBRL_PATTERN.match(line_clean) or ("Member" in line_clean and ":" in line_clean):
            continue

        # Filter out boilerplate running headers/footers, TOC lines, standalone page numbers
        if is_boilerplate_line(line_clean):
            continue

        is_item = ITEM_HEADER_PATTERN.match(line_clean) and len(line_clean) < 95
        is_note = NOTE_HEADER_PATTERN.match(line_clean) and len(line_clean) < 95

        if is_item or is_note:
            if is_item:
                clean_heading = re.sub(r"\s*\|\s*", " - ", line_clean).strip(" -")
            else:
                m_note = NOTE_HEADER_PATTERN.match(line_clean)
                note_num = m_note.group(1)
                note_title = m_note.group(2).strip(" —–-:")
                clean_heading = f"Item 8 - Note {note_num}" + (f": {note_title}" if note_title else "")

            if current_sec_lines:
                sec_content = "\n".join(current_sec_lines).strip()
                if len(sec_content.split()) >= MIN_CHUNK_WORDS:
                    sections.append({
                        "section": current_sec_name,
                        "text": sec_content
                    })
                current_sec_lines = []
            current_sec_name = clean_heading
        else:
            current_sec_lines.append(line_clean)

    if current_sec_lines:
        sec_content = "\n".join(current_sec_lines).strip()
        if len(sec_content.split()) >= MIN_CHUNK_WORDS:
            sections.append({
                "section": current_sec_name,
                "text": sec_content
            })

    # 3. Chunk each section with sentence-aware sliding window
    file_chunks = []
    chunk_idx = 0

    for sec in sections:
        sec_name = sec["section"]
        sec_text = sec["text"]

        windows = split_into_sentence_windows(
            sec_text,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            min_chunk_words=MIN_CHUNK_WORDS
        )

        for win_text in windows:
            chunk_idx += 1
            chunk_id = f"{ticker}_{year}_chunk_{chunk_idx:04d}"
            situational_header = f"[{ticker} - {company} Form 10-K (FY{year}) | {sec_name}]\n"
            file_chunks.append({
                "chunk_id": chunk_id,
                "ticker": ticker,
                "company": company,
                "fiscal_year": year,
                "section": sec_name,
                "text": situational_header + win_text,
                "raw_text": win_text,
                "source_file": os.path.basename(file_path)
            })

    return file_chunks


def process_all_10k_filings(
    data_dir: str = TECH_10K_DIR,
    output_path: str = CHUNKS_JSON_PATH,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS
) -> List[Dict[str, Any]]:
    """Chunks all Form 10-K files in data/tech_10k/ and saves data/chunks.json."""
    if not os.path.exists(data_dir):
        print(f"[-] Directory not found: {data_dir}")
        return []

    files = [f for f in sorted(os.listdir(data_dir)) if f.endswith(".txt")]
    print("=" * 80)
    print(f"📦 SENTENCE-AWARE CHUNKING: {len(files)} Form 10-K Filings")
    print(f"   Target chunk size: ~{chunk_size_words} words (overlap: ~{overlap_words} words)")
    print(f"   Boilerplate & footer filtering: ACTIVE | Min chunk words: {MIN_CHUNK_WORDS}")
    print("=" * 80)

    all_chunks = []
    summary_by_company = {}

    for fname in files:
        fpath = os.path.join(data_dir, fname)
        print(f"  * Processing: {fname}...")
        chunks = chunk_10k_file(fpath, chunk_size_words, overlap_words)
        all_chunks.extend(chunks)

        ticker = fname.split("_")[0]
        summary_by_company[ticker] = len(chunks)
        print(f"    -> Generated {len(chunks)} clean sentence-aware chunks.")

    # Save to data/chunks.json
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"✅ Total clean chunks created: {len(all_chunks):,} across {len(files)} companies.")
    print(f"💾 Saved chunks to: {output_path}")
    print("=" * 80)
    print("Chunks breakdown per company:")
    for ticker, count in summary_by_company.items():
        print(f"  - {ticker:6s}: {count:4d} chunks")
    print("=" * 80)

    return all_chunks


if __name__ == "__main__":
    process_all_10k_filings()
