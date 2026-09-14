import os
import re
import json
from pathlib import Path
from typing import Optional
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, CHUNKS_JSON_PATH, METADATA_CATALOG_PATH


# Academic section keywords (for validating custom headings)
ACADEMIC_KEYWORDS = {
    "abstract", "introduction", "related", "background", "preliminaries",
    "method", "methodology", "approach", "architecture", "framework",
    "model", "system", "experiments", "experimental", "setup", "results",
    "evaluation", "analysis", "ablation", "discussion", "conclusion",
    "conclusions", "limitations", "future", "overview", "design",
    "benchmark", "dataset", "training", "inference", "theory", "proof"
}

# Unnumbered terminal sections that mark the end of the paper content
BIBLIOGRAPHY_SECTIONS = {"references", "bibliography"}


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text from a PDF with two-column layout awareness.
    Sorts blocks: Top spanning headers -> Left column -> Right column -> Bottom spanning.
    Prevents text scrambling across two-column arXiv paper layouts.
    """
    pages_data = []

    # Try PyMuPDF (fitz) with column-aware block sorting
    try:
        import pymupdf
        doc = pymupdf.open(pdf_path)

        for page_idx, page in enumerate(doc):
            rect = page.rect
            width, height = rect.width, rect.height
            mid_x = width / 2.0

            # Get text blocks: (x0, y0, x1, y1, text, block_no, block_type)
            blocks = page.get_text("blocks")
            text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

            if not text_blocks:
                continue

            top_spanning = []
            left_col = []
            right_col = []
            bottom_spanning = []

            for b in text_blocks:
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                block_width = x1 - x0

                # Block spans across both columns (e.g. Title, Abstract, wide table)
                if block_width > width * 0.65 or (x0 < mid_x * 0.6 and x1 > mid_x * 1.4):
                    if y0 < height * 0.35:
                        top_spanning.append(b)
                    else:
                        bottom_spanning.append(b)
                elif x1 <= mid_x + 20:
                    left_col.append(b)
                else:
                    right_col.append(b)

            # Sort each group top-to-bottom by y0 coordinate
            top_spanning.sort(key=lambda b: b[1])
            left_col.sort(key=lambda b: b[1])
            right_col.sort(key=lambda b: b[1])
            bottom_spanning.sort(key=lambda b: b[1])

            # Combine in true human reading order
            ordered_blocks = top_spanning + left_col + right_col + bottom_spanning
            page_text = "\n".join(b[4].strip() for b in ordered_blocks)

            if page_text.strip():
                pages_data.append({"page_num": page_idx + 1, "text": page_text})

        doc.close()
        return pages_data
    except ImportError:
        pass

    # Fallback to pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_data.append({"page_num": page_idx + 1, "text": text})
        return pages_data
    except ImportError:
        raise ImportError(
            "Neither PyMuPDF ('pymupdf') nor 'pypdf' is installed. "
            "Please run: pip install pymupdf pypdf"
        )


def is_page_header_or_footer(line: str, paper_title: str = "") -> bool:
    """
    Detects running page headers, footers, preprint watermarks, and lone page numbers.
    """
    cleaned = line.strip()
    if not cleaned:
        return True

    # Lone page number or short number
    if cleaned.isdigit() or re.match(r"^page\s+\d+$", cleaned, re.IGNORECASE):
        return True

    lower = cleaned.lower()

    # Common preprint/header watermarks
    if re.search(r"arxiv:\d+\.\d+", lower) or "under review" in lower or "preprint" in lower:
        return True

    # Check title overlap (running headers repeating the paper's title or author names)
    if paper_title:
        title_words = set(re.findall(r"\w+", paper_title.lower()))
        line_words = set(re.findall(r"\w+", lower))
        if len(line_words) >= 3 and title_words:
            overlap = len(line_words & title_words) / len(line_words)
            if overlap >= 0.5:
                return True

    return False


def is_table_like(line: str) -> bool:
    """
    Detects lines from PDF tables, benchmark grids, numerical results, and dense table legends.
    Filters out:
    1. Table/Figure captions ('Table 1:', 'Fig. 2')
    2. Dense lists of short comma-separated items (e.g. 'Dataset, Memory text, top-k 25...')
    3. Grid rows with high numeric/stat density (>35%)
    Leaves normal prose sentences untouched.
    """
    cleaned = line.strip()
    if not cleaned:
        return False

    # Direct table/figure caption match
    if re.match(r"^(table|tab\.|figure|fig\.)\s+\d+[:\.]?", cleaned, re.IGNORECASE):
        return True

    # Filter 1: Dense comma-separated or semicolon-separated table/legend phrases
    # e.g., 'Dataset, Memory text, primary vectors, top-k 25, requested reader protocol'
    comma_segments = [s.strip() for s in cleaned.split(",") if s.strip()]
    if len(comma_segments) >= 4:
        short_segments = sum(1 for s in comma_segments if len(s.split()) <= 4)
        if short_segments / len(comma_segments) >= 0.75:
            return True

    # Filter 2: Table columns with vertical bars or multiple consecutive spaces/tabs
    if cleaned.count("|") >= 2 or "\t\t" in line:
        return True

    tokens = cleaned.split()
    if len(tokens) < 3:
        num_count = sum(
            1 for t in tokens 
            if any(c.isdigit() for c in t) or t in {"-", "—", "–", "+", "±", "|", "/", "=", "<", ">"}
        )
        return num_count == len(tokens)

    # Filter 3: High density of numeric, percentage, range, or table separator tokens (>35%)
    numeric_or_symbol_count = 0
    for t in tokens:
        if any(c.isdigit() for c in t):
            numeric_or_symbol_count += 1
        elif t in {"-", "—", "–", "+", "±", "|", "/", "\\", "=", "<", ">", "%"}:
            numeric_or_symbol_count += 1
        elif len(t) <= 2 and not t.isalpha():
            numeric_or_symbol_count += 1

    ratio = numeric_or_symbol_count / len(tokens)
    if ratio >= 0.35:
        return True

    # Filter 4: Benchmark table rows with multiple decimal floats / ranges / stats
    stat_tokens = [t for t in tokens if re.search(r"(\d+\.\d+|\d+–\d+|\+\d+|\-\d+)", t)]
    if len(stat_tokens) >= 3:
        return True

    return False


# Common academic standalone section headers
STANDALONE_HEADERS = {
    "abstract", "introduction", "related work", "background", "preliminaries",
    "method", "methods", "methodology", "proposed method", "proposed approach",
    "architecture", "model architecture", "system design", "experiments",
    "experimental setup", "experiments and results", "results", "evaluation",
    "analysis", "ablation study", "discussion", "limitations", "conclusion",
    "conclusions", "future work", "ethical considerations", "acknowledgements",
    "acknowledgments", "appendix", "references", "bibliography"
}


def identify_section(line: str, paper_title: str = "") -> Optional[str]:
    """
    Detects section headers accurately while eliminating false-positives:
    - Matches numbered sections with or without dots (e.g. '1. Introduction', '3.2. Model Architecture')
    - Matches standalone academic headings ('Introduction', 'Method', 'Results', 'Conclusion')
    - Rejects table-like numerical lines and legends
    - Rejects running page headers/footers matching the paper title
    - Rejects sentences starting with numbers ('4 NVIDIA GPUs were used...')
    - Rejects lines ending with sentence punctuation (., ;, :, !)
    """
    cleaned = line.strip()

    # Reject table rows and running page headers/footers immediately
    if is_table_like(cleaned) or is_page_header_or_footer(cleaned, paper_title):
        return None

    # Section headers are concise (3 to 65 characters, max 8 words)
    if not cleaned or len(cleaned) < 3 or len(cleaned) > 65:
        return None

    words = cleaned.split()
    if len(words) > 8:
        return None

    # Sentence filter: Real headings do NOT end with sentence punctuation (except when it's just '1.')
    if cleaned.endswith((",", ";", ":", "!", "?")):
        return None

    lower = cleaned.lower()

    # Check 1: Standalone unnumbered common sections (Abstract, Introduction, Method, Results, Conclusion, etc.)
    if lower in STANDALONE_HEADERS:
        return cleaned.title()

    # Sentence filter: Reject lines containing mid-sentence verbs/pronouns
    sentence_clues = {" were ", " was ", " are ", " is ", " we ", " they ", " our ", " with ", " that ", " using "}
    if any(clue in f" {lower} " for clue in sentence_clues):
        return None

    # Check 2: Numbered section pattern (e.g., '1. Introduction', '1 Introduction', '3.2. Model Architecture', 'IV. Experiments')
    # Match: Top-level section number 1-12 or Roman numeral I-XII, with optional trailing dot, followed by heading text
    numbered_match = re.match(
        r"^((?:[1-9]|1[0-2])(?:\.\d+)*\.?|[ivxlcdm]+[\.\)]|section\s+(?:[1-9]|1[0-2])[:\.]?)\s+(.+)$",
        cleaned,
        re.IGNORECASE
    )

    if numbered_match:
        sec_prefix = numbered_match.group(1).strip()
        raw_heading_body = numbered_match.group(2).strip()

        # If heading is followed inline by a sentence (e.g. '5 Method. We propose...'), take only the heading part before the period
        if ". " in raw_heading_body:
            heading_body = raw_heading_body.split(". ")[0].strip()
        else:
            heading_body = raw_heading_body

        body_lower = heading_body.lower()
        body_words = body_lower.split()

        # Reject if heading body repeats paper title
        if paper_title and is_page_header_or_footer(heading_body, paper_title):
            return None

        # Must contain actual words (letters), not just numbers/decimals from table rows
        alpha_words = [w for w in heading_body.split() if any(c.isalpha() for c in w)]
        if not alpha_words:
            return None

        # Must have at least one academic keyword OR all words capitalized (Title Case/ALL CAPS) OR in STANDALONE_HEADERS
        has_academic_word = any(w in ACADEMIC_KEYWORDS for w in body_words)
        is_title_cased = all(w[0].isupper() for w in alpha_words if w[0].isalpha())
        is_known_heading = body_lower in STANDALONE_HEADERS

        if has_academic_word or is_title_cased or is_known_heading:
            clean_prefix = sec_prefix.rstrip(".")
            return f"{clean_prefix} {heading_body.title()}"

    return None


def split_text_into_windows(text: str, chunk_size_words: int = CHUNK_SIZE_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """
    Splits long text into overlapping word windows to prevent cutting thoughts.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += chunk_size_words - overlap_words

    return chunks


def parse_and_chunk_paper(
    pdf_path: str,
    paper_id: str,
    paper_title: str,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS
) -> list[dict]:
    """
    Section-aware chunking of an academic paper.
    Groups text by section, then chunks each section into manageable windows.
    """
    pages_data = extract_text_from_pdf(pdf_path)
    if not pages_data:
        return []

    sections = []
    current_section_title = "Abstract / Header"
    current_section_text = []
    current_page = 1

    # First pass: collect lines and group by detected sections
    for page in pages_data:
        lines = page["text"].split("\n")
        stop_parsing = False
        for line in lines:
            cleaned_line = line.strip()
            # Skip empty lines, page headers/footers, and numerical table/legend rows
            if (
                not cleaned_line 
                or is_page_header_or_footer(cleaned_line, paper_title=paper_title) 
                or is_table_like(cleaned_line)
            ):
                continue

            detected = identify_section(cleaned_line, paper_title=paper_title)
            if detected:
                # If we reached References/Bibliography, stop collecting further sections
                if any(ref_word in detected.lower() for ref_word in ["reference", "bibliography"]):
                    stop_parsing = True
                    break

                # Save previous section if it has text
                full_text = " ".join(current_section_text).strip()
                if full_text:
                    sections.append({
                        "section": current_section_title,
                        "text": full_text,
                        "page_num": current_page
                    })
                current_section_title = detected
                current_section_text = []
                current_page = page["page_num"]
            else:
                current_section_text.append(cleaned_line)

        if stop_parsing:
            break

    # Add the last active section before references
    full_text = " ".join(current_section_text).strip()
    if full_text and not any(ref_word in current_section_title.lower() for ref_word in ["reference", "bibliography"]):
        sections.append({
            "section": current_section_title,
            "text": full_text,
            "page_num": current_page
        })

    # Second pass: chunk each section
    all_chunks = []
    chunk_idx = 0

    for sec in sections:

        windows = split_text_into_windows(
            sec["text"], 
            chunk_size_words=chunk_size_words, 
            overlap_words=overlap_words
        )

        for w_idx, window_text in enumerate(windows):
            chunk_idx += 1
            chunk_obj = {
                "chunk_id": f"{paper_id}_chunk_{chunk_idx:03d}",
                "paper_id": paper_id,
                "paper_title": paper_title,
                "section": sec["section"],
                "page_num": sec["page_num"],
                "text": window_text
            }
            all_chunks.append(chunk_obj)

    return all_chunks


def process_all_papers(
    metadata_json_path: str = METADATA_CATALOG_PATH,
    output_chunks_path: str = CHUNKS_JSON_PATH
) -> list[dict]:
    """
    Reads metadata catalog of downloaded papers, chunks all of them, and saves to chunks.json.
    """
    meta_path = Path(metadata_json_path)
    if not meta_path.exists():
        print(f"[-] Metadata file not found at: {metadata_json_path}")
        return []

    with open(meta_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    all_processed_chunks = []

    print(f"[*] Processing {len(papers)} papers with section-aware chunking...")

    for paper in papers:
        pdf_path = paper.get("pdf_path")
        paper_id = paper.get("paper_id")
        paper_title = paper.get("title")

        if not pdf_path or not os.path.exists(pdf_path):
            print(f"[-] Skipping {paper_id}: PDF file not found at {pdf_path}")
            continue

        print(f"    - Parsing: {paper_title[:60]}... (ID: {paper_id})")
        chunks = parse_and_chunk_paper(
            pdf_path=pdf_path,
            paper_id=paper_id,
            paper_title=paper_title
        )
        print(f"      -> Created {len(chunks)} section-aware chunks.")
        all_processed_chunks.extend(chunks)

    # Save all chunks
    out_path = Path(output_chunks_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_processed_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Total chunks created: {len(all_processed_chunks)}")
    print(f"[+] Chunks saved to: {output_chunks_path}")
    return all_processed_chunks


if __name__ == "__main__":
    process_all_papers()
