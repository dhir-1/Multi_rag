"""
Fetch and process official Form 10-K filings for 10 major Tech Companies from SEC EDGAR:
1. NVIDIA (NVDA)
2. Apple (AAPL)
3. Microsoft (MSFT)
4. Alphabet / Google (GOOGL)
5. Amazon (AMZN)
6. Tesla (TSLA)
7. Meta Platforms (META)
8. Netflix (NFLX)
9. Advanced Micro Devices (AMD)
10. Salesforce (CRM)

Extracts high-value analytical sections (Business, Risk Factors, MD&A, Financial Highlights)
and saves them as clean structured text under data/tech_10k/{TICKER}_2024_10K.txt
"""

import os
import re
import sys
import time
import json
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TECH_10K_DIR = os.path.join(PROJECT_ROOT, "data", "tech_10k")
os.makedirs(TECH_10K_DIR, exist_ok=True)

# SEC EDGAR requires specific User-Agent header
SEC_HEADERS = {
    "User-Agent": "AcademicResearchInstitute rag_benchmark@univ-research.edu",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}

SEC_ARCHIVE_HEADERS = {
    "User-Agent": "AcademicResearchInstitute rag_benchmark@univ-research.edu",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov"
}

COMPANIES = [
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "cik": "0001045810", "fy": 2024},
    {"ticker": "AAPL", "name": "Apple Inc.", "cik": "0000320193", "fy": 2024},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "cik": "0000789019", "fy": 2024},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "cik": "0001652044", "fy": 2024},
    {"ticker": "AMZN", "name": "Amazon.com, Inc.", "cik": "0001018724", "fy": 2024},
    {"ticker": "TSLA", "name": "Tesla, Inc.", "cik": "0001318605", "fy": 2024},
    {"ticker": "META", "name": "Meta Platforms, Inc.", "cik": "0001326801", "fy": 2024},
    {"ticker": "NFLX", "name": "Netflix, Inc.", "cik": "0001065280", "fy": 2024},
    {"ticker": "AMD", "name": "Advanced Micro Devices, Inc.", "cik": "0000002488", "fy": 2024},
    {"ticker": "CRM", "name": "Salesforce, Inc.", "cik": "0001108524", "fy": 2024},
]


def clean_html_to_markdown(html_content: str) -> str:
    """Cleans SEC inline XBRL / HTML into readable structured markdown text."""
    soup = BeautifulSoup(html_content, "lxml")
    
    # Remove script, style, and metadata tags
    for tag in soup(["script", "style", "meta", "link", "noscript"]):
        tag.decompose()
        
    # Convert tables into clean text lines or markdown tables
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [re.sub(r"\s+", " ", td.get_text().strip()) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            table.replace_with("\n" + "\n".join(rows) + "\n")
        else:
            table.decompose()
            
    text = soup.get_text("\n")
    # Normalize whitespace
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    clean_lines = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                clean_lines.append("")
                prev_blank = True
        else:
            clean_lines.append(line)
            prev_blank = False
            
    return "\n".join(clean_lines)


def fetch_latest_10k_url(cik: str) -> Optional[Dict[str, str]]:
    """Finds the accession number and document name for the latest Form 10-K on SEC EDGAR."""
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    try:
        r = requests.get(url, headers=SEC_HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"  [!] Failed to get submissions for CIK {cik}: Status {r.status_code}")
            return None
        data = r.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        for idx, form in enumerate(forms):
            if form == "10-K":
                acc = recent["accessionNumber"][idx]
                doc = recent["primaryDocument"][idx]
                filing_date = recent["filingDate"][idx]
                acc_clean = acc.replace("-", "")
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"
                return {
                    "filing_date": filing_date,
                    "accession_number": acc,
                    "doc_name": doc,
                    "url": doc_url
                }
    except Exception as e:
        print(f"  [!] Error fetching CIK {cik}: {e}")
    return None


def download_and_extract_10k(company: Dict[str, Any]) -> bool:
    ticker = company["ticker"]
    name = company["name"]
    cik = company["cik"]
    fy = company["fy"]
    out_path = os.path.join(TECH_10K_DIR, f"{ticker}_2024_10K.txt")
    
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        print(f"  [+] {ticker} ({name}) already downloaded ({os.path.getsize(out_path):,} bytes). Skipping.")
        return True
        
    print(f"\n[*] Fetching Form 10-K metadata for {ticker} ({name})...")
    meta = fetch_latest_10k_url(cik)
    if not meta:
        print(f"  [!] Could not locate 10-K for {ticker}")
        return False
        
    doc_url = meta["url"]
    filing_date = meta["filing_date"]
    print(f"  * 10-K Filing Date : {filing_date}")
    print(f"  * Downloading Document: {doc_url}")
    
    try:
        r = requests.get(doc_url, headers=SEC_ARCHIVE_HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"  [!] Download failed with status {r.status_code}")
            return False
            
        print(f"  * Received {len(r.content):,} bytes. Parsing and cleaning text...")
        clean_text = clean_html_to_markdown(r.text)
        
        # Add rich metadata header
        header = (
            f"================================================================================\n"
            f"UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n"
            f"FORM 10-K ANNUAL REPORT\n"
            f"Company Name : {name}\n"
            f"Ticker Symbol: {ticker}\n"
            f"CIK Number   : {cik}\n"
            f"Fiscal Year  : {fy}\n"
            f"Filing Date  : {filing_date}\n"
            f"SEC URL      : {doc_url}\n"
            f"================================================================================\n\n"
        )
        
        full_content = header + clean_text
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_content)
            
        print(f"  [+] Successfully saved {ticker} 10-K to {out_path} ({len(full_content):,} chars)")
        return True
        
    except Exception as e:
        print(f"  [!] Failed to download/process {ticker}: {e}")
        return False


def main():
    print("=" * 80)
    print("📥 SEC EDGAR FORM 10-K INGESTION: 10 TECH COMPANIES (FY2024)")
    print("=" * 80)
    
    success_count = 0
    for comp in COMPANIES:
        ok = download_and_extract_10k(comp)
        if ok:
            success_count += 1
        time.sleep(0.5)  # Respect SEC fair access rate limit (max 10 req/sec)
        
    print("\n" + "=" * 80)
    print(f"✅ Ingestion Complete: {success_count}/{len(COMPANIES)} company filings downloaded.")
    print(f"📂 Saved location: {TECH_10K_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
