import os
import time
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path


def fetch_arxiv_papers(
    query: str = "cat:cs.CL AND (ti:RAG OR ti:Retrieval OR ti:Reasoning)",
    max_results: int = 10,
    output_dir: str = "data/raw_papers",
) -> list[dict]:
    """
    Fetches real ML research papers from arXiv API and downloads their PDFs.
    
    Args:
        query: Search query for arXiv API (e.g. category cs.CL for NLP).
        max_results: Number of papers to fetch.
        output_dir: Folder path where PDFs and metadata will be saved.
        
    Returns:
        List of paper metadata dictionaries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Querying arXiv API for: '{query}' (Max results: {max_results})")

    # Base URL for arXiv API
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "RAG-Research-Assistant/1.0 (Educational research project)"}
    )

    with urllib.request.urlopen(req) as response:
        xml_data = response.read()

    # Parse XML response
    root = ET.fromstring(xml_data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    entries = root.findall("atom:entry", ns)

    if not entries:
        print("[-] No papers found for the query.")
        return []

    print(f"[+] Found {len(entries)} papers. Starting download...\n")

    for i, entry in enumerate(entries, 1):
        # Extract metadata
        paper_id_url = entry.find("atom:id", ns).text
        paper_id = paper_id_url.split("/abs/")[-1]
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        published = entry.find("atom:published", ns).text[:10]
        
        authors = [
            author.find("atom:name", ns).text 
            for author in entry.findall("atom:author", ns)
        ]

        # Clean filename for the PDF
        clean_id = paper_id.replace("/", "_")
        pdf_filename = f"{clean_id}.pdf"
        pdf_path = output_path / pdf_filename
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

        print(f"[{i}/{len(entries)}] {title}")
        print(f"    - ID: {paper_id} | Published: {published}")
        print(f"    - Authors: {', '.join(authors[:3])}{' et al.' if len(authors) > 3 else ''}")

        # Download PDF if not already present
        if not pdf_path.exists():
            print(f"    - Downloading PDF from: {pdf_url} ...")
            try:
                pdf_req = urllib.request.Request(
                    pdf_url,
                    headers={"User-Agent": "RAG-Research-Assistant/1.0"}
                )
                with urllib.request.urlopen(pdf_req) as resp, open(pdf_path, "wb") as f:
                    f.write(resp.read())
                print(f"    - Saved to: {pdf_path}")
            except Exception as e:
                print(f"    - [Warning] Failed to download PDF: {e}")
            
            # Respect arXiv API rate limit policy (3s delay between downloads)
            time.sleep(3.0)
        else:
            print(f"    - PDF already exists locally at: {pdf_path}")

        paper_meta = {
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "published": published,
            "summary": summary,
            "pdf_url": pdf_url,
            "pdf_path": str(pdf_path),
        }
        papers.append(paper_meta)
        print()

    # Save metadata JSON catalog
    meta_file = output_path / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"[+] Metadata catalog saved to: {meta_file}")
    return papers


if __name__ == "__main__":
    # Fetch 5 sample papers on RAG / LLM Reasoning from arXiv
    fetch_arxiv_papers(
        query="cat:cs.CL AND (ti:RAG OR ti:Retrieval OR ti:Reasoning)",
        max_results=5,
        output_dir="data/raw_papers"
    )
