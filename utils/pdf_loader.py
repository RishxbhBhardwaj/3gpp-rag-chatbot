"""
PDF Loader for 3GPP Specification Documents.
Extracts text from PDFs while preserving structure metadata.
Uses pypdf (pure Python, works on all Python versions).
"""
import re
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader


def extract_spec_number(filename: str) -> str:
    """Extract 3GPP spec number from filename.
    
    Examples:
        ts_123501v170600p.pdf -> TS 23.501
        ts_138300v170300p.pdf -> TS 38.300
    """
    # Try ETSI filename pattern: ts_1XXYYYV...
    # Where XX is series and YYY is number
    match = re.search(r'ts[_]?1(\d{2})(\d{3})', filename, re.IGNORECASE)
    if match:
        series = match.group(1)
        number = match.group(2)
        return f"TS {series}.{number}"
    
    # Try direct pattern: TS XX.YYY
    match = re.search(r'TS[_ ]?(\d{2})\.?(\d{3})', filename, re.IGNORECASE)
    if match:
        return f"TS {match.group(1)}.{match.group(2)}"
    
    # Fallback: use filename
    return Path(filename).stem


def extract_section_title(text: str) -> str:
    """Try to extract section title from chunk text."""
    lines = text.strip().split('\n')
    for line in lines[:3]:
        line = line.strip()
        if re.match(r'^\d+(\.\d+)*\s+[A-Z]', line):
            return line
    return ""


def load_pdf(pdf_path: str) -> List[Dict]:
    """
    Load a 3GPP PDF and extract text with page-level metadata.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        List of dicts with 'text', 'page', 'source', 'spec_number' keys.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    spec_number = extract_spec_number(pdf_path.name)
    documents = []
    
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF {pdf_path}: {e}")
    
    # Skip first N pages (cover, legal, ToC) — 3GPP specs typically have
    # 8-15 pages of front matter before actual technical content begins
    skip_pages = _detect_content_start(reader)
    
    for page_num, page in enumerate(reader.pages):
        # Skip front matter
        if page_num < skip_pages:
            continue
        
        try:
            text = page.extract_text()
        except Exception:
            continue
        
        # Skip empty pages or pages with only headers/footers
        if not text or len(text.strip()) < 50:
            continue
        
        # Clean up common PDF artifacts
        text = _clean_text(text)
        
        # Skip pages that are mostly table of contents or list of figures
        if _is_toc_page(text):
            continue
        
        documents.append({
            "text": text,
            "page": page_num + 1,
            "source": pdf_path.name,
            "spec_number": spec_number,
        })
    
    return documents


def _clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove page headers/footers (common 3GPP patterns)
    text = re.sub(r'3GPP TS \d+\.\d+ version .+Release \d+', '', text)
    text = re.sub(r'ETSI TS \d+ \d+ V\d+\.\d+\.\d+.*', '', text)
    text = re.sub(r'\d+\s*ETSI', '', text)
    # Remove excessive spaces
    text = re.sub(r' {3,}', '  ', text)
    return text.strip()


def _detect_content_start(reader) -> int:
    """
    Detect where actual technical content starts in a 3GPP spec.
    Skips cover pages, legal notices, table of contents.
    Returns the page index to start from.
    """
    # 3GPP specs typically have content starting after "1 Scope" section
    for page_num in range(min(20, len(reader.pages))):
        try:
            text = reader.pages[page_num].extract_text()
            if not text:
                continue
            # Look for "1 Scope" or "1  Scope" — marks start of real content
            if re.search(r'^1\s+Scope', text, re.MULTILINE):
                return page_num
        except Exception:
            continue
    # Default: skip first 5 pages (conservative)
    return min(5, len(reader.pages) - 1)


def _is_toc_page(text: str) -> bool:
    """Check if a page is mostly table of contents."""
    lines = text.strip().split('\n')
    if not lines:
        return True
    
    # Count lines that look like TOC entries (ending with page numbers or dots)
    toc_patterns = 0
    for line in lines:
        line = line.strip()
        # Lines like "5.2.1 Network Functions..........23" or "5.2.1 Network Functions 23"
        if re.match(r'^[\d.]+\s+.+?\.{3,}\s*\d+', line):
            toc_patterns += 1
        elif re.match(r'^[\d.]+\s+.+\s+\d+$', line) and len(line) < 100:
            toc_patterns += 1
        # Lines like "Table 5.1: blah........23"
        elif re.match(r'^(Table|Figure)\s+[\d.]+', line):
            toc_patterns += 1
    
    # If more than 40% of lines look like TOC, skip this page
    return toc_patterns > len(lines) * 0.4


def load_all_pdfs(directory: str) -> List[Dict]:
    """
    Load all PDFs from a directory.
    
    Args:
        directory: Path to directory containing PDF files.
        
    Returns:
        List of all extracted documents with metadata.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {directory}")
    
    all_documents = []
    for pdf_file in sorted(pdf_files):
        print(f"  Loading: {pdf_file.name}")
        try:
            docs = load_pdf(str(pdf_file))
            all_documents.extend(docs)
            print(f"    → Extracted {len(docs)} pages")
        except Exception as e:
            print(f"    ⚠ Error: {e}")
    
    print(f"\n  Total: {len(all_documents)} pages from {len(pdf_files)} PDFs")
    return all_documents


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        docs = load_pdf(sys.argv[1])
        print(f"Extracted {len(docs)} pages")
        if docs:
            print(f"First page preview:\n{docs[0]['text'][:500]}")
    else:
        print("Usage: python pdf_loader.py <path_to_pdf>")
