"""
Text Chunker for 3GPP documents.
Splits documents into overlapping chunks while preserving metadata.
"""
import re
from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: List[Dict], 
                    chunk_size: int = CHUNK_SIZE, 
                    chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Split documents into overlapping chunks with metadata.
    
    Uses a recursive splitting strategy:
    1. Split on section headers (double newline)
    2. Split on paragraphs (single newline)
    3. Split on sentences (period + space)
    4. Split on words (space) as last resort
    
    Args:
        documents: List of dicts from pdf_loader.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        List of chunk dicts with text, metadata, and chunk_id.
    """
    separators = ["\n\n", "\n", ". ", " "]
    all_chunks = []
    chunk_id = 0
    
    for doc in documents:
        text = doc["text"]
        if not text.strip():
            continue
        
        # Split text into chunks
        text_chunks = _recursive_split(text, separators, chunk_size, chunk_overlap)
        
        for chunk_text in text_chunks:
            if len(chunk_text.strip()) < 30:  # Skip tiny chunks
                continue
            
            # Extract section info if available
            section = _extract_section(chunk_text)
            
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text.strip(),
                "metadata": {
                    "source": doc["source"],
                    "spec_number": doc["spec_number"],
                    "page": doc["page"],
                    "section": section,
                    "chunk_size": len(chunk_text.strip()),
                }
            })
            chunk_id += 1
    
    return all_chunks


def _recursive_split(text: str, separators: List[str], 
                     chunk_size: int, chunk_overlap: int) -> List[str]:
    """Recursively split text using different separators."""
    if len(text) <= chunk_size:
        return [text]
    
    # Find the best separator that actually exists in the text
    separator = separators[-1]  # default to space
    for sep in separators:
        if sep in text:
            separator = sep
            break
    
    # Split by chosen separator
    parts = text.split(separator)
    
    chunks = []
    current_chunk = ""
    
    for part in parts:
        # If adding this part would exceed chunk_size
        if len(current_chunk) + len(part) + len(separator) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                # Keep overlap from end of current chunk
                overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
                current_chunk = overlap_text + separator + part
            else:
                # Single part is too long, split further with next separator
                if len(separators) > 1:
                    sub_chunks = _recursive_split(
                        part, separators[1:], chunk_size, chunk_overlap
                    )
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    # Last resort: hard split
                    for i in range(0, len(part), chunk_size - chunk_overlap):
                        chunks.append(part[i:i + chunk_size])
                    current_chunk = ""
        else:
            current_chunk = current_chunk + separator + part if current_chunk else part
    
    if current_chunk.strip():
        chunks.append(current_chunk)
    
    return chunks


def _extract_section(text: str) -> str:
    """Extract section number and title from chunk if present."""
    lines = text.strip().split('\n')
    for line in lines[:5]:
        line = line.strip()
        # Match: "5.2.1  Network Functions" or "5.2.1 Network Functions"
        match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)', line)
        if match:
            section_num = match.group(1)
            section_title = match.group(2).strip()
            # Validate it looks like a title (starts with uppercase, not too long)
            if section_title[0].isupper() and len(section_title) < 100:
                return f"{section_num} {section_title}"
    return ""


def get_chunk_stats(chunks: List[Dict]) -> Dict:
    """Get statistics about the chunked documents."""
    if not chunks:
        return {"total_chunks": 0}
    
    sizes = [c["metadata"]["chunk_size"] for c in chunks]
    specs = set(c["metadata"]["spec_number"] for c in chunks)
    sections_found = sum(1 for c in chunks if c["metadata"]["section"])
    
    return {
        "total_chunks": len(chunks),
        "avg_chunk_size": sum(sizes) // len(sizes),
        "min_chunk_size": min(sizes),
        "max_chunk_size": max(sizes),
        "specs_covered": list(specs),
        "chunks_with_sections": sections_found,
    }


if __name__ == "__main__":
    # Quick test with sample text
    sample_docs = [{
        "text": "5.2.1  Network Functions\nThe 5G System architecture consists of the following network functions...\n\n5.2.2  Network Slicing\nNetwork slicing allows the operator to create multiple logical networks...",
        "page": 1,
        "source": "test.pdf",
        "spec_number": "TS 23.501",
    }]
    
    chunks = chunk_documents(sample_docs, chunk_size=100, chunk_overlap=20)
    print(f"Created {len(chunks)} chunks from sample")
    for chunk in chunks:
        print(f"  [{chunk['chunk_id']}] ({chunk['metadata']['chunk_size']} chars) {chunk['text'][:60]}...")
