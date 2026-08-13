"""
Ingestion Pipeline for 3GPP RAG Chatbot.
Loads PDFs → Chunks text → Creates embeddings → Stores in FAISS vector database.

Usage:
    python ingest.py                    # Ingest all PDFs from data/specs/
    python ingest.py --path /some/dir   # Ingest from custom directory
"""
import sys
import json
import time
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR, VECTORSTORE_DIR, EMBEDDING_MODEL, 
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_DIMENSION
)
from utils.pdf_loader import load_all_pdfs
from utils.chunker import chunk_documents, get_chunk_stats


def load_embedding_model(model_name: str = EMBEDDING_MODEL):
    """Load the sentence-transformer embedding model."""
    print(f"\n📦 Loading embedding model: {model_name}")
    print("   (First run will download ~90MB model — one-time only)")
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    print(f"   ✓ Model loaded (dimension: {model.get_sentence_embedding_dimension()})")
    return model


def create_faiss_index(dimension: int = EMBEDDING_DIMENSION):
    """Create a new FAISS index."""
    import faiss
    # Using L2 (Euclidean) distance — lower = more similar
    index = faiss.IndexFlatL2(dimension)
    return index


def run_ingestion(data_dir: str = None, force: bool = False):
    """
    Run the full ingestion pipeline.
    
    Steps:
    1. Load all PDFs from data directory
    2. Chunk text with overlap and metadata
    3. Generate embeddings using sentence-transformers
    4. Store embeddings in FAISS index
    5. Save chunk metadata alongside
    """
    data_path = Path(data_dir) if data_dir else DATA_DIR
    
    print("=" * 60)
    print("🚀 3GPP RAG Chatbot — Ingestion Pipeline")
    print("=" * 60)
    
    # Check if vectorstore already exists
    index_path = VECTORSTORE_DIR / "index.faiss"
    meta_path = VECTORSTORE_DIR / "chunks_metadata.pkl"
    
    if index_path.exists() and not force:
        print(f"\n⚠️  Vector store already exists at {VECTORSTORE_DIR}")
        print("   Use --force to rebuild.")
        response = input("   Rebuild? (y/N): ").strip().lower()
        if response != 'y':
            print("   Aborted.")
            return
    
    # Step 1: Load PDFs
    print(f"\n📄 Step 1: Loading PDFs from {data_path}")
    print("-" * 40)
    
    if not data_path.exists():
        print(f"   ❌ Directory not found: {data_path}")
        print(f"   Please add 3GPP PDF files to: {data_path}")
        return
    
    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        print(f"   ❌ No PDF files found in {data_path}")
        print(f"   Please download 3GPP specs and place them in: {data_path}")
        print(f"\n   Run: python download_specs.py")
        return
    
    documents = load_all_pdfs(str(data_path))
    
    # Step 2: Chunk documents
    print(f"\n✂️  Step 2: Chunking documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print("-" * 40)
    
    chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    stats = get_chunk_stats(chunks)
    
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Avg chunk size: {stats['avg_chunk_size']} chars")
    print(f"   Specs covered: {', '.join(stats['specs_covered'])}")
    print(f"   Chunks with section info: {stats['chunks_with_sections']}")
    
    # Step 3: Generate embeddings
    print(f"\n🔢 Step 3: Generating embeddings")
    print("-" * 40)
    
    model = load_embedding_model()
    
    # Extract texts for embedding
    texts = [chunk["text"] for chunk in chunks]
    
    print(f"   Embedding {len(texts)} chunks...")
    start_time = time.time()
    
    # Batch encode for efficiency
    embeddings = model.encode(
        texts, 
        show_progress_bar=True, 
        batch_size=64,
        normalize_embeddings=False,  # FAISS L2 works without normalization
    )
    
    elapsed = time.time() - start_time
    print(f"   ✓ Done in {elapsed:.1f}s ({len(texts)/elapsed:.0f} chunks/sec)")
    
    # Step 4: Create and populate FAISS index
    print(f"\n💾 Step 4: Building FAISS index")
    print("-" * 40)
    
    import faiss
    import numpy as np
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to index
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    
    print(f"   Index size: {index.ntotal} vectors")
    print(f"   Dimension: {dimension}")
    
    # Step 5: Save everything
    print(f"\n💾 Step 5: Saving to {VECTORSTORE_DIR}")
    print("-" * 40)
    
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, str(index_path))
    print(f"   ✓ FAISS index: {index_path}")
    
    # Save chunk metadata (everything except embeddings)
    metadata = []
    for chunk in chunks:
        metadata.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"],
        })
    
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"   ✓ Metadata: {meta_path}")
    
    # Save stats for reference
    stats_path = VECTORSTORE_DIR / "ingestion_stats.json"
    ingestion_info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_pdfs": len(pdf_files),
        "total_pages": len(documents),
        "total_chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": dimension,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "stats": stats,
    }
    with open(stats_path, 'w') as f:
        json.dump(ingestion_info, f, indent=2, default=str)
    print(f"   ✓ Stats: {stats_path}")
    
    # Done!
    print("\n" + "=" * 60)
    print("✅ Ingestion complete!")
    print(f"   {len(pdf_files)} PDFs → {len(documents)} pages → {len(chunks)} chunks")
    print(f"   Vector store ready at: {VECTORSTORE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest 3GPP PDFs into vector store")
    parser.add_argument("--path", type=str, help="Path to directory containing PDFs")
    parser.add_argument("--force", action="store_true", help="Force rebuild vector store")
    args = parser.parse_args()
    
    run_ingestion(data_dir=args.path, force=args.force)
