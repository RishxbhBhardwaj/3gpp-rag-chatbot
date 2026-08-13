"""
FastAPI Server for 3GPP Standards Intelligence.
Serves the RAG chatbot API + static frontend.

Run: uvicorn server:app --reload --port 8000
"""
import sys
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from config import VECTORSTORE_DIR, LLM_MODEL, EMBEDDING_MODEL, SUPPORTED_SPECS
from rag_chain import RAGChain

# ─── Initialize App ──────────────────────────────────────────────────────────
app = FastAPI(title="3GPP Standards Intelligence", version="1.0.0")

# ─── Lazy-load RAG Chain (saves memory on startup) ────────────────────────────
rag_chain = None

def get_rag_chain():
    global rag_chain
    if rag_chain is None:
        # Auto-run ingestion if vector store doesn't exist
        index_path = VECTORSTORE_DIR / "index.faiss"
        if not index_path.exists():
            import subprocess
            print("⚡ Vector store not found. Running ingestion...")
            subprocess.run([sys.executable, "download_specs.py"], check=True)
            subprocess.run([sys.executable, "ingest.py", "--force"], check=True)
        rag_chain = RAGChain()
        rag_chain.load()
    return rag_chain

@app.on_event("startup")
async def startup():
    """Pre-load RAG chain on startup."""
    try:
        get_rag_chain()
    except Exception as e:
        print(f"Warning: Failed to pre-load RAG chain: {e}")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

# ─── Request/Response Models ──────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    spec: str
    page: object
    section: str

class QueryResponse(BaseModel):
    response: str
    confidence: float
    decision: str
    sources: list
    indicator: str

# ─── API Routes ───────────────────────────────────────────────────────────────
@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Send a question to the RAG chain."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    chain = get_rag_chain()
    result = chain.query(req.question)
    
    # Get the response text (backend uses different keys)
    response_text = result.get("response", result.get("final_response", ""))
    
    # Strip inline source bullets from response (we show them separately)
    if "📚 **Sources:**" in response_text:
        response_text = response_text.split("📚 **Sources:**")[0].strip()
    if "📚 Sources:" in response_text:
        response_text = response_text.split("📚 Sources:")[0].strip()
    
    return QueryResponse(
        response=response_text,
        confidence=result.get("confidence", 0),
        decision=result.get("decision", ""),
        sources=result.get("sources", []),
        indicator=result.get("indicator", ""),
    )

@app.get("/api/status")
async def status():
    """System status and stats."""
    stats_path = VECTORSTORE_DIR / "ingestion_stats.json"
    stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    
    return {
        "status": "online",
        "model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "total_specs": stats.get("total_pdfs", 0),
        "total_chunks": stats.get("total_chunks", 0),
        "embedding_dimension": stats.get("embedding_dimension", 384),
        "last_updated": stats.get("timestamp", ""),
        "specs": SUPPORTED_SPECS,
    }

# ─── Static Files & Frontend ─────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")
