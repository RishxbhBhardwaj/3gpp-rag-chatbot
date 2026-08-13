"""
Configuration settings for 3GPP RAG Chatbot.
All free-tier compatible settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "specs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# ─── Groq API (Free Tier) ────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.3-70b-versatile"  # Best free model on Groq
LLM_TEMPERATURE = 0.1  # Low temperature for factual answers
LLM_MAX_TOKENS = 1024

# ─── Embedding Model (Local, Free) ───────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~90MB, runs on CPU
EMBEDDING_DIMENSION = 384

# ─── Chunking Settings ────────────────────────────────────────────────────────
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks for context continuity

# ─── Retrieval Settings ───────────────────────────────────────────────────────
TOP_K = 5  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.35  # Minimum similarity score (0-1). Below this = refuse to answer.

# ─── Hallucination Control ────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.4  # If average retrieval score < this, flag as low confidence
REFUSAL_THRESHOLD = 0.25  # If best score < this, refuse entirely
MAX_CONTEXT_LENGTH = 4000  # Max characters of context to pass to LLM

# ─── 3GPP Spec Metadata ──────────────────────────────────────────────────────
SUPPORTED_SPECS = {
    "TS 23.501": "System architecture for the 5G System (5GS)",
    "TS 23.502": "Procedures for the 5G System (5GS)",
    "TS 23.503": "Policy and charging control framework for the 5G System",
    "TS 38.300": "NR; NR and NG-RAN Overall Description",
    "TS 38.331": "NR; Radio Resource Control (RRC) protocol specification",
    "TS 24.501": "Non-Access-Stratum (NAS) protocol for 5G System",
    "TS 29.500": "5G System; Technical Realization of Service Based Architecture",
    "TS 33.501": "Security architecture and procedures for 5G System",
}
