# ⚡ 3GPP Standards Intelligence

**Evidence-Grounded Telecom AI Assistant** — A RAG-based chatbot that answers questions using 3GPP telecom specifications with near-zero hallucinations and traceable citations.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-yellow)
![Groq](https://img.shields.io/badge/LLM-Llama_3.3_70B-purple)

---

## 🎯 Project Overview

This system answers questions about 3GPP telecom standards using **Retrieval-Augmented Generation (RAG)**, with a strong focus on achieving **minimal to near-zero hallucinations**.

Every response is:
- Grounded in retrieved 3GPP specification text
- Accompanied by source citations (specification, section, page)
- Validated through a confidence scoring pipeline
- Refused gracefully when evidence is insufficient

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Frontend (HTML/CSS/JS)                          │
│         Premium chat UI with voice support                  │
└─────────────────────────────┬───────────────────────────────┘
                              │ REST API
┌─────────────────────────────▼───────────────────────────────┐
│                  FastAPI Server                              │
│                                                             │
│  POST /api/query  →  RAG Chain                              │
│  GET  /api/status →  System Info                            │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                   RAG Pipeline                               │
│                                                             │
│  Query → Embed → FAISS Search → Hallucination Guard         │
│       → Context Assembly → LLM Generation → Citations       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛡️ Hallucination Control (Key Differentiator)

| Layer | Technique | Implementation |
|-------|-----------|----------------|
| 1 | Context-Only Generation | System prompt forbids prior knowledge |
| 2 | Confidence Scoring | Retrieval similarity scores evaluated before answering |
| 3 | Graceful Refusal | Below-threshold queries return "insufficient evidence" |
| 4 | Source Citations | Every answer cites spec number, section, and page |
| 5 | Retrieval Validation | Chunks filtered by similarity threshold |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Free Groq API key → [console.groq.com](https://console.groq.com) (no credit card)

### Setup & Run

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/3gpp-rag-chatbot.git
cd 3gpp-rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Download 3GPP specifications (free from ETSI)
python download_specs.py

# Build the vector store
python ingest.py

# Launch the application
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

---

## 📂 Project Structure

```
3gpp-rag-chatbot/
├── server.py               # FastAPI backend (API endpoints)
├── rag_chain.py            # RAG query pipeline
├── ingest.py               # Document ingestion pipeline
├── config.py               # Configuration
├── prompts.py              # System prompts & templates
├── download_specs.py       # 3GPP PDF downloader
├── utils/
│   ├── pdf_loader.py       # PDF text extraction
│   ├── chunker.py          # Recursive text chunking
│   └── hallucination.py    # Confidence scoring & refusal logic
├── static/
│   ├── index.html          # Frontend UI
│   ├── style.css           # Styling (dark/light themes)
│   └── app.js              # Chat logic & interactions
├── vectorstore/            # FAISS index (generated)
├── data/specs/             # 3GPP PDFs (downloaded)
├── requirements.txt
└── README.md
```

---

## 🔧 Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| LLM | Llama 3.3 70B (Groq) | Response generation |
| Embeddings | all-MiniLM-L6-v2 | Semantic text embeddings |
| Vector Store | FAISS | Fast similarity search |
| Backend | FastAPI | REST API server |
| Frontend | HTML/CSS/JS | Premium chat interface |
| PDF Parsing | PyPDF | Text extraction |
| Knowledge Source | 3GPP/ETSI Specs | Telecom standards |

---

## 📚 Indexed Specifications

| Specification | Topic |
|--------------|-------|
| TS 23.501 | 5G System Architecture |
| TS 23.502 | 5G System Procedures |
| TS 23.503 | Policy & Charging Control |
| TS 38.300 | NR & NG-RAN Overview |
| TS 33.501 | 5G Security Architecture |
| TS 29.500 | Service Based Architecture |
| TS 24.501 | 5G NAS Protocol |

---

## ✨ Features

- **Evidence-Grounded Answers** — Every response is sourced from retrieved 3GPP text
- **Source Citations Table** — Specification, section, and page for traceability
- **Graceful Refusal** — Refuses to answer when evidence is insufficient
- **Voice Input** — Speak your question via microphone (Chrome/Edge)
- **Read Aloud** — Text-to-speech for responses
- **Light/Dark Mode** — Toggle between themes
- **Responsive Design** — Works on desktop, tablet, and mobile
- **Fast Inference** — Sub-3s responses via Groq

---

## 💡 Example Queries

- "Explain the 5G registration procedure"
- "What is Service Based Architecture in 5G Core?"
- "Describe the 5G-AKA authentication mechanism"
- "How does network slicing work according to 3GPP?"
- "Which network functions are defined in TS 23.501?"

---

## 📈 System Stats

- **7** specifications indexed
- **3,004** text chunks in vector store
- **384** embedding dimensions
- **<3s** average response time

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

- [3GPP](https://www.3gpp.org/) — Telecom standards
- [ETSI](https://www.etsi.org/) — Free specification hosting
- [Groq](https://groq.com/) — Fast LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [Sentence Transformers](https://www.sbert.net/) — Embedding models
