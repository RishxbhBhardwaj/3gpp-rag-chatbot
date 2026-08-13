---
title: 3GPP Standards Intelligence
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: true
license: mit
short_description: Evidence-Grounded 3GPP Telecom AI Assistant
---

# ⚡ 3GPP Standards Intelligence

**Evidence-Grounded Telecom AI Assistant** — A RAG-based chatbot that answers questions using 3GPP telecom specifications with near-zero hallucinations and traceable citations.

## Features
- Retrieval-Augmented Generation using 7 official 3GPP specifications
- 5-layer hallucination control
- Evidence-backed answers with source citations
- Voice input/output
- Light/Dark mode

## Tech Stack
- **LLM:** Llama 3.3 70B (Groq)
- **Embeddings:** all-MiniLM-L6-v2
- **Vector Store:** FAISS
- **Backend:** FastAPI
- **Frontend:** HTML/CSS/JS

See [DOCS.md](DOCS.md) for full documentation.
