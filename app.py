"""
3GPP Standards Intelligence — Evidence-Grounded Telecom AI Assistant
Enterprise-grade RAG interface with full retrieval transparency.
"""
import streamlit as st
import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import VECTORSTORE_DIR, LLM_MODEL, EMBEDDING_MODEL, TOP_K, SUPPORTED_SPECS
from rag_chain import RAGChain

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="3GPP Standards Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load Stats ───────────────────────────────────────────────────────────────
@st.cache_data
def load_stats():
    stats_path = VECTORSTORE_DIR / "ingestion_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return {}

STATS = load_stats()

# ─── Comprehensive CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ══════════════════════════════════════════════════════════════
   GLOBAL
   ══════════════════════════════════════════════════════════════ */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #111118;
    --bg-tertiary: #16161f;
    --bg-elevated: #1c1c27;
    --border-subtle: rgba(255,255,255,0.06);
    --border-medium: rgba(255,255,255,0.1);
    --text-primary: #f0f0f5;
    --text-secondary: #a0a0b0;
    --text-muted: #6b6b7b;
    --accent: #6366f1;
    --accent-soft: rgba(99,102,241,0.12);
    --accent-border: rgba(99,102,241,0.3);
    --green: #10b981;
    --green-soft: rgba(16,185,129,0.1);
    --green-border: rgba(16,185,129,0.3);
    --yellow: #f59e0b;
    --yellow-soft: rgba(245,158,11,0.1);
    --yellow-border: rgba(245,158,11,0.3);
    --red: #ef4444;
    --red-soft: rgba(239,68,68,0.1);
    --red-border: rgba(239,68,68,0.3);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
}

.stApp {
    background: var(--bg-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

#MainMenu, footer, header { visibility: hidden; }

.main .block-container {
    max-width: 920px;
    padding: 1.5rem 1.5rem 5rem 1.5rem;
}

/* ══════════════════════════════════════════════════════════════
   SIDEBAR — using native Streamlit theme, no overrides
   ══════════════════════════════════════════════════════════════ */

/* Sidebar brand */
.sb-brand {
    padding: 0.5rem 0 0.8rem 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 1rem;
}
.sb-brand-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.3px;
}
.sb-brand-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 2px;
    letter-spacing: 0.3px;
}

/* Sidebar sections */
.sb-section {
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--border-subtle);
}
.sb-section:last-child { border-bottom: none; }
.sb-label {
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.6rem;
}
.sb-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.3rem 0;
    font-size: 0.83rem;
    color: var(--text-secondary);
}
.sb-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.sb-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.25rem 0;
    font-size: 0.83rem;
}
.sb-stat-key { color: var(--text-muted); }
.sb-stat-val { color: var(--text-primary); font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }

/* Sidebar status */
.sb-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.6rem 0.8rem;
    background: var(--green-soft);
    border: 1px solid var(--green-border);
    border-radius: var(--radius-sm);
    font-size: 0.78rem;
    color: var(--green);
    font-weight: 500;
    margin-top: 0.5rem;
}
.sb-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ══════════════════════════════════════════════════════════════
   CHAT MESSAGES
   ══════════════════════════════════════════════════════════════ */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 1.2rem 0 !important;
    margin: 0 !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: var(--text-primary);
    font-size: 0.92rem;
    line-height: 1.75;
}

[data-testid="stChatMessage"] code {
    background: var(--bg-tertiary);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}

[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    color: var(--text-primary) !important;
}

/* ══════════════════════════════════════════════════════════════
   CHAT INPUT
   ══════════════════════════════════════════════════════════════ */
[data-testid="stChatInput"] > div {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: var(--radius-md) !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--accent-border) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}

[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
}

/* ══════════════════════════════════════════════════════════════
   BUTTONS
   ══════════════════════════════════════════════════════════════ */
.stButton > button {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-secondary) !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.15s ease !important;
    text-align: left !important;
}

.stButton > button:hover {
    background: var(--bg-elevated) !important;
    border-color: var(--accent-border) !important;
    color: var(--text-primary) !important;
}

/* ══════════════════════════════════════════════════════════════
   HERO / LANDING
   ══════════════════════════════════════════════════════════════ */
.hero {
    text-align: center;
    padding: 4rem 0 2rem 0;
}
.hero-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.9;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.5px;
    margin-bottom: 0.5rem;
}
.hero-desc {
    font-size: 0.95rem;
    color: var(--text-muted);
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ══════════════════════════════════════════════════════════════
   EXAMPLE CARDS
   ══════════════════════════════════════════════════════════════ */
.examples-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.7rem;
    margin: 2rem 0;
}

/* ══════════════════════════════════════════════════════════════
   GROUNDING BADGE
   ══════════════════════════════════════════════════════════════ */
.grounding-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.2px;
    margin-top: 0.6rem;
}
.gb-grounded {
    background: var(--green-soft);
    border: 1px solid var(--green-border);
    color: var(--green);
}
.gb-limited {
    background: var(--yellow-soft);
    border: 1px solid var(--yellow-border);
    color: var(--yellow);
}
.gb-insufficient {
    background: var(--red-soft);
    border: 1px solid var(--red-border);
    color: var(--red);
}

/* ══════════════════════════════════════════════════════════════
   SOURCES & EVIDENCE
   ══════════════════════════════════════════════════════════════ */
.evidence-container {
    margin-top: 1rem;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    overflow: hidden;
}
.evidence-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.7rem 1rem;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.evidence-card {
    padding: 0.8rem 1rem;
    border-bottom: 1px solid var(--border-subtle);
    transition: background 0.15s;
}
.evidence-card:last-child { border-bottom: none; }
.evidence-card:hover { background: var(--bg-tertiary); }
.ev-spec {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
}
.ev-section {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 2px;
}
.ev-page {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 2px;
}

/* ══════════════════════════════════════════════════════════════
   RETRIEVAL PIPELINE
   ══════════════════════════════════════════════════════════════ */
.pipeline-container {
    margin-top: 0.8rem;
    padding: 1rem;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
}
.pipeline-title {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.8rem;
}
.pipeline-steps {
    display: flex;
    align-items: center;
    gap: 0;
    flex-wrap: wrap;
}
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    font-size: 0.7rem;
    color: var(--text-secondary);
    font-weight: 500;
}
.pipeline-step.active {
    border-color: var(--green-border);
    color: var(--green);
    background: var(--green-soft);
}
.pipeline-arrow {
    color: var(--text-muted);
    font-size: 0.7rem;
    padding: 0 4px;
}

/* ══════════════════════════════════════════════════════════════
   NO-ANSWER STATE
   ══════════════════════════════════════════════════════════════ */
.no-evidence {
    padding: 1.2rem;
    background: var(--red-soft);
    border: 1px solid var(--red-border);
    border-radius: var(--radius-md);
    margin-top: 0.5rem;
}
.no-evidence-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--red);
    margin-bottom: 0.4rem;
}
.no-evidence-text {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.6;
}
.no-evidence-suggestions {
    margin-top: 0.6rem;
    padding-top: 0.6rem;
    border-top: 1px solid var(--red-border);
}
.no-evidence-suggestions p {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin: 0.2rem 0;
}

/* ══════════════════════════════════════════════════════════════
   CONTROLS BAR
   ══════════════════════════════════════════════════════════════ */
.controls-bar {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.5rem 0;
    margin-bottom: 0.5rem;
    font-size: 0.72rem;
    color: var(--text-muted);
}
.control-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 5px;
    font-size: 0.7rem;
    color: var(--text-muted);
}
.control-chip-val {
    color: var(--text-secondary);
    font-weight: 500;
}

/* ══════════════════════════════════════════════════════════════
   EXPANDER
   ══════════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-secondary) !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.82rem !important;
    color: var(--text-secondary) !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--text-primary) !important;
}

/* ══════════════════════════════════════════════════════════════
   SCROLLBAR
   ══════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }

/* ══════════════════════════════════════════════════════════════
   RESPONSIVE
   ══════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    .hero-title { font-size: 1.5rem; }
    .examples-grid { grid-template-columns: 1fr; }
    .pipeline-steps { flex-wrap: wrap; gap: 4px; }
    .controls-bar { flex-wrap: wrap; }
}
</style>
""", unsafe_allow_html=True)


# ─── Load RAG Chain ───────────────────────────────────────────────────────────
@st.cache_resource
def load_rag_chain():
    chain = RAGChain()
    chain.load()
    return chain


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("## ⚡ 3GPP Intelligence")
    st.caption("Evidence-Grounded Telecom AI")
    st.markdown("---")

    # New Chat
    if st.button("➕  New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.markdown("---")

    # ── System ──
    st.markdown("**SYSTEM**")
    st.markdown("""
    | | |
    |---|---|
    | Model | Llama 3.3 70B |
    | Provider | Groq |
    | Retrieval | Semantic (FAISS) |
    | Embeddings | MiniLM-L6 |
    """)

    st.markdown("---")

    # ── Knowledge Base ──
    st.markdown("**KNOWLEDGE BASE**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", STATS.get('total_pdfs', 7))
    with col2:
        st.metric("Chunks", f"{STATS.get('total_chunks', 0):,}")
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Dimensions", STATS.get('embedding_dimension', 384))
    with col4:
        st.metric("Updated", STATS.get('timestamp', '—')[:10])

    st.markdown("---")

    # ── Indexed Specifications ──
    st.markdown("**INDEXED SPECIFICATIONS**")
    for spec, title in SUPPORTED_SPECS.items():
        short_title = title.split(';')[-1].strip() if ';' in title else title
        if len(short_title) > 30:
            short_title = short_title[:28] + "…"
        st.markdown(f"📄 `{spec}` — {short_title}")

    st.markdown("---")

    # ── Hallucination Control ──
    st.markdown("**HALLUCINATION CONTROL**")
    st.markdown("✅ Context-only generation")
    st.markdown("✅ Confidence scoring")
    st.markdown("✅ Source citations")
    st.markdown("✅ Retrieval validation")
    st.markdown("✅ Graceful refusal")

    st.markdown("---")

    # ── Status ──
    st.success("● RAG System Online")


# ─── Main Area ────────────────────────────────────────────────────────────────

# Check vector store
if not (VECTORSTORE_DIR / "index.faiss").exists():
    st.error("Vector store not found. Run `python ingest.py` to build the knowledge base.")
    st.stop()

try:
    rag_chain = load_rag_chain()
except Exception as e:
    st.error(f"System initialization failed: {e}")
    st.stop()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ─── Landing Page ─────────────────────────────────────────────────────────────
if not st.session_state.messages and not st.session_state.pending_query:
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">⚡</div>
        <div class="hero-title">Ask the 3GPP Standards</div>
        <div class="hero-desc">
            Get evidence-backed answers from telecom specifications with traceable citations.
            Every response is grounded in retrieved 3GPP documentation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Controls bar
    st.markdown(f"""
    <div class="controls-bar" style="justify-content: center;">
        <div class="control-chip">Retrieval: <span class="control-chip-val">Semantic</span></div>
        <div class="control-chip">Knowledge: <span class="control-chip-val">3GPP Standards</span></div>
        <div class="control-chip">Chunks: <span class="control-chip-val">{STATS.get('total_chunks', '—'):,}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Example prompts
    examples = [
        ("🏗️", "Explain the 5G registration procedure"),
        ("📡", "What happens during PDU session establishment?"),
        ("🔐", "Describe the 5G-AKA authentication mechanism"),
        ("🌐", "What is Service Based Architecture in 5G Core?"),
        ("📋", "Which network functions are defined in TS 23.501?"),
        ("🔀", "How does network slicing work according to 3GPP?"),
    ]

    st.markdown('<div class="examples-grid">', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (icon, text) in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"{icon}  {text}", key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_query = text
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # ── Compact header in chat mode ──
    st.markdown(f"""
    <div class="controls-bar">
        <span style="font-weight: 600; color: var(--text-primary);">⚡ 3GPP Standards Intelligence</span>
        <div class="control-chip">Model: <span class="control-chip-val">Llama 3.3 70B</span></div>
        <div class="control-chip">Retrieval: <span class="control-chip-val">Semantic</span></div>
        <div class="control-chip">KB: <span class="control-chip-val">{STATS.get('total_chunks', '—'):,} chunks</span></div>
    </div>
    """, unsafe_allow_html=True)


# ─── Chat History Display ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "⚡"
    with st.chat_message(msg["role"], avatar=avatar):
        # Clean inline sources from display
        display_content = msg["content"]
        if "📚 **Sources:**" in display_content:
            display_content = display_content.split("📚 **Sources:**")[0].strip()
        if "📚 Sources:" in display_content:
            display_content = display_content.split("📚 Sources:")[0].strip()
        st.markdown(display_content)

        if msg["role"] == "assistant" and msg.get("decision") != "greeting":
            confidence = msg.get("confidence", 0)
            decision = msg.get("decision", "")
            sources = msg.get("sources", [])

            # Sources & Evidence
            if sources:
                with st.expander("📄 Sources & Evidence", expanded=True):
                    import pandas as pd
                    source_data = []
                    for s in sources:
                        source_data.append({
                            "Specification": s["spec"],
                            "Section": s.get("section", "—"),
                            "Page": s["page"],
                        })
                    df = pd.DataFrame(source_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)


# ─── Chat Input & Processing ─────────────────────────────────────────────────
prompt = st.chat_input("Ask about 3GPP standards, procedures, specifications...")

if st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None

if prompt:
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant", avatar="⚡"):
        with st.spinner("Retrieving relevant 3GPP evidence..."):
            result = rag_chain.query(prompt)

        response = result.get("response", result.get("final_response", ""))
        confidence = result.get("confidence", 0)
        decision = result.get("decision", "")
        sources = result.get("sources", [])

        # Clean response: remove inline source citations added by backend
        # (we show them in the table instead)
        if "📚 **Sources:**" in response:
            response = response.split("📚 **Sources:**")[0].strip()
        if "📚 Sources:" in response:
            response = response.split("📚 Sources:")[0].strip()

        # Display response
        st.markdown(response)

        # Post-response elements (only for non-greetings)
        if decision != "greeting":

            # Sources & Evidence panel
            if sources:
                with st.expander("📄 Sources & Evidence", expanded=True):
                    import pandas as pd
                    source_data = []
                    for s in sources:
                        source_data.append({
                            "Specification": s["spec"],
                            "Section": s.get("section", "—"),
                            "Page": s["page"],
                        })
                    df = pd.DataFrame(source_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)

            # No-answer state for refusals
            if decision == "refuse":
                st.markdown("""
                <div class="no-evidence">
                    <div class="no-evidence-title">Insufficient Evidence in Knowledge Base</div>
                    <div class="no-evidence-text">
                        The indexed 3GPP standards do not contain enough supporting information to answer this reliably.
                    </div>
                    <div class="no-evidence-suggestions">
                        <p>💡 Try:</p>
                        <p>• Asking a more specific question</p>
                        <p>• Specifying a 3GPP specification (e.g., TS 23.501)</p>
                        <p>• Mentioning a specific procedure or section</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Save to session
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "confidence": confidence,
        "decision": decision,
        "sources": sources,
    })
