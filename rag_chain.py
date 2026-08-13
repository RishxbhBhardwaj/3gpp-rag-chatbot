"""
RAG Query Chain for 3GPP Chatbot.
Handles: Query → Embed → Retrieve → Evaluate → Generate → Validate response.
"""
import sys
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    VECTORSTORE_DIR, EMBEDDING_MODEL, TOP_K, MAX_CONTEXT_LENGTH
)
from prompts import SYSTEM_PROMPT, QA_PROMPT_TEMPLATE, REFUSAL_RESPONSE, GREETING_RESPONSE
from utils.hallucination import guard, check_and_respond


class RAGChain:
    """
    Retrieval-Augmented Generation chain for 3GPP specifications.
    
    Flow:
    1. Embed user query
    2. Search FAISS for top-K similar chunks
    3. Hallucination guard evaluates retrieval quality
    4. If quality sufficient: pass context to LLM
    5. If quality insufficient: refuse gracefully
    6. Format response with citations
    """
    
    def __init__(self):
        self.index = None
        self.metadata = None
        self.embedding_model = None
        self.groq_client = None
        self._loaded = False
    
    def load(self):
        """Load all components: FAISS index, metadata, embedding model, LLM client."""
        if self._loaded:
            return
        
        print("Loading RAG chain components...")
        
        # Load FAISS index
        import faiss
        index_path = VECTORSTORE_DIR / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                f"Run 'python ingest.py' first to build the vector store."
            )
        self.index = faiss.read_index(str(index_path))
        print(f"  ✓ FAISS index loaded ({self.index.ntotal} vectors)")
        
        # Load chunk metadata
        meta_path = VECTORSTORE_DIR / "chunks_metadata.pkl"
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"  ✓ Metadata loaded ({len(self.metadata)} chunks)")
        
        # Load embedding model
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"  ✓ Embedding model loaded ({EMBEDDING_MODEL})")
        
        # Initialize Groq client
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set. Create a .env file with:\n"
                "GROQ_API_KEY=your_key_here\n\n"
                "Get your free key at: https://console.groq.com/"
            )
        
        from groq import Groq
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        print(f"  ✓ Groq client initialized (model: {LLM_MODEL})")
        
        self._loaded = True
        print("  ✓ RAG chain ready!\n")
    
    def query(self, user_question: str) -> Dict:
        """
        Process a user question through the full RAG pipeline.
        
        Args:
            user_question: The user's question about 3GPP specs.
            
        Returns:
            Dict with: response, confidence, decision, sources, metadata.
        """
        if not self._loaded:
            self.load()
        
        # Handle greetings/casual messages
        if self._is_greeting(user_question):
            return {
                "response": GREETING_RESPONSE,
                "confidence": 1.0,
                "decision": "greeting",
                "sources": [],
                "indicator": "🟢 Greeting",
            }
        
        # Step 1: Embed the query
        import numpy as np
        query_embedding = self.embedding_model.encode([user_question])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Step 2: Search FAISS for similar chunks
        distances, indices = self.index.search(query_embedding, TOP_K)
        distances = distances[0].tolist()  # Flatten
        indices = indices[0].tolist()
        
        # Step 3: Retrieve chunk data
        retrieved_chunks = []
        for idx in indices:
            if idx < len(self.metadata) and idx >= 0:
                retrieved_chunks.append(self.metadata[idx])
        
        # Step 4: Hallucination guard evaluation
        evaluation = guard.evaluate_retrieval(distances, retrieved_chunks)
        
        if evaluation["decision"] == "refuse":
            return {
                "response": REFUSAL_RESPONSE,
                "confidence": evaluation["confidence"],
                "decision": "refuse",
                "sources": [],
                "indicator": guard.get_confidence_indicator(evaluation["confidence"]),
            }
        
        # Step 5: Filter relevant chunks and build context
        filtered_chunks, filtered_scores = guard.filter_relevant_chunks(
            distances, retrieved_chunks
        )
        
        # Use filtered chunks if available, otherwise fall back to all retrieved
        context_chunks = filtered_chunks if filtered_chunks else retrieved_chunks
        context = self._build_context(context_chunks)
        
        # Step 6: Generate response with LLM
        llm_response = self._call_llm(user_question, context)
        
        # Step 7: Compose final response with hallucination check
        result = check_and_respond(distances, retrieved_chunks, llm_response)
        
        # Add source info
        sources = []
        for chunk in context_chunks:
            meta = chunk.get("metadata", {})
            sources.append({
                "spec": meta.get("spec_number", "Unknown"),
                "page": meta.get("page", "?"),
                "section": meta.get("section", ""),
            })
        
        result["sources"] = sources
        return result
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks, respecting max length."""
        context_parts = []
        total_length = 0
        
        for chunk in chunks:
            text = chunk["text"]
            meta = chunk.get("metadata", {})
            
            # Add source info to each chunk
            source_prefix = f"[Source: {meta.get('spec_number', 'Unknown')}"
            if meta.get("section"):
                source_prefix += f", {meta['section']}"
            source_prefix += f", Page {meta.get('page', '?')}]\n"
            
            chunk_text = source_prefix + text
            
            if total_length + len(chunk_text) > MAX_CONTEXT_LENGTH:
                # Truncate if needed
                remaining = MAX_CONTEXT_LENGTH - total_length
                if remaining > 100:
                    context_parts.append(chunk_text[:remaining] + "...")
                break
            
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
        
        return "\n\n---\n\n".join(context_parts)
    
    def _call_llm(self, question: str, context: str) -> str:
        """Call Groq LLM with the question and context."""
        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _is_greeting(self, text: str) -> bool:
        """Check if the input is a greeting rather than a question."""
        greetings = [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "howdy", "what can you do", "help",
            "what are you", "who are you", "start", "yo", "sup",
            "hi there", "hey there", "hello there",
        ]
        text_lower = text.strip().lower().rstrip("!?.,")
        
        # Exact match
        if text_lower in greetings:
            return True
        
        # Short text (less than 4 chars)
        if len(text_lower) < 4:
            return True
        
        # Repeated greetings like "hello hello" or "hi hi hi"
        words = text_lower.split()
        if words and all(w in ["hello", "hi", "hey", "yo", "hii", "hiii", "helloo"] for w in words):
            return True
        
        # Starts with a greeting word and is short (< 20 chars)
        greeting_starts = ["hello", "hi ", "hey ", "good morning", "good afternoon", "good evening"]
        if len(text_lower) < 20 and any(text_lower.startswith(g) for g in greeting_starts):
            return True
        
        return False


# Singleton instance for use across the app
rag = RAGChain()


def ask(question: str) -> Dict:
    """Convenience function: ask a question to the RAG chain."""
    return rag.query(question)


if __name__ == "__main__":
    # Interactive test mode
    print("=" * 60)
    print("🤖 3GPP RAG Chatbot — CLI Mode")
    print("=" * 60)
    print("Type your questions about 3GPP specifications.")
    print("Type 'quit' to exit.\n")
    
    rag.load()
    
    while True:
        question = input("\n❓ You: ").strip()
        if question.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        if not question:
            continue
        
        result = rag.query(question)
        print(f"\n{result.get('indicator', '')}")
        print(f"🤖 Assistant: {result['response']}")
