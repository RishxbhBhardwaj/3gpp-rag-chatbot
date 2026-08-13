"""
Hallucination Control Module.
Implements confidence scoring, source verification, and refusal logic.
"""
import numpy as np
from typing import List, Dict, Tuple
from config import (
    CONFIDENCE_THRESHOLD, 
    REFUSAL_THRESHOLD, 
    SIMILARITY_THRESHOLD,
    TOP_K
)
from prompts import REFUSAL_RESPONSE, LOW_CONFIDENCE_PREFIX


class HallucinationGuard:
    """
    Guards against hallucination by evaluating retrieval quality
    and deciding whether to answer, caveat, or refuse.
    """
    
    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        refusal_threshold: float = REFUSAL_THRESHOLD,
    ):
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.refusal_threshold = refusal_threshold
    
    def evaluate_retrieval(
        self, scores: List[float], chunks: List[Dict]
    ) -> Dict:
        """
        Evaluate the quality of retrieved chunks.
        
        Args:
            scores: Similarity scores from vector search (0-1, higher = more similar).
            chunks: Retrieved chunk data.
            
        Returns:
            Dict with decision, confidence, and reasoning.
        """
        if not scores or not chunks:
            return {
                "decision": "refuse",
                "confidence": 0.0,
                "reason": "No relevant documents found in the knowledge base.",
                "response_prefix": "",
            }
        
        # Convert distances to similarity if needed (FAISS returns L2 distances)
        # Lower distance = more similar. Convert to 0-1 similarity score.
        similarities = self._normalize_scores(scores)
        
        best_score = max(similarities)
        avg_score = np.mean(similarities[:min(3, len(similarities))])  # Top 3 average
        
        # Decision logic
        if best_score < self.refusal_threshold:
            return {
                "decision": "refuse",
                "confidence": best_score,
                "reason": f"Best retrieval score ({best_score:.3f}) is below refusal threshold ({self.refusal_threshold}). No relevant context found.",
                "response_prefix": "",
            }
        
        if avg_score < self.confidence_threshold:
            return {
                "decision": "low_confidence",
                "confidence": avg_score,
                "reason": f"Average retrieval score ({avg_score:.3f}) is below confidence threshold ({self.confidence_threshold}). Context may be partially relevant.",
                "response_prefix": LOW_CONFIDENCE_PREFIX,
            }
        
        # Filter chunks below similarity threshold
        relevant_count = sum(1 for s in similarities if s >= self.similarity_threshold)
        
        if relevant_count == 0:
            return {
                "decision": "refuse",
                "confidence": best_score,
                "reason": "No chunks above similarity threshold after filtering.",
                "response_prefix": "",
            }
        
        return {
            "decision": "answer",
            "confidence": avg_score,
            "reason": f"Good retrieval quality. {relevant_count}/{len(similarities)} chunks above threshold.",
            "response_prefix": "",
        }
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize FAISS L2 distance scores to 0-1 similarity.
        FAISS returns L2 distances (lower = more similar).
        We convert: similarity = 1 / (1 + distance)
        """
        return [1.0 / (1.0 + score) for score in scores]
    
    def filter_relevant_chunks(
        self, scores: List[float], chunks: List[Dict]
    ) -> Tuple[List[Dict], List[float]]:
        """
        Filter chunks that are below the similarity threshold.
        
        Returns:
            Tuple of (filtered_chunks, filtered_scores)
        """
        similarities = self._normalize_scores(scores)
        
        filtered_chunks = []
        filtered_scores = []
        
        for sim, chunk in zip(similarities, chunks):
            if sim >= self.similarity_threshold:
                filtered_chunks.append(chunk)
                filtered_scores.append(sim)
        
        return filtered_chunks, filtered_scores
    
    def format_citations(self, chunks: List[Dict]) -> str:
        """Format source citations from chunks."""
        if not chunks:
            return ""
        
        sources = set()
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            spec = meta.get("spec_number", "Unknown")
            page = meta.get("page", "?")
            section = meta.get("section", "")
            
            if section:
                sources.add(f"{spec}, {section} (p.{page})")
            else:
                sources.add(f"{spec} (p.{page})")
        
        citation_text = "\n\n📚 **Sources:**\n"
        for source in sorted(sources):
            citation_text += f"- {source}\n"
        
        return citation_text
    
    def get_confidence_indicator(self, confidence: float) -> str:
        """Return a visual confidence indicator."""
        if confidence >= 0.7:
            return "🟢 High Confidence"
        elif confidence >= 0.5:
            return "🟡 Medium Confidence"
        elif confidence >= self.refusal_threshold:
            return "🟠 Low Confidence"
        else:
            return "🔴 Insufficient Context"


# Singleton instance
guard = HallucinationGuard()


def check_and_respond(
    scores: List[float], 
    chunks: List[Dict], 
    llm_response: str = ""
) -> Dict:
    """
    Main entry point: evaluate retrieval and prepare final response.
    
    Args:
        scores: Raw FAISS distance scores.
        chunks: Retrieved chunks with metadata.
        llm_response: The LLM's generated response (if already generated).
        
    Returns:
        Dict with final_response, confidence, decision, citations.
    """
    evaluation = guard.evaluate_retrieval(scores, chunks)
    
    if evaluation["decision"] == "refuse":
        return {
            "final_response": REFUSAL_RESPONSE,
            "confidence": evaluation["confidence"],
            "decision": "refuse",
            "citations": "",
            "indicator": guard.get_confidence_indicator(evaluation["confidence"]),
        }
    
    # Get citations
    filtered_chunks, _ = guard.filter_relevant_chunks(scores, chunks)
    citations = guard.format_citations(filtered_chunks)
    
    # Prepare response
    prefix = evaluation.get("response_prefix", "")
    final_response = prefix + llm_response + citations
    
    return {
        "final_response": final_response,
        "confidence": evaluation["confidence"],
        "decision": evaluation["decision"],
        "citations": citations,
        "indicator": guard.get_confidence_indicator(evaluation["confidence"]),
    }
