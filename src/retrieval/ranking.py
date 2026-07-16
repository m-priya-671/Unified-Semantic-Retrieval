from typing import List, Dict, Any, Tuple
from src.retrieval.metadata import RetrievalChunk
from src.utils.logger import logger

class RankingProcessor:
    """Handles vector score processing, duplicate chunk removals, threshold filtering, 
    and confidence level mappings.
    """
    
    @staticmethod
    def process(
        results: List[Dict[str, Any]], 
        threshold: float, 
        duplicate_removal: bool
    ) -> Tuple[List[RetrievalChunk], int, bool]:
        """Filters, deduplicates, and classifies candidate chunks.
        
        Args:
            results: List of raw SQLite dictionaries mapped by IndexManager.
            threshold: Similarity threshold float.
            duplicate_removal: True to filter duplicate chunk IDs.
            
        Returns:
            Tuple of (list of ranked RetrievalChunk objects, duplicates_removed_count, is_low_confidence)
        """
        seen_ids = set()
        retrieved_chunks = []
        duplicates_removed = 0
        is_low_confidence = False
        
        if not results:
            return [], 0, False
            
        # Determine highest candidate score
        highest_score = float(results[0].get("similarity_score", 0.0))
        
        # Determine active threshold. If no candidate exceeds threshold, trigger fallback
        active_threshold = threshold
        if highest_score < threshold:
            # Low confidence fallback! Grab top matches within a small bracket
            active_threshold = max(highest_score - 0.02, 0.0)
            is_low_confidence = True
            logger.warning(f"Low confidence fallback triggered. Lowering threshold from {threshold:.4f} to {active_threshold:.4f}")
            
        for r in results:
            chunk_id = r.get("chunk_id")
            score = float(r.get("similarity_score", 0.0))
            
            # 1. Deduplication
            if duplicate_removal and chunk_id:
                if chunk_id in seen_ids:
                    duplicates_removed += 1
                    continue
                seen_ids.add(chunk_id)
                
            # 2. Similarity Threshold Filter
            if score < active_threshold:
                logger.debug(f"Chunk '{chunk_id}' filtered out. Score {score:.4f} is below active threshold {active_threshold:.4f}")
                continue
                
            # 3. Configurable Confidence Level Assignment
            if is_low_confidence:
                confidence = "Low"
            else:
                if score >= 0.80:
                    confidence = "High"
                elif score >= 0.70:
                    confidence = "Medium"
                else:
                    confidence = "Low"
                
            retrieved_chunks.append(
                RetrievalChunk(
                    chunk_id=chunk_id,
                    document_id=r.get("document_id", ""),
                    source_file=r.get("source_file", "Unknown"),
                    source_reference=r.get("source_reference", "N/A"),
                    similarity_score=score,
                    confidence=confidence,
                    chunk_text=r.get("chunk_text", "")
                )
            )
            
        logger.info(f"Ranking processing complete. Kept {len(retrieved_chunks)} / {len(results)} chunks "
                    f"(Duplicates removed: {duplicates_removed}, low_confidence: {is_low_confidence})")
        return retrieved_chunks, duplicates_removed, is_low_confidence
