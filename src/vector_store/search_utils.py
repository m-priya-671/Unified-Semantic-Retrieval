from typing import List, Dict, Any
import numpy as np
from src.vector_store.metadata_store import MetadataStore
from src.utils.logger import logger

class SearchUtils:
    """Provides utility methods to map raw numerical search outputs back into structured metadata results."""
    
    @staticmethod
    def map_results(
        similarities: np.ndarray, 
        indices: np.ndarray, 
        metadata_store: MetadataStore
    ) -> List[Dict[str, Any]]:
        """Maps FAISS search results to structured metadata dictionaries.
        
        Args:
            similarities: 2D numpy array of float inner-product scores (e.g. from index.search).
            indices: 2D numpy array of mapped 64-bit integer IDs.
            metadata_store: Mapped database manager.
            
        Returns:
            A list of dictionary results containing:
                - chunk_id: Unique index code.
                - document_id: Parent hash link.
                - source_file: Upload file name.
                - source_reference: Coordinate citation.
                - similarity_score: float matching score.
                - chunk_text: Raw context text payload.
        """
        # similarities and indices are 2D arrays (usually matching single batch query inputs)
        if len(indices) == 0 or len(indices[0]) == 0:
            return []
            
        flat_ids = [int(x) for x in indices[0] if x != -1]
        flat_scores = [float(s) for s in similarities[0]]
        
        if not flat_ids:
            return []
            
        # Bulk query matching SQLite records
        metadata_rows = metadata_store.get_metadata_by_ids(flat_ids)
        
        # Build quick dictionary mapping
        id_to_metadata = {row["faiss_id"]: row for row in metadata_rows}
        
        structured_results = []
        for idx, fid in enumerate(flat_ids):
            if fid in id_to_metadata:
                row = id_to_metadata[fid]
                structured_results.append({
                    "faiss_id": fid,
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "source_file": row["source_file"],
                    "source_reference": row["source_reference"],
                    "similarity_score": round(flat_scores[idx], 6),
                    "chunk_text": row["chunk_text"]
                })
                
        logger.debug(f"Mapped {len(structured_results)} FAISS vector IDs to SQLite metadata.")
        return structured_results
