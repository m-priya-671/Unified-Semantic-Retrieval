import time
from typing import Dict, Any

class EmbeddingMetadataGenerator:
    """Collates performance and model metrics to record traceability metadata for every chunk embedding."""
    
    @staticmethod
    def generate(
        chunk_id: str,
        document_id: str,
        model_name: str,
        dimension: int,
        processing_time_ms: float,
        norm: float
    ) -> Dict[str, Any]:
        """Compiles embedding metadata dictionary parameters.
        
        Args:
            chunk_id: Unique sequential chunk ID.
            document_id: Parent document content ID.
            model_name: Name of the SentenceTransformer model used.
            dimension: Length of the embedding vector.
            processing_time_ms: Embedding duration in milliseconds.
            norm: L2 normalization magnitude of the vector (should be close to 1.0).
            
        Returns:
            A metadata dictionary.
        """
        return {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "embedding_model": model_name,
            "embedding_dimension": dimension,
            "embedding_version": "1.0",
            "embedding_norm": round(float(norm), 6),
            "embedding_created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_time_ms": round(processing_time_ms, 2),
            "embedding_status": "Success"
        }
