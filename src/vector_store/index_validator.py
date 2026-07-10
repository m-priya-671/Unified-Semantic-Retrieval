import numpy as np
import faiss
from src.utils.logger import logger

class IndexValidator:
    """Enforces validation checks on indexing inputs, search queries, and binary structure integrity."""
    
    @staticmethod
    def validate_vectors(vectors: np.ndarray, expected_dimension: int):
        """Asserts that shape and dimension boundaries are correctly set.
        
        Args:
            vectors: 2D float32 numpy array.
            expected_dimension: Configured index dimension.
        """
        if vectors is None or len(vectors) == 0:
            msg = "Cannot index an empty vector array."
            logger.error(msg)
            raise ValueError(msg)
            
        if len(vectors.shape) != 2:
            msg = f"Expected 2D vector matrix array, got shape {vectors.shape}."
            logger.error(msg)
            raise ValueError(msg)
            
        actual_dim = vectors.shape[1]
        if actual_dim != expected_dimension:
            msg = f"Vector dimension mismatch. Expected {expected_dimension}, got {actual_dim} dimensions."
            logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def validate_search(query_vector: np.ndarray, index_total: int, expected_dimension: int):
        """Validates state before attempting similarity search.
        
        Args:
            query_vector: 1D or 2D float32 array.
            index_total: Total items loaded in FAISS.
            expected_dimension: Required vector size.
        """
        if index_total == 0:
            msg = "Similarity search failed: Vector index is empty."
            logger.warning(msg)
            raise ValueError(msg)
            
        if query_vector is None:
            msg = "Query vector is None."
            logger.error(msg)
            raise ValueError(msg)
            
        shape = query_vector.shape
        actual_dim = shape[0] if len(shape) == 1 else shape[1]
        if actual_dim != expected_dimension:
            msg = f"Search query dimension mismatch. Expected {expected_dimension}, got {actual_dim} dimensions."
            logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def validate_index_file(index: faiss.Index, expected_dimension: int) -> bool:
        """Checks if the loaded FAISS index is valid and matches model configs.
        
        Args:
            index: Loaded FAISS index.
            expected_dimension: Configured vector size.
        """
        if index is None:
            return False
            
        try:
            if index.d != expected_dimension:
                msg = f"Loaded FAISS index dimension {index.d} does not match configured dimension {expected_dimension}."
                logger.error(msg)
                raise ValueError(msg)
            return True
        except Exception as e:
            logger.error(f"FAISS index validation exception: {str(e)}")
            return False
