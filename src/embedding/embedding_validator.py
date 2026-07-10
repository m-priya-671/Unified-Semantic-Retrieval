import numpy as np
from src.utils.logger import logger

class EmbeddingValidator:
    """Enforces boundaries and shape validations on input texts and generated vector dimensions."""
    
    @staticmethod
    def validate_chunk_text(text: str) -> bool:
        """Ensures that chunk text is non-empty and has sufficient content.
        
        Args:
            text: String chunk segment.
            
        Returns:
            True if valid, raises ValueError otherwise.
        """
        if not text or not text.strip():
            msg = "Cannot generate embedding for an empty or whitespace-only chunk."
            logger.error(msg)
            raise ValueError(msg)
            
        if len(text.strip()) < 3:
            msg = f"Chunk text is too short ('{text}') to represent semantic meaning."
            logger.warning(msg)
            
        return True

    @staticmethod
    def validate_dimensions(vector: np.ndarray, expected_dim: int) -> bool:
        """Asserts that the output vector dimensions match the configured model dimensions.
        
        Args:
            vector: Generated numpy vector array.
            expected_dim: Model output dimensions.
            
        Returns:
            True if matching, raises ValueError otherwise.
        """
        if vector is None:
            msg = "Generated vector is None."
            logger.error(msg)
            raise ValueError(msg)
            
        actual_shape = vector.shape
        # Check if 1D vector matching dimension
        if len(actual_shape) != 1:
            msg = f"Expected 1D vector shape, got multi-dimensional shape: {actual_shape}"
            logger.error(msg)
            raise ValueError(msg)
            
        if actual_shape[0] != expected_dim:
            msg = f"Vector dimension mismatch. Expected {expected_dim}, got {actual_shape[0]} dimensions."
            logger.error(msg)
            raise ValueError(msg)
            
        return True
