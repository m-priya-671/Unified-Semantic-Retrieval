from src.vector_store import IndexManager
from src.utils.logger import logger

class QueryValidator:
    """Enforces query validation constraints, including size, empty checks, and index status."""
    
    @staticmethod
    def validate(query: str, max_length: int, index_manager: IndexManager):
        """Validates query content and checks vector store state.
        
        Raises:
            ValueError: If validation bounds are violated.
        """
        # 1. Empty check
        if not query or not query.strip():
            msg = "Query cannot be empty."
            logger.error(msg)
            raise ValueError(msg)
            
        # 2. Maximum length limit check
        if len(query) > max_length:
            msg = f"Query length ({len(query)} characters) exceeds maximum limit of {max_length} characters."
            logger.error(msg)
            raise ValueError(msg)
            
        # 3. Vector Index check
        if index_manager is None or index_manager.engine is None or index_manager.engine.total == 0:
            msg = "Vector index is empty. Please upload and index documents first."
            logger.error(msg)
            raise ValueError(msg)
            
        logger.debug("Query validation passed successfully.")
