from src.text_processing.unified_document import UnifiedDocument
from src.utils.logger import logger

class TextProcessingValidator:
    """Validator for verifying document sizes, content presence, and schema specifications."""
    
    @staticmethod
    def validate_document(doc: UnifiedDocument) -> bool:
        """Validates that a UnifiedDocument has adequate content and valid fields.
        
        Args:
            doc: UnifiedDocument instance.
            
        Returns:
            True if valid, raises ValueError otherwise.
        """
        logger.info(f"Validating UnifiedDocument: {doc.source_file} (id={doc.document_id})")
        
        # 1. Content check
        if not doc.text or not doc.text.strip():
            msg = f"Document '{doc.source_file}' has no text content."
            logger.error(msg)
            raise ValueError(msg)
            
        # 2. Length lower boundary check
        text_len = len(doc.text)
        if text_len < 10:
            msg = f"Document '{doc.source_file}' contains too little text ({text_len} chars) to process."
            logger.error(msg)
            raise ValueError(msg)
            
        # 3. Warning boundary check
        if text_len > 1_000_000:
            logger.warning(f"Document '{doc.source_file}' is very large ({text_len} characters). Processing may be slow.")
            
        # 4. Mandatory fields checks
        if not doc.document_id:
            msg = f"Document ID is missing for '{doc.source_file}'."
            logger.error(msg)
            raise ValueError(msg)
            
        if not doc.source_type:
            msg = f"Source type is missing for '{doc.source_file}'."
            logger.error(msg)
            raise ValueError(msg)
            
        # 5. Metadata verification
        if doc.metadata is None or not isinstance(doc.metadata, dict):
            msg = f"Document metadata is invalid or missing for '{doc.source_file}'."
            logger.error(msg)
            raise ValueError(msg)
            
        logger.info(f"UnifiedDocument validation successful: {doc.source_file}")
        return True
