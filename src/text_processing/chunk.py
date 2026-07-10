from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Chunk:
    """Represents a text chunk generated from a UnifiedDocument, 
    complete with lineage references and human-readable citations.
    """
    chunk_id: str             # Readable pattern: {document_id}_chunk_{chunk_index:04d}
    document_id: str          # Reference link to the parent UnifiedDocument
    chunk_index: int          # Index in the document sequence (0-indexed)
    text: str                 # Extracted segment text
    metadata: Dict[str, Any]  # Inherited parent metadata + source_reference & coordinates
    token_estimate: int       # Debug approximate token metric (characters // 4)
    character_count: int      # Length of this chunk's text in characters
