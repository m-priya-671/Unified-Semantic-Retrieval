from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class UnifiedDocument:
    """Standardized document representation matching heterogeneous ingestion outputs 
    before splitting into embeddings chunks.
    """
    document_id: str                  # SHA-256 hash derived from file contents
    source_file: str                  # Original file name (e.g. "manual.pdf")
    source_type: str                  # "pdf" | "docx" | "image" | "audio"
    text: str                         # Continuous, consolidated text contents
    metadata: Dict[str, Any]          # Parent document metadata parameters
    languages: List[Dict[str, Any]]   # Multilingual array: [{'language': 'en', 'probability': 0.9}]
    created_at: str                   # ISO timestamp of document parse
    processing_time: float            # Ingestion latency in seconds
