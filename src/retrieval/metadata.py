from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class RetrievalChunk:
    """Represents a validated, ranked chunk retrieved from index, preserving citation details."""
    chunk_id: str
    document_id: str
    source_file: str
    source_reference: str
    similarity_score: float
    confidence: str
    chunk_text: str

@dataclass
class RetrievalResult:
    """Structured container encapsulating context packaging status, metadata, and performance metrics."""
    success: bool
    reason: str  # "SUCCESS", "NO_RELEVANT_CONTEXT", "EMPTY_QUERY", "QUERY_TOO_LONG", "EMPTY_INDEX"
    message: str
    query: str
    language: str
    language_confidence: float
    retrieved_chunks: List[RetrievalChunk] = field(default_factory=list)
    combined_context: str = ""
    total_chunks: int = 0
    statistics: Dict[str, Any] = field(default_factory=dict)  # top_k_requested, top_k_returned, threshold, duplicates_removed
    latency_metrics: Dict[str, Any] = field(default_factory=dict)  # query_embedding_time_ms, faiss_search_time_ms, metadata_lookup_time_ms, context_assembly_time_ms, total_latency_ms
