from dataclasses import dataclass
from src.config.settings import DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD, MAX_QUERY_LENGTH

@dataclass
class RetrievalConfig:
    """Structure that groups semantic retrieval hyperparameters and query limits."""
    top_k: int = DEFAULT_TOP_K
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    max_query_length: int = MAX_QUERY_LENGTH
    duplicate_removal: bool = True
    retrieval_mode: str = "Automatic"
