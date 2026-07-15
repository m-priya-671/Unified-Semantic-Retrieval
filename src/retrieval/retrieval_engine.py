import time
from typing import List, Dict, Any, Tuple
import numpy as np
from src.embedding import EmbeddingManager
from src.embedding.embedding_engine import EmbeddingEngine
from src.vector_store import IndexManager
from src.utils.logger import logger

class RetrievalEngine:
    """Manages raw vector embedding generation and performs search via IndexManager.
    
    Enforces isolation: Communicates only with IndexManager, never directly with FAISS or SQLite.
    """
    
    def __init__(self, embedding_manager: EmbeddingManager, index_manager: IndexManager):
        """Initializes dependencies.
        
        Args:
            embedding_manager: Shared embedding manager instance.
            index_manager: Main index coordinator.
        """
        self.embedding_manager = embedding_manager
        self.index_manager = index_manager

    def retrieve(self, query: str, top_k: int) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, float]]:
        """Generates a query embedding and queries the index manager for top candidates.
        
        Args:
            query: Normalised user query text.
            top_k: Number of documents requested.
            
        Returns:
            Tuple containing:
                - query_vector: 1D float32 numpy array of query embedding.
                - raw_results: List of raw SQLite dictionaries mapped by IndexManager.
                - latencies: Dict containing embedding, search, and db lookup durations in ms.
        """
        latencies = {}
        
        # 1. Lazy load embedding engine if not initialized
        if self.embedding_manager.engine is None:
            logger.info("Lazily instantiating local EmbeddingEngine for query parsing...")
            self.embedding_manager.engine = EmbeddingEngine(self.embedding_manager.model_name)
            
        # 2. Query Embedding Time
        start_emb = time.time()
        query_vectors = self.embedding_manager.engine.generate([query])
        query_vector = query_vectors[0]
        latencies["query_embedding_time_ms"] = (time.time() - start_emb) * 1000.0
        
        # 3. FAISS Search Time (raw indexes only)
        start_search = time.time()
        similarities, indices = self.index_manager.raw_search(query_vector, top_k=top_k)
        latencies["faiss_search_time_ms"] = (time.time() - start_search) * 1000.0
        
        # 4. Database Lookup Time
        start_lookup = time.time()
        raw_results = self.index_manager.map_search_results(similarities, indices)
        latencies["metadata_lookup_time_ms"] = (time.time() - start_lookup) * 1000.0
        
        logger.info(f"Retrieval timing breakdown: Embedding={latencies['query_embedding_time_ms']:.2f}ms, "
                    f"FAISS Search={latencies['faiss_search_time_ms']:.2f}ms, "
                    f"Metadata Query={latencies['metadata_lookup_time_ms']:.2f}ms")
                    
        return query_vector, raw_results, latencies
