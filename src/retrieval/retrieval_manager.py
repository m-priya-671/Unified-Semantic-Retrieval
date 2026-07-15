import time
from typing import Dict, Any, List
import numpy as np

from src.embedding import EmbeddingManager
from src.vector_store import IndexManager
from src.retrieval.retrieval_config import RetrievalConfig
from src.retrieval.metadata import RetrievalChunk, RetrievalResult
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.query_validator import QueryValidator
from src.retrieval.language_detector import LanguageDetector
from src.retrieval.retrieval_engine import RetrievalEngine
from src.retrieval.ranking import RankingProcessor
from src.retrieval.context_builder import ContextBuilder
from src.utils.logger import logger

class RetrievalManager:
    """Orchestrates query validation, normalization, vector matching, ranking, and context packaging."""
    
    def __init__(self, embedding_manager: EmbeddingManager = None, index_manager: IndexManager = None):
        """Initializes components.
        
        Args:
            embedding_manager: Shared EmbeddingManager instance.
            index_manager: Main IndexManager instance.
        """
        self.embedding_manager = embedding_manager if embedding_manager is not None else EmbeddingManager()
        self.index_manager = index_manager if index_manager is not None else IndexManager()
        self.engine = RetrievalEngine(self.embedding_manager, self.index_manager)

    def search(self, raw_query: str, config: RetrievalConfig = None) -> RetrievalResult:
        """Executes the complete semantic search and context builder pipeline.
        
        Args:
            raw_query: Raw input text query.
            config: Specific config overrides (defaults to settings properties).
            
        Returns:
            A populated RetrievalResult dataclass object.
        """
        start_total = time.time()
        
        if config is None:
            config = RetrievalConfig()
            
        # 1. Cleaning & Unicode Normalization
        processed_query = QueryProcessor.process(raw_query)
        
        # 2. Language Detection
        lang, lang_conf = LanguageDetector.detect(processed_query)
        
        # 3. Sanity Checks and Validations
        try:
            QueryValidator.validate(processed_query, config.max_query_length, self.index_manager)
        except ValueError as val_err:
            msg = str(val_err)
            reason = "EMPTY_QUERY"
            if "maximum limit" in msg:
                reason = "QUERY_TOO_LONG"
            elif "index is empty" in msg:
                reason = "EMPTY_INDEX"
                
            latency_metrics = {
                "query_embedding_time_ms": 0.0,
                "faiss_search_time_ms": 0.0,
                "metadata_lookup_time_ms": 0.0,
                "context_assembly_time_ms": 0.0,
                "total_latency_ms": (time.time() - start_total) * 1000.0
            }
            
            return RetrievalResult(
                success=False,
                reason=reason,
                message=msg,
                query=raw_query,
                language=lang,
                language_confidence=lang_conf,
                latency_metrics=latency_metrics
            )

        # 4. Search Retrieval Engine
        try:
            query_vector, raw_candidates, latencies = self.engine.retrieve(processed_query, config.top_k)
        except Exception as e:
            msg = f"Inference retrieval execution failed: {str(e)}"
            logger.error(msg)
            return RetrievalResult(
                success=False,
                reason="RETRIEVAL_ERROR",
                message=msg,
                query=processed_query,
                language=lang,
                language_confidence=lang_conf,
                latency_metrics={"total_latency_ms": (time.time() - start_total) * 1000.0}
            )

        # 5. Ranking (Filtering & Deduplicating)
        retrieved_chunks, duplicates_removed = RankingProcessor.process(
            results=raw_candidates,
            threshold=config.similarity_threshold,
            duplicate_removal=config.duplicate_removal
        )
        
        stats = {
            "top_k_requested": config.top_k,
            "top_k_returned": len(retrieved_chunks),
            "threshold": config.similarity_threshold,
            "duplicates_removed": duplicates_removed
        }

        # 6. No-Result Fallback Handling
        if not retrieved_chunks:
            latencies["total_latency_ms"] = (time.time() - start_total) * 1000.0
            return RetrievalResult(
                success=False,
                reason="NO_RELEVANT_CONTEXT",
                message="No relevant information matching your query was found in the indexed documents. Please upload documents related to this topic.",
                query=processed_query,
                language=lang,
                language_confidence=lang_conf,
                retrieved_chunks=[],
                combined_context="",
                total_chunks=0,
                statistics=stats,
                latency_metrics=latencies
            )

        # 7. Context Builder Assembly
        start_context = time.time()
        combined_context = ContextBuilder.build(retrieved_chunks)
        latencies["context_assembly_time_ms"] = (time.time() - start_context) * 1000.0
        latencies["total_latency_ms"] = (time.time() - start_total) * 1000.0

        logger.info(f"Retrieval operation completed successfully. Hits: {len(retrieved_chunks)} / {config.top_k}")
        
        return RetrievalResult(
            success=True,
            reason="SUCCESS",
            message="Retrieval completed successfully.",
            query=processed_query,
            language=lang,
            language_confidence=lang_conf,
            retrieved_chunks=retrieved_chunks,
            combined_context=combined_context,
            total_chunks=len(retrieved_chunks),
            statistics=stats,
            latency_metrics=latencies
        )
