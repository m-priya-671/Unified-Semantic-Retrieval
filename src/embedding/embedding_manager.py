import hashlib
import time
import numpy as np
from typing import List, Tuple, Dict, Any
from src.text_processing.chunk import Chunk
from src.embedding.embedding_engine import EmbeddingEngine
from src.embedding.embedding_cache import EmbeddingCache
from src.embedding.embedding_validator import EmbeddingValidator
from src.embedding.metadata import EmbeddingMetadataGenerator
from src.utils.logger import logger

class EmbeddingManager:
    """Facade orchestrator that validates chunk texts, checks the database cache, 
    coordinates batch inference, and records embedding tracking parameters.
    """
    
    def __init__(self, model_name: str = None):
        """Initializes the cache and prepares the lazy engine properties."""
        self.model_name = model_name
        self.engine = None
        self.cache = EmbeddingCache()
        
        # Track execution metrics
        self.total_processed = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_inference_time_ms = 0.0

    def embed_chunks(
        self, 
        chunks: List[Chunk], 
        batch_size: int = 32
    ) -> Tuple[np.ndarray, List[Chunk]]:
        """Validates and generates unit-length embeddings for a list of Chunk objects.
        
        Args:
            chunks: List of Chunk objects from text_processing.
            batch_size: Configurable batch chunk capacity (default: 32).
            
        Returns:
            A tuple (vectors, chunks) where:
                - vectors: A 2D float32 numpy array of shape (len(chunks), dimension).
                - chunks: The updated chunks list, enriched with embedding metadata.
        """
        if not chunks:
            dim = self.engine.dimension if self.engine else 384
            return np.empty((0, dim), dtype=np.float32), []
            
        logger.info(f"Preparing to embed {len(chunks)} chunks (batch size: {batch_size})")
        
        # 1. Lazy load embedding engine to get model specifications
        if self.engine is None:
            self.engine = EmbeddingEngine(self.model_name)
            
        model_tag = self.engine.model_name
        dimension = self.engine.dimension
        
        # Initialize results placeholders
        num_chunks = len(chunks)
        vectors_list = [None] * num_chunks
        
        missed_indices = []
        missed_hashes = []
        missed_texts = []
        
        # 2. Cache Lookup and Input Validation
        for idx, chunk in enumerate(chunks):
            # Validate input content
            EmbeddingValidator.validate_chunk_text(chunk.text)
            
            # Derive SHA-256 hash of the chunk text
            chunk_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            
            # Query Database Cache
            cached_vector = self.cache.get(chunk_hash, model_tag)
            
            if cached_vector is not None:
                # Cache Hit! Validate cached shape
                EmbeddingValidator.validate_dimensions(cached_vector, dimension)
                
                # Retrieve norm
                norm = np.linalg.norm(cached_vector)
                
                # Append metadata (0ms processing latency on cache hit)
                meta = EmbeddingMetadataGenerator.generate(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    model_name=model_tag,
                    dimension=dimension,
                    processing_time_ms=0.0,
                    norm=norm
                )
                chunk.metadata.update(meta)
                vectors_list[idx] = cached_vector
                
                self.cache_hits += 1
                self.total_processed += 1
            else:
                # Cache Miss
                missed_indices.append(idx)
                missed_hashes.append(chunk_hash)
                missed_texts.append(chunk.text)
                
                self.cache_misses += 1
                self.total_processed += 1
                
        logger.info(f"Cache check complete. Hits: {self.cache_hits}, Misses: {self.cache_misses}")
        
        # 3. Batch Inference on Cache Misses
        if missed_texts:
            num_misses = len(missed_texts)
            logger.info(f"Executing CPU model inference for {num_misses} chunks...")
            
            for b_idx in range(0, num_misses, batch_size):
                batch_slice = slice(b_idx, b_idx + batch_size)
                batch_chunk_texts = missed_texts[batch_slice]
                batch_chunk_hashes = missed_hashes[batch_slice]
                batch_chunk_indices = missed_indices[batch_slice]
                
                # Measure batch inference time
                start_time = time.time()
                batch_vectors = self.engine.generate(batch_chunk_texts)
                duration_ms = (time.time() - start_time) * 1000.0
                
                self.total_inference_time_ms += duration_ms
                time_per_chunk = duration_ms / len(batch_chunk_texts)
                
                # Process and cache each generated vector
                for i_in_batch, vector in enumerate(batch_vectors):
                    original_idx = batch_chunk_indices[i_in_batch]
                    chunk_hash = batch_chunk_hashes[i_in_batch]
                    target_chunk = chunks[original_idx]
                    
                    # Validate vector dimensions
                    EmbeddingValidator.validate_dimensions(vector, dimension)
                    
                    # Verify L2 normalization
                    norm = np.linalg.norm(vector)
                    
                    # Write to database cache
                    self.cache.set(
                        chunk_hash=chunk_hash,
                        document_hash=target_chunk.document_id,
                        vector=vector,
                        model_name=model_tag
                    )
                    
                    # Collate metadata
                    meta = EmbeddingMetadataGenerator.generate(
                        chunk_id=target_chunk.chunk_id,
                        document_id=target_chunk.document_id,
                        model_name=model_tag,
                        dimension=dimension,
                        processing_time_ms=time_per_chunk,
                        norm=norm
                    )
                    target_chunk.metadata.update(meta)
                    vectors_list[original_idx] = vector
                    
            logger.info(f"Batch inference complete. Total inference time: {self.total_inference_time_ms:.1f}ms")
            
        # 4. Stack results into 2D float32 numpy array
        stacked_vectors = np.vstack(vectors_list).astype(np.float32)
        return stacked_vectors, chunks

    def get_stats(self) -> Dict[str, Any]:
        """Returns processing metrics for the current session."""
        total = self.total_processed
        hit_ratio = (self.cache_hits / total * 100.0) if total > 0 else 0.0
        avg_speed = (self.total_inference_time_ms / self.cache_misses) if self.cache_misses > 0 else 0.0
        
        return {
            "total_processed": total,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio_percent": round(hit_ratio, 2),
            "total_inference_time_ms": round(self.total_inference_time_ms, 2),
            "avg_inference_speed_ms_per_chunk": round(avg_speed, 2)
        }
