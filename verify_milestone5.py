import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path

# Add workspace directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.text_processing.chunk import Chunk
from src.embedding.embedding_engine import EmbeddingEngine
from src.embedding.embedding_cache import EmbeddingCache
from src.embedding.embedding_manager import EmbeddingManager
from src.embedding.embedding_validator import EmbeddingValidator
from src.utils.logger import logger

def test_engine_inference():
    logger.info("=== 1. TESTING LOCAL EMBEDDING ENGINE ===")
    engine = EmbeddingEngine()
    
    # Verify properties
    assert engine.dimension == 384, f"Expected 384 dimensions for MiniLM, got {engine.dimension}"
    
    # Test texts
    eng_text = "This is a local semantic search test."
    tam_text = "செயற்கை நுண்ணறிவுத் தொழில்நுட்பம் வேகமாக வளர்ந்து வருகிறது."
    mix_text = "RAG சிஸ்டம் embedding test."
    
    vectors = engine.generate([eng_text, tam_text, mix_text])
    
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (3, 384)
    assert vectors.dtype == np.float32
    
    # Verify L2 normalization: norm should be extremely close to 1.0
    for idx, vec in enumerate(vectors):
        norm = np.linalg.norm(vec)
        logger.info(f"Vector {idx} L2 Norm: {norm:.6f}")
        assert np.isclose(norm, 1.0, atol=1e-5), f"Vector {idx} is not L2 normalized. Norm: {norm}"
        
    logger.info("Local Embedding Engine validation: OK\n" + "-"*50)

def test_caching_and_manager():
    logger.info("=== 2. TESTING EMBEDDING MANAGER & CACHE ===")
    
    # Create test chunks
    chunk_1 = Chunk(
        chunk_id="doc1_chunk_0000",
        document_id="doc1",
        chunk_index=0,
        text="The quick brown fox jumps over the lazy dog.",
        metadata={"source_reference": "Block 1"},
        token_estimate=10,
        character_count=44
    )
    chunk_2 = Chunk(
        chunk_id="doc1_chunk_0001",
        document_id="doc1",
        chunk_index=1,
        text="செயற்கை நுண்ணறிவுத் தொழில்நுட்பம்.",
        metadata={"source_reference": "Block 2"},
        token_estimate=8,
        character_count=33
    )
    
    # Init manager
    manager = EmbeddingManager()
    
    # Clear local cache entries if they exist for clean test state
    try:
        import sqlite3
        from src.config.settings import DATABASE_PATH
        with sqlite3.connect(str(DATABASE_PATH)) as conn:
            conn.cursor().execute("DELETE FROM embedding_cache")
            conn.commit()
        logger.info("Database cache cleared for clean test state.")
    except Exception as e:
        logger.warning(f"Could not clear cache DB: {e}")
        
    # First execution (Cache Misses)
    vectors_1, updated_chunks_1 = manager.embed_chunks([chunk_1, chunk_2], batch_size=2)
    stats_1 = manager.get_stats()
    
    assert vectors_1.shape == (2, 384)
    assert stats_1["cache_hits"] == 0
    assert stats_1["cache_misses"] == 2
    
    # Verify metadata fields are appended correctly
    meta1 = updated_chunks_1[0].metadata
    assert meta1["embedding_model"] == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert meta1["embedding_dimension"] == 384
    assert meta1["embedding_version"] == "1.0"
    assert "embedding_norm" in meta1
    assert "embedding_created_at" in meta1
    assert meta1["embedding_time_ms"] > 0.0
    assert meta1["embedding_status"] == "Success"
    
    logger.info("First run metrics (Cache Misses): OK")
    
    # Second execution (Cache Hits)
    # Re-instantiate identical chunks
    chunk_1_dup = Chunk(
        chunk_id="doc1_chunk_0000",
        document_id="doc1",
        chunk_index=0,
        text="The quick brown fox jumps over the lazy dog.",
        metadata={"source_reference": "Block 1"},
        token_estimate=10,
        character_count=44
    )
    chunk_2_dup = Chunk(
        chunk_id="doc1_chunk_0001",
        document_id="doc1",
        chunk_index=1,
        text="செயற்கை நுண்ணறிவுத் தொழில்நுட்பம்.",
        metadata={"source_reference": "Block 2"},
        token_estimate=8,
        character_count=33
    )
    
    manager_dup = EmbeddingManager()
    vectors_2, updated_chunks_2 = manager_dup.embed_chunks([chunk_1_dup, chunk_2_dup], batch_size=2)
    stats_2 = manager_dup.get_stats()
    
    # Assert all hits, 0 misses, vectors are identical
    assert np.allclose(vectors_1, vectors_2, atol=1e-5), "Cached vectors do not match original vectors!"
    assert stats_2["cache_hits"] == 2
    assert stats_2["cache_misses"] == 0
    
    # Verify metadata shows 0ms latency on cache hit
    assert updated_chunks_2[0].metadata["embedding_time_ms"] == 0.0
    
    logger.info("Second run metrics (Cache Hits): OK")
    logger.info("Embedding Manager & Caching tests: OK\n" + "-"*50)

def test_validation():
    logger.info("=== 3. TESTING EMBEDDING VALIDATOR ===")
    
    # Empty string check
    try:
        EmbeddingValidator.validate_chunk_text("")
        raise AssertionError("Validator failed to reject empty string")
    except ValueError as e:
        logger.info(f"Validator correctly rejected empty chunk: {e}")
        
    # Dimension check
    bad_vector = np.array([0.1, 0.2, 0.3])
    try:
        EmbeddingValidator.validate_dimensions(bad_vector, expected_dim=384)
        raise AssertionError("Validator failed to reject invalid dimensions")
    except ValueError as e:
        logger.info(f"Validator correctly rejected dimension mismatch: {e}")
        
    logger.info("Embedding Validator tests: OK\n" + "-"*50)

def run_all_tests():
    logger.info("Starting Milestone 5 automated tests...")
    test_engine_inference()
    test_caching_and_manager()
    test_validation()
    logger.info("ALL MILESTONE 5 TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_all_tests()
