import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import sqlite3
import json
from pathlib import Path

# Add workspace directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.text_processing.chunk import Chunk
from src.vector_store import IndexManager, MetadataStore, IndexPersistence, IndexValidator, SearchUtils
from src.utils.logger import logger

def clean_indices():
    """Wipes out physical index folders and database tables for testing consistency."""
    manager = IndexManager()
    manager.clear_all()
    # Delete index file if exists
    idx_file = manager.index_dir / "index.faiss"
    if idx_file.exists():
        os.remove(idx_file)
    info_file = manager.index_dir / "index_info.json"
    if info_file.exists():
        os.remove(info_file)
    logger.info("Indices and databases fully cleared for test run.")

def test_index_pipeline():
    logger.info("=== 1. TESTING MAIN INDEX PIPELINE & PERSISTENCE ===")
    clean_indices()
    
    manager = IndexManager()
    
    # Generate 2 mock chunks and L2 normalized vectors
    c1 = Chunk(
        chunk_id="test_doc_chunk_0000",
        document_id="test_doc",
        chunk_index=0,
        text="The quick brown fox jumps over the lazy dog.",
        metadata={"source_file": "sample.pdf", "source_reference": "Page 1"},
        token_estimate=10,
        character_count=44
    )
    c2 = Chunk(
        chunk_id="test_doc_chunk_0001",
        document_id="test_doc",
        chunk_index=1,
        text="செயற்கை நுண்ணறிவுத் தொழில்நுட்பம் வேகமாக வளர்கிறது.",
        metadata={"source_file": "sample.pdf", "source_reference": "Page 2"},
        token_estimate=12,
        character_count=52
    )
    
    # Create normalized vectors (dimension: 384)
    v1 = np.random.rand(384).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    
    v2 = np.random.rand(384).astype(np.float32)
    v2 /= np.linalg.norm(v2)
    
    vectors = np.vstack([v1, v2])
    
    # Ingest vectors
    manager.add_vectors_and_metadata(vectors, [c1, c2])
    
    # Assert counts
    assert manager.engine.total == 2
    
    # Verify metadata entries exist in SQLite
    row1 = manager.metadata_store.get_metadata(faiss_id=0)
    assert row1 is not None
    assert row1["chunk_id"] == "test_doc_chunk_0000"
    assert row1["source_reference"] == "Page 1"
    
    # Verify document-level table has been populated
    with sqlite3.connect(manager.metadata_store.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM document_index_status WHERE document_hash = ?", ("test_doc",))
        doc_row = cursor.fetchone()
        assert doc_row is not None
        assert doc_row[1] == 2  # indexed_chunks
        assert doc_row[2] == 2  # total_chunks
        assert doc_row[4] == "Completed"
        
    logger.info("Main Insertion & Transaction Sync: OK")
    
    # Test Cosine Similarity Search
    # Search with v1 vector -> should return c1 as top neighbor with score close to 1.0
    results = manager.search(v1, top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "test_doc_chunk_0000"
    assert np.isclose(results[0]["similarity_score"], 1.0, atol=1e-4)
    assert results[0]["chunk_text"] == "The quick brown fox jumps over the lazy dog."
    
    logger.info("Similarity Dot Product Retrieval Search: OK")
    
    # Test Reload consistency
    manager_reload = IndexManager()
    assert manager_reload.engine.total == 2
    
    results_reload = manager_reload.search(v1, top_k=1)
    assert results_reload[0]["chunk_id"] == "test_doc_chunk_0000"
    assert np.isclose(results_reload[0]["similarity_score"], 1.0, atol=1e-4)
    
    # Check index_info.json structure
    info = manager_reload.get_index_info()
    assert info["index_version"] == "1.0"
    assert info["embedding_version"] == "1.0"
    assert info["schema_version"] == "1.0"
    assert info["total_documents"] == 1
    assert info["average_chunks_per_document"] == 2.0
    assert info["normalized_vectors"] is True
    
    logger.info("Persistence Save/Reload Consistency: OK\n" + "-"*50)

def test_edge_cases():
    logger.info("=== 2. TESTING FAISS INDEX EDGE CASES ===")
    
    # Edge Case A: Empty Index Search
    clean_indices()
    manager = IndexManager()
    assert manager.engine.total == 0
    
    empty_vector = np.random.rand(384).astype(np.float32)
    empty_vector /= np.linalg.norm(empty_vector)
    
    # Should safely return empty list without crashing
    empty_res = manager.search(empty_vector, top_k=5)
    assert len(empty_res) == 0
    logger.info("Edge Case: Empty Index Search -> OK")
    
    # Edge Case B: Duplicate Insert
    c1 = Chunk(
        chunk_id="duplicate_chunk_0000",
        document_id="doc_dup",
        chunk_index=0,
        text="Duplicate chunk text content.",
        metadata={"source_file": "dup.docx"},
        token_estimate=5,
        character_count=29
    )
    v1 = np.random.rand(384).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    
    # Insert first time
    manager.add_vectors_and_metadata(np.expand_dims(v1, axis=0), [c1])
    assert manager.engine.total == 1
    
    # Insert identical chunk_id a second time -> should skip insertion cleanly
    manager.add_vectors_and_metadata(np.expand_dims(v1, axis=0), [c1])
    assert manager.engine.total == 1
    logger.info("Edge Case: Duplicate Insert skipped -> OK")
    
    # Edge Case C: Invalid Vector Dimensions
    bad_vectors = np.random.rand(1, 128).astype(np.float32)  # Dimension 128 instead of 384
    c_bad = Chunk(
        chunk_id="bad_dim_chunk",
        document_id="doc_bad",
        chunk_index=0,
        text="Bad dimension test.",
        metadata={},
        token_estimate=5,
        character_count=20
    )
    try:
        manager.add_vectors_and_metadata(bad_vectors, [c_bad])
        raise AssertionError("Failed to raise dimension mismatch ValueError")
    except ValueError as e:
        logger.info(f"Edge Case: Invalid Vector Dimension rejected -> {e} -> OK")
        
    # Edge Case D: Missing Index File Load
    # Move index file away to simulate deletion
    idx_file = manager.index_dir / "index.faiss"
    if idx_file.exists():
        os.remove(idx_file)
        
    # Manager reload should fallback to new empty index cleanly
    manager_fallback = IndexManager()
    assert manager_fallback.engine.total == 0
    logger.info("Edge Case: Missing Index File fallback -> OK")
    
    # Edge Case E: Corrupted Index File
    # Write garbage byte content to simulate corruption
    with open(idx_file, "wb") as f:
        f.write(b"CORRUPTED JUNKB YTES DATA")
        
    # Reload should catch error and fall back to clean index
    manager_corrupt = IndexManager()
    assert manager_corrupt.engine.total == 0
    logger.info("Edge Case: Corrupted Index File fallback -> OK")
    
    # Cleanup corrupted index file
    if idx_file.exists():
        os.remove(idx_file)
        
    logger.info("Edge Cases: ALL PASSED SUCCESSFULLY\n" + "-"*50)

def run_all_tests():
    logger.info("Starting Milestone 6 automated tests...")
    test_index_pipeline()
    test_edge_cases()
    logger.info("ALL MILESTONE 6 TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_all_tests()
