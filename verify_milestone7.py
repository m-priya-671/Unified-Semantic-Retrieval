import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import sqlite3
from pathlib import Path

# Add workspace directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.text_processing.chunk import Chunk
from src.embedding import EmbeddingManager
from src.vector_store import IndexManager
from src.retrieval import RetrievalManager, RetrievalConfig
from src.retrieval.language_detector import LanguageDetector
from src.utils.logger import logger

class MockEmbeddingEngine:
    """Mock EmbeddingEngine to simulate queries without loading PyTorch weights."""
    def __init__(self, vector: np.ndarray):
        self.vector = vector
        self.dimension = 384
        
    def generate(self, texts):
        return [self.vector]

def setup_mock_index() -> IndexManager:
    """Sets up a clean mock index with predictable vectors for testing."""
    manager = IndexManager()
    manager.clear_all()
    
    # Ingest 3 chunks
    c1 = Chunk(
        chunk_id="doc1_chunk_0000",
        document_id="doc1",
        chunk_index=0,
        text="The artificial intelligence pipeline processes semantic retrieval queries.",
        metadata={"source_file": "doc1.pdf", "source_reference": "Page 1"},
        token_estimate=10,
        character_count=71
    )
    c2 = Chunk(
        chunk_id="doc2_chunk_0000",
        document_id="doc2",
        chunk_index=0,
        text="செயற்கை நுண்ணறிவுத் தொழில்நுட்பம் வேகமாக வளர்கிறது.",
        metadata={"source_file": "doc2.pdf", "source_reference": "Page 2"},
        token_estimate=12,
        character_count=52
    )
    c3 = Chunk(
        chunk_id="doc3_chunk_0000",
        document_id="doc3",
        chunk_index=0,
        text="Deep learning handles mixed input structures in Python and Tamil.",
        metadata={"source_file": "doc3.docx", "source_reference": "Paragraph 10"},
        token_estimate=12,
        character_count=65
    )
    
    # Generate constant/orthogonal unit vectors of size 384
    v1 = np.zeros(384, dtype=np.float32)
    v1[0] = 1.0  # Points to dim 0
    
    v2 = np.zeros(384, dtype=np.float32)
    v2[1] = 1.0  # Points to dim 1
    
    v3 = np.zeros(384, dtype=np.float32)
    v3[2] = 1.0  # Points to dim 2
    
    vectors = np.vstack([v1, v2, v3])
    manager.add_vectors_and_metadata(vectors, [c1, c2, c3])
    return manager

def test_language_detection():
    logger.info("=== Testing Language Detection ===")
    
    # English
    lang, conf = LanguageDetector.detect("What is semantic retrieval?")
    assert lang == "en"
    assert conf > 0.8
    
    # Tamil
    lang, conf = LanguageDetector.detect("செயற்கை நுண்ணறிவு என்றால் என்ன?")
    assert lang == "ta"
    assert conf > 0.8
    
    # Mixed
    lang, conf = LanguageDetector.detect("How does செயற்கை நுண்ணறிவு work?")
    assert lang == "mixed"
    logger.info(f"Language detection passed: Mixed query returned code '{lang}' with confidence {conf}")

def test_retrieval_edge_cases():
    logger.info("=== Testing Retrieval Edge Cases ===")
    idx_manager = setup_mock_index()
    ret_manager = RetrievalManager(index_manager=idx_manager)
    
    # 1. Empty query check
    res = ret_manager.search("")
    assert res.success is False
    assert res.reason == "EMPTY_QUERY"
    assert "cannot be empty" in res.message
    
    # 2. Maximum query length check
    long_query = "a" * 600
    res = ret_manager.search(long_query)
    assert res.success is False
    assert res.reason == "QUERY_TOO_LONG"
    assert "exceeds maximum limit" in res.message
    
    # 3. Empty index check
    idx_manager.clear_all()
    res = ret_manager.search("What is AI?")
    assert res.success is False
    assert res.reason == "EMPTY_INDEX"
    assert "index is empty" in res.message
    logger.info("Edge Cases: Empty validation, Long check, and Empty Index fallback -> OK")

def test_retrieval_ranking_and_thresholding():
    logger.info("=== Testing Ranking & Thresholding ===")
    idx_manager = setup_mock_index()
    ret_manager = RetrievalManager(index_manager=idx_manager)
    
    # We want a query vector pointing to dim 0 (v1)
    mock_vector = np.zeros(384, dtype=np.float32)
    mock_vector[0] = 1.0
    
    # Set the mock engine
    ret_manager.engine.embedding_manager.engine = MockEmbeddingEngine(mock_vector)
    
    # Search with threshold 0.70
    res = ret_manager.search("Mock English Query matching c1", config=RetrievalConfig(similarity_threshold=0.70))
    assert res.success is True
    assert res.total_chunks == 1
    assert res.retrieved_chunks[0].chunk_id == "doc1_chunk_0000"
    assert res.retrieved_chunks[0].confidence == "High"
    
    # Test No-Result Scenario: Search with similarity threshold 0.95, but query has dim 0 = 0.5 (similarity will be 0.5)
    mock_low = np.zeros(384, dtype=np.float32)
    mock_low[0] = 0.5
    ret_manager.engine.embedding_manager.engine = MockEmbeddingEngine(mock_low)
    
    res_low = ret_manager.search("Query with low similarity", config=RetrievalConfig(similarity_threshold=0.70))
    assert res_low.success is False
    assert res_low.reason == "NO_RELEVANT_CONTEXT"
    assert "No relevant information matching your query was found" in res_low.message
    assert "Please upload documents related to this topic." in res_low.message
    assert res_low.combined_context == ""
    logger.info("Ranking, thresholding, and no-result recovery -> OK")

def test_missing_metadata():
    logger.info("=== Testing Missing Metadata Handling ===")
    idx_manager = setup_mock_index()
    ret_manager = RetrievalManager(index_manager=idx_manager)
    
    # Manually delete metadata row from SQLite to simulate corrupted sync
    with sqlite3.connect(idx_manager.metadata_store.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM vector_metadata WHERE faiss_id = 0")
        conn.commit()
        
    # Query vector matches dim 0 (which was mapped to faiss_id = 0)
    mock_vector = np.zeros(384, dtype=np.float32)
    mock_vector[0] = 1.0
    
    ret_manager.engine.embedding_manager.engine = MockEmbeddingEngine(mock_vector)
    
    # Search should run but return 0 chunks since mapped metadata was deleted
    res = ret_manager.search("Missing metadata target query", config=RetrievalConfig(similarity_threshold=0.50))
    assert res.success is False
    assert res.reason == "NO_RELEVANT_CONTEXT"
    assert res.total_chunks == 0
    logger.info("Missing database metadata gracefully resolved -> OK")

def run_tests():
    logger.info("Starting Milestone 7 automated checks...")
    test_language_detection()
    test_retrieval_edge_cases()
    test_retrieval_ranking_and_thresholding()
    test_missing_metadata()
    logger.info("ALL MILESTONE 7 VERIFICATION CHECKS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
