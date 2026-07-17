import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path

# Add workspace directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.retrieval.metadata import RetrievalChunk, RetrievalResult
from src.llm import LLMManager, GroundedAnswer
from src.llm.prompt_builder import PromptBuilder
from src.llm.citation_formatter import CitationFormatter
from src.utils.logger import logger

class MockOllamaClient:
    """Mock HTTP client simulating server and model states offline."""
    def __init__(self, running=True, model_installed=True, response_text="Mocked grounded answer."):
        self.running = running
        self.model_installed = model_installed
        self.response_text = response_text
        self.base_url = "http://localhost:11434"
        
    def is_server_running(self):
        return self.running
        
    def is_model_installed(self, model_name):
        return self.model_installed
        
    def generate(self, model_name, prompt):
        return self.response_text, {"prompt_eval_count": 25, "eval_count": 50}, {
            "model": model_name,
            "prompt_length": len(prompt),
            "context_limit": 4000,
            "request_time": "2026-07-17T15:00:00Z",
            "inference_time_ms": 10.0,
            "http_status": 200,
            "returned_characters": len(self.response_text),
            "ollama_error": None
        }

def test_grounded_answer_success():
    logger.info("=== Testing Grounded Answer (English) ===")
    
    # 1. Setup matching chunk and result
    c1 = RetrievalChunk(
        chunk_id="c1", document_id="doc1", source_file="sample.pdf",
        source_reference="Page 2", similarity_score=0.92,
        confidence="High", chunk_text="The quick brown fox jumps over the lazy dog."
    )
    
    ret_res = RetrievalResult(
        success=True, reason="SUCCESS", message="Retrieval OK",
        query="What does the fox do?", language="en", language_confidence=0.99,
        retrieved_chunks=[c1], combined_context=c1.chunk_text, total_chunks=1
    )
    
    manager = LLMManager()
    manager.client = MockOllamaClient(response_text="The fox jumps over the lazy dog.")
    
    ans = manager.generate_grounded_answer(ret_res)
    assert ans.success is True
    assert "Sources:" in ans.answer
    assert "sample.pdf (Page 2)" in ans.answer
    assert ans.token_statistics["eval_count"] == 50
    logger.info("Grounded Answer Success: OK")

def test_tamil_mixed_queries():
    logger.info("=== Testing Multilingual Questions ===")
    c_ta = RetrievalChunk(
        chunk_id="c_ta", document_id="doc2", source_file="tamil.docx",
        source_reference="Paragraph 3", similarity_score=0.88,
        confidence="High", chunk_text="செயற்கை நுண்ணறிவுத் தொழில்நுட்பம் வேகமாக வளர்கிறது."
    )
    
    ret_res = RetrievalResult(
        success=True, reason="SUCCESS", message="Retrieval OK",
        query="செயற்கை நுண்ணறிவு எப்படி வளர்கிறது?", language="ta", language_confidence=0.95,
        retrieved_chunks=[c_ta], combined_context=c_ta.chunk_text, total_chunks=1
    )
    
    manager = LLMManager()
    manager.client = MockOllamaClient(response_text="செயற்கை நுண்ணறிவுத் தொழில்நுட்பம் வேகமாக வளர்கிறது.")
    
    ans = manager.generate_grounded_answer(ret_res)
    assert ans.success is True
    assert "tamil.docx" in ans.answer
    logger.info("Multilingual groundings check: OK")

def test_no_context_retrieval():
    logger.info("=== Testing No-Context Retrieval Fallback ===")
    
    # Retrieval failed (no chunks match threshold)
    ret_res = RetrievalResult(
        success=False, reason="NO_RELEVANT_CONTEXT", message="No matches",
        query="Out of scope query", language="en", language_confidence=1.0,
        retrieved_chunks=[], combined_context="", total_chunks=0
    )
    
    manager = LLMManager()
    manager.client = MockOllamaClient()
    
    ans = manager.generate_grounded_answer(ret_res)
    assert ans.success is False
    assert ans.reason == "NO_RELEVANT_CONTEXT"
    assert "Please upload documents related to this topic." in ans.answer
    logger.info("No-Context Fallback boundary check: OK")

def test_pre_inference_availability():
    logger.info("=== Testing Ollama/Model Availability ===")
    c1 = RetrievalChunk(
        chunk_id="c1", document_id="doc1", source_file="sample.pdf",
        source_reference="Page 2", similarity_score=0.92,
        confidence="High", chunk_text="Context content."
    )
    ret_res = RetrievalResult(
        success=True, reason="SUCCESS", message="Retrieval OK",
        query="Query text", language="en", language_confidence=1.0,
        retrieved_chunks=[c1], combined_context="Context content.", total_chunks=1
    )
    
    # Case A: Ollama server offline
    manager = LLMManager()
    manager.client = MockOllamaClient(running=False)
    ans = manager.generate_grounded_answer(ret_res)
    assert ans.success is False
    assert ans.reason == "OLLAMA_UNAVAILABLE"
    assert "Please start Ollama" in ans.answer
    
    # Case B: Model not installed
    manager.client = MockOllamaClient(model_installed=False)
    ans2 = manager.generate_grounded_answer(ret_res)
    assert ans2.success is False
    assert ans2.reason == "MODEL_MISSING"
    assert "not installed in your local Ollama registry" in ans2.answer
    logger.info("Service Offline and Model Missing check: OK")

def test_long_context_truncation():
    logger.info("=== Testing Long Context Truncation ===")
    
    # Setup chunks of size 3000 chars each.
    # Total context characters limit is 4000 (from settings.py)
    # The builder should include the first chunk, but skip the second one since adding it would cross 4000.
    c1 = RetrievalChunk(
        chunk_id="c1", document_id="doc1", source_file="c1.pdf",
        source_reference="Page 1", similarity_score=0.90, confidence="High",
        chunk_text="A" * 3000
    )
    c2 = RetrievalChunk(
        chunk_id="c2", document_id="doc1", source_file="c2.pdf",
        source_reference="Page 2", similarity_score=0.85, confidence="High",
        chunk_text="B" * 2000
    )
    
    prompt, context, count = PromptBuilder.build("Query", [c1, c2], max_context_chars=4000)
    assert count == 1
    assert "c1.pdf" in context
    assert "c2.pdf" not in context
    logger.info("Context length truncation: OK")

def test_duplicate_citations():
    logger.info("=== Testing Citation Deduplication ===")
    
    # Two chunks referencing the exact same page of a PDF
    c1 = RetrievalChunk(
        chunk_id="c1", document_id="doc1", source_file="sample.pdf",
        source_reference="Page 2", similarity_score=0.90, confidence="High",
        chunk_text="Text 1"
    )
    c2 = RetrievalChunk(
        chunk_id="c2", document_id="doc1", source_file="sample.pdf",
        source_reference="Page 2", similarity_score=0.88, confidence="High",
        chunk_text="Text 2"
    )
    
    cite_md, cite_list = CitationFormatter.format([c1, c2])
    assert len(cite_list) == 1
    assert cite_list[0] == "sample.pdf (Page 2)"
    logger.info("Citation deduplication check: OK")

def test_response_validators():
    logger.info("=== Testing Response Validators ===")
    
    # Case A: Empty response
    manager = LLMManager()
    manager.client = MockOllamaClient(response_text="")
    c1 = RetrievalChunk(
        chunk_id="c1", document_id="doc1", source_file="sample.pdf",
        source_reference="Page 2", similarity_score=0.90, confidence="High",
        chunk_text="The artificial intelligence model maps vectors."
    )
    ret_res = RetrievalResult(
        success=True, reason="SUCCESS", message="Retrieval OK",
        query="Query text", language="en", language_confidence=1.0,
        retrieved_chunks=[c1], combined_context="The artificial intelligence model maps vectors.", total_chunks=1
    )
    ans = manager.generate_grounded_answer(ret_res)
    assert ans.success is False
    assert ans.reason == "VALIDATION_FAILED"
    
    # Case B: Hallucinated entities check (introducing 'Microsoft' which is missing from context)
    manager.client = MockOllamaClient(response_text="Microsoft released an intelligence model.")
    ans_hallucinate = manager.generate_grounded_answer(ret_res)
    assert ans_hallucinate.success is False
    assert ans_hallucinate.reason == "VALIDATION_FAILED"
    assert "could not be validated against the retrieved documents" in ans_hallucinate.answer
    logger.info("Validation catches empty texts and hallucinated entity names -> OK")

def test_summaries_and_overviews_neighbors():
    logger.info("=== Testing Summaries & Overviews Neighbor Chunks ===")
    from src.retrieval import RetrievalManager, RetrievalConfig
    from src.vector_store import IndexManager
    import numpy as np
    
    idx_manager = IndexManager()
    idx_manager.clear_all()
    
    # Index 5 contiguous chunks for a document
    chunks = []
    from src.text_processing import Chunk
    for i in range(1, 6):
        chunks.append(Chunk(
            chunk_id=f"doc1_chunk_{i}",
            document_id="doc1",
            chunk_index=i-1,
            text=f"This is segment number {i} of the test document text.",
            metadata={
                "source_file": "neighbors_test.pdf",
                "source_reference": f"Page {i}",
                "embedding_norm": 1.0
            },
            character_count=50,
            token_estimate=10
        ))
        
    from src.embedding import EmbeddingManager
    emb_manager = EmbeddingManager()
    vectors, updated_chunks = emb_manager.embed_chunks(chunks)
    idx_manager.add_vectors_and_metadata(vectors, updated_chunks)
    
    ret_manager = RetrievalManager(embedding_manager=emb_manager, index_manager=idx_manager)
    config = RetrievalConfig(top_k=2, similarity_threshold=0.5, retrieval_mode="Document Summary")
    
    result = ret_manager.search("segment number 3", config=config)
    assert result.success is True
    assert result.intent == "SUMMARY"
    
    assert len(result.retrieved_chunks) == 5
    texts = [c.chunk_text for c in result.retrieved_chunks]
    assert "segment number 1" in texts[0]
    assert "segment number 5" in texts[-1]
    logger.info("Summaries & Overviews Neighbors fetch check: OK")

def test_low_confidence_fallback():
    logger.info("=== Testing Low-Confidence Retrieval Fallback ===")
    from src.retrieval import RetrievalManager, RetrievalConfig
    from src.vector_store import IndexManager
    import numpy as np
    
    idx_manager = IndexManager()
    idx_manager.clear_all()
    
    from src.text_processing import Chunk
    c = Chunk(
        chunk_id="fallback_chunk",
        document_id="doc_fb",
        chunk_index=0,
        text="Special target secret text for testing confidence threshold fallbacks.",
        metadata={
            "source_file": "fb.pdf",
            "source_reference": "Page 1",
            "embedding_norm": 1.0
        },
        character_count=70,
        token_estimate=15
    )
    from src.embedding import EmbeddingManager
    emb_manager = EmbeddingManager()
    vector, updated_chunks = emb_manager.embed_chunks([c])
    idx_manager.add_vectors_and_metadata(vector, updated_chunks)
    
    ret_manager = RetrievalManager(embedding_manager=emb_manager, index_manager=idx_manager)
    config = RetrievalConfig(top_k=1, similarity_threshold=0.99, retrieval_mode="Semantic Search")
    
    result = ret_manager.search("something completely unrelated", config=config)
    assert result.success is True
    assert result.is_low_confidence is True
    assert len(result.retrieved_chunks) == 1
    assert "Special target secret text" in result.retrieved_chunks[0].chunk_text
    logger.info("Low-confidence fallback check: OK")

def test_rebuild_based_deletions():
    logger.info("=== Testing Rebuild-Based Document Deletions ===")
    from src.vector_store import IndexManager
    from src.text_processing import Chunk
    import numpy as np
    
    idx_manager = IndexManager()
    idx_manager.clear_all()
    
    c_a = Chunk(
        chunk_id="docA_chunk_1", document_id="docA", chunk_index=0,
        text="Text from document A content.",
        metadata={
            "source_file": "docA.pdf",
            "source_reference": "Page 1"
        },
        character_count=30, token_estimate=10
    )
    c_b = Chunk(
        chunk_id="docB_chunk_1", document_id="docB", chunk_index=0,
        text="Text from document B content.",
        metadata={
            "source_file": "docB.pdf",
            "source_reference": "Page 1"
        },
        character_count=30, token_estimate=10
    )
    
    from src.embedding import EmbeddingManager
    emb_manager = EmbeddingManager()
    v_a, updated_a = emb_manager.embed_chunks([c_a])
    v_b, updated_b = emb_manager.embed_chunks([c_b])
    
    idx_manager.add_vectors_and_metadata(v_a, updated_a)
    idx_manager.add_vectors_and_metadata(v_b, updated_b)
    
    assert idx_manager.engine.total == 2
    
    idx_manager.delete_document("docA")
    assert idx_manager.engine.total == 1
    
    stats = idx_manager.metadata_store.get_index_stats()
    assert stats["total_documents"] == 1
    assert stats["total_vectors"] == 1
    
    all_docs = idx_manager.metadata_store.get_all_documents()
    assert len(all_docs) == 1
    assert all_docs[0]["document_hash"] == "docB"
    logger.info("Rebuild-based document deletion and FAISS synchronization check: OK")

def run_tests():
    logger.info("Starting Milestone 8 automated tests...")
    test_grounded_answer_success()
    test_tamil_mixed_queries()
    test_no_context_retrieval()
    test_pre_inference_availability()
    test_long_context_truncation()
    test_duplicate_citations()
    test_response_validators()
    test_summaries_and_overviews_neighbors()
    test_low_confidence_fallback()
    test_rebuild_based_deletions()
    logger.info("ALL MILESTONE 8 TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
