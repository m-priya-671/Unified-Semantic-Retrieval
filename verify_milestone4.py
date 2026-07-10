import sys
from pathlib import Path

# Add workspace directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.ingestion.base import Document
from src.text_processing.unified_document import UnifiedDocument
from src.text_processing.chunk import Chunk
from src.text_processing.document_converter import DocumentConverter
from src.text_processing.chunking_engine import ChunkingEngine
from src.text_processing.validator import TextProcessingValidator
from src.utils.logger import logger

def test_pdf_conversion_and_chunking():
    logger.info("=== 1. TESTING PDF CONVERSION & CHUNKING ===")
    
    # Mock PDF: 2 pages
    doc_page1 = Document(
        text="This is sentence one on page one. This is sentence two on page one.",
        metadata={"page_number": 1, "total_pages": 2}
    )
    doc_page2 = Document(
        text="This is sentence three on page two. This is sentence four on page two.",
        metadata={"page_number": 2, "total_pages": 2}
    )
    
    # Convert
    ud = DocumentConverter.convert(
        documents=[doc_page1, doc_page2],
        source_file="sample.pdf",
        source_type="pdf",
        processing_time=0.1
    )
    
    assert TextProcessingValidator.validate_document(ud) is True
    assert ud.source_type == "pdf"
    assert ud.languages == [{"language": "en", "probability": 1.0}]
    assert len(ud.metadata["sentences"]) == 4
    
    # Chunk: target size 50 chars, overlap 10 chars
    chunks = ChunkingEngine.chunk_document(ud, chunk_size=80, chunk_overlap=15)
    
    assert len(chunks) > 0
    first_chunk = chunks[0]
    
    # Verify readable chunk ID
    assert first_chunk.chunk_id == f"{ud.document_id}_chunk_0000"
    assert first_chunk.document_id == ud.document_id
    assert first_chunk.character_count == len(first_chunk.text)
    assert first_chunk.token_estimate == len(first_chunk.text) // 4
    
    # Verify page citation reference
    assert "pages" in first_chunk.metadata
    assert "source_reference" in first_chunk.metadata
    assert first_chunk.metadata["source_reference"].startswith("Page")
    
    logger.info(f"PDF Chunking passed. Generated {len(chunks)} chunks.")
    logger.info(f"First Chunk ID: {first_chunk.chunk_id}, citation: {first_chunk.metadata['source_reference']}")
    logger.info("PDF tests: OK\n" + "-"*50)

def test_docx_table_preservation():
    logger.info("=== 2. TESTING DOCX TABLE PRESERVATION ===")
    
    # Mock DOCX: 1 paragraph, 1 table, 1 paragraph
    doc_p1 = Document(
        text="Introduction paragraph here.",
        metadata={"block_index": 1, "block_type": "paragraph"}
    )
    doc_tbl = Document(
        text="Col1 | Col2\nVal1 | Val2\nVal3 | Val4",
        metadata={"block_index": 2, "block_type": "table"}
    )
    doc_p2 = Document(
        text="Concluding paragraph here.",
        metadata={"block_index": 3, "block_type": "paragraph"}
    )
    
    ud = DocumentConverter.convert(
        documents=[doc_p1, doc_tbl, doc_p2],
        source_file="report.docx",
        source_type="docx",
        processing_time=0.05
    )
    
    # Assert table is mapped as a single sentence block to avoid splitting
    sentences = ud.metadata["sentences"]
    table_sentence = next(s for s in sentences if s["block_index"] == 2)
    assert table_sentence["text"] == "Col1 | Col2\nVal1 | Val2\nVal3 | Val4"
    logger.info("Table parsed as unified sentence entry: OK")
    
    # Chunking
    chunks = ChunkingEngine.chunk_document(ud, chunk_size=150, chunk_overlap=20)
    
    # Verify table block index is kept in chunk metadata
    tbl_chunk = next(c for c in chunks if "Col1" in c.text)
    assert 2 in tbl_chunk.metadata["block_ids"]
    assert "source_reference" in tbl_chunk.metadata
    assert "Block" in tbl_chunk.metadata["source_reference"]
    
    logger.info(f"DOCX Chunking passed. Generated {len(chunks)} chunks.")
    logger.info(f"Table Chunk metadata source reference: {tbl_chunk.metadata['source_reference']}")
    logger.info("DOCX tests: OK\n" + "-"*50)

def test_audio_timestamps():
    logger.info("=== 3. TESTING AUDIO TIMESTAMPS LINEAGE ===")
    
    # Mock audio parsed documents
    doc = Document(
        text="Hello, this is standard English dialogue. Second audio block segment.",
        metadata={
            "languages": [{"language": "en", "probability": 0.98}],
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Hello, this is standard English dialogue."},
                {"start": 2.5, "end": 6.2, "text": "Second audio block segment."}
            ]
        }
    )
    
    ud = DocumentConverter.convert(
        documents=[doc],
        source_file="recording.mp3",
        source_type="audio",
        processing_time=1.2
    )
    
    assert ud.languages[0]["language"] == "en"
    
    # Chunking
    chunks = ChunkingEngine.chunk_document(ud, chunk_size=100, chunk_overlap=10)
    
    assert len(chunks) > 0
    chunk = chunks[0]
    
    # Verify timestamps are preserved in metadata
    assert "start_time" in chunk.metadata
    assert "end_time" in chunk.metadata
    assert chunk.metadata["start_time"] == 0.0
    assert chunk.metadata["end_time"] == 6.2
    assert "Timestamp" in chunk.metadata["source_reference"]
    
    logger.info(f"Audio Chunking passed. Generated {len(chunks)} chunks.")
    logger.info(f"Audio Chunk citation reference: {chunk.metadata['source_reference']}")
    logger.info("Audio tests: OK\n" + "-"*50)

def test_tamil_sentence_splitting():
    logger.info("=== 4. TESTING TAMIL SENTENCE BOUNDARIES ===")
    
    tamil_text = "செயற்கை நுண்ணறிவுத் தொழில்நுட்பம் வேகமாக வளர்ந்து வருகிறது. இது உலகை மாற்றியமைக்கிறது!"
    
    doc = Document(
        text=tamil_text,
        metadata={"block_index": 1}
    )
    
    ud = DocumentConverter.convert(
        documents=[doc],
        source_file="tamil.txt",
        source_type="image",
        processing_time=0.2
    )
    
    sentences = ud.metadata["sentences"]
    assert len(sentences) == 2, f"Expected 2 sentences, got {len(sentences)}"
    assert sentences[0]["text"] == "செயற்கை நுண்ணறிவுத் தொழில்நுட்பம் வேகமாக வளர்ந்து வருகிறது."
    assert sentences[1]["text"] == "இது உலகை மாற்றியமைக்கிறது!"
    
    logger.info("Tamil sentence splitting boundary check: OK")
    logger.info("Tamil tests: OK\n" + "-"*50)

def run_all_tests():
    logger.info("Starting Milestone 4 automated tests...")
    test_pdf_conversion_and_chunking()
    test_docx_table_preservation()
    test_audio_timestamps()
    test_tamil_sentence_splitting()
    logger.info("ALL MILESTONE 4 TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_all_tests()
