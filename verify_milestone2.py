import os
import sys
import time
from pathlib import Path

# Add the workspace directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

from PIL import Image, ImageDraw, ImageFont
from src.config.settings import UPLOAD_DIR
from src.utils.logger import logger
from src.image_processing.image_processor import ImageProcessor

def get_system_font() -> str:
    """Attempts to find a system font that supports English and Tamil characters on Windows."""
    candidates = [
        "C:\\Windows\\Fonts\\Nirmala.ttf",  # Supports all Indic scripts
        "C:\\Windows\\Fonts\\latha.ttf",    # Standard Tamil font
        "C:\\Windows\\Fonts\\arial.ttf"     # Standard Arial (English fallback)
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""

def create_mock_ocr_images():
    """Generates mock images containing text for testing OCR capabilities."""
    logger.info("Generating mock test images for Milestone 2...")
    
    eng_image_path = UPLOAD_DIR / "ocr_test_english.png"
    tam_image_path = UPLOAD_DIR / "ocr_test_tamil.png"
    mix_image_path = UPLOAD_DIR / "ocr_test_mixed.png"
    
    font_path = get_system_font()
    
    # Define text strings
    eng_text = "Unified Semantic Retrieval System OCR Verification"
    # Tamil: "செயற்கை நுண்ணறிவுத் தொழில்நுட்பம்" (Artificial Intelligence Technology)
    tam_text = "செயற்கை நுண்ணறிவுத் தொழில்நுட்பம்"
    mix_text = "Offline RAG சிஸ்டம் OCR Test"

    def draw_text_to_image(text_to_draw: str, target_path: Path):
        # Create a white canvas
        img = Image.new("RGB", (1000, 180), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Load font
        if font_path:
            try:
                font = ImageFont.truetype(font_path, 28)
            except Exception as e:
                logger.warning(f"Failed to load TrueType font from {font_path}: {e}. Using default.")
                font = ImageFont.load_default()
        else:
            logger.warning("No system font found. Defaulting to standard font.")
            font = ImageFont.load_default()
            
        # Draw text centered vertically and padded left
        draw.text((40, 70), text_to_draw, fill=(0, 0, 0), font=font)
        img.save(str(target_path))
        logger.info(f"Created mock image: {target_path.name}")
        
    draw_text_to_image(eng_text, eng_image_path)
    draw_text_to_image(tam_text, tam_image_path)
    draw_text_to_image(mix_text, mix_image_path)
    
    return eng_image_path, tam_image_path, mix_image_path

def test_refinements():
    """Verifies Milestone 2 refinements (upload limits, file validation message schema)."""
    logger.info("=== TESTING MILESTONE 2 CONFIG & VALIDATION REFINEMENTS ===")
    
    # 1. Assert new constants are defined in settings
    from src.config import settings
    from src.utils.file_manager import FileManager
    
    assert hasattr(settings, "MAX_PDF_SIZE_MB"), "MAX_PDF_SIZE_MB not found"
    assert hasattr(settings, "MAX_DOCX_SIZE_MB"), "MAX_DOCX_SIZE_MB not found"
    assert hasattr(settings, "MAX_IMAGE_SIZE_MB"), "MAX_IMAGE_SIZE_MB not found"
    assert hasattr(settings, "MAX_AUDIO_SIZE_MB"), "MAX_AUDIO_SIZE_MB not found"
    
    assert settings.MAX_PDF_SIZE_MB == 100
    assert settings.MAX_DOCX_SIZE_MB == 50
    assert settings.MAX_IMAGE_SIZE_MB == 20
    assert settings.MAX_AUDIO_SIZE_MB == 200
    logger.info("Configuration size constants: OK")
    
    # 2. Test FileManager.validate_file on valid file size
    valid_size = 5 * 1024 * 1024  # 5 MB
    is_valid, details = FileManager.validate_file("test.pdf", valid_size)
    assert is_valid is True, "Valid PDF file validation failed"
    assert details == {}, f"Expected empty dict, got {details}"
    
    # 3. Test FileManager.validate_file on oversized file
    oversized_size = 101 * 1024 * 1024  # 101 MB (limit is 100 MB)
    is_valid, details = FileManager.validate_file("oversized_test.pdf", oversized_size)
    assert is_valid is False, "Oversized PDF file validation should fail"
    assert isinstance(details, dict), "Validation details should be a dictionary"
    assert details["file_name"] == "oversized_test.pdf"
    assert details["size_mb"] == 101.0
    assert details["limit_mb"] == 100.0
    assert "exceeds" in details["reason"]
    assert "settings.py" in details["suggestion"]
    logger.info("Oversized file validation response: OK")
    
    # 4. Test FileManager.validate_file on unsupported extension
    is_valid, details = FileManager.validate_file("unsupported.txt", 1000)
    assert is_valid is False, "Unsupported format should fail validation"
    assert isinstance(details, dict)
    assert details["limit_mb"] == 0.0
    assert "not supported" in details["reason"]
    logger.info("Unsupported extension validation response: OK")
    logger.info("REFINEMENT UNIT TESTS COMPLETED SUCCESSFULLY!")

def run_tests():
    """Tests the entire image parser pipeline on mock images."""
    test_refinements()
    eng_path, tam_path, mix_path = create_mock_ocr_images()
    
    processor = ImageProcessor()
    
    # 1. Test English Ingestion
    logger.info("=== TESTING ENGLISH IMAGE OCR ===")
    eng_docs = processor.parse(str(eng_path))
    assert len(eng_docs) == 1, "Expected exactly 1 document"
    eng_doc = eng_docs[0]
    logger.info(f"Engine Used: {eng_doc.metadata['ocr_engine']}")
    logger.info(f"Avg Confidence: {eng_doc.metadata['confidence']}")
    logger.info(f"Extracted Text: {eng_doc.text}")
    logger.info(f"Metadata blocks count: {len(eng_doc.metadata['blocks'])}")
    
    # Perform soft checks to see if text was extracted
    # (OCR results might have small errors, so we check for keywords)
    assert "Semantic" in eng_doc.text or "Retrieval" in eng_doc.text or "Verification" in eng_doc.text, \
        f"Failed to extract English keywords. Text got: {eng_doc.text}"
    logger.info("English image parsing assertion check: SUCCESS")
    logger.info("-" * 50)
    
    # 2. Test Tamil Ingestion
    logger.info("=== TESTING TAMIL IMAGE OCR ===")
    tam_docs = processor.parse(str(tam_path))
    assert len(tam_docs) == 1, "Expected exactly 1 document"
    tam_doc = tam_docs[0]
    logger.info(f"Engine Used: {tam_doc.metadata['ocr_engine']}")
    logger.info(f"Avg Confidence: {tam_doc.metadata['confidence']}")
    logger.info(f"Extracted Text: {tam_doc.text}")
    
    # Tamil keyword presence (at least some character recognition)
    # Check for basic Tamil character sequences if possible
    # We do a print warning if empty, but we assert text is non-empty
    assert len(tam_doc.text.strip()) > 0, "Tamil OCR returned empty text."
    logger.info("Tamil image parsing assertion check: SUCCESS")
    logger.info("-" * 50)
    
    # 3. Test Mixed Ingestion
    logger.info("=== TESTING MIXED ENGLISH + TAMIL OCR ===")
    mix_docs = processor.parse(str(mix_path))
    assert len(mix_docs) == 1, "Expected exactly 1 document"
    mix_doc = mix_docs[0]
    logger.info(f"Engine Used: {mix_doc.metadata['ocr_engine']}")
    logger.info(f"Avg Confidence: {mix_doc.metadata['confidence']}")
    logger.info(f"Extracted Text: {mix_doc.text}")
    
    assert "Offline" in mix_doc.text or "OCR" in mix_doc.text or len(mix_doc.text.strip()) > 5, \
        "Mixed language OCR failed."
    logger.info("Mixed language image parsing assertion check: SUCCESS")
    logger.info("-" * 50)
    
    logger.info("OCR Metadata block details:")
    # Print the coordinates of first block for verification
    if mix_doc.metadata['blocks']:
        first_block = mix_doc.metadata['blocks'][0]
        logger.info(f"Sample block bounding box: {first_block['bbox']}")
        logger.info(f"Sample block text: {first_block['text']}")
        logger.info(f"Sample block confidence: {first_block['confidence']}")
        
    logger.info("ALL MILESTONE 2 TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
