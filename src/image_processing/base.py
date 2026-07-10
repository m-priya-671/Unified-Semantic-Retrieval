from abc import ABC, abstractmethod
from typing import List, Dict, Any

class OCRResult:
    """Standardized result containing text, confidence, and metadata for OCR extractions."""
    
    def __init__(
        self, 
        text: str, 
        confidence: float, 
        processing_time: float, 
        language: str, 
        engine_used: str,
        blocks: List[Dict[str, Any]] = None
    ):
        """Initializes the OCRResult.
        
        Args:
            text: The full concatenated text extracted from the image.
            confidence: The average confidence score (0.0 to 1.0).
            processing_time: Time taken in seconds for OCR execution.
            language: Detected or configured language(s) (e.g., 'en+ta').
            engine_used: The engine name used for the extraction ('EasyOCR' or 'Tesseract').
            blocks: List of detailed blocks containing:
                    - 'text': text segment
                    - 'bbox': bounding box coordinates [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
                    - 'confidence': segment confidence score
        """
        self.text = text
        self.confidence = confidence
        self.processing_time = processing_time
        self.language = language
        self.engine_used = engine_used
        self.blocks = blocks if blocks is not None else []

    def __repr__(self) -> str:
        return (f"OCRResult(engine={self.engine_used}, text_len={len(self.text)}, "
                f"confidence={self.confidence:.3f}, time={self.processing_time:.2f}s, "
                f"blocks_count={len(self.blocks)})")


class BaseOCREngine(ABC):
    """Abstract Base Class for OCR engines."""
    
    @abstractmethod
    def extract_text(self, image_path: str, lang: str = "eng+tam") -> OCRResult:
        """Extracts text and coordinates from an image.
        
        Args:
            image_path: Path to the input image file.
            lang: Language parameter (e.g., 'eng+tam').
            
        Returns:
            An OCRResult containing extracted text, average confidence, and block positions.
        """
        pass
