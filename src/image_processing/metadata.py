import os
import datetime
from typing import Dict, Any, List

class OCRMetadataGenerator:
    """Generates structured metadata for processed OCR images."""

    @staticmethod
    def generate(
        file_path: str,
        original_width: int,
        original_height: int,
        ocr_engine: str,
        confidence: float,
        duration: float,
        preprocessing_meta: Dict[str, Any],
        blocks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compiles standard OCR metadata.
        
        Args:
            file_path: Absolute or relative path to the source image.
            original_width: Original width of the image before scaling.
            original_height: Original height of the image before scaling.
            ocr_engine: The name of the engine used ('EasyOCR' or 'Tesseract').
            confidence: The average confidence score (0.0 to 1.0).
            duration: The processing time in seconds.
            preprocessing_meta: Preprocessing stages summary.
            blocks: Detailed text segments with bounding boxes and segment confidences.
            
        Returns:
            A dictionary containing structural image metadata.
        """
        return {
            "source": os.path.basename(file_path),
            "file_path": os.path.abspath(file_path),
            "file_type": "image",
            "dimensions": {
                "width": original_width,
                "height": original_height
            },
            "ocr_engine": ocr_engine,
            "confidence": round(confidence, 4),
            "processing_duration_sec": round(duration, 4),
            "extracted_at": datetime.datetime.utcnow().isoformat() + "Z",
            "preprocessing": preprocessing_meta,
            "blocks": blocks  # Include block-level bounding boxes, text, and confidences
        }
