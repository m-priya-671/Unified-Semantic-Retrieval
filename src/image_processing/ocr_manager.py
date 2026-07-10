from typing import Optional
from src.image_processing.base import OCRResult, BaseOCREngine
from src.image_processing.ocr_engine import EasyOCREngine, TesseractOCREngine
from src.utils.logger import logger

class OCRManager:
    """Coordinates and executes OCR engines, managing primary run flows and fallbacks."""

    def __init__(self):
        self.easy_ocr = EasyOCREngine()
        self.tesseract = TesseractOCREngine()

    def run_ocr(
        self, 
        image_path: str, 
        lang: str = "eng+tam", 
        force_engine: Optional[str] = None
    ) -> OCRResult:
        """Executes OCR on the image, handling fallback logic or forced engine choices.
        
        Args:
            image_path: Path to the preprocessed image.
            lang: Language parameter (e.g. 'eng+tam').
            force_engine: Explicitly choose engine ('EasyOCR' or 'Tesseract'). 
                          If None, uses EasyOCR as primary with Tesseract fallback.
                          
        Returns:
            An OCRResult containing extracted text, scores, and block positions.
        """
        # 1. Forced Engine Execution
        if force_engine:
            engine_choice = force_engine.strip().lower()
            if engine_choice == "easyocr":
                logger.info("Forcing OCR engine selection: EasyOCR")
                return self.easy_ocr.extract_text(image_path, lang)
            elif engine_choice == "tesseract":
                logger.info("Forcing OCR engine selection: Tesseract")
                return self.tesseract.extract_text(image_path, lang)
            else:
                logger.warning(f"Unknown forced engine '{force_engine}'. Defaulting to dynamic selection.")

        # 2. Dynamic Selection / Graceful Fallback Flow
        logger.info("Starting dynamic OCR engine selection (Primary: EasyOCR)")
        try:
            # Attempt EasyOCR
            return self.easy_ocr.extract_text(image_path, lang)
        except Exception as e:
            logger.warning(
                f"Primary engine (EasyOCR) failed due to error: {str(e)}. "
                f"Gracefully falling back to Tesseract OCR..."
            )
            
            try:
                # Fallback to Tesseract
                return self.tesseract.extract_text(image_path, lang)
            except Exception as t_err:
                logger.error(f"Fallback engine (Tesseract) also failed: {str(t_err)}")
                raise RuntimeError(
                    f"OCR failed. Both primary (EasyOCR) and fallback (Tesseract) engines "
                    f"encountered errors. EasyOCR error: {str(e)} | Tesseract error: {str(t_err)}"
                )
