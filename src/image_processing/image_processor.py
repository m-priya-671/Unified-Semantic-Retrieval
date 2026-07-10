import os
import time
from typing import List
from PIL import Image as PILImage
from src.ingestion.base import BaseParser, Document
from src.image_processing.validator import ImageValidator
from src.image_processing.preprocessing import ImagePreprocessor
from src.image_processing.ocr_manager import OCRManager
from src.image_processing.metadata import OCRMetadataGenerator
from src.utils.logger import logger

class ImageProcessor(BaseParser):
    """Facade orchestrator that coordinates validation, preprocessing, OCR execution, 
    and metadata collation. Returns standardized Document lists.
    """

    def __init__(self):
        self.ocr_manager = OCRManager()

    def parse(self, file_path: str) -> List[Document]:
        """Validates, preprocesses, and extracts text from an image, returning a Document list.
        
        Args:
            file_path: Absolute or relative path to the image file.
            
        Returns:
            A list containing a single Document object with OCR text and metadata.
        """
        logger.info(f"Starting image parsing workflow for: {file_path}")
        start_time = time.time()
        
        # 1. Validate Image File
        is_valid, err_msg = ImageValidator.validate(file_path)
        if not is_valid:
            logger.error(f"Image validation failed for {file_path}: {err_msg}")
            raise ValueError(f"Invalid image file: {err_msg}")
            
        # Retrieve original dimensions before preprocessing
        try:
            with PILImage.open(file_path) as img:
                w_orig, h_orig = img.size
        except Exception as e:
            logger.error(f"Failed to read image dimensions from {file_path}: {e}")
            raise e

        temp_img_path = None
        ocr_result = None
        preprocessing_meta = {}

        try:
            # 2. Run Preprocessing optimized for EasyOCR (Grayscale/CLAHE/No Binarization)
            logger.info("Running image preprocessing optimized for EasyOCR...")
            temp_img_path, preprocessing_meta = ImagePreprocessor.preprocess_pipeline(
                file_path=file_path, 
                for_easy_ocr=True
            )
            
            # 3. Attempt OCR execution via OCRManager (EasyOCR by default)
            ocr_result = self.ocr_manager.run_ocr(temp_img_path)
            
        except Exception as ocr_err:
            logger.warning(
                f"EasyOCR pipeline failed: {str(ocr_err)}. "
                f"Re-routing image to Tesseract fallback pipeline..."
            )
            
            # Remove the EasyOCR temp image if it exists
            if temp_img_path and os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                except Exception:
                    pass

            try:
                # 4. Run Preprocessing optimized for Tesseract (Grayscale/CLAHE/Binarization)
                logger.info("Running image preprocessing optimized for Tesseract...")
                temp_img_path, preprocessing_meta = ImagePreprocessor.preprocess_pipeline(
                    file_path=file_path, 
                    for_easy_ocr=False
                )
                
                # 5. Run Tesseract via OCRManager
                ocr_result = self.ocr_manager.run_ocr(temp_img_path, force_engine="Tesseract")
                
            except Exception as fallback_err:
                logger.error(f"Fallback Tesseract pipeline also failed: {str(fallback_err)}")
                raise fallback_err
                
        finally:
            # Clean up temporary preprocessed image from data/uploads to save disk space
            if temp_img_path and os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                    logger.debug("Successfully cleaned up temporary preprocessed image file.")
                except Exception as cleanup_err:
                    logger.warning(f"Could not remove temporary image {temp_img_path}: {cleanup_err}")

        # 6. Collate Metadata & Package as Document representation
        total_duration = time.time() - start_time
        logger.info(f"Image parsing workflow completed successfully in {total_duration:.2f}s.")
        
        metadata = OCRMetadataGenerator.generate(
            file_path=file_path,
            original_width=w_orig,
            original_height=h_orig,
            ocr_engine=ocr_result.engine_used,
            confidence=ocr_result.confidence,
            duration=total_duration,
            preprocessing_meta=preprocessing_meta,
            blocks=ocr_result.blocks
        )
        
        return [Document(text=ocr_result.text, metadata=metadata)]
