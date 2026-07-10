import time
import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Any
from src.image_processing.base import BaseOCREngine, OCRResult
from src.config.settings import OCR_USE_GPU, OCR_EASYOCR_MODEL_DIR
from src.utils.logger import logger

class EasyOCREngine(BaseOCREngine):
    """OCR Engine using EasyOCR as the primary extraction tool."""
    
    def __init__(self):
        self.reader = None
        self.gpu = OCR_USE_GPU
        self.model_dir = str(OCR_EASYOCR_MODEL_DIR)

    def _init_reader(self):
        """Initializes the EasyOCR reader lazily to save startup resources."""
        if self.reader is None:
            logger.info("Monkey-patching EasyOCR Tamil config to resolve size mismatch bug...")
            try:
                import easyocr.config
                correct_tamil_chars = (
                    "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "\u0b83\u0b85\u0b86\u0b87\u0b88\u0b89\u0b8a\u0b8e\u0b8f\u0b90\u0b92\u0b93\u0b94"
                    "\u0b95\u0b99\u0b9a\u0b9c\u0b9e\u0b9f\u0ba3\u0ba4\u0ba8\u0ba9\u0baa\u0bae\u0baf"
                    "\u0bb0\u0bb1\u0bb2\u0bb3\u0bb4\u0bb5\u0bb7\u0bb8\u0bb9\u0bbe\u0bbf\u0bc0"
                    "\u0bc1\u0bc2\u0bc6\u0bc7\u0bc8\u0bca\u0bcb\u0bcc\u0bcd"
                )
                easyocr.config.recognition_models['gen1']['tamil_g1']['characters'] = correct_tamil_chars
            except Exception as e:
                logger.warning(f"Failed to apply EasyOCR Tamil patch: {e}")

            logger.info(f"Initializing EasyOCR Reader (gpu={self.gpu}, cache_dir={self.model_dir})...")
            import easyocr
            # We initialize EasyOCR for both English and Tamil
            self.reader = easyocr.Reader(
                lang_list=["en", "ta"],
                gpu=self.gpu,
                model_storage_directory=self.model_dir,
                download_enabled=True
            )
            logger.info("EasyOCR Reader successfully initialized.")

    def extract_text(self, image_path: str, lang: str = "eng+tam") -> OCRResult:
        """Extracts text and word/phrase bounding boxes from an image using EasyOCR.
        
        Args:
            image_path: Path to the image file.
            lang: Language parameter (ignored in EasyOCR, configured at init).
            
        Returns:
            An OCRResult holding text and parsed block structures.
        """
        self._init_reader()
        start_time = time.time()
        
        logger.info(f"Executing EasyOCR on: {image_path}")
        try:
            # easyocr.readtext accepts file path directly
            # Returns list of tuples: (bbox, text, confidence)
            # bbox is [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
            results = self.reader.readtext(image_path)
            
            blocks = []
            text_lines = []
            conf_sum = 0.0
            
            for bbox, text_str, conf in results:
                # Ensure all bbox points are standard ints for JSON metadata serialization
                clean_bbox = [[int(coord[0]), int(coord[1])] for coord in bbox]
                blocks.append({
                    "text": text_str.strip(),
                    "bbox": clean_bbox,
                    "confidence": float(conf)
                })
                text_lines.append(text_str.strip())
                conf_sum += float(conf)
                
            full_text = "\n".join(text_lines)
            avg_conf = conf_sum / len(results) if results else 0.0
            duration = time.time() - start_time
            
            logger.info(f"EasyOCR parsing complete. Extracted {len(blocks)} blocks in {duration:.2f}s (avg conf={avg_conf:.2f}).")
            
            return OCRResult(
                text=full_text,
                confidence=avg_conf,
                processing_time=duration,
                language="en+ta",
                engine_used="EasyOCR",
                blocks=blocks
            )
        except Exception as e:
            logger.error(f"EasyOCR extraction crashed: {str(e)}")
            raise e


class TesseractOCREngine(BaseOCREngine):
    """OCR Engine using Tesseract OCR as a fallback/optional extraction tool."""

    def extract_text(self, image_path: str, lang: str = "eng+tam") -> OCRResult:
        """Extracts text and groups bounding boxes into sequential line blocks using Tesseract.
        
        Args:
            image_path: Path to the image file.
            lang: Language code for Tesseract (e.g. 'eng+tam').
            
        Returns:
            An OCRResult holding text and structured line block metadata.
        """
        start_time = time.time()
        logger.info(f"Executing Tesseract OCR (lang={lang}) on: {image_path}")
        
        try:
            # Read image using OpenCV to verify loading
            img = cv2.imread(image_path)
            if img is None:
                raise IOError(f"Tesseract unable to load image file: {image_path}")
                
            # Get detailed word-level data
            data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
            
            # Group word entries by (block_num, par_num, line_num) to form cohesive line blocks
            lines = {}
            for i in range(len(data["text"])):
                text_str = data["text"][i].strip()
                conf_val = float(data["conf"][i])
                
                # Filter out container blocks and low confidence/empty text entries
                if not text_str or conf_val == -1:
                    continue
                    
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                if key not in lines:
                    lines[key] = []
                    
                left = int(data["left"][i])
                top = int(data["top"][i])
                width = int(data["width"][i])
                height = int(data["height"][i])
                
                lines[key].append({
                    "text": text_str,
                    "left": left,
                    "top": top,
                    "right": left + width,
                    "bottom": top + height,
                    "conf": conf_val / 100.0  # normalize to 0.0 - 1.0
                })
                
            blocks = []
            text_lines = []
            total_conf_sum = 0.0
            total_word_count = 0
            
            for key, words in lines.items():
                # Sort words left-to-right to guarantee correct reading order
                words_sorted = sorted(words, key=lambda w: w["left"])
                
                min_left = min(w["left"] for w in words_sorted)
                min_top = min(w["top"] for w in words_sorted)
                max_right = max(w["right"] for w in words_sorted)
                max_bottom = max(w["bottom"] for w in words_sorted)
                
                line_text = " ".join(w["text"] for w in words_sorted)
                line_conf = sum(w["conf"] for w in words_sorted) / len(words_sorted)
                
                bbox = [
                    [min_left, min_top],
                    [max_right, min_top],
                    [max_right, max_bottom],
                    [min_left, max_bottom]
                ]
                
                blocks.append({
                    "text": line_text.strip(),
                    "bbox": bbox,
                    "confidence": float(line_conf)
                })
                text_lines.append(line_text.strip())
                total_conf_sum += sum(w["conf"] for w in words_sorted)
                total_word_count += len(words_sorted)
                
            full_text = "\n".join(text_lines)
            avg_conf = total_conf_sum / total_word_count if total_word_count > 0 else 0.0
            duration = time.time() - start_time
            
            logger.info(f"Tesseract parsing complete. Extracted {len(blocks)} lines in {duration:.2f}s (avg conf={avg_conf:.2f}).")
            
            return OCRResult(
                text=full_text,
                confidence=avg_conf,
                processing_time=duration,
                language=lang.replace("+", ","),
                engine_used="Tesseract",
                blocks=blocks
            )
        except Exception as e:
            logger.error(f"Tesseract OCR extraction failed: {str(e)}")
            raise e
