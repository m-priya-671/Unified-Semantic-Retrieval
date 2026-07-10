import os
import cv2
import re
import numpy as np
import pytesseract
from PIL import Image
from typing import Tuple, Dict, Any
from src.utils.logger import logger
from pathlib import Path
from src.config.settings import UPLOAD_DIR

class ImagePreprocessor:
    """Provides image enhancement and correction tools to optimize OCR readability."""

    @staticmethod
    def load_image(file_path: str) -> np.ndarray:
        """Loads an image from disk using OpenCV."""
        # Using cv2.imdecode to support non-ASCII paths on Windows
        try:
            with open(file_path, "rb") as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode returned None")
            return img
        except Exception as e:
            logger.error(f"Failed to load image via OpenCV from {file_path}: {e}")
            raise e

    @staticmethod
    def resize(img: np.ndarray, max_dim: int = 1600) -> Tuple[np.ndarray, bool]:
        """Resizes the image if its dimensions exceed max_dim while keeping aspect ratio.
        
        Returns:
            A tuple of (resized_image: np.ndarray, was_resized: bool)
        """
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h} (scale={scale:.2f})")
            return resized, True
        return img, False

    @staticmethod
    def to_grayscale(img: np.ndarray) -> np.ndarray:
        """Converts a BGR image to Grayscale."""
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def reduce_noise(img: np.ndarray) -> np.ndarray:
        """Applies Gaussian Blur to smooth out high frequency scan/compression noise."""
        return cv2.GaussianBlur(img, (3, 3), 0)

    @staticmethod
    def enhance_contrast(img: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        # img must be grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    @staticmethod
    def adaptive_threshold(img: np.ndarray) -> np.ndarray:
        """Binarizes the image using Gaussian adaptive thresholding (ideal for Tesseract)."""
        return cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

    @staticmethod
    def correct_rotation(img: np.ndarray) -> Tuple[np.ndarray, int]:
        """Detects image rotation using Tesseract OSD and rotates the image to upright.
        
        Returns:
            A tuple of (rotated_image: np.ndarray, detected_angle: int)
        """
        logger.info("Analyzing image orientation...")
        angle = 0
        try:
            # Pytesseract OSD requires a RGB/PIL Image
            if len(img.shape) == 3:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
            else:
                pil_img = Image.fromarray(img)
                
            osd = pytesseract.image_to_osd(pil_img)
            
            # Parse the rotation angle
            match = re.search(r"Rotate:\s*(\d+)", osd)
            if match:
                angle = int(match.group(1))
                
            if angle in [90, 180, 270]:
                logger.info(f"Detected rotation: {angle} degrees. Re-orienting upright...")
                if angle == 90:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    img = cv2.rotate(img, cv2.ROTATE_180)
                elif angle == 270:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                logger.info("Orientation matches default (0 degrees).")
                angle = 0
        except Exception as e:
            # OSD fails if there is insufficient text to detect orientation script; fallback to no rotation.
            logger.debug(f"Orientation detection skipped (insufficient text or no Tesseract OSD): {str(e)}")
            angle = 0
            
        return img, angle

    @staticmethod
    def preprocess_pipeline(file_path: str, for_easy_ocr: bool = True) -> Tuple[str, Dict[str, Any]]:
        """Orchestrates the preprocessing sequence and writes a temporary preprocessed image.
        
        Args:
            file_path: Location of the input source image.
            for_easy_ocr: If True, halts before binarization since deep learning 
                          models (EasyOCR) perform better with gradient/grayscale.
                          If False, applies binarization (for Tesseract).
                          
        Returns:
            A tuple of (temp_image_path: str, preprocessing_metadata: dict)
        """
        logger.info(f"Starting image preprocessing pipeline for: {file_path}")
        
        # Load
        img = ImagePreprocessor.load_image(file_path)
        h_orig, w_orig = img.shape[:2]
        
        # 1. Resize
        img, resized = ImagePreprocessor.resize(img, max_dim=1600)
        h_new, w_new = img.shape[:2]
        
        # 2. Convert to Grayscale
        gray = ImagePreprocessor.to_grayscale(img)
        
        # 3. Correct Rotation
        rotated_gray, rotation_angle = ImagePreprocessor.correct_rotation(gray)
        
        # 4. Enhance Contrast
        contrast = ImagePreprocessor.enhance_contrast(rotated_gray)
        
        # 5. Reduce Noise
        clean = ImagePreprocessor.reduce_noise(contrast)
        
        # 6. Binarize if Tesseract
        final_img = clean
        binarized = False
        if not for_easy_ocr:
            final_img = ImagePreprocessor.adaptive_threshold(clean)
            binarized = True
            
        # Write preprocessed file to temp location
        temp_name = f"temp_preprocessed_{os.path.basename(file_path)}"
        temp_path = UPLOAD_DIR / temp_name
        
        # Using cv2.imencode to support Windows Unicode paths
        ext = Path(file_path).suffix
        success, encoded_img = cv2.imencode(ext, final_img)
        if not success:
            raise IOError(f"Failed to encode preprocessed image to {ext}")
            
        with open(temp_path, "wb") as f:
            f.write(encoded_img.tobytes())
            
        logger.info(f"Saved preprocessed image to: {temp_path}")
        
        prep_metadata = {
            "original_dimensions": {"width": w_orig, "height": h_orig},
            "processed_dimensions": {"width": w_new, "height": h_new},
            "resized": resized,
            "rotation_corrected_angle": rotation_angle,
            "contrast_enhanced": True,
            "noise_reduced": True,
            "binarized": binarized
        }
        
        return str(temp_path), prep_metadata
