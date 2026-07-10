import os
from pathlib import Path
from typing import Tuple
from PIL import Image
from src.config.settings import UPLOAD_LIMITS
from src.utils.logger import logger

class ImageValidator:
    """Handles verification and integrity checks for uploaded image files before OCR."""

    @staticmethod
    def validate(file_path: str) -> Tuple[bool, str]:
        """Validates that the file is an image, supports the format, and is not corrupted.
        
        Args:
            file_path: Path to the image file to validate.
            
        Returns:
            A tuple of (is_valid: bool, error_message: str)
        """
        path = Path(file_path)
        
        # 1. Existence Check
        if not path.exists():
            msg = f"File does not exist: {file_path}"
            logger.warning(msg)
            return False, msg

        # 2. File Format Check
        ext = path.suffix.lower().lstrip(".")
        supported_formats = ["jpg", "jpeg", "png", "bmp", "tiff"]
        if ext not in supported_formats:
            msg = f"Unsupported image extension: .{ext}. Supported: {', '.join(supported_formats)}"
            logger.warning(f"Validation failed for {path.name}: {msg}")
            return False, msg

        # 3. File Size Check
        size_limit = UPLOAD_LIMITS.get("image", 5 * 1024 * 1024)
        file_size = os.path.getsize(file_path)
        if file_size > size_limit:
            msg = f"Image size ({file_size / (1024*1024):.2f} MB) exceeds size limit ({size_limit / (1024*1024):.1f} MB)"
            logger.warning(f"Validation failed for {path.name}: {msg}")
            return False, msg

        # 4. Open and Verify File Integrity (using PIL)
        try:
            with Image.open(file_path) as img:
                img.verify()  # verify checks if the file is corrupted
            
            # Re-open because verify() closes the file but can leave it in an unusable state
            with Image.open(file_path) as img:
                _ = img.size  # access dimensions to force load headers
            
            return True, ""
        except Exception as e:
            msg = f"Corrupted image or invalid format. PIL Error: {str(e)}"
            logger.error(f"Validation failed for {path.name}: {msg}")
            return False, msg
