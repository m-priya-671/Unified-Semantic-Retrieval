import os
from pathlib import Path
from typing import Tuple, Optional, Any
from src.config.settings import UPLOAD_DIR, UPLOAD_LIMITS
from src.utils.logger import logger

class FileManager:
    """Manages file storage, validation, and metadata for uploaded user files."""

    @staticmethod
    def validate_file(file_name: str, file_size: int) -> Tuple[bool, dict]:
        """Validates the file extension and size, returning structured error data if invalid.
        
        Args:
            file_name: Name of the file.
            file_size: Size of the file in bytes.
            
        Returns:
            A tuple of (is_valid: bool, error_details: dict)
        """
        # Extract extension
        ext = file_name.split(".")[-1].lower() if "." in file_name else ""
        
        # Map specific extensions to validation categories in settings
        ext_to_category = {
            "pdf": "pdf",
            "docx": "docx",
            "png": "image",
            "jpg": "image",
            "jpeg": "image",
            "bmp": "image",
            "tiff": "image",
            "wav": "audio",
            "mp3": "audio",
            "m4a": "audio",
            "flac": "audio"
        }
        
        category = ext_to_category.get(ext)
        if not category or category not in UPLOAD_LIMITS:
            details = {
                "file_name": file_name,
                "size_mb": file_size / (1024 * 1024),
                "limit_mb": 0.0,
                "reason": f"The file extension '.{ext}' is not supported by the system.",
                "suggestion": "Please upload a supported format: PDF, DOCX, or Images (PNG, JPG, JPEG, BMP, TIFF)."
            }
            logger.warning(f"File validation failed for {file_name}: Unsupported extension '.{ext}'")
            return False, details

        limit = UPLOAD_LIMITS[category]
        if file_size > limit:
            details = {
                "file_name": file_name,
                "size_mb": file_size / (1024 * 1024),
                "limit_mb": limit / (1024 * 1024),
                "reason": f"The uploaded {ext.upper()} exceeds the configured upload limit.",
                "suggestion": f"Increase the upload limit in settings.py or upload a smaller file."
            }
            logger.warning(f"File validation failed for {file_name}: Size {details['size_mb']:.2f}MB exceeds limit {details['limit_mb']:.2f}MB")
            return False, details

        return True, {}

    @staticmethod
    def save_upload(file_name: str, file_bytes: bytes) -> Tuple[Optional[Path], Any]:
        """Validates and saves the uploaded file bytes to the upload directory.
        Handles naming conflicts by appending a counter.
        
        Args:
            file_name: Original name of the uploaded file.
            file_bytes: Content of the file in bytes.
            
        Returns:
            A tuple of (saved_path: Optional[Path], status_message_or_error_details: Any)
        """
        # Validate size and extension
        is_valid, err_details = FileManager.validate_file(file_name, len(file_bytes))
        if not is_valid:
            return None, err_details

        try:
            # Secure name and resolve collisions
            file_path = Path(file_name)
            base_name = file_path.stem
            suffix = file_path.suffix
            
            target_path = UPLOAD_DIR / file_name
            counter = 1
            while target_path.exists():
                new_name = f"{base_name}_{counter}{suffix}"
                target_path = UPLOAD_DIR / new_name
                counter += 1
            
            # Save file bytes
            with open(target_path, "wb") as f:
                f.write(file_bytes)
                
            logger.info(f"Successfully saved uploaded file: {target_path.name} ({len(file_bytes)} bytes)")
            return target_path, f"File saved successfully as {target_path.name}"
            
        except Exception as e:
            error_msg = f"Error saving file {file_name}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
