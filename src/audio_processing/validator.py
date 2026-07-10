import os
import av
from pathlib import Path
from src.config.settings import UPLOAD_LIMITS
from src.utils.logger import logger

class AudioValidator:
    """Validates audio file characteristics like extension, size limit, and codec readability."""
    
    @staticmethod
    def validate(file_path: str) -> bool:
        """Runs the validation suite on the given audio file path.
        
        Args:
            file_path: Absolute or relative path to the audio file.
            
        Returns:
            True if valid, raises an exception (ValueError or FileNotFoundError) otherwise.
        """
        logger.info(f"Validating audio file: {file_path}")
        
        # 1. Check path exists
        if not os.path.exists(file_path):
            msg = f"Audio file not found: {file_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
            
        path = Path(file_path)
        ext = path.suffix.lower().lstrip(".")
        
        # 2. Check extension support
        supported_formats = ["mp3", "wav", "m4a", "flac"]
        if ext not in supported_formats:
            msg = f"Unsupported audio extension: .{ext}. Supported formats: {', '.join(supported_formats)}"
            logger.error(msg)
            raise ValueError(msg)
            
        # 3. Check file size limits
        file_size = os.path.getsize(file_path)
        limit = UPLOAD_LIMITS.get("audio", 200 * 1024 * 1024)
        if file_size > limit:
            limit_mb = limit / (1024 * 1024)
            size_mb = file_size / (1024 * 1024)
            msg = f"Audio file size ({size_mb:.2f} MB) exceeds the limit of {limit_mb:.1f} MB for .{ext} files."
            logger.error(msg)
            raise ValueError(msg)
            
        # 4. Check for header corruption using PyAV
        try:
            with av.open(file_path) as container:
                # Must contain at least one audio stream
                audio_streams = [s for s in container.streams if s.type == "audio"]
                if not audio_streams:
                    msg = f"File {file_path} does not contain any valid audio streams."
                    logger.error(msg)
                    raise ValueError(msg)
                    
                # Inspect first stream's properties
                stream = audio_streams[0]
                if stream.duration is None and stream.frames == 0:
                    msg = f"Audio stream in {file_path} is empty or unreadable."
                    logger.error(msg)
                    raise ValueError(msg)
                    
        except Exception as e:
            msg = f"Audio file header is corrupted or unreadable: {str(e)}"
            logger.error(msg)
            raise ValueError(msg)
            
        logger.info(f"Audio validation successful: {file_path}")
        return True
