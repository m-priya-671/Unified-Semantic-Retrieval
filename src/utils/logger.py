import logging
import sys
from src.config.settings import LOG_FILE_PATH

def setup_logger(name: str = "offline_rag") -> logging.Logger:
    """Sets up a logger with a file handler and a console handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Formatter configuration
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File Handler - using UTF-8 to support Tamil characters in logs
    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Reconfigure console streams on Windows to prevent UnicodeEncodeErrors with Tamil characters
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Default app-wide logger instance
logger = setup_logger()
