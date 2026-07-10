import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data & Models Directories
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DATABASE_DIR = DATA_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "metadata.db"
INDEX_PATH = DATA_DIR / "indices" / "faiss.index"

MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DATA_DIR, UPLOAD_DIR, DATABASE_DIR, MODELS_DIR, LOGS_DIR, INDEX_PATH.parent, MODELS_DIR / "easyocr"]:
    directory.mkdir(parents=True, exist_ok=True)

# Configuration Parameters
DEFAULT_CHUNK_SIZE = 500  # Default character limit per chunk
DEFAULT_CHUNK_OVERLAP = 100  # Overlap between consecutive chunks
MAX_CHUNK_LENGTH = 1000  # Absolute maximum limit for a single chunk

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
WHISPER_MODEL_NAME = "base"

# OCR Settings
OCR_USE_GPU = False  # EasyOCR CPU execution by default
OCR_EASYOCR_MODEL_DIR = MODELS_DIR / "easyocr"

SUPPORTED_LANGUAGES = {
    "en": "English",
    "ta": "Tamil"
}

# Configurable Upload limits in MB
MAX_PDF_SIZE_MB = 100
MAX_DOCX_SIZE_MB = 50
MAX_IMAGE_SIZE_MB = 20
MAX_AUDIO_SIZE_MB = 200

# Upload Limits (in bytes)
UPLOAD_LIMITS = {
    "pdf": MAX_PDF_SIZE_MB * 1024 * 1024,
    "docx": MAX_DOCX_SIZE_MB * 1024 * 1024,
    "image": MAX_IMAGE_SIZE_MB * 1024 * 1024,
    "audio": MAX_AUDIO_SIZE_MB * 1024 * 1024
}

# Logging settings
LOG_FILE_PATH = LOGS_DIR / "app.log"
