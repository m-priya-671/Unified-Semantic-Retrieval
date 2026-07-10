import sqlite3
import numpy as np
import time
from src.config.settings import DATABASE_PATH
from src.utils.logger import logger

class EmbeddingCache:
    """Provides persistent SQLite-based caching for generated embedding vectors 
    to avoid redundant local CPU inference runs.
    """
    
    def __init__(self):
        """Initializes the database connection and creates cache tables if not present."""
        self.db_path = str(DATABASE_PATH)
        logger.info(f"Connecting to embedding cache database: {self.db_path}")
        self._init_db()

    def _init_db(self):
        """Creates the cache table schema if not already initialized."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS embedding_cache (
                        chunk_hash TEXT PRIMARY KEY,
                        document_hash TEXT,
                        vector BLOB,
                        model_name TEXT,
                        created_at TEXT
                    )
                """)
                conn.commit()
            logger.info("Embedding cache database initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize embedding cache database: {str(e)}")
            raise e

    def get(self, chunk_hash: str, model_name: str) -> np.ndarray:
        """Retrieves a cached normalized float32 vector matching the text content hash.
        
        Args:
            chunk_hash: SHA-256 hash of the chunk text.
            model_name: Configured model name to avoid stale model cache hits.
            
        Returns:
            A numpy array representing the vector, or None if a cache miss occurs.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT vector FROM embedding_cache WHERE chunk_hash = ? AND model_name = ?",
                    (chunk_hash, model_name)
                )
                row = cursor.fetchone()
                if row:
                    # Deserialize binary blob back to float32 numpy array
                    vector_data = np.frombuffer(row[0], dtype=np.float32)
                    logger.debug(f"Cache hit for chunk hash: {chunk_hash}")
                    return vector_data
        except Exception as e:
            logger.warning(f"Failed to read from embedding cache database: {str(e)}")
            
        return None

    def set(self, chunk_hash: str, document_hash: str, vector: np.ndarray, model_name: str):
        """Saves a newly generated float32 normalized vector to the database cache.
        
        Args:
            chunk_hash: SHA-256 hash of the chunk text.
            document_hash: SHA-256 hash of the parent document.
            vector: Float32 numpy array vector to cache.
            model_name: Embedding model name tag.
        """
        try:
            # Serialize numpy array to binary format
            vector_blob = vector.tobytes()
            created_at = time.strftime("%Y-%m-%d %H:%M:%S")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO embedding_cache 
                    (chunk_hash, document_hash, vector, model_name, created_at) 
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (chunk_hash, document_hash, sqlite3.Binary(vector_blob), model_name, created_at)
                )
                conn.commit()
            logger.debug(f"Cached chunk vector successfully (hash: {chunk_hash})")
        except Exception as e:
            logger.warning(f"Failed to write to embedding cache database: {str(e)}")
