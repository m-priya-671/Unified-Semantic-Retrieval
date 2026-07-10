import os
import time
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from src.config.settings import MODELS_DIR, EMBEDDING_MODEL_NAME
from src.utils.logger import logger

class EmbeddingEngine:
    """Wrapper class around local SentenceTransformers to perform high-performance 
    offline text-to-vector embedding generation.
    """
    
    def __init__(self, model_name: str = None):
        """Initializes the local SentenceTransformer model.
        
        Args:
            model_name: HuggingFace model tag or local path (defaults to settings).
        """
        name = model_name if model_name is not None else EMBEDDING_MODEL_NAME
        self.embeddings_dir = MODELS_DIR / "embeddings"
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        logger.info(f"Loading local embedding model '{name}' (cache folder: {self.embeddings_dir})...")
        start_time = time.time()
        
        # Load SentenceTransformer locally using target cache directory
        self.model = SentenceTransformer(
            model_name_or_path=name,
            cache_folder=str(self.embeddings_dir),
            device="cpu"  # Force CPU execution for consistent offline behavior
        )
        self.model_name = name
        self.dimension = self.model.get_embedding_dimension()
        
        logger.info(f"Embedding model loaded successfully in {time.time() - start_time:.2f}s. "
                    f"Dimensions: {self.dimension}")

    def generate(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of text segments into unit-length L2-normalized vectors.
        
        Args:
            texts: List of string segments.
            
        Returns:
            A 2D numpy array of shape (len(texts), dimension) containing float32 vectors.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
            
        logger.debug(f"Generating vectors for batch of size {len(texts)}")
        
        # normalize_embeddings=True automatically normalizes the outputs to L2 norm = 1.0
        vectors = self.model.encode(
            inputs=texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return vectors.astype(np.float32)
