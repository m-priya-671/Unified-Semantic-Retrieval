import numpy as np
import faiss
from src.utils.logger import logger

class FaissEngine:
    """Wraps raw FAISS index construction, indexing additions, and similarity queries on CPU."""
    
    def __init__(self, dimension: int = 384):
        """Initializes a flat inner product index wrapped in ID Map 2.
        
        Args:
            dimension: Float vector size (default: 384 for MiniLM-L12-v2).
        """
        self.dimension = dimension
        # FlatIP performs inner product dot-products (exact Cosine Similarity for unit vectors)
        self.flat_index = faiss.IndexFlatIP(dimension)
        # IDMap2 allows tracking arbitrary sequential integer IDs mapped to vectors
        self.index = faiss.IndexIDMap2(self.flat_index)
        logger.info(f"Initialized FAISS IndexIDMap2(IndexFlatIP) with dimension {dimension}")

    def add(self, vectors: np.ndarray, ids: np.ndarray):
        """Inserts unit-length normalized vectors with unique 64-bit integer IDs.
        
        Args:
            vectors: 2D float32 numpy array of shape (num_vectors, dimension).
            ids: 1D int64 numpy array of shape (num_vectors,).
        """
        if len(vectors) == 0:
            return
            
        # Ensure vectors are float32
        vectors = vectors.astype(np.float32)
        ids = ids.astype(np.int64)
        
        logger.debug(f"Inserting {len(vectors)} vectors into FAISS index...")
        self.index.add_with_ids(vectors, ids)
        logger.debug("Vectors successfully added to FAISS index.")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> tuple:
        """Retrieves nearest neighbor IDs and similarity scores for a given query vector.
        
        Args:
            query_vector: 1D or 2D float32 numpy array.
            top_k: Number of nearest neighbors to retrieve.
            
        Returns:
            A tuple of (similarities, indices) where:
                - similarities: List/Array of float Inner Product similarity scores.
                - indices: List/Array of matching 64-bit integer IDs.
        """
        if self.index.ntotal == 0:
            return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
            
        # Ensure correct shape
        if len(query_vector.shape) == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
            
        query_vector = query_vector.astype(np.float32)
        similarities, indices = self.index.search(query_vector, top_k)
        
        return similarities, indices

    def remove(self, ids: np.ndarray) -> int:
        """Removes a list of vector IDs from the index.
        
        Args:
            ids: 1D int64 numpy array of IDs to delete.
            
        Returns:
            Number of successfully removed vectors.
        """
        ids = ids.astype(np.int64)
        removed_count = self.index.remove_ids(ids)
        logger.debug(f"Removed {removed_count} vectors from FAISS index.")
        return removed_count

    def reset(self):
        """Clears all vectors from the index."""
        self.index.reset()
        logger.info("FAISS index successfully reset/cleared.")

    @property
    def total(self) -> int:
        """Returns the total number of indexed vectors."""
        return self.index.ntotal
