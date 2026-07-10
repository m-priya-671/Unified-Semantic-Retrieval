import os
import sqlite3
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from src.text_processing.chunk import Chunk
from src.config.settings import BASE_DIR, EMBEDDING_MODEL_NAME
from src.vector_store.faiss_engine import FaissEngine
from src.vector_store.metadata_store import MetadataStore
from src.vector_store.persistence import IndexPersistence
from src.vector_store.index_validator import IndexValidator
from src.vector_store.search_utils import SearchUtils
from src.utils.logger import logger

class IndexManager:
    """Orchestrates validation, transactional SQLite mapping, FAISS vector indexing, 
    and persistent loading operations.
    """
    
    def __init__(self, model_name: str = None, dimension: int = 384):
        """Initializes components and attempts to reload index from disk.
        
        Args:
            model_name: Name of embedding model (defaults to settings configuration).
            dimension: Mapped model vector length (default: 384).
        """
        self.model_name = model_name if model_name is not None else EMBEDDING_MODEL_NAME
        self.dimension = dimension
        self.index_dir = BASE_DIR / "data" / "indices"
        self.metadata_store = MetadataStore()
        
        # Load or initialize FAISS engine
        self.engine = None
        self.reload_index()

    def reload_index(self):
        """Loads index.faiss binary from disk, falling back to a clean index if missing or corrupted."""
        try:
            loaded_index = IndexPersistence.load(self.index_dir)
            if loaded_index is not None and IndexValidator.validate_index_file(loaded_index, self.dimension):
                self.engine = FaissEngine(self.dimension)
                self.engine.index = loaded_index
                logger.info("Successfully loaded existing FAISS index from disk.")
            else:
                self.engine = FaissEngine(self.dimension)
                logger.info("Created clean in-memory FAISS index.")
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {str(e)}. Fallback to clean index.")
            self.engine = FaissEngine(self.dimension)

    def add_vectors_and_metadata(self, vectors: np.ndarray, chunks: List[Chunk]):
        """Executes explicit transaction parsing, database inserts, and FAISS indexing.
        
        Args:
            vectors: 2D float32 numpy array of shape (num_vectors, dimension).
            chunks: List of Chunk objects to synchronize.
        """
        if not chunks:
            logger.warning("Empty chunks list received. Skipping index insertion.")
            return
            
        # 1. Validate vector shape and dimensions
        IndexValidator.validate_vectors(vectors, self.dimension)
        
        if len(vectors) != len(chunks):
            msg = f"Size mismatch. Vectors count: {len(vectors)}, chunks count: {len(chunks)}"
            logger.error(msg)
            raise ValueError(msg)
            
        # Check for duplicate chunk insertions beforehand
        for chunk in chunks:
            if self.metadata_store.is_chunk_indexed(chunk.chunk_id):
                logger.warning(f"Chunk '{chunk.chunk_id}' is already indexed. Skipping entire batch to avoid duplicates.")
                return
                
        # 2. Begin SQLite Connection and Transaction block
        conn = sqlite3.connect(self.metadata_store.db_path)
        conn.isolation_level = None  # Disable auto-commit
        cursor = conn.cursor()
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # Determine start faiss_id mapping offset sequentially
            cursor.execute("SELECT IFNULL(MAX(faiss_id), -1) FROM vector_metadata")
            max_id = cursor.fetchone()[0]
            start_id = max_id + 1
            
            logger.info(f"Indexing {len(chunks)} chunks starting at FAISS ID: {start_id}")
            
            # 3. Write metadata to SQLite
            document_hash = chunks[0].document_id
            source_file = chunks[0].metadata.get("source_file", "Unknown")
            
            for idx, chunk in enumerate(chunks):
                faiss_id = start_id + idx
                self.metadata_store.add_chunk_metadata_tx(
                    cursor=cursor,
                    faiss_id=faiss_id,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_file=source_file,
                    source_reference=chunk.metadata.get("source_reference", "N/A"),
                    chunk_text=chunk.text,
                    embedding_model=self.model_name
                )
                
            # Insert document index status logs
            self.metadata_store.update_document_status_tx(
                cursor=cursor,
                document_hash=document_hash,
                indexed_chunks=len(chunks),
                total_chunks=len(chunks),
                index_status="Completed"
            )
            
            # 4. Insert vectors into FAISS in-memory index
            ids = np.arange(start_id, start_id + len(chunks), dtype=np.int64)
            self.engine.add(vectors, ids)
            
            # 5. Persist FAISS index binary dump to disk
            temp_stats = {
                "total_documents": self._get_doc_count_tx(cursor),
                "average_chunks_per_document": self._get_avg_chunks_tx(cursor)
            }
            IndexPersistence.save(self.index_dir, self.engine.index, temp_stats, self.model_name)
            
            # 6. Commit SQLite transaction
            cursor.execute("COMMIT")
            
            # 8. Enrich Python objects chunk metadata
            indexed_at = time.strftime("%Y-%m-%d %H:%M:%S")
            for idx, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "faiss_id": int(start_id + idx),
                    "index_status": "Indexed",
                    "indexed_at": indexed_at
                })
                
            logger.info(f"Batch index transaction completed successfully. Mapped IDs: {start_id} to {start_id + len(chunks) - 1}")
            
        except Exception as e:
            # 7. Rollback SQLite and reload previous valid FAISS index from disk
            logger.error(f"Transaction failed: {str(e)}. Executing rollback and index state restoration.")
            try:
                cursor.execute("ROLLBACK")
            except Exception as sql_err:
                logger.error(f"SQL Rollback failed: {str(sql_err)}")
            self.reload_index()  # Restores in-memory FAISS state to match disk
            raise e
        finally:
            conn.close()

    def _get_doc_count_tx(self, cursor: sqlite3.Cursor) -> int:
        """Retrieves active document count within current transaction."""
        cursor.execute("SELECT COUNT(*) FROM document_index_status")
        return cursor.fetchone()[0]

    def _get_avg_chunks_tx(self, cursor: sqlite3.Cursor) -> float:
        """Retrieves average chunk count within transaction."""
        cursor.execute("SELECT COUNT(*) FROM document_index_status")
        docs = cursor.fetchone()[0]
        if docs == 0:
            return 0.0
        cursor.execute("SELECT AVG(indexed_chunks) FROM document_index_status")
        return round(float(cursor.fetchone()[0]), 2)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Validates and executes nearest-neighbor similarity searches.
        
        Args:
            query_vector: 1D or 2D query embedding vector.
            top_k: Number of nearest neighbor contexts to retrieve.
        """
        if self.engine.total == 0:
            logger.warning("Empty index queried. Returning empty results.")
            return []
            
        # Validate query parameters
        IndexValidator.validate_search(query_vector, self.engine.total, self.dimension)
        
        # Raw index search
        similarities, indices = self.engine.search(query_vector, top_k)
        
        # Map IDs to metadata
        return SearchUtils.map_results(similarities, indices, self.metadata_store)

    def get_index_info(self) -> Dict[str, Any]:
        """Loads configuration properties from JSON info settings."""
        return IndexPersistence.load_info(self.index_dir)

    def clear_all(self):
        """Clears SQLite records and resets FAISS index files on disk."""
        logger.info("Resetting entire vector store and cache metadata...")
        
        # Reset SQLite tables
        self.metadata_store.clear_all()
        
        # Reset raw FAISS
        self.engine.reset()
        
        # Overwrite persistence file
        empty_stats = {"total_documents": 0, "average_chunks_per_document": 0.0}
        IndexPersistence.save(self.index_dir, self.engine.index, empty_stats, self.model_name)
        logger.info("Vector store reset complete.")
