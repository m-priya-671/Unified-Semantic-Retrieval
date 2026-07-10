import os
import json
import time
import faiss
from pathlib import Path
from typing import Dict, Any
from src.utils.logger import logger

class IndexPersistence:
    """Manages saving and loading the FAISS binary index and JSON statistics files."""
    
    @staticmethod
    def save(
        index_dir: Path, 
        index: faiss.Index, 
        stats: Dict[str, Any], 
        model_name: str
    ) -> Path:
        """Serializes the FAISS index to a binary file and writes index_info.json.
        
        Args:
            index_dir: Target directory (e.g. data/indices).
            index: FAISS index object to serialize.
            stats: Index metrics dictionary from MetadataStore.
            model_name: Configured model tag.
            
        Returns:
            The path to the saved FAISS index file.
        """
        os.makedirs(index_dir, exist_ok=True)
        index_file = index_dir / "index.faiss"
        info_file = index_dir / "index_info.json"
        
        logger.info(f"Saving FAISS index to persistent store: {index_file}")
        start_time = time.time()
        
        try:
            # Save raw FAISS index
            faiss.write_index(index, str(index_file))
            file_size = os.path.getsize(index_file)
            
            # Collate statistics info JSON
            info_data = {
                "index_type": "IndexIDMap2(FlatIP)",
                "dimension": index.d,
                "total_vectors": index.ntotal,
                "total_documents": stats.get("total_documents", 0),
                "average_chunks_per_document": stats.get("average_chunks_per_document", 0.0),
                "normalized_vectors": True,
                "embedding_model": model_name,
                "index_version": "1.0",
                "embedding_version": "1.0",
                "schema_version": "1.0",
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "index_file_size_bytes": file_size
            }
            
            with open(info_file, "w", encoding="utf-8") as f:
                json.dump(info_data, f, indent=2)
                
            logger.info(f"FAISS index successfully persisted in {time.time() - start_time:.2f}s "
                        f"({file_size / 1024:.1f} KB).")
            return index_file
        except Exception as e:
            logger.error(f"Failed to persist index files: {str(e)}")
            raise e

    @staticmethod
    def load(index_dir: Path) -> faiss.Index:
        """Deserializes the binary FAISS index from disk.
        
        Args:
            index_dir: Directory containing index.faiss.
            
        Returns:
            Loaded FAISS index object, or None if the file does not exist.
        """
        index_file = index_dir / "index.faiss"
        if not index_file.exists():
            logger.warning(f"FAISS index file does not exist: {index_file}")
            return None
            
        logger.info(f"Loading FAISS index from {index_file}...")
        start_time = time.time()
        
        try:
            index = faiss.read_index(str(index_file))
            logger.info(f"FAISS index loaded successfully in {time.time() - start_time:.2f}s. "
                        f"Total vectors: {index.ntotal}")
            return index
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
            raise e

    @staticmethod
    def load_info(index_dir: Path) -> Dict[str, Any]:
        """Loads configuration properties from index_info.json.
        
        Args:
            index_dir: Directory containing index_info.json.
            
        Returns:
            Info dictionary, or empty dict if not found.
        """
        info_file = index_dir / "index_info.json"
        if not info_file.exists():
            return {}
            
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read index info file: {str(e)}")
            return {}
