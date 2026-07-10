import sqlite3
import time
from typing import List, Dict, Any, Tuple
from src.config.settings import DATABASE_PATH
from src.utils.logger import logger

class MetadataStore:
    """Manages metadata mappings in SQLite to synchronize with FAISS indexing."""
    
    def __init__(self):
        """Initializes database path and sets up tables."""
        self.db_path = str(DATABASE_PATH)
        self._init_db()

    def _init_db(self):
        """Creates vector_metadata and document_index_status schemas if they do not exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Table 1: Chunk-level Metadata mapping to FAISS ID
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS vector_metadata (
                        faiss_id INTEGER PRIMARY KEY,
                        chunk_id TEXT UNIQUE,
                        document_id TEXT,
                        source_file TEXT,
                        source_reference TEXT,
                        chunk_text TEXT,
                        embedding_model TEXT,
                        indexed_at TEXT,
                        status TEXT
                    )
                """)
                
                # Table 2: Document-level indexing status
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS document_index_status (
                        document_hash TEXT PRIMARY KEY,
                        indexed_chunks INTEGER,
                        total_chunks INTEGER,
                        last_indexed TEXT,
                        index_status TEXT
                    )
                """)
                conn.commit()
            logger.info("MetadataStore database tables initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize MetadataStore tables: {str(e)}")
            raise e

    def is_chunk_indexed(self, chunk_id: str) -> bool:
        """Checks if a chunk has already been indexed to avoid duplicate insertions.
        
        Args:
            chunk_id: String chunk identifier.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM vector_metadata WHERE chunk_id = ?", (chunk_id,))
                return cursor.fetchone() is not None
        except Exception as e:
            logger.warning(f"Failed to check chunk indexed status: {str(e)}")
        return False

    def add_chunk_metadata_tx(
        self, 
        cursor: sqlite3.Cursor, 
        faiss_id: int, 
        chunk_id: str, 
        document_id: str, 
        source_file: str, 
        source_reference: str, 
        chunk_text: str, 
        embedding_model: str
    ):
        """Inserts a single chunk metadata mapping record within an active SQLite transaction.
        
        Args:
            cursor: Active database transaction cursor.
            faiss_id: Sequence key assigned to FAISS index vector.
            chunk_id: Unique chunk ID format.
            document_id: Parent document content ID.
            source_file: Upload file basename.
            source_reference: Citations page/block/timestamp.
            chunk_text: Text snippet payload.
            embedding_model: Model name.
        """
        indexed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT OR REPLACE INTO vector_metadata 
            (faiss_id, chunk_id, document_id, source_file, source_reference, chunk_text, embedding_model, indexed_at, status) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (faiss_id, chunk_id, document_id, source_file, source_reference, chunk_text, embedding_model, indexed_at, "Indexed")
        )

    def update_document_status_tx(
        self, 
        cursor: sqlite3.Cursor, 
        document_hash: str, 
        indexed_chunks: int, 
        total_chunks: int, 
        index_status: str
    ):
        """Inserts or updates document level index logs within an active SQLite transaction.
        
        Args:
            cursor: Active database transaction cursor.
            document_hash: Parent document unique identifier.
            indexed_chunks: Number of chunks indexed.
            total_chunks: Parse chunks parsed.
            index_status: Execution status (e.g. Completed, Partial).
        """
        last_indexed = time.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT OR REPLACE INTO document_index_status 
            (document_hash, indexed_chunks, total_chunks, last_indexed, index_status) 
            VALUES (?, ?, ?, ?, ?)
            """,
            (document_hash, indexed_chunks, total_chunks, last_indexed, index_status)
        )

    def get_metadata(self, faiss_id: int) -> Dict[str, Any]:
        """Retrieves a single chunk metadata record.
        
        Args:
            faiss_id: Integer key.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM vector_metadata WHERE faiss_id = ?", (faiss_id,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
        except Exception as e:
            logger.error(f"Error fetching metadata for faiss_id {faiss_id}: {str(e)}")
        return None

    def get_metadata_by_ids(self, faiss_ids: List[int]) -> List[Dict[str, Any]]:
        """Retrieves multiple metadata records, preserving input ID order.
        
        Args:
            faiss_ids: List of sequential integer keys.
        """
        if not faiss_ids:
            return []
            
        try:
            # Prepare query placeholders
            placeholders = ",".join("?" for _ in faiss_ids)
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT * FROM vector_metadata WHERE faiss_id IN ({placeholders})", 
                    faiss_ids
                )
                rows = cursor.fetchall()
                
                # Group by faiss_id for fast lookup
                lookup = {row["faiss_id"]: dict(row) for row in rows}
                
                # Maintain original query list order
                results = []
                for fid in faiss_ids:
                    if fid in lookup:
                        results.append(lookup[fid])
                return results
        except Exception as e:
            logger.error(f"Error fetching bulk metadata: {str(e)}")
            
        return []

    def remove_metadata_by_document(self, cursor: sqlite3.Cursor, document_id: str) -> List[int]:
        """Deletes chunk and document logs for a file inside an active transaction.
        
        Returns:
            List of deleted faiss_id keys.
        """
        # Fetch matching faiss_id keys first
        cursor.execute("SELECT faiss_id FROM vector_metadata WHERE document_id = ?", (document_id,))
        faiss_ids = [row[0] for row in cursor.fetchall()]
        
        if faiss_ids:
            # Delete chunks metadata
            placeholders = ",".join("?" for _ in faiss_ids)
            cursor.execute(f"DELETE FROM vector_metadata WHERE faiss_id IN ({placeholders})", faiss_ids)
            
        # Delete document status
        cursor.execute("DELETE FROM document_index_status WHERE document_hash = ?", (document_id,))
        
        return faiss_ids

    def get_index_stats(self) -> Dict[str, Any]:
        """Computes statistical metrics across indexed collections."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM vector_metadata")
                total_vectors = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM document_index_status")
                total_docs = cursor.fetchone()[0]
                
                avg_chunks = 0.0
                if total_docs > 0:
                    cursor.execute("SELECT AVG(indexed_chunks) FROM document_index_status")
                    avg_chunks = round(float(cursor.fetchone()[0]), 2)
                    
                return {
                    "total_vectors": total_vectors,
                    "total_documents": total_docs,
                    "average_chunks_per_document": avg_chunks
                }
        except Exception as e:
            logger.warning(f"Failed to calculate metadata index statistics: {str(e)}")
            
        return {
            "total_vectors": 0,
            "total_documents": 0,
            "average_chunks_per_document": 0.0
        }

    def clear_all(self):
        """Clears both SQLite tables completely."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM vector_metadata")
                cursor.execute("DELETE FROM document_index_status")
                conn.commit()
            logger.info("Cleared all records from vector_metadata and document_index_status.")
        except Exception as e:
            logger.error(f"Failed to clear MetadataStore records: {str(e)}")
            raise e
