import hashlib
import time
from typing import List, Dict, Any
from src.ingestion.base import Document
from src.text_processing.unified_document import UnifiedDocument
from src.text_processing.sentence_splitter import SentenceSplitter

class DocumentConverter:
    """Converts List[Document] collections from ingestion parsers into a UnifiedDocument, 
    preserving structured blocks and sentence boundaries.
    """
    
    @staticmethod
    def convert(
        documents: List[Document],
        source_file: str,
        source_type: str,
        processing_time: float
    ) -> UnifiedDocument:
        """Consolidates text contents and maps sentence-level source coordinates.
        
        Args:
            documents: List of Document instances returned by parsers.
            source_file: Original filename.
            source_type: Ingestion file format ("pdf" | "docx" | "image" | "audio").
            processing_time: Parse duration in seconds.
            
        Returns:
            A UnifiedDocument object.
        """
        if not documents:
            raise ValueError("Cannot convert empty documents list.")
            
        # 1. Map sentences/blocks with their original coordinate identifiers
        sentences_map = []
        
        for doc in documents:
            meta = doc.metadata
            
            if source_type == "audio":
                # For audio, segments are already sentence-sliced with timestamp coordinates
                for seg in meta.get("segments", []):
                    sentences_map.append({
                        "text": seg["text"].strip(),
                        "start": seg["start"],
                        "end": seg["end"]
                    })
            elif source_type == "docx" and meta.get("block_type") == "table":
                # Treat tables as single, indivisible blocks to preserve structure
                sentences_map.append({
                    "text": doc.text.strip(),
                    "block_index": meta.get("block_index")
                })
            else:
                # Regular text blocks: PDF pages, DOCX paragraphs, OCR images
                raw_sentences = SentenceSplitter.split(doc.text)
                for sentence in raw_sentences:
                    sent_info = {"text": sentence.strip()}
                    
                    if source_type == "pdf":
                        sent_info["page_number"] = meta.get("page_number")
                    elif source_type == "docx":
                        sent_info["block_index"] = meta.get("block_index")
                    elif source_type == "image":
                        sent_info["block_index"] = meta.get("block_index", 0)
                        
                    sentences_map.append(sent_info)
                    
        # Filter out empty entries
        sentences_map = [s for s in sentences_map if s["text"]]
        
        # 2. Reconstruct consolidated continuous text
        full_text = "\n\n".join(doc.text for doc in documents).strip()
        
        # 3. Content hashing for document ID
        sha = hashlib.sha256(full_text.encode("utf-8"))
        document_id = sha.hexdigest()[:16]  # Compact 16-character hash
        
        # 4. Extract multilingual info
        languages = []
        for doc in documents:
            if "languages" in doc.metadata:
                languages = doc.metadata["languages"]
                break
            elif "language" in doc.metadata:
                languages = [{"language": doc.metadata["language"], "probability": 1.0}]
                break
                
        if not languages:
            languages = [{"language": "en", "probability": 1.0}]
            
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 5. Collate parent metadata
        parent_metadata = {
            "source_type": source_type,
            "total_extracted_items": len(documents),
            "sentences": sentences_map  # Preserved coordinates map
        }
        
        return UnifiedDocument(
            document_id=document_id,
            source_file=source_file,
            source_type=source_type,
            text=full_text,
            metadata=parent_metadata,
            languages=languages,
            created_at=created_at,
            processing_time=processing_time
        )
