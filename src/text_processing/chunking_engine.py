import hashlib
from typing import List, Dict, Any
from src.text_processing.unified_document import UnifiedDocument
from src.text_processing.chunk import Chunk
from src.config.settings import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, MAX_CHUNK_LENGTH
from src.utils.logger import logger

class ChunkingEngine:
    """Intelligently groups UnifiedDocument elements into semantic chunks, 
    respecting sentence boundaries, preserving tables, and calculating citations.
    """
    
    @staticmethod
    def chunk_document(
        doc: UnifiedDocument,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[Chunk]:
        """Segments a UnifiedDocument into overlapping chunks.
        
        Args:
            doc: UnifiedDocument to segment.
            chunk_size: Ideal character size limit.
            chunk_overlap: Desired backtrack overlap in characters.
            
        Returns:
            A list of Chunk objects.
        """
        size = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
        overlap = chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP
        
        logger.info(f"Starting chunking for '{doc.source_file}' (size limit: {size}, overlap: {overlap})")
        
        sentences = doc.metadata.get("sentences", [])
        if not sentences:
            # Fallback if no pre-mapped sentences exist
            from src.text_processing.sentence_splitter import SentenceSplitter
            raw_sents = SentenceSplitter.split(doc.text)
            sentences = [{"text": s} for s in raw_sents if s.strip()]
            
        chunks = []
        chunk_index = 0
        total_sentences = len(sentences)
        i = 0
        
        while i < total_sentences:
            curr_sentences = []
            curr_len = 0
            
            # Aggregate sentences into the current chunk
            j = i
            while j < total_sentences:
                sent = sentences[j]
                sent_len = len(sent["text"])
                
                # Must include at least the first sentence to avoid infinite loops
                if len(curr_sentences) > 0 and (curr_len + sent_len > size or curr_len + sent_len > MAX_CHUNK_LENGTH):
                    break
                    
                curr_sentences.append(sent)
                curr_len += sent_len + 1  # Account for spacing
                j += 1
                
            chunk_text = " ".join(s["text"] for s in curr_sentences).strip()
            
            # Setup chunk metadata
            chunk_meta = {
                "source_file": doc.source_file,
                "source_type": doc.source_type,
                "parent_id": doc.document_id,
                "languages": doc.languages
            }
            
            source_ref = "Unknown"
            
            # Calculate human-readable citation references
            if doc.source_type == "pdf":
                pages = sorted(list({s["page_number"] for s in curr_sentences if "page_number" in s and s["page_number"] is not None}))
                if pages:
                    chunk_meta["pages"] = pages
                    source_ref = f"Page {pages[0]}" if len(pages) == 1 else f"Pages {pages[0]}-{pages[-1]}"
                    
            elif doc.source_type == "docx":
                blocks = sorted(list({s["block_index"] for s in curr_sentences if "block_index" in s and s["block_index"] is not None}))
                if blocks:
                    chunk_meta["block_ids"] = blocks
                    source_ref = f"Block {blocks[0]}" if len(blocks) == 1 else f"Blocks {blocks[0]}-{blocks[-1]}"
                    
            elif doc.source_type == "image":
                blocks = sorted(list({s["block_index"] for s in curr_sentences if "block_index" in s and s["block_index"] is not None}))
                if blocks:
                    chunk_meta["ocr_blocks"] = blocks
                    source_ref = f"OCR Block {blocks[0]}" if len(blocks) == 1 else f"OCR Blocks {blocks[0]}-{blocks[-1]}"
                    
            elif doc.source_type == "audio":
                starts = [s["start"] for s in curr_sentences if "start" in s and s["start"] is not None]
                ends = [s["end"] for s in curr_sentences if "end" in s and s["end"] is not None]
                if starts and ends:
                    start_time = min(starts)
                    end_time = max(ends)
                    chunk_meta["start_time"] = round(start_time, 2)
                    chunk_meta["end_time"] = round(end_time, 2)
                    
                    s_min, s_sec = divmod(int(start_time), 60)
                    e_min, e_sec = divmod(int(end_time), 60)
                    source_ref = f"Timestamp [{s_min:02d}:{s_sec:02d} - {e_min:02d}:{e_sec:02d}]"
            
            chunk_meta["source_reference"] = source_ref
            
            # Readable sequential chunk ID
            chunk_id = f"{doc.document_id}_chunk_{chunk_index:04d}"
            char_count = len(chunk_text)
            token_est = char_count // 4  # Approximation for debugging only
            
            new_chunk = Chunk(
                chunk_id=chunk_id,
                document_id=doc.document_id,
                chunk_index=chunk_index,
                text=chunk_text,
                metadata=chunk_meta,
                token_estimate=token_est,
                character_count=char_count
            )
            
            chunks.append(new_chunk)
            chunk_index += 1
            
            if j >= total_sentences:
                break
                
            # Backtrack sentences to satisfy requested overlap limit
            backtrack_len = 0
            backtrack_count = 0
            
            for k in range(j - 1, i, -1):
                s_len = len(sentences[k]["text"])
                if backtrack_len + s_len > overlap:
                    break
                backtrack_len += s_len + 1
                backtrack_count += 1
                
            # Advance base cursor pointer
            if backtrack_count == 0:
                i = j
            else:
                i = j - backtrack_count
                
        logger.info(f"Chunking finished: {len(chunks)} chunks created for '{doc.source_file}'.")
        return chunks
