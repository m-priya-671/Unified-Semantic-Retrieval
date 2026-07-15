from typing import List
from src.retrieval.metadata import RetrievalChunk

class ContextBuilder:
    """Assembles ranked text chunks into an LLM-agnostic context payload."""
    
    @staticmethod
    def build(chunks: List[RetrievalChunk]) -> str:
        """Concatenates chunk texts formatted with source attribution headers.
        
        Args:
            chunks: List of ranked RetrievalChunk objects.
            
        Returns:
            Structured context block string.
        """
        if not chunks:
            return ""
            
        blocks = []
        for idx, c in enumerate(chunks, 1):
            block = (
                f"[Chunk {idx}]\n"
                f"Source: {c.source_file}\n"
                f"Reference: {c.source_reference}\n"
                f"Similarity: {c.similarity_score:.2f}\n\n"
                f"{c.chunk_text}"
            )
            blocks.append(block)
            
        # Join chunks with standard boundary lines
        return "\n\n----------------------------------\n\n".join(blocks)
