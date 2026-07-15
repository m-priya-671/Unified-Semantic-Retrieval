from typing import List, Tuple
from src.retrieval.metadata import RetrievalChunk
from src.utils.logger import logger

class CitationFormatter:
    """Formats unique source citations under a dedicated references section, maintaining relevance order."""
    
    @staticmethod
    def format(chunks: List[RetrievalChunk]) -> Tuple[str, List[str]]:
        """Deduplicates attributions and formats markdown lists.
        
        Args:
            chunks: List of retrieved RetrievalChunk objects.
            
        Returns:
            Tuple containing:
                - formatted_md: Markdown list string (empty if no sources).
                - sources_list: List of deduplicated citation strings.
        """
        seen = set()
        sources_list = []
        
        for c in chunks:
            # Combined file name and coordinate references
            citation = f"{c.source_file} ({c.source_reference})"
            if citation not in seen:
                seen.add(citation)
                sources_list.append(citation)
                
        if not sources_list:
            return "", []
            
        # Format sources section
        formatted_md = "\n\nSources:\n" + "\n".join(f"• {src}" for src in sources_list)
        logger.debug(f"Compiled {len(sources_list)} citations successfully.")
        return formatted_md, sources_list
