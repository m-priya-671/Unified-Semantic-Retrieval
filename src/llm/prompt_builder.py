from typing import List, Tuple
from src.retrieval.metadata import RetrievalChunk
from src.utils.logger import logger

class PromptBuilder:
    """Assembles prompt templates and manages context length boundaries by selecting whole chunks."""
    
    @staticmethod
    def build(
        query: str, 
        chunks: List[RetrievalChunk], 
        max_context_chars: int
    ) -> Tuple[str, str, int]:
        """Filters chunks to fit context limit and formats the prompt payload.
        
        Args:
            query: Sanitized user query string.
            chunks: List of retrieved chunks.
            max_context_chars: Character limit for combined context text.
            
        Returns:
            Tuple of (full_prompt_string, limited_context_string, chunks_used_count)
        """
        context_blocks = []
        current_len = 0
        chunks_used = 0
        
        # Build context blocks sequentially
        for idx, c in enumerate(chunks, 1):
            block = (
                f"[Chunk {idx}]\n"
                f"Source: {c.source_file}\n"
                f"Reference: {c.source_reference}\n"
                f"Similarity: {c.similarity_score:.2f}\n\n"
                f"{c.chunk_text}"
            )
            
            # Check length limit before adding (never truncate chunk text in the middle)
            # Add separation block length if not first item
            sep_len = 36 if idx > 1 else 0
            if current_len + len(block) + sep_len > max_context_chars:
                logger.warning(f"Context character limit reached ({current_len} chars). Skipping chunk {idx} onward.")
                break
                
            context_blocks.append(block)
            current_len += len(block) + sep_len
            chunks_used += 1
            
        limited_context = "\n\n----------------------------------\n\n".join(context_blocks)
        
        # Assemble standard prompt template
        prompt = (
            "You are an expert AI assistant providing strictly grounded answers based ONLY on the provided context.\n"
            "Follow these response rules:\n"
            "1. Answer the user question relying ONLY on the clear facts present in the Context section below.\n"
            "2. If the Context does not contain the answer, state: \"I am sorry, but the provided documents do not contain the information to answer this question.\" Do not attempt to guess or fabricate.\n"
            "3. Do NOT use any pre-existing or external general knowledge outside of the Context.\n"
            "4. Keep the response concise, objective, and directly relevant to the query.\n"
            "5. If the query or context is in Tamil, answer in Tamil. If in English, answer in English. Maintain multilingual alignment.\n\n"
            f"[Context]\n{limited_context}\n\n"
            f"[Question]\n{query}\n\n"
            "Answer:"
        )
        
        logger.debug(f"Assembled prompt of size {len(prompt)} characters using {chunks_used} chunks.")
        return prompt, limited_context, chunks_used
