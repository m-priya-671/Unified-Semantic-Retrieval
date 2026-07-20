from typing import List, Tuple, Dict, Any
from src.retrieval.metadata import RetrievalChunk
from src.utils.logger import logger

class PromptBuilder:
    """Assembles prompt templates and manages context length boundaries by calculating prompt budgets and selecting whole chunks."""
    
    SYSTEM_PROMPT = (
        "You are an expert AI assistant providing strictly grounded answers based ONLY on the provided context.\n"
        "Follow these response rules:\n"
        "1. Answer the user request relying ONLY on the clear facts present in the Context section below.\n"
        "2. Natural paraphrasing, rewording, sentence restructuring, and summarization are encouraged. Do NOT copy verbatim unless necessary.\n"
        "3. Do NOT invent facts, numbers, dates, names, or unsupported claims.\n"
        "4. If the Context does not contain the necessary information, state: \"The uploaded document contains limited information related to your question, so a detailed answer cannot be generated from the available content.\"\n"
        "5. Keep the response concise, objective, and directly relevant to the query.\n"
        "6. If the query or context is in Tamil, answer in Tamil. If in English, answer in English. Maintain multilingual alignment."
    )

    SUMMARY_SYSTEM_PROMPT = (
        "You are an expert AI assistant providing strictly grounded document summaries based ONLY on the provided context.\n"
        "Follow these summary rules:\n"
        "1. Generate a clear, concise, and well-structured summary using key facts, main topics, and essential details present in the Context below.\n"
        "2. Paraphrasing, rewording, and synthesis are encouraged. Do NOT look for a pre-existing summary section; synthesize from the available content.\n"
        "3. Do NOT invent facts, statistics, names, or external information.\n"
        "4. If the Context contains insufficient meaningful text, state: \"The uploaded document contains limited information related to your question, so a detailed answer cannot be generated from the available content.\"\n"
        "5. If the query or context is in Tamil, answer in Tamil. If in English, answer in English."
    )

    @classmethod
    def build(
        cls,
        query: str,
        chunks: List[RetrievalChunk],
        max_context_chars: int = 4000,
        return_diagnostics: bool = False,
        intent: str = "QUESTION"
    ) -> Any:
        """Filters chunks using prompt budgeting and skip-and-continue selection.
        
        Args:
            query: Sanitized user query string.
            chunks: List of retrieved chunks.
            max_context_chars: Hard limit for final prompt character count.
            return_diagnostics: If True, returns a 4-tuple including diagnostics dictionary.
            intent: Query intent ("QUESTION" or "SUMMARY").
            
        Returns:
            Tuple of (full_prompt_string, limited_context_string, chunks_used_count)
            or 4-tuple (full_prompt_string, limited_context_string, chunks_used_count, diagnostics_dict)
        """
        prompt, limited_context, chunks_used, diag = cls.build_with_diagnostics(
            query=query,
            chunks=chunks,
            max_context_chars=max_context_chars,
            intent=intent
        )
        if return_diagnostics:
            return prompt, limited_context, chunks_used, diag
        return prompt, limited_context, chunks_used

    @classmethod
    def build_with_diagnostics(
        cls,
        query: str,
        chunks: List[RetrievalChunk],
        max_context_chars: int = 4000,
        intent: str = "QUESTION"
    ) -> Tuple[str, str, int, Dict[str, Any]]:
        """Calculates prompt budget and assembles context using skip-and-continue selection.
        
        Args:
            query: Sanitized user query string.
            chunks: List of retrieved chunks.
            max_context_chars: Hard limit for final prompt character count.
            intent: Query intent string.
            
        Returns:
            Tuple of (final_prompt, limited_context, chunks_included_count, diagnostics_dict)
        """
        safe_query = query if query is not None else ""
        system_text = cls.SUMMARY_SYSTEM_PROMPT if intent == "SUMMARY" else cls.SYSTEM_PROMPT
        
        system_prompt_chars = len(system_text)
        question_chars = len(safe_query)
        
        # Template prefix & suffix
        template_prefix = f"{system_text}\n\n[Context]\n"
        template_suffix = f"\n\n[Question]\n{safe_query}\n\nAnswer:"
        
        template_chars = len(template_prefix) + len(template_suffix) - system_prompt_chars - question_chars
        base_prompt_chars = system_prompt_chars + question_chars + template_chars
        
        available_context_budget = max(0, max_context_chars - base_prompt_chars)
        
        chunks = chunks or []
        chunks_retrieved = len(chunks)
        
        included_chunks = []
        current_context_len = 0
        chunks_trimmed = 0
        
        # Skip-and-continue chunk assembly algorithm
        for idx, c in enumerate(chunks, 1):
            block = (
                f"[Chunk {idx}]\n"
                f"Source: {c.source_file}\n"
                f"Reference: {c.source_reference}\n"
                f"Similarity: {c.similarity_score:.2f}\n\n"
                f"{c.chunk_text}"
            )
            
            sep_len = 36 if len(included_chunks) > 0 else 0
            projected_len = current_context_len + len(block) + sep_len
            
            if projected_len <= available_context_budget:
                included_chunks.append((idx, c, block))
                current_context_len = projected_len
            else:
                chunks_trimmed += 1
                logger.info(
                    f"Chunk {idx} (size: {len(block)} chars) exceeds available context budget "
                    f"({available_context_budget} chars). Skipping chunk {idx} and checking remaining chunks."
                )

        # Re-assemble limited_context from included chunks
        context_blocks = [item[2] for item in included_chunks]
        limited_context = "\n\n----------------------------------\n\n".join(context_blocks)
        
        # Build final prompt
        final_prompt = f"{template_prefix}{limited_context}{template_suffix}"
        
        # Defensive assertion & trimming fallback
        if len(final_prompt) > max_context_chars and included_chunks:
            logger.warning(
                f"Defensive validation triggered: final_prompt length {len(final_prompt)} "
                f"exceeds limit {max_context_chars}. Trimming trailing chunks."
            )
            while len(final_prompt) > max_context_chars and included_chunks:
                included_chunks.pop()
                chunks_trimmed += 1
                context_blocks = [item[2] for item in included_chunks]
                limited_context = "\n\n----------------------------------\n\n".join(context_blocks)
                final_prompt = f"{template_prefix}{limited_context}{template_suffix}"

        chunks_included = len(included_chunks)
        retrieved_context_chars = len(limited_context)
        final_prompt_chars = len(final_prompt)
        remaining_budget = max(0, max_context_chars - final_prompt_chars)
        
        context_utilization_percent = (
            round((retrieved_context_chars / available_context_budget * 100.0), 2)
            if available_context_budget > 0 else 0.0
        )
        trim_reason = "Exceeded context budget" if chunks_trimmed > 0 else "None"
        
        diagnostics = {
            "max_context_characters": max_context_chars,
            "system_prompt_chars": system_prompt_chars,
            "question_chars": question_chars,
            "template_chars": template_chars,
            "base_prompt_chars": base_prompt_chars,
            "available_context_budget": available_context_budget,
            "retrieved_context_chars": retrieved_context_chars,
            "final_prompt_chars": final_prompt_chars,
            "remaining_budget": remaining_budget,
            "context_utilization_percent": context_utilization_percent,
            "chunks_retrieved": chunks_retrieved,
            "chunks_included": chunks_included,
            "chunks_trimmed": chunks_trimmed,
            "trim_reason": trim_reason
        }
        
        logger.info(
            f"Prompt Budget: {max_context_chars} | "
            f"Base Prompt Size: {base_prompt_chars} | "
            f"Available Context Budget: {available_context_budget} | "
            f"Retrieved Context Chars: {retrieved_context_chars} | "
            f"Final Prompt Length: {final_prompt_chars} | "
            f"Remaining Budget: {remaining_budget} | "
            f"Context Utilization: {context_utilization_percent:.2f}% | "
            f"Chunks Retrieved: {chunks_retrieved} | "
            f"Chunks Included: {chunks_included} | "
            f"Chunks Trimmed: {chunks_trimmed} | "
            f"Trim Reason: {trim_reason}"
        )
        
        return final_prompt, limited_context, chunks_included, diagnostics
