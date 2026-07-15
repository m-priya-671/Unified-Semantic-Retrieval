class AnswerFormatter:
    """Merges validated LLM text responses with compiled citations."""
    
    @staticmethod
    def format(answer: str, citations_md: str) -> str:
        """Appends formatted markdown citation listings to the answer text.
        
        Args:
            answer: Cleaned response text.
            citations_md: Formatted sources section string.
            
        Returns:
            Formatted output answer payload.
        """
        if not citations_md:
            return answer
        return f"{answer}{citations_md}"
