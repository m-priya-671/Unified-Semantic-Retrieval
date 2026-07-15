import unicodedata

class QueryProcessor:
    """Applies cleaning, whitespace trimming, and Unicode NFKC normalization to queries."""
    
    @staticmethod
    def process(query: str) -> str:
        """Trims whitespace and applies NFKC normalization.
        
        Args:
            query: User's raw text query.
            
        Returns:
            Sanitized query string.
        """
        if not query:
            return ""
        # NFKC normalizes characters, mapping mixed symbol formats consistently
        normalized = unicodedata.normalize("NFKC", query)
        return normalized.strip()
