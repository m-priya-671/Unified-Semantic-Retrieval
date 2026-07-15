import re
from typing import Tuple

class LanguageDetector:
    """Performs offline language classification by analyzing character block ratios (English, Tamil, or Mixed)."""
    
    @staticmethod
    def detect(query: str) -> Tuple[str, float]:
        """Classifies the language of the query.
        
        Args:
            query: Sanitized query string.
            
        Returns:
            Tuple of (language_code, confidence_score) where:
                - language_code: "en" (English), "ta" (Tamil), or "mixed"
                - confidence_score: Confidence probability between 0.0 and 1.0
        """
        if not query:
            return "en", 1.0
            
        # Parse characters using regex Unicode ranges
        tamil_letters = len(re.findall(r"[\u0b80-\u0bff]", query))
        english_letters = len(re.findall(r"[a-zA-Z]", query))
        total_letters = tamil_letters + english_letters
        
        if total_letters == 0:
            # Fallback to English for number/punctuation strings
            return "en", 1.0
            
        ta_ratio = tamil_letters / total_letters
        en_ratio = english_letters / total_letters
        
        # Mixed categorization if both languages cross a threshold of 10%
        if ta_ratio > 0.1 and en_ratio > 0.1:
            # Max possible balance is 0.5 each. Normalize to 1.0
            conf = min(ta_ratio, en_ratio) * 2.0
            return "mixed", round(float(conf), 2)
        elif ta_ratio > 0.1:
            return "ta", round(float(ta_ratio), 2)
        else:
            return "en", round(float(en_ratio), 2)
