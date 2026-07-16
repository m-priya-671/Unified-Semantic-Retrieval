import re
from src.utils.logger import logger

class IntentClassifier:
    """Classifies user queries into semantic intents (SUMMARY, OVERVIEW, TRANSLATION, GENERAL, QUESTION)."""
    
    @staticmethod
    def classify(query: str) -> str:
        """Determines query intent based on pattern matching and keywords.
        
        Args:
            query: User search query.
            
        Returns:
            The classified intent code.
        """
        if not query:
            return "QUESTION"
            
        clean_query = query.strip().lower()
        
        # 1. Summary keywords
        summary_kws = [
            "summarize", "summary", "tl;dr", "tldr", "digest", "condense", 
            "synopsis", "shorten", "சுருக்கம்", "சுருக்கமாக", "சுருக்கி"
        ]
        if any(kw in clean_query for kw in summary_kws):
            logger.info("Detected SUMMARY intent.")
            return "SUMMARY"
            
        # 2. Overview keywords
        overview_kws = [
            "overview", "about this", "document about", "what is this", "main topic", 
            "what does this cover", "pattriya", "பற்றிய", "பற்றி", "தொகுப்பு", "விளக்கம்"
        ]
        if any(kw in clean_query for kw in overview_kws):
            logger.info("Detected OVERVIEW intent.")
            return "OVERVIEW"
            
        # 3. Translation keywords
        translation_kws = [
            "translate", "translation", "in tamil", "in english", 
            "மொழிபெயர்ப்பு", "மொழிபெயர்", "ஆங்கிலத்தில்", "தமிழில்"
        ]
        if any(kw in clean_query for kw in translation_kws):
            logger.info("Detected TRANSLATION intent.")
            return "TRANSLATION"
            
        # 4. General greetings/chat interactions
        general_kws = [
            "hello", "hi", "hey", "who are you", "what can you do", 
            "வணக்கம்", "நன்றி", "thanks", "thank you"
        ]
        if any(clean_query.startswith(kw) or clean_query == kw for kw in general_kws):
            logger.info("Detected GENERAL intent.")
            return "GENERAL"
            
        # 5. Default
        logger.info("Detected QUESTION intent.")
        return "QUESTION"
