import re
from src.utils.logger import logger

class ResponseValidator:
    """Validates generation properties, catching empty text, short snippets, and hallucinated entities."""
    
    @staticmethod
    def validate(answer: str, context: str, query: str) -> bool:
        """Inspects generation text against source text inputs.
        
        Args:
            answer: Generated LLM response text.
            context: Text context payload formatted by ContextBuilder.
            query: User's cleaned question.
            
        Returns:
            True if output passes all validations.
        """
        # 1. Empty/Whitespace check
        if not answer or not answer.strip():
            logger.warning("Validation Failed: Empty generation received.")
            return False
            
        cleaned_answer = answer.strip()
        
        # Determine if it's a standard decline message
        declines = [
            "sorry", "provided documents", "information is not", 
            "do not contain", "not present", "நாங்கள்", "தகவல் இல்லை"
        ]
        is_decline = any(dec in cleaned_answer.lower() for dec in declines)
        
        # 2. Reasonable length check
        if len(cleaned_answer) < 10 and not is_decline:
            logger.warning(f"Validation Failed: Excessively short answer ({len(cleaned_answer)} characters).")
            return False
            
        # 3. Entity Validation: Detect English proper nouns introduced in generation
        # Matches capitalized words of length 2 or more
        entities = set(re.findall(r"\b[A-Z][a-z]+\b", cleaned_answer))
        
        # Standard English proper noun stopwords to skip
        stopwords = {
            "The", "A", "An", "And", "But", "Or", "If", "Because", "As", 
            "Until", "While", "Of", "At", "By", "For", "With", "About", 
            "Against", "Between", "Into", "Through", "During", "Before", 
            "After", "Above", "Below", "To", "From", "Up", "Down", "In", 
            "Out", "On", "Off", "Over", "Under", "Again", "Further", 
            "Then", "Once", "Here", "There", "When", "Where", "Why", "How",
            "I", "You", "He", "She", "It", "We", "They", "What", "This",
            "No", "Yes", "Not", "Is", "Are", "Was", "Were", "Be", "Been", "Being"
        }
        filtered_entities = entities - stopwords
        
        reference_pool = (context + " " + query).lower()
        
        for entity in filtered_entities:
            # Check if entity proper noun is absent in query and context
            if entity.lower() not in reference_pool:
                logger.warning(f"Validation Failed: Hallucinated entity '{entity}' is not present in retrieved context.")
                return False
                
        logger.info("Response validator check passed successfully.")
        return True
