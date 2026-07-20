import re
from typing import Tuple, Dict, Any, List, Set
from src.utils.logger import logger

class ResponseValidator:
    """Validates generation properties using semantic grounding checks, preventing false validation failures."""
    
    # Common English stopwords, connectives, and formatting words to ignore during entity checks
    IGNORED_WORDS: Set[str] = {
        "The", "A", "An", "And", "But", "Or", "If", "Because", "As", 
        "Until", "While", "Of", "At", "By", "For", "With", "About", 
        "Against", "Between", "Into", "Through", "During", "Before", 
        "After", "Above", "Below", "To", "From", "Up", "Down", "In", 
        "Out", "On", "Off", "Over", "Under", "Again", "Further", 
        "Then", "Once", "Here", "There", "When", "Where", "Why", "How",
        "I", "You", "He", "She", "It", "We", "They", "What", "This", "That", "These", "Those",
        "No", "Yes", "Not", "Is", "Are", "Was", "Were", "Be", "Been", "Being", "Have", "Has", "Had",
        "Do", "Does", "Did", "Can", "Could", "Should", "Would", "May", "Might", "Must",
        "According", "Context", "Document", "Summary", "Overview", "Section", "Question", "Answer",
        "Furthermore", "However", "Therefore", "Moreover", "Additionally", "Overall", "Note",
        "Result", "Information", "Details", "Key", "Main", "Points", "Based", "Provided", "System"
    }

    DECLINE_PHRASES = [
        "sorry", "provided documents", "information is not", "do not contain", 
        "not present", "limited information", "unavailable", "நாங்கள்", "தகவல் இல்லை"
    ]

    @classmethod
    def validate(cls, answer: str, context: str, query: str, intent: str = "QUESTION") -> bool:
        """Validates generation text against source text inputs.
        
        Returns:
            True if output passes grounding validation checks.
        """
        is_valid, _ = cls.validate_detailed(answer, context, query, intent=intent)
        return is_valid

    @classmethod
    def validate_detailed(
        cls, 
        answer: str, 
        context: str, 
        query: str, 
        intent: str = "QUESTION"
    ) -> Tuple[bool, Dict[str, Any]]:
        """Inspects response for semantic grounding, extracting unsupported entities and numbers.
        
        Returns:
            Tuple of (is_valid_boolean, diagnostics_dictionary)
        """
        # 1. Empty/Whitespace check
        if not answer or not answer.strip():
            diag = {
                "validation_passed": False,
                "validation_failed": True,
                "validation_reason": "Empty or whitespace response received",
                "unsupported_claims_found": ["Empty response"],
                "unsupported_entities_found": [],
                "semantic_similarity_score": 0.0
            }
            logger.warning("Validation Failed: Empty generation received.")
            return False, diag

        cleaned_answer = answer.strip()
        reference_pool = f"{context} {query}".lower()

        # 2. Check standard decline responses
        is_decline = any(phrase in cleaned_answer.lower() for phrase in cls.DECLINE_PHRASES)
        if is_decline:
            diag = {
                "validation_passed": True,
                "validation_failed": False,
                "validation_reason": "Valid decline response",
                "unsupported_claims_found": [],
                "unsupported_entities_found": [],
                "semantic_similarity_score": 1.0
            }
            logger.info("Validation Passed: Standard decline response recognized.")
            return True, diag

        # 3. Extract Unsupported Numbers & Dates
        # Ignore markdown formatting like [1], [Chunk 1], 1., 2.
        sanitized_for_nums = re.sub(r"\[(?:Chunk\s*)?\d+\]", " ", cleaned_answer)
        sanitized_for_nums = re.sub(r"^\s*\d+[\.\)]\s+", " ", sanitized_for_nums, flags=re.MULTILINE)
        
        # Extract distinct numbers / dates (2 or more digits, or decimals, or years)
        numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", sanitized_for_nums))
        unsupported_numbers = []
        for num in numbers:
            clean_num = num.replace("%", "").strip()
            if clean_num not in reference_pool and num.lower() not in reference_pool:
                unsupported_numbers.append(num)

        # 4. Extract Unsupported Named Entities (Proper Nouns / Capitalized Phrases)
        candidate_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", cleaned_answer))
        unsupported_entities = []
        
        for ent in candidate_entities:
            if ent in cls.IGNORED_WORDS:
                continue
            if ent.lower() not in reference_pool:
                unsupported_entities.append(ent)

        # 5. Semantic Similarity / Overlap Score
        content_words = [
            w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", cleaned_answer)
            if w.title() not in cls.IGNORED_WORDS
        ]
        if content_words:
            matched_words = [w for w in content_words if w in reference_pool]
            semantic_score = round(len(matched_words) / len(content_words), 2)
        else:
            semantic_score = 1.0

        # Combine unsupported claims
        unsupported_claims = []
        if unsupported_numbers:
            unsupported_claims.append(f"Unsupported numbers/dates: {', '.join(unsupported_numbers)}")
        if unsupported_entities:
            unsupported_claims.append(f"Unsupported entities: {', '.join(unsupported_entities)}")

        # Validation Decision
        is_valid = True
        reason = "Passed semantic grounding checks"

        if unsupported_entities:
            is_valid = False
            reason = f"Unsupported entity found: {', '.join(unsupported_entities)}"
        elif unsupported_numbers:
            is_valid = False
            reason = f"Unsupported number/date found: {', '.join(unsupported_numbers)}"
        elif semantic_score < 0.25 and not is_decline:
            is_valid = False
            reason = f"Low semantic content overlap ({semantic_score})"

        diag = {
            "validation_passed": is_valid,
            "validation_failed": not is_valid,
            "validation_reason": reason,
            "unsupported_claims_found": unsupported_claims,
            "unsupported_entities_found": unsupported_entities + unsupported_numbers,
            "semantic_similarity_score": semantic_score
        }

        if is_valid:
            logger.info(f"Response validator check passed successfully (Semantic Score: {semantic_score}).")
        else:
            logger.warning(f"Response validation failed: {reason}")

        return is_valid, diag
