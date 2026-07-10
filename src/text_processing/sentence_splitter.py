import re
from typing import List

class SentenceSplitter:
    """Splits continuous text into coherent sentences while respecting multilingual boundaries 
    and common abbreviation patterns.
    """
    
    @staticmethod
    def split(text: str) -> List[str]:
        """Splits input text into individual sentences.
        
        Args:
            text: Input text string.
            
        Returns:
            A list of sentence strings.
        """
        if not text:
            return []
            
        # Common abbreviation tokens to prevent splitting (without trailing periods)
        abbreviations = {
            "mr", "ms", "mrs", "dr", "prof", "vs", "eg", "ie", "etc", "ta", "en", 
            "approx", "gen", "vol", "p.m", "a.m", "p", "pg", "al", "jan", "feb", 
            "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
        }
        
        # Regex to split on . or ? or ! followed by one or more spaces
        pattern = re.compile(r'(?<=[.!?])\s+')
        raw_splits = pattern.split(text.strip())
        
        sentences = []
        buffer_sent = ""
        
        for segment in raw_splits:
            segment_clean = segment.strip()
            if not segment_clean:
                continue
                
            # If buffer contains text, merge it
            candidate = f"{buffer_sent} {segment_clean}".strip() if buffer_sent else segment_clean
            
            # Check the last word of the candidate sentence
            words = candidate.split()
            if words:
                last_word = words[-1].lower().rstrip(".!?")
                if last_word in abbreviations:
                    # Keep accumulating in the buffer
                    buffer_sent = candidate
                    continue
                    
            # Complete sentence, append and reset buffer
            sentences.append(candidate)
            buffer_sent = ""
            
        # If there's any remaining segment in the buffer
        if buffer_sent:
            sentences.append(buffer_sent)
            
        return sentences
