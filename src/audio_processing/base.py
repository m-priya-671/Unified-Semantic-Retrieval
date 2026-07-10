from abc import ABC, abstractmethod
from typing import List, Dict, Any

class TranscriptionResult:
    """Standardized result containing full text, timestamps, language details, 
    and processing performance metrics for speech-to-text transcriptions.
    """
    
    def __init__(
        self, 
        text: str, 
        segments: List[Dict[str, Any]], 
        languages: List[Dict[str, Any]], 
        processing_time: float, 
        engine_used: str
    ):
        """Initializes the TranscriptionResult.
        
        Args:
            text: The concatenated clean text transcript.
            segments: List of segment dictionaries containing:
                      - 'start': float (start time in seconds)
                      - 'end': float (end time in seconds)
                      - 'text': str (text of the segment)
            languages: List of language dictionaries containing:
                       - 'language': str (detected language ISO code)
                       - 'probability': float (detection confidence)
            processing_time: Execution latency in seconds.
            engine_used: Name of the transcription engine used.
        """
        self.text = text
        self.segments = segments
        self.languages = languages
        self.processing_time = processing_time
        self.engine_used = engine_used

    def __repr__(self) -> str:
        return (f"TranscriptionResult(engine={self.engine_used}, text_len={len(self.text)}, "
                f"segments_count={len(self.segments)}, languages={self.languages}, "
                f"time={self.processing_time:.2f}s)")


class BaseTranscriptionEngine(ABC):
    """Abstract Base Class for local offline speech-to-text engines."""
    
    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribes the input audio file.
        
        Args:
            audio_path: Absolute or relative path to the audio file.
            
        Returns:
            A TranscriptionResult containing full text and segment details.
        """
        pass
