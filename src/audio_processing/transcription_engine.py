import os
import time
import numpy as np
from typing import Union, List, Dict, Any
from faster_whisper import WhisperModel
from src.audio_processing.base import BaseTranscriptionEngine, TranscriptionResult
from src.config.settings import MODELS_DIR, WHISPER_MODEL_NAME
from src.utils.logger import logger

class FasterWhisperEngine(BaseTranscriptionEngine):
    """Faster-Whisper engine for fast local offline transcription on CPU/GPU."""
    
    def __init__(self, model_size: str = None, cpu_threads: int = 4):
        """Initializes the Faster-Whisper model in local offline mode.
        
        Args:
            model_size: Size of the Whisper model (e.g., 'tiny', 'base', 'small').
            cpu_threads: Number of CPU threads to allocate for inference.
        """
        # Read from settings if not explicitly passed
        size = model_size if model_size is not None else WHISPER_MODEL_NAME
        self.model_dir = MODELS_DIR / "faster_whisper"
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info(f"Loading local Faster-Whisper model ({size}) from cache dir: {self.model_dir}")
        start_time = time.time()
        
        # Load Faster-Whisper model. Uses CPU and INT8 quantization by default.
        self.model = WhisperModel(
            model_size_or_path=size,
            device="cpu",
            compute_type="int8",
            download_root=str(self.model_dir),
            cpu_threads=cpu_threads
        )
        self.model_size = size
        logger.info(f"Faster-Whisper engine loaded successfully in {time.time() - start_time:.2f}s.")

    def transcribe(self, audio: Union[str, np.ndarray]) -> TranscriptionResult:
        """Transcribes an audio file or preprocessed audio waveform numpy array.
        
        Args:
            audio: File path to the audio file, or a 1D float32 numpy array.
            
        Returns:
            A TranscriptionResult containing clean text, timeline segments, and languages list.
        """
        logger.info("Starting audio transcription...")
        start_time = time.time()
        
        # transcribe returns a generator of segments, and transcription info
        segments_gen, info = self.model.transcribe(
            audio,
            beam_size=5,
            vad_filter=True, # Voice Activity Detection to filter out long silences
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        segments = []
        clean_text_parts = []
        
        # Iterate over generator to execute transcription
        for segment in segments_gen:
            segment_text = segment.text.strip()
            if segment_text:
                clean_text_parts.append(segment_text)
                segments.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment_text
                })
                
        # Format detected language info
        languages = [
            {
                "language": info.language,
                "probability": round(info.language_probability, 4)
            }
        ]
        
        clean_text = " ".join(clean_text_parts).strip()
        processing_time = time.time() - start_time
        
        logger.info(f"Audio transcription completed in {processing_time:.2f}s. "
                    f"Segments: {len(segments)}, Language detected: {info.language}")
                    
        return TranscriptionResult(
            text=clean_text,
            segments=segments,
            languages=languages,
            processing_time=processing_time,
            engine_used=f"Faster-Whisper ({self.model_size})"
        )
