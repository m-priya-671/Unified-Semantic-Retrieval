import os
import av
import time
from typing import Dict, Any
from src.audio_processing.base import TranscriptionResult

class AudioMetadataGenerator:
    """Extracts container metadata and aggregates speech recognition metrics into standard schema."""
    
    @staticmethod
    def generate(
        file_path: str,
        transcription_result: TranscriptionResult,
        processing_duration: float
    ) -> Dict[str, Any]:
        """Gathers stream metadata and collates transcription segments.
        
        Args:
            file_path: Absolute or relative path to the audio file.
            transcription_result: Standardized STT transcription result.
            processing_duration: Total processing pipeline duration in seconds.
            
        Returns:
            A metadata dictionary matching the Offline RAG ingestion schema.
        """
        duration = 0.0
        sample_rate = 0
        channels = 0
        
        try:
            with av.open(file_path) as container:
                if container.streams.audio:
                    stream = container.streams.audio[0]
                    
                    # Convert duration to seconds if time_base is valid
                    if stream.duration and stream.time_base:
                        duration = float(stream.duration * stream.time_base)
                    else:
                        # Fallback: estimate from container duration
                        duration = float(container.duration / av.time_base) if container.duration else 0.0
                        
                    sample_rate = stream.rate or 0
                    channels = stream.channels or 0
        except Exception:
            # Silence header read exceptions to ensure robust metadata mapping
            pass
            
        metadata = {
            "source": os.path.basename(file_path),
            "file_path": os.path.abspath(file_path),
            "file_type": os.path.splitext(file_path)[1].lower().lstrip("."),
            "duration_sec": round(duration, 2),
            "sample_rate": sample_rate,
            "channels": channels,
            "languages": transcription_result.languages,
            "segments": transcription_result.segments,
            "speaker": "Unknown",          # Future diarization placeholder
            "speaker_count": 1,            # Future diarization placeholder
            "transcription_engine": transcription_result.engine_used,
            "processing_duration_sec": round(processing_duration, 2),
            "transcript_length_chars": len(transcription_result.text),
            "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return metadata
