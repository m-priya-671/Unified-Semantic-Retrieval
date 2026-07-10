import time
from typing import List
from src.ingestion.base import BaseParser, Document
from src.audio_processing.validator import AudioValidator
from src.audio_processing.preprocessing import AudioPreprocessor
from src.audio_processing.transcription_engine import FasterWhisperEngine
from src.audio_processing.metadata import AudioMetadataGenerator
from src.utils.logger import logger

class AudioProcessor(BaseParser):
    """Facade orchestrator that validates, preprocesses, and transcribes local audio files 
    into standard Document formats using Faster-Whisper.
    """
    
    def __init__(self):
        """Initializes the audio processor. The transcription engine is instantiated lazily 
        to avoid loading model weights into memory until transcription is actually executed.
        """
        self.engine = None

    def parse(self, file_path: str) -> List[Document]:
        """Runs the validation, preprocessing, and transcription pipeline on an audio file.
        
        Args:
            file_path: Absolute or relative path to the audio file.
            
        Returns:
            A list containing a single Document object with the full transcript text and metadata.
        """
        logger.info(f"Starting audio parsing workflow for: {file_path}")
        start_time = time.time()
        
        # 1. Validate Audio File
        AudioValidator.validate(file_path)
        
        # 2. Preprocess Audio to standard PCM float32 waveform
        audio_data = AudioPreprocessor.preprocess(file_path)
        
        # 3. Transcribe Audio (Lazy engine initialization)
        if self.engine is None:
            self.engine = FasterWhisperEngine()
            
        transcription_result = self.engine.transcribe(audio_data)
        
        # 4. Collate Metadata
        total_duration = time.time() - start_time
        metadata = AudioMetadataGenerator.generate(
            file_path=file_path,
            transcription_result=transcription_result,
            processing_duration=total_duration
        )
        
        doc = Document(
            text=transcription_result.text,
            metadata=metadata
        )
        
        logger.info(f"Audio parsing workflow successfully completed in {total_duration:.2f}s.")
        return [doc]
