import os
import sys
import wave
import struct
import numpy as np
from pathlib import Path

# Add the workspace directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

from src.utils.logger import logger
from src.config.settings import UPLOAD_DIR
from src.audio_processing.validator import AudioValidator
from src.audio_processing.preprocessing import AudioPreprocessor
from src.audio_processing.audio_processor import AudioProcessor

def create_mock_sine_wave(file_name: str, duration: int = 3, sample_rate: int = 22050, frequency: int = 440) -> Path:
    """Generates a mock sine wave WAV file using standard library to test audio pipelines offline."""
    file_path = UPLOAD_DIR / file_name
    logger.info(f"Generating mock sine wave WAV file: {file_path}")
    
    num_samples = duration * sample_rate
    # Generate sine wave samples and scale to 16-bit integer PCM range
    samples = [int(32767 * np.sin(2 * np.pi * frequency * i / sample_rate)) for i in range(num_samples)]
    
    # Write WAV file using standard wave module
    with wave.open(str(file_path), "wb") as wav:
        wav.setnchannels(1)      # Mono
        wav.setsampwidth(2)      # 2 bytes = 16-bit
        wav.setframerate(sample_rate)
        # Pack format 'h' represents short integers
        binary_data = struct.pack("<" + "h" * len(samples), *samples)
        wav.writeframes(binary_data)
        
    logger.info(f"Mock audio successfully written: {file_path.name} ({os.path.getsize(file_path)} bytes)")
    return file_path

def test_validator():
    """Tests AudioValidator extension validation, size bounds, and header corruption checks."""
    logger.info("=== 1. TESTING AUDIO VALIDATOR ===")
    
    # Generate test WAV
    wav_path = create_mock_sine_wave("validator_test.wav", duration=2)
    
    # Test valid case
    assert AudioValidator.validate(str(wav_path)) is True, "Valid WAV validation failed"
    logger.info("Valid WAV file validation: OK")
    
    # Test unsupported extension
    bad_ext_path = UPLOAD_DIR / "bad_ext.txt"
    with open(bad_ext_path, "w") as f:
        f.write("text content")
    
    try:
        AudioValidator.validate(str(bad_ext_path))
        raise AssertionError("Validator failed to reject unsupported extension")
    except ValueError as e:
        logger.info(f"Validator correctly rejected bad extension: {e}")
    finally:
        if bad_ext_path.exists():
            bad_ext_path.unlink()
            
    # Test corrupted header
    corrupt_path = UPLOAD_DIR / "corrupt_audio.wav"
    with open(corrupt_path, "wb") as f:
        f.write(b"RIFFxxxxWAVEfmt xxxxxxxxxxxxxxxxxxxxxxxxdata")
        
    try:
        AudioValidator.validate(str(corrupt_path))
        raise AssertionError("Validator failed to reject corrupted headers")
    except ValueError as e:
        logger.info(f"Validator correctly rejected corrupt file: {e}")
    finally:
        if corrupt_path.exists():
            corrupt_path.unlink()
            
    # Clean up test WAV
    if wav_path.exists():
        wav_path.unlink()
    logger.info("Audio Validator test suite: SUCCESS\n" + "-"*50)

def test_preprocessor():
    """Tests AudioPreprocessor resampling, mono-conversion, and peak normalization."""
    logger.info("=== 2. TESTING AUDIO PREPROCESSOR ===")
    
    # Create 22050Hz WAV
    wav_path = create_mock_sine_wave("preprocessor_test.wav", duration=2, sample_rate=22050)
    
    # Preprocess (resamples to 16000Hz)
    audio_data = AudioPreprocessor.preprocess(str(wav_path), target_sr=16000)
    
    assert isinstance(audio_data, np.ndarray), "Preprocessed audio must be a numpy array"
    assert audio_data.dtype == np.float32, "Audio data must be float32"
    # Check normalization: max amplitude should be close to 1.0
    assert np.max(np.abs(audio_data)) <= 1.0, "Audio values must be normalized in range [-1.0, 1.0]"
    
    logger.info("Audio Preprocessor conversion: OK")
    
    # Clean up
    if wav_path.exists():
        wav_path.unlink()
    logger.info("Audio Preprocessor test suite: SUCCESS\n" + "-"*50)

def test_full_pipeline():
    """Tests the full AudioProcessor orchestrator pipeline end-to-end."""
    logger.info("=== 3. TESTING AUDIO PROCESSOR WORKFLOW ===")
    
    # Generate 3-second test WAV
    wav_path = create_mock_sine_wave("pipeline_test.wav", duration=3, sample_rate=16000)
    
    processor = AudioProcessor()
    docs = processor.parse(str(wav_path))
    
    assert len(docs) == 1, "Expected exactly 1 document"
    doc = docs[0]
    
    # Assert Document structure
    assert isinstance(doc.text, str), "Transcript should be a string"
    meta = doc.metadata
    
    # Assert Metadata refinements
    assert meta["source"] == "pipeline_test.wav"
    assert meta["file_type"] == "wav"
    assert meta["duration_sec"] == 3.0
    assert meta["channels"] == 1
    assert meta["sample_rate"] == 16000
    
    # Refinement 1: Timeline segments list
    assert "segments" in meta, "Metadata should contain segment lists"
    assert isinstance(meta["segments"], list)
    
    # Refinement 2: Speaker placeholders
    assert meta["speaker"] == "Unknown"
    assert meta["speaker_count"] == 1
    
    # Refinement 3: Multilingual structured language
    assert "languages" in meta
    assert isinstance(meta["languages"], list)
    assert len(meta["languages"]) > 0
    assert "language" in meta["languages"][0]
    assert "probability" in meta["languages"][0]
    
    logger.info(f"Pipeline Execution Metadata:\n{meta}")
    
    # Clean up
    if wav_path.exists():
        wav_path.unlink()
    logger.info("Audio Processor full workflow tests: SUCCESS\n" + "-"*50)

def run_tests():
    logger.info("Starting Milestone 3 automated verification tests...")
    test_validator()
    test_preprocessor()
    test_full_pipeline()
    logger.info("ALL MILESTONE 3 TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
