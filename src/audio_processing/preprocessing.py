import av
import numpy as np
from src.utils.logger import logger

class AudioPreprocessor:
    """Preprocesses diverse audio containers into mono, 16kHz float32 arrays required by Whisper."""
    
    @staticmethod
    def preprocess(file_path: str, target_sr: int = 16000) -> np.ndarray:
        """Decodes, resamples, averages layout channels to mono, and normalizes volume.
        
        Args:
            file_path: Path to the audio file.
            target_sr: Target sample rate in Hz (default: 16000).
            
        Returns:
            A 1D numpy float32 array containing normalised audio waveforms.
        """
        logger.info(f"Preprocessing audio: {file_path}")
        container = None
        try:
            container = av.open(file_path)
            if not container.streams.audio:
                raise ValueError(f"No audio streams found in {file_path}")
                
            stream = container.streams.audio[0]
            
            # Setup resampler: target format float (flt), layout mono, rate 16000Hz
            resampler = av.AudioResampler(
                format="flt",  # Float 32-bit format
                layout="mono", # Single channel
                rate=target_sr # 16kHz
            )
            
            all_chunks = []
            for frame in container.decode(stream):
                # Resample frame
                resampled_frames = resampler.resample(frame)
                for resampled_frame in resampled_frames:
                    # Convert to ndarray. Shape for mono flt is (1, samples)
                    arr = resampled_frame.to_ndarray()
                    all_chunks.append(arr[0])
            
            # Flush resampler buffer
            flushed_frames = resampler.resample(None)
            for resampled_frame in flushed_frames:
                arr = resampled_frame.to_ndarray()
                all_chunks.append(arr[0])
                
            if not all_chunks:
                raise ValueError("Resampler output is empty. No audio frames decoded.")
                
            # Concatenate all resampled chunks
            audio_data = np.concatenate(all_chunks).astype(np.float32)
            
            # Perform Peak Volume Normalization
            max_peak = np.max(np.abs(audio_data))
            if max_peak > 0:
                audio_data = audio_data / max_peak
                logger.debug(f"Volume normalized using max peak factor: {max_peak:.4f}")
                
            # Simple threshold-based silence trimming (remove leading/trailing silence < 1% amplitude)
            silence_threshold = 0.01
            non_silent = np.where(np.abs(audio_data) > silence_threshold)[0]
            if len(non_silent) > 0:
                start_idx = non_silent[0]
                end_idx = non_silent[-1]
                # Keep some padding (e.g. 0.1s = 1600 samples)
                padding = 1600
                start_padded = max(0, start_idx - padding)
                end_padded = min(len(audio_data), end_idx + padding)
                audio_data = audio_data[start_padded:end_padded]
                logger.debug(f"Silence trimmed: original length {len(all_chunks)*1024} -> trimmed {len(audio_data)} samples.")
                
            logger.info(f"Audio preprocessing complete: {len(audio_data)} samples extracted at {target_sr}Hz.")
            return audio_data
            
        except Exception as e:
            logger.error(f"Error during audio preprocessing: {str(e)}")
            raise e
        finally:
            if container:
                container.close()
