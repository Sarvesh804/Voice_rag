import librosa
import numpy as np
from voice_processor.src.config import AudioConfig
from voice_processor.src.utils import init_logger

class AudioPreprocessor:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.logger = init_logger(__name__)

    def load_audio(self, audio_path: str):
        try:
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            self.logger.info(f"Loaded audio: {len(audio)/sr:.2f}s at {sr}Hz")
            return audio, sr
        except Exception as e:
            self.logger.error(f"Error loading audio: {e}")
            raise

    def validate_audio(self, audio: np.ndarray, sr: int) -> bool:
        duration = len(audio) / sr
        if duration < self.config.min_audio_length:
            self.logger.warning(f"Audio too short: {duration:.2f}s")
            return False
        if duration > self.config.max_audio_length:
            self.logger.warning(f"Audio too long: {duration:.2f}s")
            return False
        if np.max(np.abs(audio)) < 0.001:
            self.logger.warning("Audio appears to be silent")
            return False
        return True

    def reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        noise_sample_size = int(0.5 * sr)
        noise_floor = np.mean(np.abs(audio[:noise_sample_size]))
        threshold = noise_floor * 2
        denoised = np.where(np.abs(audio) > threshold, audio, audio * 0.1)
        return denoised

    def normalize_volume(self, audio: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 0.1
            normalized = audio * (target_rms / rms)
            return np.clip(normalized, -1.0, 1.0)
        return audio

    def segment_audio(self, audio: np.ndarray, sr: int) -> list:
        duration = len(audio) / sr
        if duration <= self.config.chunk_duration:
            return [audio]

        chunk_samples = int(self.config.chunk_duration * sr)
        overlap_samples = int(self.config.overlap_duration * sr)
        step = chunk_samples - overlap_samples

        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            if len(chunk) >= int(self.config.min_audio_length * sr):
                chunks.append(chunk)
            start += step

        return chunks

    def preprocess(self, audio_path: str) -> list:
        audio, sr = self.load_audio(audio_path)

        if not self.validate_audio(audio, sr):
            raise ValueError("Audio validation failed")

        if self.config.noise_reduction:
            audio = self.reduce_noise(audio, sr)

        if self.config.volume_normalization:
            audio = self.normalize_volume(audio)

        chunks = self.segment_audio(audio, sr)
        self.logger.info(f"Preprocessed audio into {len(chunks)} chunks")
        return chunks
