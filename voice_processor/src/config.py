from dataclasses import dataclass;

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    chunk_duration: int = 30
    overlap_duration: float = 1.0
    noise_reduction: bool = True
    volume_normalization: bool = True
    min_audio_length: int = 0.5
    max_audio_length: int = 300