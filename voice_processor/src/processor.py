from voice_processor.src.config import AudioConfig
from voice_processor.src.preprocessing import AudioPreprocessor
from voice_processor.src.transcriber import WhisperTranscriber
import gc


class VoiceInputProcessor:
    def __init__(self, model_size="base", device="cpu", config=None):
        self.config = config or AudioConfig()
        self.preprocessor = AudioPreprocessor(self.config)
        self.transcriber = WhisperTranscriber(model_size="small", device="cpu")

    def process_audio_file(self, path):
        chunks = self.preprocessor.preprocess(path)
        combined_text = []
        segments = []

        for chunk in chunks:
            result = self.transcriber.transcribe(chunk)
            if result and result.get("text", "").strip():
                combined_text.append(result["text"])
                segments.extend(result.get("segments", []))

        confidence = self.calculate_confidence(segments)
        language = segments[0].get("language", "en") if segments else "en"
        del chunks
        gc.collect()

        return {
            "text": " ".join(combined_text),
            "confidence": confidence,
            "language": language,
            "segments": segments
        }

    def calculate_confidence(self, segments):
        if not segments:
            return 0.0
        total_duration = sum(s.get("end", 0) - s.get("start", 0) for s in segments)
        avg_segment_length = total_duration / len(segments)
        return min(1.0, avg_segment_length / 2.0)