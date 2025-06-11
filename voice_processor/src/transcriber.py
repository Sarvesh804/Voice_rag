import whisper

class WhisperTranscriber:
    def __init__(self, model_size="small", device="cpu"):
        self.model = whisper.load_model(model_size, device=device)

    def transcribe(self, audio_chunk):
        return self.model.transcribe(audio_chunk)
    

    