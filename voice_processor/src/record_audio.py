import sounddevice as sd
from scipy.io.wavfile import write
import os

def record_audio(filename="D:\Voice_rag\\voice_processor\\audio-samples\mic_input.wav", duration=10, sample_rate=16000):
    print(f"🎙️ Recording for {duration} seconds...")

    try:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  
        write(filename, sample_rate, recording)
        print(f"✅ Saved recording to: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Mic recording failed: {e}")
        return None

if __name__ == "__main__":
    record_audio(duration=10)
