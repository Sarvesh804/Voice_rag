import asyncio
import edge_tts

class TextToSpeechEngine:
    def __init__(self, voice="hi-IN-SwaraNeural"):
        self.voice = voice

    async def speak(self, text: str, output_file: str = "voice_output/output/output.wav"):
        if not isinstance(text, str):
            raise TypeError("Expected `text` to be a string, got: " + str(type(text)))

        communicate = edge_tts.Communicate(text, voice=self.voice)
        await communicate.save(output_file)
        print(f"🔊 Speech saved to {output_file}")
