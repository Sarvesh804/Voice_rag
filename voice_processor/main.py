import time
from voice_processor.src.config import AudioConfig
from voice_processor.src.processor import VoiceInputProcessor
from transformers import pipeline
import os
from google import genai
from dotenv import load_dotenv
from google.genai import types
from voice_processor.src.record_audio import record_audio



load_dotenv()
USE_LLM_REFINEMENT = False

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))


def clean_transcription(text):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                system_instruction='''You are an expert transcription cleaner. Your task is to refine noisy audio transcriptions for clarity and coherence. Follow some rules given below:
                1. Correct grammar and punctuation errors. Also the original sentiment of text should be preserved.
                2. Maintain the original meaning and intent of the text. Try to predict the meaningful word sounding similar to the original noisy word.
                3. Ensure the text is coherent and flows naturally.
                4. If the text is already clear, return it as is.
                5. If the text is too noisy, and you are able to understand the context, try to reconstruct it as best as you can.
                6. If you cannot understand the context, drop that part and return rest of text having some nearst meaning.
                '''),
                contents=text,
            )
            return response.text
        except Exception as e:
            print(f"[LLM Error]: {e}")
            return text


def analyze_sentiment(text):
    try:
        return sentiment_analyzer(text[:512])[0] 
    except Exception as e:
        return {"label": "NEUTRAL", "score": 0.0}
    

def run_voice_pipeline():
    start = time.time()
    config = AudioConfig(
        sample_rate=16000,
        chunk_duration=15,
        overlap_duration=1.0,
        noise_reduction=True,
        volume_normalization=True
    )

    processor = VoiceInputProcessor(
        model_size="tiny",
        device="cpu",
        config=config
    )

    # audio_path = record_audio(duration=10)
    audio_path = "D:\Voice_rag\\voice_processor\\audio-samples\mic.wav"
    if not audio_path:
        print("❌ Failed to record audio. Exiting.")
    chunks = processor.preprocessor.preprocess(audio_path)
    print(f"[INFO] {len(chunks)} chunks created")

    
    
    results = []

    for chunk in chunks:
        results.append(processor.transcriber.transcribe(chunk))

    
    all_text = [r["text"] for r in results if r["text"].strip()]
    segments = [s for r in results for s in r.get("segments", [])]


    cleaned_text = " ".join(all_text)
    # cleaned_text = clean_transcription(combined_text)
    confidence = processor.calculate_confidence(segments)
    sentiment = analyze_sentiment(cleaned_text)
    duration = time.time() - start

    print("\n📤 Final Output:")
    print(f"📝 Transcription:\n{cleaned_text}\n")
    print(f"🧠 Sentiment: {sentiment['label']} ({sentiment['score']:.2f})")
    print(f"✅ Confidence: {confidence:.2f}")
    print(f"⏱️ Processing Time: {duration:.2f}s")


    return {
        "query": cleaned_text,
        "sentiment": sentiment,
        "confidence": confidence,
        "processing_time": duration,
        "segments": segments
    }


