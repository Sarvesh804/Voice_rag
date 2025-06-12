import streamlit as st
import os
import asyncio
import numpy as np
import soundfile as sf
from audiorecorder import audiorecorder

from voice_processor.main import run_voice_pipeline
from rag_system.system.rag import RAGSystem
from voice_output.tts_engine import TextToSpeechEngine
from rag_system.configs.chunking_config import ChunkingConfig
from rag_system.configs.embedding_config import EmbeddingConfig
from rag_system.configs.retrieval_config import RetrievalConfig
from main import ask_query

# ----------------- Streamlit UI Setup -----------------
st.set_page_config(page_title="Voice RAG Assistant", layout="wide")
st.title("🩺 Voice RAG: Assistant")

# ----------------- File Upload -----------------
uploaded_file = st.file_uploader("📄 Upload Guideline PDF (e.g., medical documents)", type=["pdf"])
if uploaded_file:
    os.makedirs("rag_system/data/knowledge", exist_ok=True)
    save_path = f"rag_system/data/knowledge/{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ PDF uploaded. You can now ask questions.")

# ----------------- Load RAG -----------------
rag = RAGSystem(ChunkingConfig(), EmbeddingConfig(), RetrievalConfig())
rag.build_knowledge_base("rag_system/data/knowledge")

# ----------------- Query Input -----------------
st.markdown("### 🔍 Query Input")
query_type = st.radio("Choose input method:", ["🗣️ Voice", "⌨️ Text"], horizontal=True)

query = ""
sentiment = ""

if query_type == "🗣️ Voice":
    st.markdown("#### 🎙️ Record Your Query")
    audio = audiorecorder("Start Recording", "Stop Recording")

    if len(audio) > 0:
        os.makedirs("voice_processor/audio-samples", exist_ok=True)
        audio_path = "voice_processor/audio-samples/mic_input.wav"

        # Convert audio to numpy array and save using soundfile
        samples = np.array(audio.get_array_of_samples())
        channels = audio.channels
        samples = samples.reshape((-1, channels)) if channels > 1 else samples

        sf.write(audio_path, samples, audio.frame_rate, format='WAV')
        st.audio(audio_path, format="audio/wav")
        st.success("✅ Voice recorded. Processing...")

        result = run_voice_pipeline(audio_path)
        query = result["query"]
        sentiment = result["sentiment"]["label"] if result["sentiment"] else "neutral"
        st.write("🧠 Detected Sentiment:", sentiment)
        st.write("📝 Transcription:", query)

elif query_type == "⌨️ Text":
    query = st.text_input("Type your question here:")

# ----------------- Process Query -----------------
if query:
    answer = ask_query(query, sentiment)
    st.markdown("### 💬 Assistant Response")
    st.success(answer)

    # ----------------- TTS -----------------
    tts = TextToSpeechEngine()
    output_path = "voice_output/output/output.wav"
    asyncio.run(tts.speak(answer, output_file=output_path))
    st.audio(output_path, format="audio/wav")

