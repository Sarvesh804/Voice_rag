import streamlit as st
import os
import tempfile
from voice_processor.main import run_voice_pipeline
from rag_system.system.rag import RAGSystem
from voice_output.tts_engine import TextToSpeechEngine
import asyncio
from rag_system.configs.chunking_config import ChunkingConfig
from rag_system.configs.embedding_config import EmbeddingConfig
from rag_system.configs.retrieval_config import RetrievalConfig
from main import ask_query

st.set_page_config(page_title="Voice RAG Assistant", layout="wide")

st.title("Voice RAG: Assistant")

uploaded_file = st.file_uploader("Upload Guideline PDF (ex: medical documents)", type=["pdf"])

if uploaded_file:
    os.makedirs("rag_system/data/knowledge", exist_ok=True)
    save_path = f"rag_system/data/knowledge/{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())


    st.success("PDF uploaded. You can now ask questions.")

rag = RAGSystem(ChunkingConfig(), EmbeddingConfig(), RetrievalConfig())
rag.build_knowledge_base("rag_system/data/knowledge")

# Query
query_type = st.radio("Choose input method", ["🗣️ Voice", "⌨️ Text"])
query = ""
sentiment = ""

if query_type == "🗣️ Voice":
    if st.button("🎙️ Record & Ask"):
        result = run_voice_pipeline()
        query = result["query"]
        sentiment = result["sentiment"]["label"] if result["sentiment"] else "Neutral"
        st.write("Detected Sentiment:", sentiment)
        st.write("You asked:", query)
elif query_type == "⌨️ Text":
    query = st.text_input("Type your query")

if query:
    
    # st.text_area("Context from medical documents:", context, height=200)

    answer = ask_query(query, sentiment)

    st.markdown("### 💬 Assistant Response")
    st.success(answer)

    
    tts = TextToSpeechEngine()
    output_path = "voice_output/output/output.wav"
    asyncio.run(tts.speak(answer, output_file=output_path))

    st.audio(output_path, format="audio/wav")

    # Optional: face animation (simplified emoji placeholder)
    st.markdown("#### 🗣️ Speaking Face ")
    # st.image("ui/assets/speaking.gif") 
