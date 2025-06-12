from rag_system.system.rag import RAGSystem
from rag_system.configs.chunking_config import ChunkingConfig
from rag_system.configs.embedding_config import EmbeddingConfig
from rag_system.configs.retrieval_config import RetrievalConfig
from voice_output.emotion_detector import EmotionDetector
from voice_output.tts_engine import TextToSpeechEngine
from google import genai
from google.genai import types
import os      


from dotenv import load_dotenv
load_dotenv()


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
def ask_query(query,sentiment):
    # Init system
    rag = RAGSystem(
        chunking_config=ChunkingConfig(),
        embedding_config=EmbeddingConfig(),
        retrieval_config=RetrievalConfig()
    )

    rag.build_knowledge_base("rag_system/data/knowledge")
    rag.load_knowledge_base()

    emotion_detector = EmotionDetector()
    tts = TextToSpeechEngine()
    


    emotion = emotion_detector.detect_emotion(query)
    result = rag.query(query, return_chunks=True)
        # print("\n📄 Retrieved Chunks:")
        # for i, chunk in enumerate(result["chunks"]):
        #     print(f"\n🔹 Chunk {i + 1}:\n{chunk['metadata'] if isinstance(chunk, dict) else chunk.metadata}")
        #     print(chunk["content"] if isinstance(chunk, dict) else chunk.page_content)


    prompt = f"""
    Context:
    {result}\n\n
        
    Question: {query}. 

    Tone of the question: {emotion}.

    Sentiment of the question: {sentiment}.
        
    """
    print("\n📚 Final Prompt Sent to LLM:\n", prompt)


    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config = types.GenerateContentConfig(
            system_instruction="You have been given a question, it's tone/emotion, sentiment and context related to query. Your task is to form a decent answer using context. Use greetings based on emotion and sentiment. Ask about him if the tone is depressed or sad. You are a helpful assistant who always tries to help the user. You are also a good listener and try to understand the user's feelings. If the user is sad, you try to cheer him up. If the user is happy, you try to celebrate with him. If the user is angry, you try to calm him down. If the user is neutral, you try to engage him in a conversation. You are also a good friend who always tries to be there for the user. And the very important point sometimes, you may not find exact key word in context, but a similar sounding word. For example, user said, 'TradeGuard' and it is transcribed as 'Freedgard'. Now 'TradeGuard' is present in context. So consider them same and answer relevantly, only from context, rather than saying, you don't know about the question's context.",
            temperature=0.5, 
        ),
        contents=prompt,
    )
    answer = response.text.strip()  
    return answer  
    
        

