from typing import List
from rag_system.core.document_processor import DocumentProcessor
from rag_system.core.embedding_models import EmbeddingModel
from rag_system.core.vector_store import VectorStore
from rag_system.core.retriever import Retriever
from rag_system.configs.chunking_config import ChunkingConfig
from rag_system.configs.embedding_config import EmbeddingConfig
from rag_system.configs.retrieval_config import RetrievalConfig

class RAGSystem:
    def __init__(self,
                 chunking_config: ChunkingConfig,
                 embedding_config: EmbeddingConfig,
                 retrieval_config: RetrievalConfig,
                 index_path: str = "rag_index"):

        self.embedder = EmbeddingModel(embedding_config)
        self.doc_processor = DocumentProcessor(chunking_config, self.embedder)
        self.vector_store = VectorStore(self.embedder, index_path=index_path)
        self.retriever = Retriever(self.vector_store, retrieval_config)

    def build_knowledge_base(self, data_dir: str):
        print("📚 Loading & chunking documents...")
        chunks = self.doc_processor.process(data_dir)
        # for i, chunk in enumerate(chunks[:10]):
        #     print(f"\n🧩 Chunk {i + 1}:\n{chunk['content'] if isinstance(chunk, dict) else chunk.page_content}")
        # print(f"🧠 Embedding & indexing {len(chunks)} chunks...")
        self.vector_store.build(chunks)
        self.vector_store.save()

    def load_knowledge_base(self):
        self.vector_store.load()

    def query(self, user_input: str, return_chunks: bool = False):
        docs = self.retriever.retrieve(user_input)
        context = "\n".join([doc["metadata"].get("text", "") or str(doc["metadata"]) for doc in docs])
        return {"context": context, "chunks": docs} if return_chunks else context
