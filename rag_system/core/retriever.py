from typing import List, Dict
from sentence_transformers import CrossEncoder
from rag_system.configs.retrieval_config import RetrievalConfig
from rag_system.core.vector_store import VectorStore

class Retriever:
    def __init__(self, vector_store: VectorStore, config: RetrievalConfig):
        self.vector_store = vector_store
        self.config = config
        self.reranker = None

        if config.rerank:
            self.reranker = CrossEncoder(config.reranker_model)

    def retrieve(self, query: str) -> List[Dict]:
        top_docs = self.vector_store.search(query, top_k=self.config.top_k)

        if self.reranker:
            rerank_inputs = [(query, doc["metadata"].get("text", "")) for doc in top_docs]
            scores = self.reranker.predict(rerank_inputs)
            for i in range(len(top_docs)):
                top_docs[i]["rerank_score"] = float(scores[i])
            top_docs = sorted(top_docs, key=lambda x: x["rerank_score"], reverse=True)

        return top_docs
