import faiss
import os
import pickle
from typing import List, Dict
import numpy as np
from rag_system.core.embedding_models import EmbeddingModel
from rag_system.configs.embedding_config import EmbeddingConfig

class VectorStore:
    def __init__(self, embedding_model: EmbeddingModel, index_path="rag_index"):
        self.embedding_model = embedding_model
        self.index_path = index_path
        self.index = None
        self.metadata = []

    def build(self, documents: List[Dict]):
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_texts(texts)

        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float32"))
        self.metadata = [
            {
                **doc.metadata,
                "text": doc.page_content if hasattr(doc, "page_content") else doc["content"]
            }
            for doc in documents
        ]


        print(f"✅ FAISS index built with {len(embeddings)} vectors.")

    def save(self):
        faiss.write_index(self.index, f"{self.index_path}.index")
        with open(f"{self.index_path}_meta.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        print("📦 Vector index and metadata saved.")

    def load(self):
        self.index = faiss.read_index(f"{self.index_path}.index")
        with open(f"{self.index_path}_meta.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        print("📂 Vector index and metadata loaded.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.embedding_model.embed_texts([query])[0]
        D, I = self.index.search(np.array([query_embedding]).astype("float32"), top_k)
        return [{
            "content": self.metadata[i].get("text", ""),  # get actual content back
            "metadata": self.metadata[i],
            "score": float(D[0][j])
        } for j, i in enumerate(I[0])]
