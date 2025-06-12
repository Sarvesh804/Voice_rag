import os
from typing import List
from rag_system.configs.embedding_config import EmbeddingConfig

class EmbeddingModel:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        if config.use_openai:
            import openai
            self.client = openai
            self.model_name = config.openai_model
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config.local_model_name)
           

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.config.use_openai:
            response = self.client.Embedding.create(
                input=texts,
                model=self.config.openai_model
            )
            return [record["embedding"] for record in response["data"]]
        else:
            return self.model.encode(texts, convert_to_numpy=True).tolist()
