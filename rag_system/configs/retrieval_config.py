from dataclasses import dataclass

@dataclass
class RetrievalConfig:
    top_k: int = 5
    similarity_metric: str = "cosine"
    rerank: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"