from dataclasses import dataclass

@dataclass
class EmbeddingConfig:
    use_openai: bool = False
    openai_model: str = "text-embedding-3-small"
    local_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu" 
