from dataclasses import dataclass

@dataclass
class ChunkingConfig:
    strategy: str = "recursive"
    max_tokens: int = 300
    overlap: int = 50
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"