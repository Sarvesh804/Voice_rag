from typing import List, Dict
from rag_system.configs.chunking_config import ChunkingConfig
from rag_system.core.embedding_models import EmbeddingModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunking_config: ChunkingConfig, embedder: EmbeddingModel):
        self.chunking_config = chunking_config
        self.embedder = embedder

    def load_documents(self, data_dir: str) -> List[Dict]:
        loader = DirectoryLoader(
            data_dir,
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
        )
        documents = loader.load()
        print(f"📄 Loaded {len(documents)} documents from {data_dir}")
        return documents

    def recursive_split(self, documents: List[Dict]) -> List[Dict]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunking_config.max_tokens,
            chunk_overlap=self.chunking_config.overlap
        )
        return splitter.split_documents(documents)

    def semantic_split(self, documents: List[Dict]) -> List[Dict]:
        chunks = []
        max_tokens = self.chunking_config.max_tokens
        overlap_sentences = 1
        similarity_threshold = 0.75

        for doc in documents:
            paragraphs = [p.strip() for p in doc.page_content.split("\n\n") if len(p.strip()) > 20]
            if len(paragraphs) <= 1:
                continue

            embeddings = self.embedder.embed_texts(paragraphs)
            group = [paragraphs[0]]
            group_tokens = len(paragraphs[0].split())

            for i in range(1, len(paragraphs)):
                sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
                tokens = len(paragraphs[i].split())

                if sim < similarity_threshold or group_tokens + tokens > max_tokens:
                    chunks.append({
                        "content": " ".join(group),
                        "metadata": doc.metadata
                    })

                    group = group[-overlap_sentences:] if overlap_sentences > 0 else []
                    group_tokens = sum(len(p.split()) for p in group)

                group.append(paragraphs[i])
                group_tokens += tokens

            if group:
                chunks.append({
                    "content": " ".join(group),
                    "metadata": doc.metadata
                })

        return chunks

    def process(self, data_dir: str) -> List[Dict]:
        raw_docs = self.load_documents(data_dir)
        if self.chunking_config.strategy == "semantic":
            return self.semantic_split(raw_docs)
        else:
            return self.recursive_split(raw_docs)
