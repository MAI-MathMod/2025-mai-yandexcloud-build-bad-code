from __future__ import annotations

from pathlib import Path

try:
    from .embeddings import EmbeddingModel
    from .knowledge_base import KnowledgeBase
    from .retrieval import HybridRetriever, LocalVectorStore, QdrantVectorStore
except ImportError:
    from embeddings import EmbeddingModel
    from knowledge_base import KnowledgeBase
    from retrieval import HybridRetriever, LocalVectorStore, QdrantVectorStore


class IndexingService:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        retriever: HybridRetriever,
        state_path: Path,
    ):
        self.knowledge_base = knowledge_base
        self.retriever = retriever
        self.state_path = state_path

    def rebuild(self) -> int:
        docs = self.knowledge_base.load_documents()
        chunks = self.knowledge_base.chunk_documents(docs)
        self.retriever.index(chunks)
        self.knowledge_base.write_state(self.state_path)
        return len(chunks)

    def update_changed(self) -> int:
        changed = self.knowledge_base.changed_files(self.state_path)
        if not changed:
            return 0
        docs = self.knowledge_base.load_documents(changed)
        chunks = self.knowledge_base.chunk_documents(docs)
        self.retriever.index(chunks, changed_sources=[str(path) for path in changed])
        self.knowledge_base.write_state(self.state_path)
        return len(chunks)


def build_vector_store(
    *,
    qdrant_url: str,
    collection: str,
    embeddings: EmbeddingModel,
    api_key: str | None,
    local_path: Path,
) -> QdrantVectorStore | LocalVectorStore:
    try:
        return QdrantVectorStore(
            url=qdrant_url,
            collection=collection,
            embeddings=embeddings,
            api_key=api_key,
        )
    except Exception:
        return LocalVectorStore(local_path, embeddings)
