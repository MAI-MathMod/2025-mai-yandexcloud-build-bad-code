from __future__ import annotations

try:
    from .config import AssistantConfig
    from .embeddings import JinaEmbeddingsV3
    from .indexing import build_vector_store
    from .knowledge_base import KnowledgeBase
    from .retrieval import HybridRetriever
except ImportError:
    from config import AssistantConfig
    from embeddings import JinaEmbeddingsV3
    from indexing import build_vector_store
    from knowledge_base import KnowledgeBase
    from retrieval import HybridRetriever


class SearchEngine:
    """Backward-compatible retrieval facade.

    The old implementation used Chroma with Yandex asymmetric embeddings. The
    production path now uses Qdrant plus jina-embeddings-v3, with a deterministic
    local vector store fallback for offline development.
    """

    def __init__(
        self,
        data_folder: str,
        sdk=None,
        persist_directory: str = "./chroma_db",
        chunk_size: int = 6000,
        chunk_overlap: int = 600,
    ):
        self.config = AssistantConfig.from_env()
        self.knowledge_base = KnowledgeBase(
            data_dir=self._resolve_data_dir(data_folder),
            chunk_chars=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embeddings = JinaEmbeddingsV3(model_name=self.config.embedding_model_name)
        self.vector_store = build_vector_store(
            qdrant_url=self.config.qdrant_url,
            collection=self.config.qdrant_collection,
            embeddings=self.embeddings,
            api_key=self.config.qdrant_api_key,
            local_path=self.config.data_dir / "local_vector_store.json",
        )
        self.retriever = HybridRetriever(self.vector_store)

    def initialize(self) -> None:
        if self.vector_store.all_texts():
            return
        docs = self.knowledge_base.load_documents()
        chunks = self.knowledge_base.chunk_documents(docs)
        self.retriever.index(chunks)

    def search(self, query: str, n_results: int = 5) -> list[str]:
        self.initialize()
        return [item.text for item in self.retriever.search(query, limit=n_results)]

    def _resolve_data_dir(self, data_folder: str):
        from pathlib import Path

        path = Path(data_folder)
        if path.exists():
            return path
        package_relative = Path(__file__).resolve().parent / data_folder
        return package_relative if package_relative.exists() else path


if __name__ == "__main__":
    engine = SearchEngine(data_folder="./data")
    engine.initialize()
    for result in engine.search("Какие проходные баллы были на ПМИ в 2024 году?", n_results=5):
        print("-" * 80)
        print(result[:2000])
