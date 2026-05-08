from __future__ import annotations

import logging
from pathlib import Path

try:
    from .admissions_db import AdmissionsDatabase
    from .agent import AdmissionsAgent, Conversation
    from .config import AssistantConfig
    from .embeddings import JinaEmbeddingsV3
    from .indexing import IndexingService, build_vector_store
    from .knowledge_base import KnowledgeBase
    from .llm import LangChainYandexGPTModel, OfflineAdmissionsModel
    from .retrieval import HybridRetriever
    from .tools import build_default_tools
except ImportError:
    from admissions_db import AdmissionsDatabase
    from agent import AdmissionsAgent, Conversation
    from config import AssistantConfig
    from embeddings import JinaEmbeddingsV3
    from indexing import IndexingService, build_vector_store
    from knowledge_base import KnowledgeBase
    from llm import LangChainYandexGPTModel, OfflineAdmissionsModel
    from retrieval import HybridRetriever
    from tools import build_default_tools


logger = logging.getLogger("mai-admissions-assistant")


class Assistant:
    """Compatibility facade used by the queue worker and local CLI."""

    def __init__(self, config: AssistantConfig | None = None):
        self.config = config or AssistantConfig.from_env()
        self.admissions_db = AdmissionsDatabase(
            db_path=self.config.sqlite_path,
            csv_dir=self.config.data_dir / "cutoff_points",
        )
        self.admissions_db.initialize()

        self.embeddings = JinaEmbeddingsV3(model_name=self.config.embedding_model_name)
        self.knowledge_base = KnowledgeBase(
            data_dir=self.config.data_dir,
            chunk_chars=self.config.chunk_chars,
            chunk_overlap=self.config.chunk_overlap,
        )
        vector_store = build_vector_store(
            qdrant_url=self.config.qdrant_url,
            collection=self.config.qdrant_collection,
            embeddings=self.embeddings,
            api_key=self.config.qdrant_api_key,
            local_path=self.config.data_dir / "local_vector_store.json",
        )
        self.retriever = HybridRetriever(vector_store)
        self.indexing = IndexingService(
            knowledge_base=self.knowledge_base,
            retriever=self.retriever,
            state_path=self.config.index_state_path,
        )
        if self.config.auto_index:
            indexed = self.indexing.update_changed()
            logger.info("Incremental index update finished: %s chunks", indexed)

        model = self._build_model()
        tools = build_default_tools(
            retriever=self.retriever,
            admissions_db=self.admissions_db,
            enable_web_search=self.config.enable_web_search,
        )
        self.agent = AdmissionsAgent(model=model, tools=tools, admissions_db=self.admissions_db)

    def create_chat(self, chat_id: str | int) -> str:
        self.agent.conversations.setdefault(str(chat_id), Conversation())
        return str(chat_id)

    def ask(self, chat_id: str | int, message: str) -> str:
        return self.agent.ask(chat_id, message)

    def rebuild_index(self) -> int:
        return self.indexing.rebuild()

    def update_index(self) -> int:
        return self.indexing.update_changed()

    def _build_model(self):
        if not self.config.folder_id or not self.config.api_key:
            return OfflineAdmissionsModel()
        try:
            return LangChainYandexGPTModel(
                folder_id=self.config.folder_id,
                api_key=self.config.api_key,
                model_name=self.config.yandex_model_name,
            )
        except Exception as exc:
            logger.warning("YandexGPT init failed, offline fallback enabled: %s", exc)
            return OfflineAdmissionsModel()


def build_assistant_from_env() -> Assistant:
    config = AssistantConfig.from_env()
    return Assistant(config=config)


if __name__ == "__main__":
    assistant = build_assistant_from_env()
    print("Ассистент готов. Формат: chat_id|сообщение")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if "|" not in line:
            print("Используйте формат chat_id|сообщение")
            continue
        chat_id, message = line.split("|", 1)
        print(assistant.ask(chat_id, message))
