from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AssistantConfig:
    folder_id: str | None
    api_key: str | None
    yandex_model_name: str
    embedding_model_name: str
    data_dir: Path
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    sqlite_path: Path
    index_state_path: Path
    auto_index: bool
    enable_web_search: bool
    chunk_chars: int
    chunk_overlap: int

    @classmethod
    def from_env(cls) -> "AssistantConfig":
        load_dotenv()
        data_dir = Path(os.getenv("RAG_DATA_DIR", RAG_ROOT / "data")).expanduser()
        sqlite_path = Path(
            os.getenv("ADMISSIONS_DB_PATH", data_dir / "admissions.sqlite")
        ).expanduser()
        return cls(
            folder_id=os.getenv("folder_id") or os.getenv("YC_FOLDER_ID"),
            api_key=os.getenv("api_key") or os.getenv("YC_API_KEY"),
            yandex_model_name=os.getenv("YC_MODEL_NAME", "yandexgpt-5-pro"),
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "jina-embeddings-v3"),
            data_dir=data_dir,
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "mai_admissions_kb"),
            sqlite_path=sqlite_path,
            index_state_path=Path(
                os.getenv("RAG_INDEX_STATE", data_dir / ".index_state.json")
            ).expanduser(),
            auto_index=os.getenv("AUTO_INDEX", "0").lower() in {"1", "true", "yes"},
            enable_web_search=os.getenv("ENABLE_WEB_SEARCH", "1").lower()
            in {"1", "true", "yes"},
            chunk_chars=int(os.getenv("RAG_CHUNK_CHARS", "6000")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "600")),
        )
