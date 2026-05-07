from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from .embeddings import EmbeddingModel, cosine_similarity
    from .knowledge_base import Chunk
except ImportError:
    from embeddings import EmbeddingModel, cosine_similarity
    from knowledge_base import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    metadata: dict[str, str]


class LocalVectorStore:
    def __init__(self, path: Path, embeddings: EmbeddingModel):
        self.path = path
        self.embeddings = embeddings
        self.records: list[dict] = []
        self._load()

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
        by_id = {record["id"]: record for record in self.records}
        for chunk, vector in zip(chunks, vectors):
            by_id[chunk.id] = {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "vector": vector,
            }
        self.records = list(by_id.values())
        self._save()

    def delete_sources(self, sources: Iterable[str]) -> None:
        source_set = set(sources)
        self.records = [
            record for record in self.records if record.get("metadata", {}).get("source") not in source_set
        ]
        self._save()

    def search(self, query: str, limit: int = 5) -> list[RetrievedChunk]:
        if not self.records:
            return []
        query_vector = self.embeddings.embed_query(query)
        scored = [
            RetrievedChunk(
                text=record["text"],
                metadata=record["metadata"],
                score=cosine_similarity(query_vector, record["vector"]),
            )
            for record in self.records
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def all_texts(self) -> list[tuple[str, dict[str, str]]]:
        return [(record["text"], record["metadata"]) for record in self.records]

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self.records = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.records = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, ensure_ascii=False), encoding="utf-8")


class QdrantVectorStore:
    def __init__(self, url: str, collection: str, embeddings: EmbeddingModel, api_key: str | None = None):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        self.models = models
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self.embeddings = embeddings
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {item.name for item in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=self.models.VectorParams(
                    size=self.embeddings.dim,
                    distance=self.models.Distance.COSINE,
                ),
            )

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
        points = [
            self.models.PointStruct(
                id=str(uuid.UUID(chunk.id[:32])),
                vector=vector,
                payload={"text": chunk.text, **chunk.metadata},
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def delete_sources(self, sources: Iterable[str]) -> None:
        for source in sources:
            self.client.delete(
                collection_name=self.collection,
                points_selector=self.models.FilterSelector(
                    filter=self.models.Filter(
                        must=[
                            self.models.FieldCondition(
                                key="source",
                                match=self.models.MatchValue(value=source),
                            )
                        ]
                    )
                ),
            )

    def search(self, query: str, limit: int = 5) -> list[RetrievedChunk]:
        vector = self.embeddings.embed_query(query)
        points = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )
        results: list[RetrievedChunk] = []
        for point in points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            results.append(RetrievedChunk(text=text, score=float(point.score), metadata=payload))
        return results

    def all_texts(self) -> list[tuple[str, dict[str, str]]]:
        points, _ = self.client.scroll(
            collection_name=self.collection,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        output = []
        for point in points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            output.append((text, payload))
        return output


class BM25Lite:
    def __init__(self, documents: list[tuple[str, dict[str, str]]]):
        self.documents = documents
        self.doc_tokens = [self._tokenize(text) for text, _ in documents]
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))
        self.avg_len = (
            sum(len(tokens) for tokens in self.doc_tokens) / len(self.doc_tokens)
            if self.doc_tokens
            else 0.0
        )

    def search(self, query: str, limit: int = 5) -> list[RetrievedChunk]:
        query_terms = self._tokenize(query)
        scored: list[RetrievedChunk] = []
        total_docs = len(self.documents)
        for (text, metadata), tokens in zip(self.documents, self.doc_tokens):
            counts = Counter(tokens)
            score = 0.0
            for term in query_terms:
                if term not in counts:
                    continue
                idf = math.log(1 + (total_docs - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
                freq = counts[term]
                denom = freq + 1.5 * (1 - 0.75 + 0.75 * len(tokens) / (self.avg_len or 1))
                score += idf * freq * 2.5 / denom
            if score > 0:
                scored.append(RetrievedChunk(text=text, metadata=metadata, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[\wа-яА-ЯёЁ]+", text.lower())


class HybridRetriever:
    def __init__(self, vector_store: LocalVectorStore | QdrantVectorStore):
        self.vector_store = vector_store
        self.bm25 = BM25Lite(vector_store.all_texts())

    def refresh_lexical_index(self) -> None:
        self.bm25 = BM25Lite(self.vector_store.all_texts())

    def index(self, chunks: list[Chunk], changed_sources: Iterable[str] = ()) -> None:
        self.vector_store.delete_sources(changed_sources)
        self.vector_store.upsert(chunks)
        self.refresh_lexical_index()

    def search(self, query: str, limit: int = 5) -> list[RetrievedChunk]:
        merged: dict[str, RetrievedChunk] = {}
        score_boosts: defaultdict[str, float] = defaultdict(float)
        for rank, item in enumerate(self.vector_store.search(query, limit=limit * 2), start=1):
            key = self._key(item)
            merged[key] = item
            score_boosts[key] += 1.0 / rank
        for rank, item in enumerate(self.bm25.search(query, limit=limit * 2), start=1):
            key = self._key(item)
            merged.setdefault(key, item)
            score_boosts[key] += 0.8 / rank
        results = [
            RetrievedChunk(text=item.text, metadata=item.metadata, score=item.score + score_boosts[key])
            for key, item in merged.items()
        ]
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def _key(self, item: RetrievedChunk) -> str:
        return f"{item.metadata.get('source')}:{item.metadata.get('chunk')}:{hash(item.text)}"
