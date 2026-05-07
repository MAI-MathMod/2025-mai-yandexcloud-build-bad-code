from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Protocol


class EmbeddingModel(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class HashingEmbeddings:
    """Deterministic local fallback for tests and offline development."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = re.findall(r"[\wа-яА-ЯёЁ]+", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


@dataclass
class JinaEmbeddingsV3:
    model_name: str = "jina-embeddings-v3"
    api_key: str | None = None
    endpoint: str = "https://api.jina.ai/v1/embeddings"
    dim: int = 1024
    batch_size: int = 32
    fallback: EmbeddingModel | None = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("JINA_API_KEY")
        if self.fallback is None:
            self.fallback = HashingEmbeddings()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            return self.fallback.embed_documents(texts)
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._request(texts[i : i + self.batch_size], task="retrieval.passage"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if not self.api_key:
            return self.fallback.embed_query(text)
        return self._request([text], task="retrieval.query")[0]

    def _request(self, texts: list[str], task: str) -> list[list[float]]:
        payload = {
            "model": self.model_name,
            "task": task,
            "input": texts,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        return [item["embedding"] for item in body["data"]]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    denom = math.sqrt(left_norm) * math.sqrt(right_norm)
    return dot / denom if denom else 0.0

