from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    from .embeddings import EmbeddingModel, JinaEmbeddingsV3, cosine_similarity
    from .knowledge_base import DataCleaner
except ImportError:
    from embeddings import EmbeddingModel, JinaEmbeddingsV3, cosine_similarity
    from knowledge_base import DataCleaner


SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+", re.MULTILINE)


@dataclass(frozen=True)
class ChatMiningResult:
    top_questions: list[tuple[str, int]]
    qa_pairs: list[dict[str, str | float]]


@dataclass(frozen=True)
class ChatMessage:
    text: str
    conversation_id: str
    message_id: str | None = None
    reply_to_message_id: str | None = None


class ChatMiningPipeline:
    def __init__(
        self,
        cleaner: DataCleaner | None = None,
        embeddings: EmbeddingModel | None = None,
        cluster_eps: float = 0.18,
        cluster_min_samples: int = 2,
        answer_similarity_threshold: float = 0.85,
        answer_candidate_window: int = 3,
    ):
        self.cleaner = cleaner or DataCleaner()
        self.embeddings = embeddings or JinaEmbeddingsV3()
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.answer_similarity_threshold = answer_similarity_threshold
        self.answer_candidate_window = answer_candidate_window

    def run(self, input_dir: Path, output_dir: Path, top_n: int = 200, dataset_size: int = 500) -> ChatMiningResult:
        messages = self._load_messages(input_dir)
        questions = [
            question
            for message in messages
            if "?" in message.text
            for question in self._extract_questions(message.text)
        ]
        top_questions = self._semantic_top_questions(questions, top_n)
        qa_pairs = self._extract_qa_pairs(messages, dataset_size)

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "top_questions.csv").write_text(
            "question,count\n"
            + "\n".join(f"{json.dumps(question, ensure_ascii=False)},{count}" for question, count in top_questions),
            encoding="utf-8",
        )
        with (output_dir / "qa_dataset.jsonl").open("w", encoding="utf-8") as handle:
            for pair in qa_pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

        return ChatMiningResult(top_questions=top_questions, qa_pairs=qa_pairs)

    def _load_messages(self, input_dir: Path) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        for path in sorted(input_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".jsonl":
                messages.extend(self._load_jsonl(path))
            elif path.suffix.lower() == ".csv":
                messages.extend(self._load_csv(path))
            elif path.suffix.lower() in {".txt", ".md"}:
                for line_no, text in enumerate(
                    path.read_text(encoding="utf-8", errors="ignore").splitlines(),
                    start=1,
                ):
                    cleaned = self.cleaner.clean(text)
                    if cleaned:
                        messages.append(
                            ChatMessage(
                                text=cleaned,
                                conversation_id=str(path),
                                message_id=str(line_no),
                            )
                        )
        return messages

    def _load_jsonl(self, path: Path) -> list[ChatMessage]:
        output: list[ChatMessage] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = payload.get("text") or payload.get("message") or payload.get("content")
            if text:
                cleaned = self.cleaner.clean(str(text))
                if cleaned:
                    output.append(self._message_from_mapping(payload, cleaned, path))
        return output

    def _load_csv(self, path: Path) -> list[ChatMessage]:
        output: list[ChatMessage] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                text = row.get("text") or row.get("message") or row.get("content")
                if text:
                    cleaned = self.cleaner.clean(text)
                    if cleaned:
                        output.append(self._message_from_mapping(row, cleaned, path))
        return output

    def _message_from_mapping(self, payload: dict, text: str, path: Path) -> ChatMessage:
        conversation_id = (
            payload.get("chat_id")
            or payload.get("dialog_id")
            or payload.get("conversation_id")
            or str(path)
        )
        message_id = payload.get("message_id") or payload.get("id")
        reply_to = payload.get("reply_to_message_id") or payload.get("reply_to")
        if isinstance(reply_to, dict):
            reply_to = reply_to.get("message_id") or reply_to.get("id")
        return ChatMessage(
            text=text,
            conversation_id=str(conversation_id),
            message_id=str(message_id) if message_id is not None else None,
            reply_to_message_id=str(reply_to) if reply_to is not None else None,
        )

    def _extract_questions(self, message: str) -> list[str]:
        """Return every sentence ending in a question mark; discard surrounding prose."""
        questions: list[str] = []
        for match in SENTENCE_RE.finditer(message):
            sentence = match.group(0).strip()
            if not sentence.endswith("?"):
                continue
            normalized = re.sub(r"\s+", " ", sentence.lower()).strip()
            if normalized:
                questions.append(normalized[:500])
        return questions

    def _semantic_top_questions(self, questions: list[str], limit: int) -> list[tuple[str, int]]:
        """Cluster equivalent questions and return representatives of largest clusters."""
        if not questions or limit <= 0:
            return []

        from sklearn.cluster import DBSCAN

        exact_counts = Counter(questions)
        unique_questions = list(exact_counts)
        vectors = self.embeddings.embed_documents(unique_questions)
        labels = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples,
            metric="cosine",
            algorithm="brute",
        ).fit_predict(vectors)

        clusters: dict[tuple[str, int], list[str]] = {}
        for index, (question, label) in enumerate(zip(unique_questions, labels)):
            # DBSCAN gives every outlier label -1. Keep each outlier as its own
            # cluster instead of incorrectly merging all noise into one topic.
            cluster_key = ("noise", index) if int(label) == -1 else ("cluster", int(label))
            clusters.setdefault(cluster_key, []).append(question)

        ranked: list[tuple[str, int]] = []
        for members in clusters.values():
            frequency = sum(exact_counts[question] for question in members)
            representative = max(
                members,
                key=lambda question: (exact_counts[question], -len(question), question),
            )
            ranked.append((representative, frequency))

        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:limit]

    def _extract_qa_pairs(
        self,
        messages: list[ChatMessage],
        limit: int,
    ) -> list[dict[str, str | float]]:
        pairs: list[dict[str, str | float]] = []
        replies: dict[tuple[str, str], list[ChatMessage]] = {}
        for message in messages:
            if message.reply_to_message_id:
                key = (message.conversation_id, message.reply_to_message_id)
                replies.setdefault(key, []).append(message)

        query_vectors: dict[str, list[float]] = {}
        answer_vectors: dict[str, list[float]] = {}

        for index, message in enumerate(messages):
            if "?" not in message.text:
                continue
            questions = self._extract_questions(message.text)
            if not questions:
                continue

            candidates = self._answer_candidates(messages, index, message, replies)
            candidates = [candidate for candidate in candidates if len(candidate.text) > 20]
            if not candidates:
                continue

            missing_answers = [
                candidate.text for candidate in candidates if candidate.text not in answer_vectors
            ]
            if missing_answers:
                embedded = self.embeddings.embed_documents(missing_answers)
                answer_vectors.update(zip(missing_answers, embedded))

            for question in questions:
                if question not in query_vectors:
                    query_vectors[question] = self.embeddings.embed_query(question)
                scored = [
                    (
                        cosine_similarity(query_vectors[question], answer_vectors[candidate.text]),
                        candidate,
                    )
                    for candidate in candidates
                ]
                similarity, answer = max(scored, key=lambda item: item[0])
                if similarity < self.answer_similarity_threshold:
                    continue
                pairs.append(
                    {
                        "question": question,
                        "answer": answer.text,
                        "similarity": round(float(similarity), 4),
                    }
                )
                if len(pairs) >= limit:
                    return pairs
        return pairs

    def _answer_candidates(
        self,
        messages: list[ChatMessage],
        question_index: int,
        question_message: ChatMessage,
        replies: dict[tuple[str, str], list[ChatMessage]],
    ) -> list[ChatMessage]:
        candidates: list[ChatMessage] = []
        for candidate in messages[question_index + 1 :]:
            if candidate.conversation_id == question_message.conversation_id:
                candidates.append(candidate)
                if len(candidates) >= self.answer_candidate_window:
                    break

        if question_message.message_id:
            reply_key = (question_message.conversation_id, question_message.message_id)
            candidates.extend(replies.get(reply_key, []))

        unique: list[ChatMessage] = []
        seen: set[tuple[str | None, str]] = set()
        for candidate in candidates:
            key = (candidate.message_id, candidate.text)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return unique
