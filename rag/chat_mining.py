from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    from .knowledge_base import DataCleaner
except ImportError:
    from knowledge_base import DataCleaner


QUESTION_RE = re.compile(r"[^.!?]*\?+")


@dataclass(frozen=True)
class ChatMiningResult:
    top_questions: list[tuple[str, int]]
    qa_pairs: list[dict[str, str]]


class ChatMiningPipeline:
    def __init__(self, cleaner: DataCleaner | None = None):
        self.cleaner = cleaner or DataCleaner()

    def run(self, input_dir: Path, output_dir: Path, top_n: int = 200, dataset_size: int = 500) -> ChatMiningResult:
        messages = self._load_messages(input_dir)
        questions = [self._normalize_question(message) for message in messages if "?" in message]
        questions = [question for question in questions if question]
        top_questions = Counter(questions).most_common(top_n)
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

    def _load_messages(self, input_dir: Path) -> list[str]:
        messages: list[str] = []
        for path in sorted(input_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".jsonl":
                messages.extend(self._load_jsonl(path))
            elif path.suffix.lower() == ".csv":
                messages.extend(self._load_csv(path))
            elif path.suffix.lower() in {".txt", ".md"}:
                messages.extend(path.read_text(encoding="utf-8", errors="ignore").splitlines())
        return [self.cleaner.clean(message) for message in messages if self.cleaner.clean(message)]

    def _load_jsonl(self, path: Path) -> list[str]:
        output: list[str] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = payload.get("text") or payload.get("message") or payload.get("content")
            if text:
                output.append(str(text))
        return output

    def _load_csv(self, path: Path) -> list[str]:
        output: list[str] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                text = row.get("text") or row.get("message") or row.get("content")
                if text:
                    output.append(text)
        return output

    def _normalize_question(self, message: str) -> str:
        match = QUESTION_RE.search(message)
        if not match:
            return ""
        question = match.group(0).strip().lower()
        question = re.sub(r"\s+", " ", question)
        return question[:500]

    def _extract_qa_pairs(self, messages: list[str], limit: int) -> list[dict[str, str]]:
        pairs: list[dict[str, str]] = []
        pending_question: str | None = None
        for message in messages:
            if "?" in message:
                pending_question = self._normalize_question(message)
                continue
            if pending_question and len(message) > 20:
                pairs.append({"question": pending_question, "answer": message})
                pending_question = None
                if len(pairs) >= limit:
                    break
        return pairs
