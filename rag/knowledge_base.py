from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".html", ".json"}


@dataclass(frozen=True)
class SourceDocument:
    text: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, str]


class DataCleaner:
    profanity = re.compile(
        r"\b(?:бля\w*|хуй\w*|пизд\w*|еба\w*|ёба\w*|сука)\b",
        re.IGNORECASE,
    )
    email = re.compile(r"[\w.\-+]+@[\w.\-]+\.\w+")
    phone = re.compile(r"(?:\+7|8)[\s\-()]?\d{3}[\s\-()]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}")
    passport = re.compile(r"\b\d{4}\s?\d{6}\b")
    snils = re.compile(r"\b\d{3}-\d{3}-\d{3}\s?\d{2}\b")

    def clean(self, text: str) -> str:
        text = html.unescape(text)
        text = self.email.sub("[EMAIL]", text)
        text = self.phone.sub("[PHONE]", text)
        text = self.passport.sub("[PASSPORT]", text)
        text = self.snils.sub("[SNILS]", text)
        text = self.profanity.sub("[REMOVED]", text)
        text = re.sub(r"[ \t]+", " ", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


class TableLinearizer:
    def csv_to_documents(self, path: Path) -> list[SourceDocument]:
        year_match = re.search(r"(20\d{2})", path.stem)
        year = year_match.group(1) if year_match else "unknown"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        documents: list[SourceDocument] = []
        for row_no, row in enumerate(rows, start=1):
            normalized = {self._normalize_header(k): (v or "").strip() for k, v in row.items()}
            code = normalized.get("код", "")
            name = (
                normalized.get("наименование конкурсной группы")
                or normalized.get("направление")
                or normalized.get("программа")
                or ""
            )
            score = next((v for k, v in normalized.items() if "проходной" in k), "")
            parts = [
                "Тип данных: структурированная строка таблицы проходных баллов МАИ.",
                f"Год приема: {year}.",
                f"Код направления: {code}.",
                f"Конкурсная группа или программа: {name}.",
                f"Проходной балл: {score}.",
            ]
            for header, value in normalized.items():
                if value and header not in {"код", "наименование конкурсной группы"} and "проходной" not in header:
                    parts.append(f"{header}: {value}.")
            documents.append(
                SourceDocument(
                    text=" ".join(parts),
                    metadata={
                        "source": str(path),
                        "type": "cutoff_table",
                        "year": year,
                        "row": str(row_no),
                        "program_code": code,
                        "program_name": name,
                    },
                )
            )
        return documents

    def markdown_tables_to_text(self, text: str, source: str) -> str:
        lines = text.splitlines()
        output: list[str] = []
        heading_stack: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                heading_stack = heading_stack[: level - 1] + [line.lstrip("#").strip()]
                output.append(line)
                i += 1
                continue
            if self._looks_like_table_at(lines, i):
                headers = [cell.strip() for cell in lines[i].strip("|").split("|")]
                i += 2
                row_no = 0
                while i < len(lines) and "|" in lines[i]:
                    cells = [cell.strip() for cell in lines[i].strip("|").split("|")]
                    if len(cells) == len(headers):
                        row_no += 1
                        context = " / ".join(heading_stack) or source
                        fields = "; ".join(
                            f"{header}: {cell}" for header, cell in zip(headers, cells) if cell
                        )
                        output.append(f"Таблица: {context}. Строка {row_no}. {fields}.")
                    i += 1
                continue
            output.append(line)
            i += 1
        return "\n".join(output)

    def _looks_like_table_at(self, lines: list[str], index: int) -> bool:
        if index + 1 >= len(lines):
            return False
        return "|" in lines[index] and re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", lines[index + 1]) is not None

    def _normalize_header(self, value: str | None) -> str:
        return re.sub(r"\s+", " ", (value or "").strip().lower())


class KnowledgeBase:
    def __init__(
        self,
        data_dir: Path,
        chunk_chars: int = 6000,
        chunk_overlap: int = 600,
        cleaner: DataCleaner | None = None,
        table_linearizer: TableLinearizer | None = None,
    ):
        self.data_dir = data_dir
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap
        self.cleaner = cleaner or DataCleaner()
        self.table_linearizer = table_linearizer or TableLinearizer()

    def iter_files(self) -> list[Path]:
        return sorted(
            path
            for path in self.data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def load_documents(self, files: Iterable[Path] | None = None) -> list[SourceDocument]:
        docs: list[SourceDocument] = []
        for path in files or self.iter_files():
            if path.name.startswith("."):
                continue
            if path.suffix.lower() == ".csv":
                docs.extend(self.table_linearizer.csv_to_documents(path))
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if path.suffix.lower() == ".md":
                text = self.table_linearizer.markdown_tables_to_text(text, str(path))
            if path.suffix.lower() == ".json":
                text = self._json_to_text(text)
            text = self.cleaner.clean(text)
            if text:
                docs.append(
                    SourceDocument(
                        text=text,
                        metadata={"source": str(path), "type": path.suffix.lower().lstrip(".")},
                    )
                )
        return self._deduplicate(docs)

    def chunk_documents(self, docs: Iterable[SourceDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in docs:
            for index, text in enumerate(self._split_text(doc.text)):
                metadata = dict(doc.metadata)
                metadata["chunk"] = str(index)
                chunk_id = hashlib.sha256(
                    f"{metadata.get('source')}:{index}:{text}".encode("utf-8")
                ).hexdigest()
                chunks.append(Chunk(id=chunk_id, text=text, metadata=metadata))
        return chunks

    def changed_files(self, state_path: Path) -> list[Path]:
        previous = self._load_state(state_path)
        changed: list[Path] = []
        for path in self.iter_files():
            digest = self.file_hash(path)
            if previous.get(str(path)) != digest:
                changed.append(path)
        return changed

    def write_state(self, state_path: Path) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {str(path): self.file_hash(path) for path in self.iter_files()}
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def file_hash(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_chars:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_chars)
            if end < len(text):
                paragraph_end = text.rfind("\n\n", start, end)
                if paragraph_end > start + self.chunk_chars // 2:
                    end = paragraph_end
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return [chunk for chunk in chunks if chunk]

    def _json_to_text(self, text: str) -> str:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _deduplicate(self, docs: list[SourceDocument]) -> list[SourceDocument]:
        seen: set[str] = set()
        unique: list[SourceDocument] = []
        for doc in docs:
            digest = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            unique.append(doc)
        return unique

    def _load_state(self, state_path: Path) -> dict[str, str]:
        if not state_path.exists():
            return {}
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

