from __future__ import annotations

import csv
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProgramChance:
    code: str
    name: str
    year: int
    cutoff_score: int
    delta: int


class AdmissionsDatabase:
    def __init__(self, db_path: Path, csv_dir: Path):
        self.db_path = db_path
        self.csv_dir = csv_dir

    def initialize(self, rebuild: bool = False) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if rebuild and self.db_path.exists():
            self.db_path.unlink()
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS programs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    track TEXT NOT NULL DEFAULT '',
                    UNIQUE(code, name, track)
                );

                CREATE TABLE IF NOT EXISTS cutoff_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_id INTEGER NOT NULL REFERENCES programs(id),
                    year INTEGER NOT NULL,
                    score INTEGER NOT NULL,
                    UNIQUE(program_id, year)
                );

                CREATE INDEX IF NOT EXISTS idx_cutoff_year_score
                    ON cutoff_scores(year, score);
                CREATE INDEX IF NOT EXISTS idx_program_name
                    ON programs(name);
                """
            )
            if rebuild or not self._has_rows(conn):
                self._load_csvs(conn)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def schema_for_llm(self) -> str:
        return (
            "SQLite schema:\n"
            "programs(id, code, name, track)\n"
            "cutoff_scores(id, program_id, year, score)\n"
            "Join: cutoff_scores.program_id = programs.id.\n"
            "Only SELECT queries are allowed."
        )

    def execute_select(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        self._validate_select(query)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def cutoff_history(self, program_query: str, limit: int = 8) -> list[dict[str, Any]]:
        rows = self.execute_select(
            """
            SELECT p.code, p.name, c.year, c.score
            FROM cutoff_scores c
            JOIN programs p ON p.id = c.program_id
            ORDER BY p.name, c.year DESC
            """
        )
        needle = program_query.casefold()
        filtered = [
            row
            for row in rows
            if needle in str(row["name"]).casefold() or needle in str(row["code"]).casefold()
        ]
        return filtered[:limit]

    def eligible_programs(self, total_score: int, year: int | None = None, limit: int = 10) -> list[ProgramChance]:
        if year is None:
            year = self.latest_year()
        rows = self.execute_select(
            """
            SELECT p.code, p.name, c.year, c.score
            FROM cutoff_scores c
            JOIN programs p ON p.id = c.program_id
            WHERE c.year = ? AND c.score <= ?
            ORDER BY c.score DESC, p.name
            LIMIT ?
            """,
            (year, total_score, limit),
        )
        return [
            ProgramChance(
                code=row["code"],
                name=row["name"],
                year=int(row["year"]),
                cutoff_score=int(row["score"]),
                delta=total_score - int(row["score"]),
            )
            for row in rows
        ]

    def latest_year(self) -> int:
        rows = self.execute_select("SELECT max(year) AS year FROM cutoff_scores")
        return int(rows[0]["year"] or 0)

    def _has_rows(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute("SELECT count(*) AS count FROM cutoff_scores").fetchone()
        return bool(row and row["count"])

    def _load_csvs(self, conn: sqlite3.Connection) -> None:
        for path in sorted(self.csv_dir.glob("*.csv")):
            year_match = re.search(r"(20\d{2})", path.stem)
            if not year_match:
                continue
            year = int(year_match.group(1))
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    normalized = {self._normalize(k): (v or "").strip() for k, v in row.items()}
                    code = normalized.get("код", "")
                    name = normalized.get("наименование конкурсной группы", "")
                    score = self._parse_score(normalized)
                    if not code or not name or score is None:
                        continue
                    track = self._extract_track(code)
                    program_id = self._upsert_program(conn, code, name, track)
                    conn.execute(
                        """
                        INSERT INTO cutoff_scores(program_id, year, score)
                        VALUES (?, ?, ?)
                        ON CONFLICT(program_id, year) DO UPDATE SET score=excluded.score
                        """,
                        (program_id, year, score),
                    )
        conn.commit()

    def _upsert_program(self, conn: sqlite3.Connection, code: str, name: str, track: str) -> int:
        conn.execute(
            """
            INSERT INTO programs(code, name, track)
            VALUES (?, ?, ?)
            ON CONFLICT(code, name, track) DO NOTHING
            """,
            (code, name, track),
        )
        row = conn.execute(
            "SELECT id FROM programs WHERE code = ? AND name = ? AND track = ?",
            (code, name, track),
        ).fetchone()
        return int(row["id"])

    def _validate_select(self, query: str) -> None:
        stripped = query.strip().rstrip(";")
        if not stripped.lower().startswith("select"):
            raise ValueError("SQL-tool accepts only SELECT queries")
        if ";" in stripped:
            raise ValueError("Only one SQL statement is allowed")
        blocked = re.search(
            r"\b(insert|update|delete|drop|alter|create|attach|pragma|vacuum|replace)\b",
            stripped,
            re.IGNORECASE,
        )
        if blocked:
            raise ValueError("SQL query contains a forbidden operation")

    def _parse_score(self, row: dict[str, str]) -> int | None:
        for key, value in row.items():
            if "проходной" in key:
                match = re.search(r"\d+", value)
                return int(match.group(0)) if match else None
        return None

    def _extract_track(self, code: str) -> str:
        match = re.search(r"\(([^)]+)\)", code)
        return match.group(1) if match else ""

    def _normalize(self, value: str | None) -> str:
        return re.sub(r"\s+", " ", (value or "").strip().lower())


def format_chances(chances: list[ProgramChance]) -> str:
    if not chances:
        return "По указанной сумме баллов не нашлось программ ниже проходного балла в базе."
    lines = [
        f"{item.name} ({item.code}), {item.year}: проходной {item.cutoff_score}, запас {item.delta}"
        for item in chances
    ]
    return "\n".join(lines)
