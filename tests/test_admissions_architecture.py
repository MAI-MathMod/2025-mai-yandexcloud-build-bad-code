from pathlib import Path

from rag.admissions_db import AdmissionsDatabase
from rag.knowledge_base import DataCleaner, TableLinearizer


def test_cutoff_csvs_build_sqlite(tmp_path):
    csv_dir = Path("rag/data/cutoff_points")
    db = AdmissionsDatabase(tmp_path / "admissions.sqlite", csv_dir)
    db.initialize(rebuild=True)

    rows = db.cutoff_history("программная инженерия")

    assert rows
    assert any(row["year"] == 2024 and row["score"] == 275 for row in rows)


def test_total_score_chances_use_latest_year(tmp_path):
    db = AdmissionsDatabase(tmp_path / "admissions.sqlite", Path("rag/data/cutoff_points"))
    db.initialize(rebuild=True)

    chances = db.eligible_programs(260)

    assert chances
    assert all(item.year == 2024 for item in chances)
    assert all(item.cutoff_score <= 260 for item in chances)


def test_cleaner_removes_personal_data_and_profanity():
    cleaned = DataCleaner().clean("Почта a@b.ru, телефон +7 999 123-45-67, блядь")

    assert "[EMAIL]" in cleaned
    assert "[PHONE]" in cleaned
    assert "[REMOVED]" in cleaned


def test_csv_table_linearizer_adds_contextual_headers():
    docs = TableLinearizer().csv_to_documents(Path("rag/data/cutoff_points/2024.csv"))

    assert docs
    assert "Год приема: 2024" in docs[0].text
    assert "Проходной балл" in docs[0].text

