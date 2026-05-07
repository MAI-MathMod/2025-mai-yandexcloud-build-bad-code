from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class EvaluationReport:
    ragas_score: float | None
    judge_accuracy: float
    average_latency_sec: float
    samples: int


class EvaluationPipeline:
    def __init__(self, assistant):
        self.assistant = assistant

    def run(self, dataset_path: Path) -> EvaluationReport:
        rows = self._load_dataset(dataset_path)
        latencies: list[float] = []
        judged: list[float] = []
        for index, row in enumerate(rows, start=1):
            started = time.perf_counter()
            answer = self.assistant.ask(f"eval-{index}", row["question"])
            latencies.append(time.perf_counter() - started)
            judged.append(self._lexical_judge(answer, row["answer"]))
        return EvaluationReport(
            ragas_score=None,
            judge_accuracy=mean(judged) if judged else 0.0,
            average_latency_sec=mean(latencies) if latencies else 0.0,
            samples=len(rows),
        )

    def _load_dataset(self, dataset_path: Path) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for line in dataset_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if "question" in payload and "answer" in payload:
                rows.append({"question": str(payload["question"]), "answer": str(payload["answer"])})
        return rows

    def _lexical_judge(self, answer: str, reference: str) -> float:
        answer_terms = set(answer.lower().split())
        reference_terms = set(reference.lower().split())
        if not reference_terms:
            return 0.0
        return 1.0 if len(answer_terms & reference_terms) / len(reference_terms) >= 0.35 else 0.0

