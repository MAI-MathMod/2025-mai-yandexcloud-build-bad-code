from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path
from statistics import mean
from typing import Any


RAGAS_METRIC_NAMES = (
    "context_recall",
    "faithfulness",
    "factual_correctness",
)


@dataclass(frozen=True)
class RagasConfig:
    api_key: str
    judge_model: str = "gpt-4o-mini"
    base_url: str | None = None
    factual_correctness_threshold: float = 0.70

    @classmethod
    def from_env(cls) -> "RagasConfig":
        api_key = os.getenv("RAGAS_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("RAGAS_API_KEY or OPENAI_API_KEY is required")
        return cls(
            api_key=api_key,
            judge_model=os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("RAGAS_BASE_URL") or None,
            factual_correctness_threshold=float(
                os.getenv("RAGAS_FACTUAL_CORRECTNESS_THRESHOLD", "0.70")
            ),
        )


@dataclass(frozen=True)
class EvaluationReport:
    ragas_score: float
    metrics: dict[str, float]
    judge_accuracy: float
    factual_correctness_threshold: float
    average_latency_sec: float
    samples: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvaluationPipeline:
    def __init__(self, assistant, config: RagasConfig | None = None):
        self.assistant = assistant
        self.config = config or RagasConfig.from_env()

    def run(
        self,
        dataset_path: Path,
        output_path: Path | None = None,
    ) -> EvaluationReport:
        self._require_production_components()
        rows = self._load_dataset(dataset_path)
        if not rows:
            raise ValueError(f"Evaluation dataset is empty: {dataset_path}")

        samples: list[dict[str, Any]] = []
        latencies: list[float] = []
        for index, row in enumerate(rows, start=1):
            started = time.perf_counter()
            answer, contexts = self.assistant.ask_with_context(
                f"ragas-eval-{index}",
                row["question"],
            )
            latencies.append(time.perf_counter() - started)
            samples.append(
                {
                    "user_input": row["question"],
                    "retrieved_contexts": contexts,
                    "response": answer,
                    "reference": row["answer"],
                }
            )

        result = self._evaluate_with_ragas(samples)
        details = self._records_from_result(result)
        metrics = {
            name: self._mean_metric(details, name)
            for name in RAGAS_METRIC_NAMES
        }
        factual_scores = [
            float(row["factual_correctness"])
            for row in details
            if row.get("factual_correctness") is not None
        ]
        if len(factual_scores) != len(rows):
            raise RuntimeError("RAGAS did not return factual_correctness for every sample")

        report = EvaluationReport(
            ragas_score=prod(metrics.values()) ** (1 / len(metrics)),
            metrics=metrics,
            judge_accuracy=mean(
                score >= self.config.factual_correctness_threshold
                for score in factual_scores
            ),
            factual_correctness_threshold=self.config.factual_correctness_threshold,
            average_latency_sec=mean(latencies),
            samples=len(rows),
        )
        if output_path is not None:
            self._write_report(output_path, report, details)
        return report

    def _require_production_components(self) -> None:
        try:
            from .llm import OfflineAdmissionsModel
            from .retrieval import QdrantVectorStore
        except ImportError:
            from llm import OfflineAdmissionsModel
            from retrieval import QdrantVectorStore

        if isinstance(self.assistant.agent.model, OfflineAdmissionsModel):
            raise RuntimeError("RAGAS evaluation refuses OfflineAdmissionsModel")
        if not getattr(self.assistant.embeddings, "api_key", None):
            raise RuntimeError("RAGAS evaluation requires JINA_API_KEY")
        if not isinstance(self.assistant.retriever.vector_store, QdrantVectorStore):
            raise RuntimeError("RAGAS evaluation requires an available Qdrant store")

    def _evaluate_with_ragas(self, samples: list[dict[str, Any]]):
        from langchain_openai import ChatOpenAI
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall

        judge = ChatOpenAI(
            model=self.config.judge_model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            temperature=0,
        )
        evaluator_llm = LangchainLLMWrapper(judge)
        dataset = EvaluationDataset.from_list(samples)
        return evaluate(
            dataset=dataset,
            metrics=[
                LLMContextRecall(llm=evaluator_llm),
                Faithfulness(llm=evaluator_llm),
                FactualCorrectness(llm=evaluator_llm, mode="f1"),
            ],
            llm=evaluator_llm,
            raise_exceptions=True,
        )

    def _load_dataset(self, dataset_path: Path) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for line_no, line in enumerate(
            dataset_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            payload = json.loads(line)
            question = str(payload.get("question", "")).strip()
            answer = str(payload.get("answer", "")).strip()
            if not question or not answer:
                raise ValueError(
                    f"Dataset row {line_no} must contain non-empty question and answer"
                )
            rows.append({"question": question, "answer": answer})
        return rows

    def _records_from_result(self, result) -> list[dict[str, Any]]:
        dataframe = result.to_pandas()
        return json.loads(dataframe.to_json(orient="records", force_ascii=False))

    def _mean_metric(self, rows: list[dict[str, Any]], name: str) -> float:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        if not values:
            raise RuntimeError(f"RAGAS metric {name!r} returned no scores")
        return mean(values)

    def _write_report(
        self,
        output_path: Path,
        report: EvaluationReport,
        details: list[dict[str, Any]],
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {"summary": report.as_dict(), "samples": details},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
