from __future__ import annotations

import argparse
from pathlib import Path

from .admissions_db import AdmissionsDatabase
from .assistant import build_assistant_from_env
from .chat_mining import ChatMiningPipeline
from .config import AssistantConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="MAI admissions assistant utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-db", help="Build SQLite admissions database from cutoff CSV files")
    subparsers.add_parser("rebuild-index", help="Rebuild the full RAG index")
    subparsers.add_parser("update-index", help="Incrementally update changed RAG sources")

    mine = subparsers.add_parser("mine-chats", help="Extract top questions and QA dataset from chat exports")
    mine.add_argument("input_dir", type=Path)
    mine.add_argument("output_dir", type=Path)
    mine.add_argument("--top-n", type=int, default=200)
    mine.add_argument("--dataset-size", type=int, default=500)
    mine.add_argument(
        "--cluster-eps",
        type=float,
        default=0.18,
        help="Maximum cosine distance between neighboring questions",
    )
    mine.add_argument("--cluster-min-samples", type=int, default=2)
    mine.add_argument("--answer-threshold", type=float, default=0.85)
    mine.add_argument("--answer-window", type=int, default=3)

    args = parser.parse_args()
    config = AssistantConfig.from_env()

    if args.command == "build-db":
        db = AdmissionsDatabase(config.sqlite_path, config.data_dir / "cutoff_points")
        db.initialize(rebuild=True)
        print(f"Built admissions DB: {config.sqlite_path}")
    elif args.command == "rebuild-index":
        assistant = build_assistant_from_env()
        chunks = assistant.rebuild_index()
        print(f"Rebuilt RAG index: {chunks} chunks")
    elif args.command == "update-index":
        assistant = build_assistant_from_env()
        chunks = assistant.update_index()
        print(f"Updated RAG index: {chunks} changed chunks")
    elif args.command == "mine-chats":
        result = ChatMiningPipeline(
            cluster_eps=args.cluster_eps,
            cluster_min_samples=args.cluster_min_samples,
            answer_similarity_threshold=args.answer_threshold,
            answer_candidate_window=args.answer_window,
        ).run(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            top_n=args.top_n,
            dataset_size=args.dataset_size,
        )
        print(
            f"Extracted {len(result.top_questions)} top questions and "
            f"{len(result.qa_pairs)} QA pairs"
        )


if __name__ == "__main__":
    main()
