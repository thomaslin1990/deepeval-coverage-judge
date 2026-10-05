import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .config import PipelineConfig
from .judge import DeepEvalCoverageJudge
from .pipeline import CoveragePipeline
from .retriever import SentenceTransformerRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge candidate CSV information coverage")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("judge_results.csv"))
    parser.add_argument("--retrieval-trigger", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--judge-threshold", type=float, default=0.80)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="Optional early-rejection gate; disabled when omitted",
    )
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--embedding-model", default=None)
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    config = PipelineConfig(
        retrieval_trigger=args.retrieval_trigger,
        top_k=args.top_k,
        judge_threshold=args.judge_threshold,
        min_similarity=args.min_similarity,
        judge_model=args.judge_model or os.getenv("JUDGE_MODEL", "gpt-4.1-mini"),
        embedding_model=args.embedding_model
        or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )

    baseline_df = pd.read_csv(args.baseline)
    candidate_df = pd.read_csv(args.candidate)
    pipeline = CoveragePipeline(
        baseline_df=baseline_df,
        config=config,
        judge=DeepEvalCoverageJudge(config.judge_model, config.judge_threshold),
        retriever=SentenceTransformerRetriever(config.embedding_model),
    )
    result = pipeline.run(candidate_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
