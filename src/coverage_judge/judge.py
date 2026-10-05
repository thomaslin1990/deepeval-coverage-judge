from __future__ import annotations

from typing import Protocol

import pandas as pd

from .text_utils import normalize_text


class CoverageJudge(Protocol):
    def evaluate(
        self,
        company_name: str,
        candidate_summary: str,
        baseline_records: pd.DataFrame,
    ) -> tuple[float, str]: ...


def build_baseline_context(records: pd.DataFrame) -> str:
    blocks = []
    for position, (record_id, row) in enumerate(records.iterrows(), start=1):
        lines = [
            f"Baseline Record {position}",
            f"Record ID: {record_id}",
            f"URL: {normalize_text(row['url'])}",
        ]
        similarity = row.get("_similarity")
        if similarity is not None and not pd.isna(similarity):
            lines.append(f"Embedding Similarity: {float(similarity):.4f}")
        lines.extend(["Content:", normalize_text(row["content_summary"])])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


class DeepEvalCoverageJudge:
    def __init__(self, model_name: str, threshold: float) -> None:
        from deepeval.metrics import GEval
        from deepeval.metrics.g_eval import Rubric
        from deepeval.test_case import SingleTurnParams

        self._test_case_type = self._load_test_case_type()
        self.metric = GEval(
            name="Information Coverage",
            evaluation_steps=[
                "Identify every material factual claim in the candidate actual output.",
                "Compare each claim with all baseline records in the expected output.",
                "Treat paraphrases and semantically equivalent statements as covered.",
                "Do not treat wording, formatting, or immaterial detail changes as new information.",
                "Reduce the score for any new material event, date, amount, location, action, cause, consequence, or risk.",
                "Score near 1 only when essentially all material candidate information is already in the baseline.",
            ],
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            rubric=[
                Rubric(score_range=(0, 2), expected_outcome="Mostly new material information."),
                Rubric(score_range=(3, 5), expected_outcome="Some overlap, but substantial information is new."),
                Rubric(score_range=(6, 7), expected_outcome="Mostly overlapping, with meaningful new information."),
                Rubric(score_range=(8, 9), expected_outcome="Almost fully covered; differences are minor."),
                Rubric(score_range=(10, 10), expected_outcome="All material information is covered."),
            ],
            threshold=threshold,
            model=model_name,
            async_mode=False,
        )

    @staticmethod
    def _load_test_case_type():
        from deepeval.test_case import LLMTestCase

        return LLMTestCase

    def evaluate(
        self,
        company_name: str,
        candidate_summary: str,
        baseline_records: pd.DataFrame,
    ) -> tuple[float, str]:
        test_case = self._test_case_type(
            input=f"Determine information coverage for {company_name}.",
            actual_output=candidate_summary,
            expected_output=build_baseline_context(baseline_records),
        )
        self.metric.measure(test_case)
        return float(self.metric.score), str(self.metric.reason or "")
