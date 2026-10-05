from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from .config import PipelineConfig
from .judge import CoverageJudge
from .retriever import RetrievalResult
from .text_utils import canonicalize_url, normalize_company_name, normalize_text


REQUIRED_COLUMNS = {"oblgr_nm", "url", "content_summary"}


class Retriever(Protocol):
    def retrieve(
        self,
        candidate_summary: str,
        company_name: str,
        company_records: pd.DataFrame,
        top_k: int,
    ) -> RetrievalResult: ...


class CoveragePipeline:
    def __init__(
        self,
        baseline_df: pd.DataFrame,
        config: PipelineConfig,
        judge: CoverageJudge,
        retriever: Retriever,
    ) -> None:
        self._validate_columns("baseline", baseline_df)
        self.config = config
        self.judge = judge
        self.retriever = retriever
        self.baseline = baseline_df.copy().reset_index(drop=True)
        self.baseline.index.name = "baseline_id"
        self.baseline["_company_key"] = self.baseline["oblgr_nm"].map(normalize_company_name)
        self.baseline["_canonical_url"] = self.baseline["url"].map(canonicalize_url)
        self._groups = {
            key: group for key, group in self.baseline.groupby("_company_key", sort=False)
        }

    @staticmethod
    def _validate_columns(label: str, frame: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing required columns: {sorted(missing)}")

    def evaluate_row(self, row: pd.Series) -> dict[str, Any]:
        company_display = normalize_text(row["oblgr_nm"])
        company_key = normalize_company_name(company_display)
        candidate_url = canonicalize_url(row["url"])
        candidate_summary = normalize_text(row["content_summary"])
        records = self._groups.get(company_key)

        if records is None or records.empty:
            return self._result(
                covered=False,
                score=0.0,
                reason="No baseline records found for this oblgr_nm.",
                matched_by="none",
                baseline_count=0,
            )

        baseline_count = len(records)
        if candidate_url:
            url_matches = records[records["_canonical_url"] == candidate_url]
            if not url_matches.empty:
                match = url_matches.iloc[[0]]
                return self._result(
                    covered=True,
                    score=1.0,
                    reason="Equivalent URL already exists in baseline.",
                    matched_by="url",
                    baseline_count=baseline_count,
                    selected=match,
                    max_similarity=1.0,
                )

        if baseline_count <= self.config.retrieval_trigger:
            selected = records.copy()
            retrieval_used = False
            retrieval_mode = "all_company_records"
            max_similarity = None
        else:
            retrieval = self.retriever.retrieve(
                candidate_summary,
                company_key,
                records,
                self.config.top_k,
            )
            selected = retrieval.records
            max_similarity = retrieval.max_similarity
            retrieval_used = True
            retrieval_mode = "embedding_top_k"

            gate = self.config.min_similarity
            if gate is not None and max_similarity < gate:
                return self._result(
                    covered=False,
                    score=0.0,
                    reason=f"Top embedding similarity {max_similarity:.4f} is below gate {gate:.4f}.",
                    matched_by="similarity_gate",
                    baseline_count=baseline_count,
                    selected=selected,
                    retrieval_used=True,
                    retrieval_mode=retrieval_mode,
                    max_similarity=max_similarity,
                )

        score, reason = self.judge.evaluate(company_display, candidate_summary, selected)
        return self._result(
            covered=score >= self.config.judge_threshold,
            score=score,
            reason=reason,
            matched_by="llm",
            baseline_count=baseline_count,
            selected=selected,
            retrieval_used=retrieval_used,
            retrieval_mode=retrieval_mode,
            max_similarity=max_similarity,
        )

    def run(self, candidate_df: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns("candidate", candidate_df)
        results = []
        total = len(candidate_df)
        for position, (_, row) in enumerate(candidate_df.iterrows(), start=1):
            print(f"[{position}/{total}] {normalize_text(row['oblgr_nm'])}")
            try:
                result = self.evaluate_row(row)
            except Exception as exc:  # preserve remaining batch rows
                result = self._result(
                    covered=None,
                    score=None,
                    reason=f"ERROR: {exc}",
                    matched_by="error",
                    baseline_count=None,
                )
            results.append(result)
        return pd.concat(
            [candidate_df.reset_index(drop=True), pd.DataFrame(results)], axis=1
        )

    @staticmethod
    def _result(
        *,
        covered: bool | None,
        score: float | None,
        reason: str,
        matched_by: str,
        baseline_count: int | None,
        selected: pd.DataFrame | None = None,
        retrieval_used: bool = False,
        retrieval_mode: str = "not_used",
        max_similarity: float | None = None,
    ) -> dict[str, Any]:
        if selected is None:
            selected_ids = ""
            selected_urls = ""
            judge_count = 0
        else:
            selected_ids = ",".join(str(item) for item in selected.index.tolist())
            selected_urls = " | ".join(selected["url"].fillna("").astype(str).tolist())
            judge_count = len(selected) if matched_by == "llm" else 0

        return {
            "covered": covered,
            "coverage_score": score,
            "reason": reason,
            "matched_by": matched_by,
            "retrieval_mode": retrieval_mode,
            "baseline_count": baseline_count,
            "retrieval_used": retrieval_used,
            "judge_baseline_count": judge_count,
            "max_embedding_similarity": max_similarity,
            "selected_baseline_ids": selected_ids,
            "selected_baseline_urls": selected_urls,
        }
