import pandas as pd

from coverage_judge.config import PipelineConfig
from coverage_judge.pipeline import CoveragePipeline
from coverage_judge.retriever import RetrievalResult


class FakeJudge:
    def __init__(self, score: float = 0.9) -> None:
        self.score = score
        self.calls = []

    def evaluate(self, company_name, candidate_summary, baseline_records):
        self.calls.append((company_name, candidate_summary, baseline_records.copy()))
        return self.score, "fake judge reason"


class FakeRetriever:
    def __init__(self, similarity: float = 0.75) -> None:
        self.similarity = similarity
        self.calls = []

    def retrieve(self, candidate_summary, company_name, company_records, top_k):
        self.calls.append((candidate_summary, company_name, len(company_records), top_k))
        selected = company_records.head(top_k).copy()
        selected["_similarity"] = self.similarity
        return RetrievalResult(selected, self.similarity)


def baseline(count: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "oblgr_nm": ["ABC Corp"] * count,
            "url": [f"https://example.com/{i}" for i in range(count)],
            "content_summary": [f"baseline fact {i}" for i in range(count)],
        }
    )


def candidate(url="https://other.example.com/new") -> pd.Series:
    return pd.Series(
        {"oblgr_nm": "  abc   CORP ", "url": url, "content_summary": "candidate fact"}
    )


def make_pipeline(count=3, trigger=20, score=0.9, similarity=0.75, gate=None):
    judge = FakeJudge(score)
    retriever = FakeRetriever(similarity)
    config = PipelineConfig(
        retrieval_trigger=trigger,
        top_k=2,
        judge_threshold=0.8,
        min_similarity=gate,
    )
    return CoveragePipeline(baseline(count), config, judge, retriever), judge, retriever


def test_same_url_short_circuits_judge_and_retrieval():
    pipeline, judge, retriever = make_pipeline()
    result = pipeline.evaluate_row(candidate("https://example.com/1/?utm_source=email#x"))
    assert result["covered"] is True
    assert result["matched_by"] == "url"
    assert judge.calls == []
    assert retriever.calls == []


def test_small_company_sends_all_records_directly_to_judge():
    pipeline, judge, retriever = make_pipeline(count=3, trigger=3)
    result = pipeline.evaluate_row(candidate())
    assert result["covered"] is True
    assert result["retrieval_used"] is False
    assert result["judge_baseline_count"] == 3
    assert len(judge.calls) == 1
    assert retriever.calls == []


def test_large_company_uses_top_k_retrieval():
    pipeline, judge, retriever = make_pipeline(count=5, trigger=3)
    result = pipeline.evaluate_row(candidate())
    assert result["retrieval_used"] is True
    assert result["judge_baseline_count"] == 2
    assert len(retriever.calls) == 1
    assert len(judge.calls) == 1


def test_similarity_gate_skips_judge():
    pipeline, judge, _ = make_pipeline(count=5, trigger=3, similarity=0.2, gate=0.35)
    result = pipeline.evaluate_row(candidate())
    assert result["covered"] is False
    assert result["matched_by"] == "similarity_gate"
    assert judge.calls == []


def test_missing_company_is_not_covered():
    pipeline, judge, retriever = make_pipeline()
    row = candidate()
    row["oblgr_nm"] = "Unknown Ltd"
    result = pipeline.evaluate_row(row)
    assert result["covered"] is False
    assert result["matched_by"] == "none"
    assert judge.calls == []
    assert retriever.calls == []
