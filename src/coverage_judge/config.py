from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    retrieval_trigger: int = 20
    top_k: int = 8
    judge_threshold: float = 0.80
    min_similarity: float | None = None
    judge_model: str = "gpt-4.1-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        if self.retrieval_trigger < 1:
            raise ValueError("retrieval_trigger must be at least 1")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0 <= self.judge_threshold <= 1:
            raise ValueError("judge_threshold must be between 0 and 1")
        if self.min_similarity is not None and not -1 <= self.min_similarity <= 1:
            raise ValueError("min_similarity must be between -1 and 1")
