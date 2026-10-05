from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RetrievalResult:
    records: pd.DataFrame
    max_similarity: float


class SentenceTransformerRetriever:
    """Lazy-loading local embedding retriever with per-company caching."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._cache: dict[str, tuple[list[int], np.ndarray]] = {}

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def retrieve(
        self,
        candidate_summary: str,
        company_name: str,
        company_records: pd.DataFrame,
        top_k: int,
    ) -> RetrievalResult:
        if company_records.empty:
            return RetrievalResult(company_records.copy(), 0.0)

        if company_name not in self._cache:
            summaries = company_records["content_summary"].fillna("").astype(str).tolist()
            embeddings = self.model.encode(
                summaries,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            self._cache[company_name] = (
                company_records.index.astype(int).tolist(),
                np.asarray(embeddings),
            )

        record_ids, corpus_embeddings = self._cache[company_name]
        query_embedding = self.model.encode(
            [candidate_summary],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]

        similarities = corpus_embeddings @ query_embedding
        k = min(top_k, len(record_ids))
        positions = np.argsort(-similarities)[:k]
        selected_ids = [record_ids[int(position)] for position in positions]
        selected = company_records.loc[selected_ids].copy()
        selected["_similarity"] = [float(similarities[int(p)]) for p in positions]

        maximum = float(similarities[int(positions[0])]) if len(positions) else 0.0
        return RetrievalResult(selected, maximum)
