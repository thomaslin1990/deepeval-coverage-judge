# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
source .venv/bin/activate
pip install -e ".[dev]"                 # install package + pytest

pytest                                  # all tests (no network, no OpenAI, no model download)
pytest tests/test_pipeline.py::test_similarity_gate_skips_judge   # single test

coverage-judge --baseline data/baseline.example.csv \
               --candidate data/candidate.example.csv \
               --output judge_results.csv
```

Running the CLI needs `OPENAI_API_KEY` (read from `.env` via `python-dotenv`) and downloads the SentenceTransformers model on first retrieval. `JUDGE_MODEL` / `EMBEDDING_MODEL` env vars set the models; CLI flags override them. There is no linter or formatter configured.

## Architecture

The package is `src/coverage_judge/`, but the distribution name is `deepeval-coverage-judge`. For each candidate CSV row, it decides whether the row's material information is already covered by baseline CSV rows for the same company. Both CSVs require the columns `oblgr_nm,url,content_summary`.

`CoveragePipeline.evaluate_row` ([pipeline.py](src/coverage_judge/pipeline.py)) is a decision cascade. Each branch returns early with a different `matched_by` value:

1. **none**: no baseline rows for the normalized company key (`normalize_company_name`: casefold + collapsed whitespace).
2. **url**: the candidate's canonical URL equals a baseline URL for that company (`canonicalize_url` strips fragments, `utm_*` and other tracking params, and trailing slashes). Neither the judge nor the retriever is called.
3. If the company has `<= retrieval_trigger` baseline rows, all of them go to the judge (`retrieval_mode=all_company_records`). Otherwise the retriever picks the Top-K (`embedding_top_k`).
4. **similarity_gate**: optional. It applies only on the retrieval path when `min_similarity` is set and the Top-1 similarity is below it.
5. **llm**: the judge returns `(score, reason)`, and `covered = score >= judge_threshold`.

`run()` wraps each row in a try/except so one failure doesn't stop the batch. A failed row is recorded as `matched_by=error`. Every branch builds its output through `_result()`, so output columns are added or changed there.

**Dependency injection.** The pipeline depends only on two `Protocol`s, `CoverageJudge` ([judge.py](src/coverage_judge/judge.py)) and `Retriever` ([pipeline.py](src/coverage_judge/pipeline.py)). The CLI wires in the real `DeepEvalCoverageJudge` and `SentenceTransformerRetriever`. Tests inject `FakeJudge` / `FakeRetriever` instead. Heavy imports (`deepeval`, `sentence_transformers`) are deferred into constructors and properties, so importing the package stays cheap and tests never load them. Keep it that way.

**Judge.** `DeepEvalCoverageJudge` uses DeepEval `GEval` with custom evaluation steps and a 0–10 rubric, which DeepEval normalizes to 0–1. The candidate summary goes in `actual_output`. The selected baseline rows are rendered by `build_baseline_context` into `expected_output`, with record IDs, URLs, and similarity when it is present. `async_mode=False`, so rows are judged serially.

**Retriever.** It caches embeddings per company key for the life of the process, so each company's baseline corpus is encoded once. Embeddings are normalized, so similarity is a dot product. It attaches a `_similarity` column to the selected rows, which the judge context reads.

**Baseline IDs.** The baseline is `reset_index`'d in the constructor with the index name `baseline_id`. These positional IDs are what `selected_baseline_ids` reports and what the retriever cache stores.

`PipelineConfig` ([config.py](src/coverage_judge/config.py)) is a frozen dataclass that validates its ranges in `__post_init__`.

## Calibration notes (from README)

The defaults (`retrieval_trigger=20`, `top_k=8`, `judge_threshold=0.80`, gate disabled) are not calibrated. Tune each one on labelled data. Do not treat the GEval score as a calibrated probability. The similarity gate is off by default for this reason.
