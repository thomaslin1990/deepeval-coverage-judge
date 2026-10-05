# DeepEval Coverage Judge

Compare every row in `candidate.csv` with records in `baseline.csv` and decide
whether the candidate's material information is already covered.

The pipeline is:

1. Match records by normalized `oblgr_nm`.
2. Return `covered=true` immediately for the same normalized URL.
3. If the company has at most `retrieval_trigger` baseline records, send all of
   them to the judge.
4. Otherwise, use local SentenceTransformers embeddings and send only Top-K
   records to DeepEval `GEval`.
5. Mark the row as covered when the DeepEval score is at least the judge
   threshold.

An optional embedding-similarity gate can skip the LLM when Top-1 similarity is
very low. It is disabled by default because the correct cutoff should be
calibrated on labelled examples.

## Project layout

```text
deepeval-coverage-judge/
├── data/
│   ├── baseline.example.csv
│   └── candidate.example.csv
├── src/coverage_judge/
│   ├── cli.py
│   ├── config.py
│   ├── judge.py
│   ├── pipeline.py
│   └── retriever.py
├── tests/test_pipeline.py
├── run_comparison.sh
├── .env.example
└── pyproject.toml
```

## Setup

Python 3.10+ is recommended.

```bash
cd deepeval-coverage-judge
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Put your OpenAI key in `.env`, or export it in the shell:

```bash
export OPENAI_API_KEY="your-api-key"
```

## Run the example

The simplest way is the helper script, which runs the example files in `data/`
and writes `judge_results.csv`:

```bash
./run_comparison.sh
```

It calls `.venv/bin/python -m coverage_judge.cli` with `PYTHONPATH=src`, so the
virtual environment does not need to be activated. Any extra flags are passed
through to the CLI:

```bash
./run_comparison.sh --top-k 5 --judge-threshold 0.75
```

To change the input or output files, edit the paths in `run_comparison.sh`.

Equivalently, with the virtual environment activated, use the installed
`coverage-judge` command:

```bash
coverage-judge \
  --baseline data/baseline.example.csv \
  --candidate data/candidate.example.csv \
  --output judge_results.csv
```

Run your own files:

```bash
coverage-judge \
  --baseline baseline.csv \
  --candidate candidate.csv \
  --output judge_results.csv \
  --retrieval-trigger 20 \
  --top-k 8 \
  --judge-threshold 0.80
```

To enable an embedding gate after calibration:

```bash
coverage-judge \
  --baseline baseline.csv \
  --candidate candidate.csv \
  --output judge_results.csv \
  --min-similarity 0.35
```

Useful environment variables:

```text
JUDGE_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OPENAI_API_KEY=...
```

CLI flags override environment variables and defaults.

## Input schema

Both CSV files must contain:

```text
oblgr_nm,url,content_summary
```

Matching `oblgr_nm` is case-insensitive and collapses repeated whitespace. URL
matching lowercases the scheme/host, removes fragments and common tracking
parameters, and ignores a trailing slash.

## Output columns

The output keeps every candidate column and adds:

- `covered`
- `coverage_score`
- `reason`
- `matched_by`
- `retrieval_mode`
- `baseline_count`
- `retrieval_used`
- `judge_baseline_count`
- `max_embedding_similarity`
- `selected_baseline_ids`
- `selected_baseline_urls`

`matched_by` is one of `none`, `url`, `llm`, `similarity_gate`, or `error`.

## Tests

Tests do not call OpenAI or download an embedding model.

```bash
pytest
```

## Recommended calibration

Before production use, label a representative sample and tune separately:

- `retrieval_trigger`: cost/latency decision.
- `top_k`: retrieval recall versus prompt size.
- `judge_threshold`: semantic coverage decision.
- `min_similarity`: optional early-rejection decision.

Do not treat the judge score as a calibrated probability.
