<div align="center">

# GameWise AI

### Explainable Steam game recommendations from natural-language queries

[Live Demo](https://gamewise-ai.streamlit.app)

</div>

---

## Overview

GameWise searches a local Steam metadata snapshot using a hybrid retrieval pipeline. Natural-language requests are converted into structured constraints and ranking signals before recommendations are returned.

Example:

```text
a cooperative survival game under $20 with at least 80% positive reviews
```

The request is interpreted as:

```text
maximum price: $20
minimum positive reviews: 80%
play mode: co-op
concept: survival
```

Hard constraints are applied before ranking. A game that violates an explicit requirement is not allowed back into the result set because of a high semantic score.

## Retrieval pipeline

```text
User query
   |
   v
Query interpretation
   |
   +-- price
   +-- review threshold
   +-- platform
   +-- release year
   +-- free status
   +-- play mode
   +-- requested concepts
   |
   v
Hard filtering
   |
   v
Candidate set
   |
   +-- semantic similarity
   +-- concept relevance
   +-- play-mode preference
   +-- review quality
   |
   v
Hybrid ranking
   |
   v
Top-k Steam records
   |
   v
Grounded explanation
```

## Features

- Natural-language search
- Price and review thresholds
- Windows, Mac, and Linux filtering
- Release-year filtering
- Free-game filtering using the explicit dataset free flag
- Official Steam category checks for single-player, co-op, and multiplayer
- Sentence Transformer retrieval
- Field-aware concept scoring
- Hybrid ranking
- Clarification for weak queries
- Strict no-result handling
- Optional grounded OpenAI summary
- Local fallback summary
- Search history and shortlist
- Sorting and CSV export
- Developer ranking details
- Traditional Chinese query support

## Traditional Chinese support

Traditional Chinese terms are expanded with English equivalents before structured parsing, concept detection, and semantic retrieval.

Examples:

```text
20 美元以下、80% 好評、支援 Mac、合作、生存遊戲
```

```text
找一款支援 Mac 的農場模擬遊戲
```

The parser is rule-assisted. It does not fully model negation, disjunction, arbitrary slang, or every possible Chinese phrasing.

See [Traditional Chinese Query Support](docs/traditional_chinese_extension.md).

## Ranking

The ranking weights depend on which signals are present in the query.

### Concept and play-mode query

```text
0.55 semantic
+ 0.20 concept
+ 0.15 play mode
+ 0.10 review quality
```

### Concept-only query

```text
0.65 semantic
+ 0.25 concept
+ 0.10 review quality
```

### Play-mode-only query

```text
0.70 semantic
+ 0.20 play mode
+ 0.10 review quality
```

### Other specific queries

```text
0.90 semantic
+ 0.10 review quality
```

These weights are heuristic. They have not been learned from human relevance labels.

## Evaluation

The repository contains 14 retrieval evaluation scenarios and 30 automated tests.

The evaluation checks:

- expected filter extraction;
- expected concept detection;
- recommendation, clarification, and no-result behavior;
- hard-filter compliance;
- valid titles;
- duplicate-free results;
- simple metadata relevance in the first five results.

A failed interpretation or retrieval check exits with a non-zero status, so CI fails on regression.

The current suite is a correctness check, not a full ranking-quality benchmark. There is no human-labeled relevance set yet, so nDCG, MRR, precision@k, and recall@k are not reported.

See [Retrieval Evaluation](docs/retrieval_evaluation.md).

## Dataset

The repository contains a 1,495-game Steam metadata snapshot.

Available fields include:

- app ID
- name
- price
- free status
- release year
- positive and negative review counts
- genres
- Steam categories
- community tags
- supported platforms
- short description
- Steam store URL

### Data provenance

The original collection source and collection date are not recorded in the repository history. This project does not claim a source that cannot be verified.

The snapshot is kept for reproducible portfolio evaluation. Prices, availability, and store metadata can become stale and should not be treated as live Steam data.

The cleaning pipeline also records price/free inconsistencies instead of assuming every zero-price record is a free-to-play game.

## Project structure

```text
gamewise-ai/
├── app.py
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── evaluation/
├── scripts/
│   ├── build_embeddings.py
│   ├── clean_dataset.py
│   ├── evaluate_retrieval.py
│   ├── formatting.py
│   ├── generate_answer.py
│   ├── inspect_dataset.py
│   ├── query_filters.py
│   ├── ranking.py
│   ├── search.py
│   ├── semantic_search.py
│   └── text_utils.py
├── tests/
├── .github/workflows/tests.yml
└── .streamlit/config.toml
```

## Setup

```powershell
git clone https://github.com/momo840505/gamewise-ai.git
cd gamewise-ai

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional generation:

```powershell
Copy-Item .env.example .env
```

```text
OPENAI_API_KEY=
OPENAI_MODEL=
```

The retrieval system works without an API key.

## Run

```powershell
python -m streamlit run .\app.py
```

## Tests

```powershell
python -m pytest -q
```

Expected:

```text
30 passed
```

## Retrieval evaluation

```powershell
python -m scripts.evaluate_retrieval
```

## CI

GitHub Actions runs:

```text
pytest
retrieval evaluation
compileall
```

The workflow runs on pushes and pull requests to `main`.

## Known limitations

- The dataset is a fixed 1,495-game snapshot, not the full Steam catalogue.
- Store prices and availability are not live.
- Ranking weights are heuristic.
- Review quality currently uses positive-review percentage without a Bayesian or Wilson adjustment.
- Traditional Chinese support is rule-assisted.
- Negation and OR-style constraints are not fully modeled.
- Ranking quality has not yet been measured against human relevance judgments.
- The first semantic query has model-loading overhead.
- Search history and shortlist data are session-only.
- The generated summary is context-constrained but does not yet use a structured factual validator.

## License

MIT
