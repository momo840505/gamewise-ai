# Retrieval Evaluation

The retrieval evaluation runs a fixed set of representative queries through the same search path used by the application.

## What is checked

Each case defines the expected parser output, expected response type, and a small set of metadata relevance terms. The evaluator checks:

- extracted hard filters against independently specified expected filters;
- detected concepts against independently specified expected concepts;
- recommendation, clarification, or no-result behavior;
- compliance with every extracted hard constraint;
- valid and non-duplicate game titles;
- simple metadata relevance in the first five results.

A failed interpretation or retrieval check exits with a non-zero status, so the GitHub Actions job fails instead of recording a green run with failed evaluation cases.

## Current coverage

The evaluation set contains 14 scenarios:

- cooperative survival;
- relaxing single-player;
- free psychological horror;
- Linux tactical strategy;
- recent co-op adventure;
- story-rich RPG;
- Windows racing;
- Mac farming simulation;
- recent multiplayer shooter;
- broad-query clarification;
- impossible-condition handling;
- Traditional Chinese cooperative survival;
- Traditional Chinese Mac farming simulation;
- Traditional Chinese broad-query clarification.

## What these metrics do not show

The suite is a correctness and regression check. It is not a ranking-quality benchmark.

Metadata relevance only checks whether at least one expected term appears in retrieved metadata. It cannot tell whether the first result is better than the fifth. A ranking benchmark would need judged query-title pairs and metrics such as nDCG@5, MRR, precision@5, and recall@5.

## CI

Every push and pull request runs:

```bash
python -m pytest -q
python -m scripts.evaluate_retrieval
python -m compileall -q app.py scripts tests
```

The evaluation command fails the job when query interpretation, search behavior, hard-filter compliance, title validity, or duplicate checks regress.
