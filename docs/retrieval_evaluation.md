# Retrieval Evaluation

GameWise AI evaluates the retrieval system as an engineering component rather than treating recommendations as a subjective demo.

## Evaluation Goals

The evaluation suite checks whether the system:

- preserves explicit user constraints before ranking;
- returns valid Steam titles without duplicates;
- asks for clarification when a query is too broad;
- reports no matching results when constraints are impossible;
- ranks games using semantic, concept, play-mode, and review-quality signals;
- keeps generated summaries grounded in retrieved metadata.

## Current Evaluation Coverage

The current evaluation set covers 11 representative scenarios:

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
- impossible-condition handling.

## Metrics

| Metric | Purpose |
|---|---|
| Expected search behaviour | Confirms that each query returns recommendations, asks for clarification, or reports no results as expected. |
| Hard-filter accuracy | Verifies that every returned game satisfies extracted constraints such as price, review threshold, platform, release year, free status, and official play mode. |
| Valid-title rate | Ensures recommendations do not contain blank, placeholder, or invalid game names. |
| Duplicate-free rate | Confirms that a recommendation set does not repeat the same title. |
| Metadata-term relevance@5 | Checks whether the top recommendations contain relevant terms in genres, tags, categories, or descriptions. |

## Why Hard Filters Run Before Ranking

GameWise applies strict filters before semantic ranking. A semantically similar game is removed if it violates an explicit user requirement.

Examples:

- a game over the requested budget is excluded;
- a game below the minimum review threshold is excluded;
- a Windows-only game is excluded from a Mac query;
- a multiplayer game is excluded when official co-op support is requested;
- a paid game is excluded from a free-game query.

This design is intentionally conservative because recommendation systems are easier to trust when stated constraints are never silently weakened.

## Recommended Next Evaluation Upgrades

- Expand the test set from 11 to 30-50 queries.
- Add Traditional Chinese query examples.
- Track recall@k against curated expected-title sets.
- Add groundedness checks for generated summaries.
- Add latency measurements for cold-start and warm-cache searches.
- Add a small human relevance review set for ambiguous subjective queries.

## CI Integration

The GitHub Actions workflow runs:

```bash
python -m pytest -q
python -m scripts.evaluate_retrieval
python -m compileall -q app.py scripts tests
```

This keeps retrieval behaviour, tests, and import health visible on every push and pull request.
