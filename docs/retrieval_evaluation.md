# Retrieval Evaluation

I wanted more than "it looks right when I try a few queries" before I called retrieval done, so this is the evaluation I actually run against it -- what it checks, what it doesn't, and where I'd take it next.

## What I'm actually checking

The evaluation suite runs a fixed set of queries through the real pipeline and checks that it:

- keeps every explicit constraint the user stated instead of quietly dropping one during ranking
- only returns valid, non-duplicate Steam titles
- asks for clarification when a query is too broad to mean anything specific
- reports no results honestly when the constraints can't all be satisfied, instead of relaxing one to have something to show
- actually uses all four ranking signals (semantic, concept, play-mode, review quality), not just semantic similarity
- keeps generated summaries grounded in the retrieved metadata rather than inventing details

## Current coverage: 14 scenarios

- cooperative survival
- relaxing single-player
- free psychological horror
- Linux tactical strategy
- recent co-op adventure
- story-rich RPG
- Windows racing
- Mac farming simulation
- recent multiplayer shooter
- broad-query clarification
- impossible-condition handling
- Traditional Chinese cooperative survival
- Traditional Chinese Mac farming simulation
- Traditional Chinese broad-query clarification

The last three got added once I built out Traditional Chinese query support -- I didn't want the evaluation set to only ever exercise the English path.

## Metrics

| Metric | What it's actually checking |
|---|---|
| Expected search behaviour | Did each query get the right kind of response -- recommendations, a clarification prompt, or a no-result message? |
| Hard-filter accuracy | Does every returned game actually satisfy every constraint I extracted from the query (price, review threshold, platform, release year, free status, official play mode)? |
| Valid-title rate | No blank, placeholder, or garbage titles in the results. |
| Duplicate-free rate | The same game doesn't show up twice in one result set. |
| Metadata-term relevance@5 | Do the top results actually contain a relevant term in their genres, tags, categories, or description? |

I go into why none of this is a ranking-quality metric (nDCG/MRR territory) in the main README's Evaluation section -- these five checks are about correctness and safety, not about whether the *order* of results is good.

## Why hard filters run before ranking, not after

If I let semantic similarity rank first and filtered afterward, an over-budget or wrong-platform game could still surface if it happened to score well semantically -- and it did, early on. So constraints get applied as a hard cut before ranking ever sees the candidates:

- over budget → excluded
- below the minimum review threshold → excluded
- wrong platform → excluded
- doesn't satisfy official co-op when co-op was requested → excluded
- paid when free was requested → excluded

I'd rather return fewer results than one that looks great but breaks something the user actually asked for.

## Where I'd take this next

- Grow the query set past 14 -- ideally into the 30-50 range, across more genre/platform/language combinations
- Track recall@k against a curated set of expected titles per query, instead of just checking that constraints hold
- Add a groundedness check specifically for the generated summaries
- Measure cold-start vs. warm-cache latency separately
- Get a small human relevance panel together for the genuinely subjective queries, where "good match" isn't something a rule can fully capture

## CI

Every push and PR runs:

```bash
python -m pytest -q
python -m scripts.evaluate_retrieval
python -m compileall -q app.py scripts tests
```

so retrieval behaviour, the test suite, and basic import health are all checked automatically rather than something I have to remember to run by hand.
