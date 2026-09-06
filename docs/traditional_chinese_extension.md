# Traditional Chinese Query Support

This started as a plan and is now actually built and shipped -- I'm keeping this doc as a record of why I added it and how it works, rather than rewriting it to erase the fact that it started as a plan.

## Why I bothered

A lot of the AI/data roles I'm looking at in Taiwan involve working with Traditional Chinese text in some form -- documents, customer questions, product metadata, support tickets. An English-only demo doesn't really show that I can handle that, so I added a Traditional Chinese path to the same query pipeline instead of building a separate project for it.

## What it needs to handle

A query like:

```text
我想找一款 20 美元以下、評價至少 80%、可以合作遊玩的生存遊戲
```

needs to extract the same structured filters the English pipeline would get from the equivalent English sentence:

- maximum price: 20
- minimum positive review percentage: 80
- play mode: co-op
- concept: survival

## How it actually works

Rather than training or wiring up a separate Chinese NLU model, I built a curated Traditional Chinese synonym dictionary (`TRADITIONAL_CHINESE_QUERY_SYNONYMS` in `scripts/text_utils.py`, 230+ terms covering budget, free-game, platform, review-threshold, release-year, and play-mode phrasing, plus the same concept vocabulary the English pipeline uses). A Chinese query gets expanded with the matching English equivalents before it hits the same `extract_filters`/`detect_requested_concepts` logic the English pipeline already used, so I didn't have to build and maintain a second parallel pipeline.

That approach is simple, but it has a real failure mode: some of these Chinese terms are ordinary words used constantly outside any gameplay context, not proper nouns/keywords the way most of the English filter terms are. "多人" (multiplayer) is also just the tail of "很多人" (a lot of people); "合作" (cooperate) shows up in phrases like a game announcing a brand tie-in that has nothing to do with co-op play. Both of those caused real false positives that are now covered by regression tests -- see the Limitations section in the main README and the "Notes From Building This" section for the actual bugs.

## Evaluation

Three of the fourteen retrieval-evaluation scenarios are Traditional Chinese queries (cooperative survival, Mac farming simulation, and a broad price-only query that should trigger clarification), covering the same three response types -- recommendations, clarification, no-result -- that the English scenarios cover. See `docs/retrieval_evaluation.md` for the full evaluation writeup.

## What's not there yet

- Full free-form Chinese language understanding -- this is a dictionary, not an NLU model, so unusual phrasing or dialectal variants can fall through
- Simplified Chinese slang beyond the terms I've explicitly added
- Any language beyond English and Traditional Chinese
