# Traditional Chinese Extension Plan

This extension turns GameWise AI from an English-only portfolio recommendation system into a stronger Taiwan-market AI engineering project.

## Motivation

Many AI engineering roles in Taiwan involve Traditional Chinese documents, customer questions, product metadata, support tickets, or internal knowledge bases. Supporting Traditional Chinese queries demonstrates practical NLP skills beyond an English demo.

## Target Behaviour

Users should be able to enter queries such as:

```text
我想找一款 20 美元以下、評價至少 80%、可以合作遊玩的生存遊戲
```

The system should extract:

- maximum price: 20;
- minimum positive review percentage: 80;
- play mode: co-op;
- concept: survival.

## Implementation Steps

1. Add Traditional Chinese keyword dictionaries for budget, free games, platforms, review thresholds, release-year phrases, and play modes.
2. Add Traditional Chinese concept synonyms for current concept groups.
3. Add evaluation cases for Chinese queries.
4. Update the UI examples to include English and Traditional Chinese prompts.
5. Document known limitations, especially mixed-language game metadata.

## Example Evaluation Cases

| Query | Expected behaviour |
|---|---|
| `我想找一款 20 美元以下、評價至少 80%、可以合作的生存遊戲` | Return co-op survival games under 20 with review percentage >= 80. |
| `推薦免費的心理恐怖遊戲` | Return free psychological horror results if available. |
| `找一款支援 Mac 的農場模擬遊戲` | Apply Mac and farming/simulation constraints. |
| `我只想找 2025 年之後推出、免費、Mac、99% 好評、可合作的遊戲` | Report no result if no game satisfies every constraint. |

## Portfolio Value

This upgrade supports resume claims around:

- multilingual NLP;
- query interpretation;
- retrieval evaluation;
- constraint-preserving recommendation;
- Taiwan-ready AI application design.
