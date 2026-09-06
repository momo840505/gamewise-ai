# Traditional Chinese Query Support

GameWise accepts a practical set of Traditional Chinese game-search phrases for budget, review score, platform, release year, play mode, and common game concepts.

## Current approach

The query parser uses a curated synonym dictionary. Matching Chinese terms are expanded with English equivalents before structured filter extraction and concept detection.

The same expanded query is also used for dense retrieval. This improves mixed Chinese/English requests while keeping the existing embedding index unchanged.

This is still a rule-assisted approach. It is not a general Chinese language-understanding model.

## Covered examples

```text
我想找一款 20 美元以下、至少 80% 好評、可以合作的生存遊戲
```

```text
找一款支援 Mac 的農場模擬遊戲
```

```text
推薦 20 美元以下的遊戲
```

The first two return recommendations when matches exist. The last one asks for more detail because price alone leaves too many candidates.

## Known limits

The parser does not fully model:

- negation such as `不要恐怖遊戲`;
- disjunction such as `Mac 或 Linux`;
- arbitrary slang or dialectal wording;
- language beyond the terms in the maintained dictionary.

Regression tests cover known ambiguous phrases such as `多人` inside `很多人`, generic uses of `合作`, and `coop` in `chicken coop`.

A future multilingual embedding benchmark should compare this approach with a dedicated multilingual sentence model using judged English and Traditional Chinese queries.
