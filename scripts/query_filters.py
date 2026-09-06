from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from scripts.ranking import detect_requested_concepts
from scripts.text_utils import (
    category_contains,
    expand_traditional_chinese_query,
    get_optional_text_column,
)


INVALID_GAME_NAMES = {
    "",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "not available",
    "unknown",
    "unknown game",
    "unknown title",
    "removed",
    "delisted",
    "unavailable",
}


def is_valid_game_name(
    value: Any,
) -> bool:
    """Return True only for usable game titles."""

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass

    normalized_name = (
        str(value)
        .strip()
        .casefold()
    )

    return (
        normalized_name
        not in INVALID_GAME_NAMES
    )


def filter_invalid_game_names(
    embeddings: np.ndarray,
    game_dataframe: pd.DataFrame,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
]:
    """
    Remove invalid titles while preserving alignment.

    The same Boolean mask is applied to both embeddings
    and dataframe rows.
    """

    if len(embeddings) != len(
        game_dataframe
    ):
        raise ValueError(
            "Embedding count does not match "
            "the dataframe row count."
        )

    if "name" not in game_dataframe.columns:
        raise ValueError(
            "The dataframe must contain "
            "a 'name' column."
        )

    valid_title_mask = (
        game_dataframe["name"]
        .map(is_valid_game_name)
        .to_numpy(dtype=bool)
    )

    filtered_embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )[
        valid_title_mask
    ]

    filtered_dataframe = (
        game_dataframe.loc[
            valid_title_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    return (
        filtered_embeddings,
        filtered_dataframe,
    )


def is_free_value(
    value: Any,
) -> bool:
    """Convert common free values into a Boolean."""

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass

    normalized_value = (
        str(value)
        .strip()
        .casefold()
    )

    return normalized_value in {
        "true",
        "1",
        "yes",
        "y",
        "free",
        "free to play",
        "free-to-play",
    }


def extract_number(
    query: str,
    patterns: list[str],
) -> float | None:
    """Return the first matching number."""

    for pattern in patterns:
        match = re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        if match:
            return float(
                match.group(1)
            )

    return None


def extract_filters(
    query: str,
) -> dict[str, object]:
    """Extract hard constraints from a query."""

    query = expand_traditional_chinese_query(
        query
    )

    normalized_query = (
        query.casefold()
    )

    filters: dict[str, object] = {}

    maximum_price = extract_number(
        query,
        [
            (
                r"(?:under|below|less than|up to|"
                r"maximum|max(?:imum)?(?: price)?)"
                r"\s*(?:usd\s*)?\$?\s*"
                r"(\d+(?:\.\d+)?)"
            ),
            (
                r"(?:usd\s*)?\$?\s*"
                r"(\d+(?:\.\d+)?)"
                r"\s*(?:or less|or under|maximum|max)"
            ),
            (
                r"(\d+(?:\.\d+)?)"
                r"\s*(?:usd|dollars?)?\s*under"
            ),
            (
                r"(?:budget|price|預算|预算|價格|价格|"
                r"價錢|价钱|售價|售价|花費|花费|"
                r"低於|低于|不到|不超過|不超过|"
                r"不能超過|不能超过|小於|小于|少於|少于)"
                r"\s*(?:usd\s*)?\$?\s*"
                r"(\d+(?:\.\d+)?)"
                r"\s*(?:美元|美金|鎂|元|塊|块|usd|dollars?)?"
            ),
            (
                r"(\d+(?:\.\d+)?)"
                r"\s*(?:美元|美金|鎂|元|塊|块|usd|dollars?)?\s*"
                r"(?:以下|以內|以内|之內|之内|內|内|"
                r"不超過|不超过|不能超過|不能超过|"
                r"以?下|以?內|以?内)"
            ),
        ],
    )

    if maximum_price is not None:
        filters[
            "maximum_price"
        ] = maximum_price

    minimum_review_percentage = (
        extract_number(
            query,
            [
                (
                    r"(?:at least|minimum|min|above|over)"
                    r"\s*(\d+(?:\.\d+)?)\s*%"
                    r"(?:\s*positive)?"
                ),
                (
                    r"(?:positive reviews?|"
                    r"positive rating|rating)"
                    r"\s*(?:of|above|over|"
                    r"at least|minimum)?"
                    r"\s*(\d+(?:\.\d+)?)\s*%"
                ),
                (
                    r"(\d+(?:\.\d+)?)\s*%"
                    r"\s*(?:positive reviews?|positive rating)"
                ),
                (
                    r"(\d+(?:\.\d+)?)\s*%"
                    r"\s*(?:以上|起)?\s*"
                    r"(?:好評|好评|正面評價|正面评价|"
                    r"正評|正评)"
                ),
                (
                    r"(\d+(?:\.\d+)?)"
                    r"\s*(?:以上|起)\s*"
                    r"(?:好評|好评|正面評價|正面评价|"
                    r"正評|正评|推薦率|推荐率|玩家評價|玩家评价)"
                ),
                (
                    r"(?:至少|起碼|起码|不低於|不低于|"
                    r"高於|高于|超過|超过|"
                    r"好評率|好评率|正評率|正评率|"
                    r"評價|评价|評分|評價率|评价率|"
                    r"推薦率|推荐率|玩家評價|玩家评价)"
                    r"\s*(\d+(?:\.\d+)?)\s*%?"
                    r"\s*(?:以上|起)?"
                ),
                (
                    r"(?:好評|好评|正面評價|正面评价|"
                    r"正評|正评|推薦率|推荐率|玩家評價|玩家评价)"
                    r"\s*(?:至少|起碼|起码|不低於|不低于|"
                    r"高於|高于|超過|超过)?"
                    r"\s*(\d+(?:\.\d+)?)\s*%?"
                    r"\s*(?:以上|起)?"
                ),
            ],
        )
    )

    if (
        minimum_review_percentage
        is not None
    ):
        filters[
            "minimum_review_percentage"
        ] = minimum_review_percentage

    release_year_after = extract_number(
        query,
        [
            (
                r"(?:released\s+)?after\s+"
                r"(19\d{2}|20\d{2})"
            ),
            (
                r"(19\d{2}|20\d{2})"
                r"\s*(?:after|or later)"
            ),
            (
                r"(19\d{2}|20\d{2})"
                r"\s*年?\s*(?:之後|以後)"
            ),
            (
                r"(19\d{2}|20\d{2})"
                r"\s*年?\s*(?:之后|以后|後|后|起|以來|以来)"
            ),
        ],
    )

    if release_year_after is not None:
        filters[
            "release_year_after"
        ] = int(
            release_year_after
        )

    release_year_since = extract_number(
        query,
        [
            (
                r"(?:released\s+)?since\s+"
                r"(19\d{2}|20\d{2})"
            ),
            (
                r"(?:released\s+)?from\s+"
                r"(19\d{2}|20\d{2})"
                r"\s+onwards?"
            ),
        ],
    )

    if release_year_since is not None:
        filters[
            "release_year_since"
        ] = int(
            release_year_since
        )

    release_year_before = extract_number(
        query,
        [
            (
                r"(?:released\s+)?before\s+"
                r"(19\d{2}|20\d{2})"
            ),
            (
                r"(19\d{2}|20\d{2})"
                r"\s*before"
            ),
            (
                r"(19\d{2}|20\d{2})"
                r"\s*年?\s*(?:以前|之前)"
            ),
            (
                r"(19\d{2}|20\d{2})"
                r"\s*年?\s*(?:前|以前|之前)"
            ),
        ],
    )

    if release_year_before is not None:
        filters[
            "release_year_before"
        ] = int(
            release_year_before
        )

    # Match English free keywords next to CJK text without substring false positives.
    if re.search(
        r"(?<![a-zA-Z])free(?:[- ]to[- ]play)?(?![a-zA-Z])",
        normalized_query,
    ) or re.search(
        r"(?:免費|免费|不用錢|不用钱|零元)",
        query,
    ):
        filters["is_free"] = True

    # Platform names must match as standalone ASCII terms, including CJK-adjacent text.
    if re.search(r"(?<![a-zA-Z])linux(?![a-zA-Z])", normalized_query):
        filters["platform"] = "Linux"

    elif re.search(
        r"(?<![a-zA-Z])(?:mac|macos|macbook)(?![a-zA-Z])",
        normalized_query,
    ) or re.search(
        r"(?:蘋果電腦|苹果电脑|蘋果|苹果|麥金塔)",
        query,
        flags=re.IGNORECASE,
    ):
        filters["platform"] = "Mac"

    elif re.search(
        r"(?<![a-zA-Z])win(?:dows)?(?![a-zA-Z])",
        normalized_query,
    ) or re.search(
        r"(?:視窗|视窗)",
        query,
        flags=re.IGNORECASE,
    ):
        filters["platform"] = "Windows"

    if any(
        phrase in normalized_query
        for phrase in [
            "single-player",
            "single player",
            "one player",
            "solo game",
            "play alone",
            "single player",
        ]
    ) or re.search(
        r"(?:單人|单人|單機|单机|一個人|一个人|"
        r"自己玩|獨自|独自)",
        query,
    ):
        filters[
            "play_mode"
        ] = "single-player"

    # Exclude known ambiguous uses of coop/multiplayer terms.
    elif any(
        phrase in normalized_query
        for phrase in [
            "online co-op",
            "online coop",
            "cooperative",
            "co-op",
            "local co-op",
        ]
    ) or re.search(
        r"(?<!chicken)(?<!chicken )\bcoop\b",
        normalized_query,
    ) or re.search(
        # Bare "合作"/"協作"/"協力" ("cooperate") and "跟朋友"/"和朋友"/
        # "與朋友" ("with a friend") used to be in this alternation too,
        # but those are ordinary Chinese words with no gameplay meaning
        # on their own (e.g. "和知名動畫合作推出" -- "released in
        # collaboration with a well-known anime" -- has nothing to do
        # with co-op). Only the specific phrases that actually mean
        # "play together" are matched here; see the matching note in
        # text_utils.expand_traditional_chinese_query.
        r"(?:多人合作|多人协作|連線合作|联机合作|"
        r"連機合作|線上合作|线上合作|本地合作|"
        r"可以合作|可合作|合作遊玩|合作游玩|一起玩|"
        r"一起打|朋友一起|"
        r"雙人|双人|兩人|两人)",
        query,
    ):
        filters[
            "play_mode"
        ] = "co-op"

    elif any(
        phrase in normalized_query
        for phrase in [
            "multiplayer",
            "multi-player",
        ]
    ) or re.search(
        r"(?:多人連線|多人联机|線上多人|线上多人|"
        r"多人連機|連機|联机|網路多人|网络多人|"
        r"多人遊戲|多人游戏|(?<![很好許许眾众])多人)",
        query,
    ):
        filters[
            "play_mode"
        ] = "multiplayer"

    return filters


def convert_to_boolean(
    series: pd.Series,
) -> pd.Series:
    """Convert Boolean-looking values."""

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(
            False
        )

    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
                "free",
                "free to play",
                "free-to-play",
            }
        )
    )


def apply_filters(
    game_dataframe: pd.DataFrame,
    filters: dict[str, object],
) -> pd.Series:
    """Apply all requested hard constraints."""

    result_mask = pd.Series(
        True,
        index=game_dataframe.index,
        dtype=bool,
    )

    if "maximum_price" in filters:
        prices = pd.to_numeric(
            game_dataframe[
                "price_usd"
            ],
            errors="coerce",
        )

        result_mask &= (
            prices.notna()
            & (
                prices
                <= float(
                    filters[
                        "maximum_price"
                    ]
                )
            )
        )

    if (
        "minimum_review_percentage"
        in filters
    ):
        review_percentages = (
            pd.to_numeric(
                game_dataframe[
                    "positive_review_percentage"
                ],
                errors="coerce",
            )
        )

        result_mask &= (
            review_percentages.notna()
            & (
                review_percentages
                >= float(
                    filters[
                        "minimum_review_percentage"
                    ]
                )
            )
        )

    if filters.get(
        "is_free"
    ) is True:
        free_game_mask = (
            convert_to_boolean(
                game_dataframe[
                    "is_free"
                ]
            )
        )

        result_mask &= free_game_mask

    if "platform" in filters:
        requested_platform = str(
            filters["platform"]
        )

        result_mask &= (
            game_dataframe[
                "platforms"
            ]
            .fillna("")
            .astype(str)
            .str.contains(
                requested_platform,
                case=False,
                regex=False,
                na=False,
            )
        )

    release_years = pd.to_numeric(
        game_dataframe[
            "release_year"
        ],
        errors="coerce",
    )

    if (
        "release_year_after"
        in filters
    ):
        result_mask &= (
            release_years.notna()
            & (
                release_years
                > int(
                    filters[
                        "release_year_after"
                    ]
                )
            )
        )

    if (
        "release_year_since"
        in filters
    ):
        result_mask &= (
            release_years.notna()
            & (
                release_years
                >= int(
                    filters[
                        "release_year_since"
                    ]
                )
            )
        )

    if (
        "release_year_before"
        in filters
    ):
        result_mask &= (
            release_years.notna()
            & (
                release_years
                < int(
                    filters[
                        "release_year_before"
                    ]
                )
            )
        )

    if "play_mode" in filters:
        categories = (
            get_optional_text_column(
                game_dataframe,
                "categories",
            )
        )

        requested_play_mode = str(
            filters["play_mode"]
        )

        if (
            requested_play_mode
            == "single-player"
        ):
            play_mode_mask = (
                category_contains(
                    categories,
                    r"single[- ]?player",
                )
            )

        elif (
            requested_play_mode
            == "co-op"
        ):
            play_mode_mask = (
                category_contains(
                    categories,
                    (
                        r"co[- ]?op|"
                        r"coop|"
                        r"cooperative"
                    ),
                )
            )

        else:
            play_mode_mask = (
                category_contains(
                    categories,
                    (
                        r"multi[- ]?player|"
                        r"cross[- ]platform multiplayer"
                    ),
                )
            )

        result_mask &= (
            play_mode_mask
        )

    return result_mask


def query_requires_clarification(
    query: str,
    filters: dict[str, object],
) -> bool:
    """
    Return True for weak or price-only queries.

    A genre, mood, platform, play mode, year,
    free condition or review condition makes
    the request specific enough.
    """

    requested_concepts = (
        detect_requested_concepts(
            query
        )
    )

    if requested_concepts:
        return False

    strong_filter_names = {
        "minimum_review_percentage",
        "is_free",
        "platform",
        "play_mode",
        "release_year_after",
        "release_year_since",
        "release_year_before",
    }

    has_strong_filter = any(
        filter_name in filters
        for filter_name
        in strong_filter_names
    )

    return not has_strong_filter


def is_query_too_broad(
    query: str,
    filters: dict[str, object],
    requested_concepts: list[str],
    candidate_count: int,
) -> bool:
    """Check whether many games match a weak query."""

    if candidate_count == 0:
        return False

    if requested_concepts:
        return False

    return (
        candidate_count >= 100
        and query_requires_clarification(
            query,
            filters,
        )
    )
