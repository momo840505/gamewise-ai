from __future__ import annotations

import argparse
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "game_embeddings.npy"
)

EMBEDDING_INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "game_embedding_index.csv"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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
}


CONCEPT_GROUPS: dict[str, dict[str, float]] = {
    "relaxing": {
        "relaxing": 1.00,
        "cozy": 1.00,
        "wholesome": 0.90,
        "family friendly": 0.70,
        "casual": 0.60,
        "atmospheric": 0.35,
    },
    "casual": {
        "casual": 1.00,
        "relaxing": 0.75,
        "cozy": 0.75,
        "family friendly": 0.65,
        "wholesome": 0.65,
    },
    "psychological horror": {
        "psychological horror": 1.00,
        "psychological": 0.75,
        "survival horror": 0.75,
        "horror": 0.65,
        "dark": 0.35,
        "atmospheric": 0.30,
    },
    "survival": {
        "survival": 1.00,
        "open world survival craft": 1.00,
        "survival horror": 0.85,
        "base building": 0.55,
        "crafting": 0.55,
        "resource management": 0.45,
    },
    "open world": {
        "open world": 1.00,
        "sandbox": 0.80,
        "exploration": 0.70,
    },
    "turn-based": {
        "turn based": 1.00,
        "turn based strategy": 1.00,
        "turn based tactics": 1.00,
        "turn based combat": 1.00,
    },
    "tactical": {
        "tactical": 1.00,
        "tactical rpg": 1.00,
        "turn based tactics": 1.00,
        "real time tactics": 0.90,
        "team based": 0.45,
    },
    "strategy": {
        "strategy": 1.00,
        "grand strategy": 1.00,
        "4x": 1.00,
        "wargame": 0.90,
        "real time strategy": 1.00,
        "turn based strategy": 1.00,
    },
    "adventure": {
        "adventure": 1.00,
        "action adventure": 1.00,
        "exploration": 0.75,
        "story rich": 0.65,
        "point and click": 0.65,
    },
    "puzzle": {
        "puzzle": 1.00,
        "logic": 0.85,
        "escape room": 0.85,
        "mystery": 0.65,
    },
    "farming": {
        "farming": 1.00,
        "farming sim": 1.00,
        "agriculture": 0.90,
        "life sim": 0.75,
    },
    "simulation": {
        "simulation": 1.00,
        "simulator": 1.00,
        "life sim": 0.80,
        "management": 0.65,
    },
    "story rich": {
        "story rich": 1.00,
        "narrative": 0.90,
        "choices matter": 0.85,
        "multiple endings": 0.70,
    },
    "rpg": {
        "rpg": 1.00,
        "role playing": 1.00,
        "action rpg": 1.00,
        "jrpg": 1.00,
        "crpg": 1.00,
        "tactical rpg": 1.00,
    },
    "racing": {
        "racing": 1.00,
        "driving": 0.85,
        "automobile sim": 0.85,
        "arcade racing": 1.00,
    },
    "shooter": {
        "shooter": 1.00,
        "fps": 1.00,
        "first person shooter": 1.00,
        "third person shooter": 1.00,
    },
}


CONCEPT_TRIGGER_TERMS: dict[str, list[str]] = {
    "relaxing": [
        "relaxing",
        "cozy",
        "wholesome",
    ],
    "casual": [
        "casual",
    ],
    "psychological horror": [
        "psychological horror",
        "psychological-horror",
    ],
    "survival": [
        "survival",
        "survival game",
        "open world survival craft",
    ],
    "open world": [
        "open world",
        "open-world",
    ],
    "turn-based": [
        "turn based",
        "turn-based",
        "turn based strategy",
        "turn-based strategy",
        "turn based tactics",
        "turn-based tactics",
    ],
    "tactical": [
        "tactical",
        "tactics",
    ],
    "strategy": [
        "strategy",
        "strategic",
        "4x",
        "wargame",
    ],
    "adventure": [
        "adventure",
        "action adventure",
        "action-adventure",
    ],
    "puzzle": [
        "puzzle",
        "logic game",
        "escape room",
    ],
    "farming": [
        "farming",
        "farm game",
        "agriculture",
    ],
    "simulation": [
        "simulation",
        "simulator",
        "sim game",
    ],
    "story rich": [
        "story rich",
        "story-rich",
        "narrative",
        "choices matter",
    ],
    "rpg": [
        "rpg",
        "role playing",
        "role-playing",
        "jrpg",
        "crpg",
    ],
    "racing": [
        "racing",
        "driving game",
        "car game",
    ],
    "shooter": [
        "shooter",
        "fps",
        "first person shooter",
        "first-person shooter",
        "third person shooter",
        "third-person shooter",
    ],
}


TRADITIONAL_CHINESE_QUERY_SYNONYMS: dict[str, str] = {
    "免費": " free ",
    "免費遊戲": " free game ",
    "以下": " under ",
    "低於": " under ",
    "不超過": " under ",
    "美元": " ",
    "美金": " ",
    "好評": " positive reviews ",
    "正面評價": " positive reviews ",
    "至少": " at least ",
    "支援": " for ",
    "合作": " co-op ",
    "多人合作": " co-op ",
    "單人": " single-player ",
    "獨自": " single-player ",
    "多人": " multiplayer ",
    "生存": " survival ",
    "心理恐怖": " psychological horror ",
    "恐怖": " horror ",
    "放鬆": " relaxing ",
    "療癒": " cozy ",
    "休閒": " casual ",
    "回合制": " turn-based ",
    "戰術": " tactical ",
    "策略": " strategy ",
    "冒險": " adventure ",
    "解謎": " puzzle ",
    "農場": " farming ",
    "模擬": " simulation ",
    "劇情": " story rich ",
    "角色扮演": " rpg ",
    "賽車": " racing ",
    "射擊": " shooter ",
    "年之後": " after ",
    "年以前": " before ",
    "之後": " after ",
    "以前": " before ",
}


def expand_traditional_chinese_query(query: str) -> str:
    """Append English equivalents for common Traditional Chinese game queries."""

    expanded_terms = [
        replacement
        for term, replacement in TRADITIONAL_CHINESE_QUERY_SYNONYMS.items()
        if term in query
    ]

    if not expanded_terms:
        return query

    return query + " " + " ".join(expanded_terms)


def normalize_text(value: Any) -> str:
    """Normalize text for phrase matching."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    normalized_value = str(value).casefold()

    normalized_value = re.sub(
        r"[-_/]+",
        " ",
        normalized_value,
    )

    normalized_value = re.sub(
        r"[^a-z0-9+]+",
        " ",
        normalized_value,
    )

    normalized_value = re.sub(
        r"\s+",
        " ",
        normalized_value,
    ).strip()

    return normalized_value


def contains_normalized_term(
    text: str,
    term: str,
) -> bool:
    """Check for a normalized whole phrase."""

    normalized_text = (
        f" {normalize_text(text)} "
    )

    normalized_term = normalize_text(
        term
    )

    if not normalized_term:
        return False

    return (
        f" {normalized_term} "
        in normalized_text
    )


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


@lru_cache(maxsize=1)
def load_search_data() -> tuple[
    np.ndarray,
    pd.DataFrame,
]:
    """Load and cache embeddings and metadata."""

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "Embeddings were not found. "
            "Run scripts/build_embeddings.py first."
        )

    if not EMBEDDING_INDEX_PATH.exists():
        raise FileNotFoundError(
            "Embedding index was not found. "
            "Run scripts/build_embeddings.py first."
        )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    game_dataframe = pd.read_csv(
        EMBEDDING_INDEX_PATH
    ).reset_index(
        drop=True
    )

    if len(embeddings) != len(
        game_dataframe
    ):
        raise ValueError(
            "Embedding count does not match "
            "the metadata row count."
        )

    (
        embeddings,
        game_dataframe,
    ) = filter_invalid_game_names(
        embeddings,
        game_dataframe,
    )

    duplicate_mask = (
        game_dataframe["name"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .duplicated(keep="first")
        .to_numpy(dtype=bool)
    )

    if duplicate_mask.any():
        keep_mask = ~duplicate_mask

        embeddings = embeddings[
            keep_mask
        ]

        game_dataframe = (
            game_dataframe.loc[
                keep_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    return (
        embeddings,
        game_dataframe,
    )


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """Load the embedding model once per process."""

    return SentenceTransformer(
        MODEL_NAME
    )


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
                r"(\d+(?:\.\d+)?)"
                r"\s*(?:美元|美金)?\s*"
                r"(?:以下|以內|內|不超過)"
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
                    r"\s*(?:好評|正面評價)"
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
        ],
    )

    if release_year_before is not None:
        filters[
            "release_year_before"
        ] = int(
            release_year_before
        )

    if re.search(
        r"\bfree(?:[- ]to[- ]play)?\b",
        normalized_query,
    ):
        filters["is_free"] = True

    if "linux" in normalized_query:
        filters["platform"] = "Linux"

    elif re.search(
        r"\bmac(?:os)?\b",
        normalized_query,
    ):
        filters["platform"] = "Mac"

    elif "windows" in normalized_query:
        filters["platform"] = "Windows"

    if any(
        phrase in normalized_query
        for phrase in [
            "single-player",
            "single player",
            "one player",
            "solo game",
            "play alone",
        ]
    ):
        filters[
            "play_mode"
        ] = "single-player"

    elif any(
        phrase in normalized_query
        for phrase in [
            "online co-op",
            "online coop",
            "cooperative",
            "co-op",
            "coop",
        ]
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


def get_optional_text_column(
    dataframe: pd.DataFrame,
    column_name: str,
) -> pd.Series:
    """
    Return a column or an aligned empty column.

    This lets tests use a smaller DataFrame without
    production-only metadata such as categories.
    """

    if column_name in dataframe.columns:
        return dataframe[
            column_name
        ]

    return pd.Series(
        "",
        index=dataframe.index,
        dtype="object",
    )


def category_contains(
    category_series: pd.Series,
    pattern: str,
) -> pd.Series:
    """Search official Steam category text."""

    return (
        category_series
        .fillna("")
        .astype(str)
        .str.contains(
            pattern,
            case=False,
            regex=True,
            na=False,
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

        zero_price_mask = (
            pd.to_numeric(
                game_dataframe[
                    "price_usd"
                ],
                errors="coerce",
            )
            .fillna(np.inf)
            == 0
        )

        result_mask &= (
            free_game_mask
            | zero_price_mask
        )

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


def detect_requested_concepts(
    query: str,
) -> list[str]:
    """Detect explicitly requested concepts."""

    normalized_query = normalize_text(
        expand_traditional_chinese_query(
            query
        )
    )

    requested_concepts: list[str] = []

    for (
        concept_name,
        trigger_terms,
    ) in CONCEPT_TRIGGER_TERMS.items():

        has_trigger = any(
            contains_normalized_term(
                normalized_query,
                trigger_term,
            )
            for trigger_term
            in trigger_terms
        )

        if has_trigger:
            requested_concepts.append(
                concept_name
            )

    return requested_concepts


def normalize_series(
    series: pd.Series,
) -> pd.Series:
    """Normalize a text column."""

    return (
        series.fillna("")
        .astype(str)
        .map(normalize_text)
        .map(
            lambda value: (
                f" {value} "
            )
        )
    )


def term_match_mask(
    normalized_series: pd.Series,
    term: str,
) -> np.ndarray:
    """Return phrase-match Boolean values."""

    normalized_term = normalize_text(
        term
    )

    if not normalized_term:
        return np.zeros(
            len(normalized_series),
            dtype=bool,
        )

    search_term = (
        f" {normalized_term} "
    )

    return (
        normalized_series
        .str.contains(
            re.escape(
                search_term
            ),
            regex=True,
            na=False,
        )
        .to_numpy(dtype=bool)
    )


def calculate_concept_scores(
    query: str,
    candidate_dataframe: pd.DataFrame,
) -> np.ndarray:
    """
    Calculate field-aware concept scores.

    Genres and tags are strong evidence.
    Categories are slightly weaker evidence.
    Description-only matches are weak evidence.
    Missing columns are handled safely.
    """

    requested_concepts = (
        detect_requested_concepts(
            query
        )
    )

    if not requested_concepts:
        return np.zeros(
            len(candidate_dataframe),
            dtype=np.float32,
        )

    normalized_genres = (
        normalize_series(
            get_optional_text_column(
                candidate_dataframe,
                "genres",
            )
        )
    )

    normalized_tags = (
        normalize_series(
            get_optional_text_column(
                candidate_dataframe,
                "tags",
            )
        )
    )

    normalized_categories = (
        normalize_series(
            get_optional_text_column(
                candidate_dataframe,
                "categories",
            )
        )
    )

    normalized_descriptions = (
        normalize_series(
            get_optional_text_column(
                candidate_dataframe,
                "short_description",
            )
        )
    )

    all_concept_scores: list[
        np.ndarray
    ] = []

    for concept_name in requested_concepts:
        concept_terms = (
            CONCEPT_GROUPS[
                concept_name
            ]
        )

        one_concept_scores = np.zeros(
            len(candidate_dataframe),
            dtype=np.float32,
        )

        for (
            term,
            term_weight,
        ) in concept_terms.items():

            genre_match = (
                term_match_mask(
                    normalized_genres,
                    term,
                )
            )

            tag_match = (
                term_match_mask(
                    normalized_tags,
                    term,
                )
            )

            category_match = (
                term_match_mask(
                    normalized_categories,
                    term,
                )
            )

            description_match = (
                term_match_mask(
                    normalized_descriptions,
                    term,
                )
            )

            strong_metadata_match = (
                genre_match
                | tag_match
            )

            term_scores = np.zeros(
                len(candidate_dataframe),
                dtype=np.float32,
            )

            term_scores = np.maximum(
                term_scores,
                (
                    strong_metadata_match
                    .astype(np.float32)
                    * float(term_weight)
                ),
            )

            term_scores = np.maximum(
                term_scores,
                (
                    category_match
                    .astype(np.float32)
                    * float(term_weight)
                    * 0.85
                ),
            )

            term_scores = np.maximum(
                term_scores,
                (
                    description_match
                    .astype(np.float32)
                    * float(term_weight)
                    * 0.25
                ),
            )

            one_concept_scores = (
                np.maximum(
                    one_concept_scores,
                    term_scores,
                )
            )

        all_concept_scores.append(
            one_concept_scores
        )

    stacked_scores = np.vstack(
        all_concept_scores
    )

    final_scores = (
        stacked_scores
        .mean(axis=0)
        .clip(0, 1)
        .astype(np.float32)
    )

    return final_scores


def calculate_play_mode_preference_scores(
    candidate_dataframe: pd.DataFrame,
    requested_play_mode: str | None,
) -> np.ndarray:
    """Score official play-mode preferences."""

    if not requested_play_mode:
        return np.zeros(
            len(candidate_dataframe),
            dtype=np.float32,
        )

    categories = (
        get_optional_text_column(
            candidate_dataframe,
            "categories",
        )
    )

    has_single_player = (
        category_contains(
            categories,
            r"single[- ]?player",
        )
        .to_numpy(dtype=bool)
    )

    has_multiplayer = (
        category_contains(
            categories,
            (
                r"multi[- ]?player|"
                r"cross[- ]platform multiplayer"
            ),
        )
        .to_numpy(dtype=bool)
    )

    has_coop = (
        category_contains(
            categories,
            (
                r"co[- ]?op|"
                r"coop|"
                r"cooperative"
            ),
        )
        .to_numpy(dtype=bool)
    )

    has_pvp = (
        category_contains(
            categories,
            (
                r"\bpvp\b|"
                r"player versus player"
            ),
        )
        .to_numpy(dtype=bool)
    )

    scores = np.zeros(
        len(candidate_dataframe),
        dtype=np.float32,
    )

    if (
        requested_play_mode
        == "single-player"
    ):
        scores[
            has_single_player
        ] = 1.00

        scores[
            has_single_player
            & (
                has_multiplayer
                | has_coop
            )
        ] -= 0.15

        scores[
            has_single_player
            & has_pvp
        ] -= 0.25

    elif (
        requested_play_mode
        == "co-op"
    ):
        scores[
            has_coop
        ] = 1.00

        scores[
            has_coop
            & has_single_player
        ] -= 0.05

        scores[
            has_coop
            & has_pvp
        ] -= 0.25

    else:
        scores[
            has_multiplayer
        ] = 1.00

        scores[
            has_multiplayer
            & has_single_player
        ] -= 0.05

    return (
        scores
        .clip(0, 1)
        .astype(np.float32)
    )


def calculate_quality_scores(
    candidate_dataframe: pd.DataFrame,
) -> np.ndarray:
    """Create a review-quality signal."""

    review_column = (
        get_optional_text_column(
            candidate_dataframe,
            "positive_review_percentage",
        )
    )

    return (
        pd.to_numeric(
            review_column,
            errors="coerce",
        )
        .fillna(0)
        .clip(0, 100)
        .to_numpy(dtype=np.float32)
        / 100
    )


def calculate_hybrid_scores(
    semantic_scores: np.ndarray,
    concept_scores: np.ndarray,
    play_mode_scores: np.ndarray,
    quality_scores: np.ndarray,
    has_requested_concepts: bool,
    has_requested_play_mode: bool,
) -> np.ndarray:
    """Combine all ranking signals."""

    normalized_semantic_scores = (
        (
            np.asarray(
                semantic_scores,
                dtype=np.float32,
            )
            + 1
        )
        / 2
    )

    if (
        has_requested_concepts
        and has_requested_play_mode
    ):
        return (
            0.55
            * normalized_semantic_scores
            + 0.20
            * concept_scores
            + 0.15
            * play_mode_scores
            + 0.10
            * quality_scores
        )

    if has_requested_concepts:
        return (
            0.65
            * normalized_semantic_scores
            + 0.25
            * concept_scores
            + 0.10
            * quality_scores
        )

    if has_requested_play_mode:
        return (
            0.70
            * normalized_semantic_scores
            + 0.20
            * play_mode_scores
            + 0.10
            * quality_scores
        )

    return (
        0.90
        * normalized_semantic_scores
        + 0.10
        * quality_scores
    )


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


def empty_search_result() -> pd.DataFrame:
    """Return an empty search result."""

    return pd.DataFrame(
        columns=[
            "semantic_score",
            "concept_score",
            "play_mode_score",
            "quality_score",
            "hybrid_score",
        ]
    )


def search_games(
    query: str,
    top_k: int = 5,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
    int,
    bool,
    list[str],
]:
    """Filter candidates and rank results."""

    cleaned_query = query.strip()

    if not cleaned_query:
        return (
            empty_search_result(),
            {},
            0,
            True,
            [],
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    (
        embeddings,
        game_dataframe,
    ) = load_search_data()

    extracted_filters = (
        extract_filters(
            cleaned_query
        )
    )

    requested_concepts = (
        detect_requested_concepts(
            cleaned_query
        )
    )

    result_mask = apply_filters(
        game_dataframe,
        extracted_filters,
    )

    candidate_indices = (
        np.flatnonzero(
            result_mask.to_numpy(
                dtype=bool
            )
        )
    )

    candidate_count = len(
        candidate_indices
    )

    if candidate_count == 0:
        return (
            empty_search_result(),
            extracted_filters,
            0,
            False,
            requested_concepts,
        )

    clarification_required = (
        is_query_too_broad(
            query=cleaned_query,
            filters=extracted_filters,
            requested_concepts=(
                requested_concepts
            ),
            candidate_count=(
                candidate_count
            ),
        )
    )

    if clarification_required:
        return (
            empty_search_result(),
            extracted_filters,
            candidate_count,
            True,
            requested_concepts,
        )

    embedding_model = (
        load_embedding_model()
    )

    query_embedding = (
        embedding_model.encode(
            [cleaned_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    candidate_embeddings = (
        embeddings[
            candidate_indices
        ]
    )

    semantic_scores = (
        candidate_embeddings
        @ query_embedding
    ).astype(
        np.float32
    )

    candidate_dataframe = (
        game_dataframe
        .iloc[
            candidate_indices
        ]
        .copy()
        .reset_index(drop=True)
    )

    concept_scores = (
        calculate_concept_scores(
            cleaned_query,
            candidate_dataframe,
        )
    )

    requested_play_mode = (
        extracted_filters.get(
            "play_mode"
        )
    )

    play_mode_scores = (
        calculate_play_mode_preference_scores(
            candidate_dataframe,
            (
                str(
                    requested_play_mode
                )
                if requested_play_mode
                is not None
                else None
            ),
        )
    )

    quality_scores = (
        calculate_quality_scores(
            candidate_dataframe
        )
    )

    hybrid_scores = (
        calculate_hybrid_scores(
            semantic_scores=(
                semantic_scores
            ),
            concept_scores=(
                concept_scores
            ),
            play_mode_scores=(
                play_mode_scores
            ),
            quality_scores=(
                quality_scores
            ),
            has_requested_concepts=bool(
                requested_concepts
            ),
            has_requested_play_mode=(
                requested_play_mode
                is not None
            ),
        )
    )

    number_of_results = min(
        int(top_k),
        candidate_count,
    )

    ranked_positions = (
        np.argsort(
            hybrid_scores,
            kind="stable",
        )[::-1][
            :number_of_results
        ]
    )

    search_results = (
        candidate_dataframe
        .iloc[
            ranked_positions
        ]
        .copy()
        .reset_index(drop=True)
    )

    search_results[
        "semantic_score"
    ] = semantic_scores[
        ranked_positions
    ]

    search_results[
        "concept_score"
    ] = concept_scores[
        ranked_positions
    ]

    search_results[
        "play_mode_score"
    ] = play_mode_scores[
        ranked_positions
    ]

    search_results[
        "quality_score"
    ] = quality_scores[
        ranked_positions
    ]

    search_results[
        "hybrid_score"
    ] = hybrid_scores[
        ranked_positions
    ]

    return (
        search_results,
        extracted_filters,
        candidate_count,
        False,
        requested_concepts,
    )


def format_filter_value(
    filter_name: str,
    filter_value: object,
) -> str:
    """Format an extracted filter."""

    if filter_name == "maximum_price":
        return (
            "Maximum price: "
            f"${float(filter_value):.2f}"
        )

    if (
        filter_name
        == "minimum_review_percentage"
    ):
        return (
            "Minimum positive reviews: "
            f"{float(filter_value):.2f}%"
        )

    if filter_name == "is_free":
        return "Free games only"

    if filter_name == "platform":
        return (
            f"Platform: {filter_value}"
        )

    if filter_name == "play_mode":
        return (
            f"Play mode: {filter_value}"
        )

    if (
        filter_name
        == "release_year_after"
    ):
        return (
            f"Released after: "
            f"{filter_value}"
        )

    if (
        filter_name
        == "release_year_since"
    ):
        return (
            f"Released since: "
            f"{filter_value}"
        )

    if (
        filter_name
        == "release_year_before"
    ):
        return (
            f"Released before: "
            f"{filter_value}"
        )

    return (
        f"{filter_name}: "
        f"{filter_value}"
    )


def format_release_year(
    value: Any,
) -> str:
    """Format release year without decimals."""

    converted_value = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(
        converted_value
    ):
        return "Unknown"

    return str(
        int(converted_value)
    )


def format_price(
    row: pd.Series,
) -> str:
    """Format a game price."""

    price = pd.to_numeric(
        row.get(
            "price_usd"
        ),
        errors="coerce",
    )

    if (
        is_free_value(
            row.get(
                "is_free"
            )
        )
        or (
            not pd.isna(price)
            and float(price) == 0
        )
    ):
        return "Free"

    if pd.isna(price):
        return "Unknown"

    return (
        f"${float(price):.2f}"
    )


def print_results(
    query: str,
    search_results: pd.DataFrame,
    filters: dict[str, object],
    candidate_count: int,
    clarification_required: bool,
    requested_concepts: list[str],
) -> None:
    """Display results in the terminal."""

    print("=" * 72)
    print(
        "GAMEWISE HYBRID SEARCH"
    )
    print("=" * 72)

    print(
        f"Query: {query}"
    )

    print(
        "\nEXTRACTED FILTERS"
    )

    if filters:
        for (
            filter_name,
            filter_value,
        ) in filters.items():
            print(
                "- "
                + format_filter_value(
                    filter_name,
                    filter_value,
                )
            )

    else:
        print(
            "No structured filters detected."
        )

    print(
        "\nDETECTED CONCEPTS"
    )

    if requested_concepts:
        for concept_name in (
            requested_concepts
        ):
            print(
                f"- {concept_name}"
            )

    else:
        print(
            "No clear genre, mood, or "
            "gameplay concept detected."
        )

    print(
        "\nCandidates after filtering: "
        f"{candidate_count:,}"
    )

    if clarification_required:
        print(
            "\nThe query is too broad. "
            f"{candidate_count:,} games "
            "satisfy the current conditions."
        )

        print(
            "\nPlease add at least one preference:"
        )

        print(
            "- Genre: RPG, strategy, "
            "horror, racing"
        )

        print(
            "- Mood: relaxing, scary, "
            "story-rich"
        )

        print(
            "- Play mode: single-player, "
            "co-op, multiplayer"
        )

        print(
            "- Platform: Windows, Mac, Linux"
        )

        print(
            "- Quality: at least 80% "
            "positive reviews"
        )

        print(
            "\nExample:"
        )

        print(
            "a relaxing single-player "
            "farming game under $20"
        )

        return

    if search_results.empty:
        print(
            "\nNo games satisfy all "
            "requested conditions."
        )

        print(
            "Try increasing the budget, "
            "lowering the review requirement, "
            "or removing one condition."
        )

        return

    for (
        result_number,
        (_, row),
    ) in enumerate(
        search_results.iterrows(),
        start=1,
    ):
        print(
            "\n" + "-" * 72
        )

        print(
            f"{result_number}. "
            f"{row['name']}"
        )

        print(
            "Hybrid score: "
            f"{float(row['hybrid_score']):.4f}"
        )

        print(
            "Semantic score: "
            f"{float(row['semantic_score']):.4f}"
        )

        print(
            "Concept score: "
            f"{float(row['concept_score']):.4f}"
        )

        if "play_mode" in filters:
            print(
                "Play-mode score: "
                f"{float(row['play_mode_score']):.4f}"
            )

        print(
            f"Price: "
            f"{format_price(row)}"
        )

        review_percentage = (
            pd.to_numeric(
                row.get(
                    "positive_review_percentage"
                ),
                errors="coerce",
            )
        )

        total_reviews = pd.to_numeric(
            row.get(
                "total_reviews"
            ),
            errors="coerce",
        )

        if pd.isna(
            review_percentage
        ):
            review_text = "Unknown"

        elif pd.isna(
            total_reviews
        ):
            review_text = (
                f"{float(review_percentage):.2f}% "
                "positive"
            )

        else:
            review_text = (
                f"{float(review_percentage):.2f}% "
                f"positive from "
                f"{int(total_reviews):,} reviews"
            )

        print(
            f"Positive reviews: "
            f"{review_text}"
        )

        print(
            f"Genres: "
            f"{row.get('genres', '')}"
        )

        print(
            f"Categories: "
            f"{row.get('categories', '')}"
        )

        print(
            f"Tags: "
            f"{row.get('tags', '')}"
        )

        print(
            f"Platforms: "
            f"{row.get('platforms', '')}"
        )

        print(
            "Release year: "
            f"{format_release_year(row.get('release_year'))}"
        )

        print(
            "Steam URL: "
            f"{row.get('steam_store_url', '')}"
        )


def warn_about_missing_price(
    query: str,
) -> None:
    """Warn when PowerShell removes a price."""

    suspicious_pattern = (
        r"\b(?:under|below|less than|up to)"
        r"\s*(?:with|for|released|$)"
    )

    if re.search(
        suspicious_pattern,
        query,
        flags=re.IGNORECASE,
    ):
        print(
            "WARNING: A price value may have "
            "been removed by PowerShell. "
            "Put the query inside single quotes, "
            "for example: "
            "'a game under $20'.\n"
        )


def main() -> None:
    """Run hybrid search from the command line."""

    argument_parser = (
        argparse.ArgumentParser(
            description=(
                "Search Steam games with hard "
                "filters, semantic similarity, "
                "field-aware concept matching, "
                "play-mode preferences, and "
                "review quality."
            )
        )
    )

    argument_parser.add_argument(
        "query",
        type=str,
        help=(
            "Natural-language game request."
        ),
    )

    argument_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "Number of results to return."
        ),
    )

    arguments = (
        argument_parser.parse_args()
    )

    warn_about_missing_price(
        arguments.query
    )

    (
        search_results,
        extracted_filters,
        candidate_count,
        clarification_required,
        requested_concepts,
    ) = search_games(
        query=arguments.query,
        top_k=arguments.top_k,
    )

    print_results(
        query=arguments.query,
        search_results=search_results,
        filters=extracted_filters,
        candidate_count=candidate_count,
        clarification_required=(
            clarification_required
        ),
        requested_concepts=(
            requested_concepts
        ),
    )


if __name__ == "__main__":
    main()
