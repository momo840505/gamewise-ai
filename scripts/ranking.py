from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from scripts.text_utils import (
    category_contains,
    expand_traditional_chinese_query,
    get_optional_text_column,
)


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
    "horror": {
        "horror": 1.00,
        "survival horror": 0.95,
        "psychological horror": 0.90,
        "dark": 0.45,
        "atmospheric": 0.35,
        "gore": 0.30,
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
    "horror": [
        "horror",
        "scary",
        "creepy",
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
