from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


EMBEDDINGS_PATH = Path(
    "data/processed/game_embeddings.npy"
)

EMBEDDING_INDEX_PATH = Path(
    "data/processed/game_embedding_index.csv"
)

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
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
}


CONCEPT_RULES = {
    "psychological horror": {
        "query_terms": [
            "psychological horror",
            "psychological scary",
        ],
        "metadata_terms": {
            "psychological horror": 1.00,
            "psychological": 0.65,
            "survival horror": 0.35,
            "horror": 0.25,
            "dark": 0.10,
        },
    },
    "horror": {
        "query_terms": [
            "horror",
            "scary",
            "creepy",
            "frightening",
        ],
        "metadata_terms": {
            "horror": 1.00,
            "survival horror": 0.90,
            "psychological horror": 0.90,
            "dark": 0.30,
            "atmospheric": 0.20,
        },
    },
    "relaxing": {
        "query_terms": [
            "relaxing",
            "cozy",
            "peaceful",
            "calming",
            "wholesome",
        ],
        "metadata_terms": {
            "relaxing": 1.00,
            "cozy": 0.95,
            "wholesome": 0.85,
            "peaceful": 0.80,
            "family friendly": 0.45,
            "casual": 0.35,
            "atmospheric": 0.20,
        },
    },
    "casual": {
        "query_terms": [
            "casual",
            "easy game",
        ],
        "metadata_terms": {
            "casual": 1.00,
            "family friendly": 0.60,
            "relaxing": 0.50,
        },
    },
    "survival": {
        "query_terms": [
            "survival",
            "survive",
        ],
        "metadata_terms": {
            "survival": 1.00,
            "open world survival craft": 1.00,
            "survival horror": 0.75,
            "crafting": 0.55,
            "base-building": 0.50,
        },
    },
    "open world": {
        "query_terms": [
            "open world",
            "open-world",
            "sandbox world",
        ],
        "metadata_terms": {
            "open world": 1.00,
            "open world survival craft": 1.00,
            "sandbox": 0.75,
            "exploration": 0.45,
        },
    },
    "turn-based": {
        "query_terms": [
            "turn-based",
            "turn based",
        ],
        "metadata_terms": {
            "turn-based": 1.00,
            "turn-based strategy": 1.00,
            "turn-based tactics": 1.00,
            "turn-based combat": 1.00,
        },
    },
    "tactical": {
        "query_terms": [
            "tactical",
            "tactics",
        ],
        "metadata_terms": {
            "tactical": 1.00,
            "turn-based tactics": 1.00,
            "tactical rpg": 0.95,
            "real time tactics": 0.80,
        },
    },
    "strategy": {
        "query_terms": [
            "strategy",
            "strategic",
        ],
        "metadata_terms": {
            "strategy": 1.00,
            "turn-based strategy": 1.00,
            "grand strategy": 0.90,
            "real time strategy": 0.90,
            "4x": 0.80,
            "wargame": 0.75,
        },
    },
    "adventure": {
        "query_terms": [
            "adventure",
            "adventurous",
        ],
        "metadata_terms": {
            "adventure": 1.00,
            "action-adventure": 0.90,
            "exploration": 0.65,
            "story rich": 0.45,
        },
    },
    "puzzle": {
        "query_terms": [
            "puzzle",
            "logic game",
            "escape room",
        ],
        "metadata_terms": {
            "puzzle": 1.00,
            "logic": 0.80,
            "escape room": 0.90,
            "mystery": 0.45,
            "hidden object": 0.35,
        },
    },
    "farming": {
        "query_terms": [
            "farming",
            "farm game",
            "agriculture",
        ],
        "metadata_terms": {
            "farming": 1.00,
            "farming sim": 1.00,
            "agriculture": 0.80,
            "life sim": 0.65,
            "simulation": 0.30,
        },
    },
    "story rich": {
        "query_terms": [
            "story rich",
            "story-rich",
            "story driven",
            "story-driven",
            "narrative",
        ],
        "metadata_terms": {
            "story rich": 1.00,
            "narrative": 0.90,
            "choices matter": 0.75,
            "interactive fiction": 0.60,
        },
    },
    "role-playing": {
        "query_terms": [
            "rpg",
            "role-playing",
            "role playing",
        ],
        "metadata_terms": {
            "rpg": 1.00,
            "role-playing": 1.00,
            "action rpg": 0.95,
            "jrpg": 0.95,
            "strategy rpg": 0.90,
        },
    },
    "simulation": {
        "query_terms": [
            "simulation",
            "simulator",
            "sim game",
        ],
        "metadata_terms": {
            "simulation": 1.00,
            "simulator": 1.00,
            "life sim": 0.85,
            "management": 0.50,
        },
    },
    "shooter": {
        "query_terms": [
            "shooter",
            "shooting",
            "fps",
        ],
        "metadata_terms": {
            "shooter": 1.00,
            "fps": 1.00,
            "first-person shooter": 1.00,
            "third-person shooter": 0.90,
        },
    },
    "racing": {
        "query_terms": [
            "racing",
            "race game",
            "driving game",
        ],
        "metadata_terms": {
            "racing": 1.00,
            "driving": 0.80,
            "automobile sim": 0.75,
        },
    },
    "sports": {
        "query_terms": [
            "sports",
            "sport game",
        ],
        "metadata_terms": {
            "sports": 1.00,
            "football": 0.80,
            "basketball": 0.80,
            "soccer": 0.80,
            "golf": 0.70,
        },
    },
}


def build_valid_game_name_mask(
    game_names: pd.Series,
) -> pd.Series:
    """Return True only for usable game names."""

    normalized_game_names = (
        game_names.astype("string")
        .fillna("")
        .str.strip()
        .str.casefold()
    )

    return ~normalized_game_names.isin(
        INVALID_GAME_NAMES
    )


def filter_invalid_game_names(
    embeddings: np.ndarray,
    game_dataframe: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Remove invalid names while preserving vector-row alignment."""

    valid_name_mask = build_valid_game_name_mask(
        game_dataframe["name"]
    )

    valid_positions = np.flatnonzero(
        valid_name_mask.to_numpy()
    )

    filtered_embeddings = embeddings[
        valid_positions
    ]

    filtered_dataframe = (
        game_dataframe.iloc[
            valid_positions
        ]
        .copy()
        .reset_index(drop=True)
    )

    return (
        filtered_embeddings,
        filtered_dataframe,
    )


def load_search_data() -> tuple[
    np.ndarray,
    pd.DataFrame,
]:
    """Load embeddings and their matching game metadata."""

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
    ).reset_index(drop=True)

    if len(embeddings) != len(
        game_dataframe
    ):
        raise ValueError(
            "The number of embeddings does not match "
            "the number of game records."
        )

    return filter_invalid_game_names(
        embeddings,
        game_dataframe,
    )


def extract_number(
    query: str,
    patterns: list[str],
) -> float | None:
    """Extract the first number matching the supplied patterns."""

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
    """Extract hard constraints from a natural-language query."""

    normalized_query = query.lower()

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
                r"\s*(?:or less|maximum|max)"
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
                    r"(?:positive reviews?|rating)"
                    r"\s*(?:of|above|over|at least)?"
                    r"\s*(\d+(?:\.\d+)?)\s*%"
                ),
            ],
        )
    )

    if minimum_review_percentage is not None:
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
    """Convert Boolean-looking values into Boolean values."""

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )


def apply_filters(
    game_dataframe: pd.DataFrame,
    filters: dict[str, object],
) -> pd.Series:
    """Apply all hard constraints to the game records."""

    result_mask = pd.Series(
        True,
        index=game_dataframe.index,
    )

    prices = pd.to_numeric(
        game_dataframe["price_usd"],
        errors="coerce",
    )

    review_percentages = pd.to_numeric(
        game_dataframe[
            "positive_review_percentage"
        ],
        errors="coerce",
    )

    release_years = pd.to_numeric(
        game_dataframe["release_year"],
        errors="coerce",
    )

    if "maximum_price" in filters:
        result_mask &= (
            prices
            <= float(
                filters[
                    "maximum_price"
                ]
            )
        )

    if (
        "minimum_review_percentage"
        in filters
    ):
        result_mask &= (
            review_percentages
            >= float(
                filters[
                    "minimum_review_percentage"
                ]
            )
        )

    if filters.get("is_free") is True:
        free_game_mask = (
            convert_to_boolean(
                game_dataframe["is_free"]
            )
        )

        zero_price_mask = (
            prices.fillna(np.inf)
            == 0
        )

        result_mask &= (
            free_game_mask
            | zero_price_mask
        )

    if "platform" in filters:
        platform_name = str(
            filters["platform"]
        )

        result_mask &= (
            game_dataframe["platforms"]
            .fillna("")
            .astype(str)
            .str.contains(
                platform_name,
                case=False,
                regex=False,
            )
        )

    if "release_year_after" in filters:
        result_mask &= (
            release_years
            > int(
                filters[
                    "release_year_after"
                ]
            )
        )

    if "release_year_since" in filters:
        result_mask &= (
            release_years
            >= int(
                filters[
                    "release_year_since"
                ]
            )
        )

    if "release_year_before" in filters:
        result_mask &= (
            release_years
            < int(
                filters[
                    "release_year_before"
                ]
            )
        )

    if "play_mode" in filters:
        category_text = (
            game_dataframe["categories"]
            .fillna("")
            .astype(str)
        )

        requested_play_mode = filters[
            "play_mode"
        ]

        if (
            requested_play_mode
            == "single-player"
        ):
            play_mode_mask = (
                category_text.str.contains(
                    r"single[- ]?player",
                    case=False,
                    regex=True,
                )
            )

        elif requested_play_mode == "co-op":
            play_mode_mask = (
                category_text.str.contains(
                    r"co[- ]?op|coop|cooperative",
                    case=False,
                    regex=True,
                )
            )

        else:
            play_mode_mask = (
                category_text.str.contains(
                    r"multi[- ]?player",
                    case=False,
                    regex=True,
                )
            )

        result_mask &= play_mode_mask

    return result_mask


def detect_requested_concepts(
    query: str,
) -> list[str]:
    """Identify concepts explicitly requested by the user."""

    normalized_query = query.lower()

    requested_concepts = []

    psychological_horror_requested = any(
        query_term in normalized_query
        for query_term in CONCEPT_RULES[
            "psychological horror"
        ]["query_terms"]
    )

    for (
        concept_name,
        concept_rule,
    ) in CONCEPT_RULES.items():

        if (
            concept_name == "horror"
            and psychological_horror_requested
        ):
            continue

        concept_requested = any(
            query_term in normalized_query
            for query_term in concept_rule[
                "query_terms"
            ]
        )

        if concept_requested:
            requested_concepts.append(
                concept_name
            )

    return requested_concepts


def build_term_pattern(
    term: str,
) -> str:
    """Create a pattern that avoids partial-word matches."""

    escaped_term = re.escape(
        term
    )

    return (
        rf"(?<!\w){escaped_term}(?!\w)"
    )


def calculate_single_concept_score(
    strong_metadata: pd.Series,
    description_text: pd.Series,
    concept_name: str,
) -> np.ndarray:
    """
    Calculate field-aware matches for one concept.

    Genres and tags are strong evidence.
    Descriptions are weaker evidence because they can contain
    incidental words such as 'fight for survival'.
    """

    metadata_terms = CONCEPT_RULES[
        concept_name
    ]["metadata_terms"]

    strong_scores = np.zeros(
        len(strong_metadata),
        dtype=np.float32,
    )

    description_scores = np.zeros(
        len(description_text),
        dtype=np.float32,
    )

    for (
        metadata_term,
        metadata_weight,
    ) in metadata_terms.items():

        term_pattern = build_term_pattern(
            metadata_term
        )

        strong_matches = (
            strong_metadata.str.contains(
                term_pattern,
                case=False,
                regex=True,
                na=False,
            )
            .to_numpy()
        )

        strong_weighted_scores = (
            strong_matches.astype(
                np.float32
            )
            * float(
                metadata_weight
            )
        )

        strong_scores = np.maximum(
            strong_scores,
            strong_weighted_scores,
        )

        term_word_count = len(
            metadata_term.split()
        )

        if term_word_count >= 2:
            description_multiplier = 0.55
        else:
            description_multiplier = 0.25

        description_matches = (
            description_text.str.contains(
                term_pattern,
                case=False,
                regex=True,
                na=False,
            )
            .to_numpy()
        )

        description_weighted_scores = (
            description_matches.astype(
                np.float32
            )
            * float(
                metadata_weight
            )
            * description_multiplier
        )

        description_scores = np.maximum(
            description_scores,
            description_weighted_scores,
        )

    return np.maximum(
        strong_scores,
        description_scores,
    )


def calculate_concept_scores(
    query: str,
    candidate_dataframe: pd.DataFrame,
) -> np.ndarray:
    """Calculate field-aware concept relevance scores."""

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

    strong_metadata = (
        candidate_dataframe["genres"]
        .fillna("")
        .astype(str)
        + " "
        + candidate_dataframe["tags"]
        .fillna("")
        .astype(str)
    ).str.lower()

    description_text = (
        candidate_dataframe[
            "short_description"
        ]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    total_concept_scores = np.zeros(
        len(candidate_dataframe),
        dtype=np.float32,
    )

    for concept_name in requested_concepts:
        total_concept_scores += (
            calculate_single_concept_score(
                strong_metadata,
                description_text,
                concept_name,
            )
        )

    total_concept_scores /= len(
        requested_concepts
    )

    return total_concept_scores


def calculate_quality_scores(
    candidate_dataframe: pd.DataFrame,
) -> np.ndarray:
    """Create a small ranking signal from reviews."""

    review_percentage = (
        pd.to_numeric(
            candidate_dataframe[
                "positive_review_percentage"
            ],
            errors="coerce",
        )
        .fillna(0)
        .clip(0, 100)
        .to_numpy(
            dtype=np.float32
        )
        / 100
    )

    total_reviews = (
        pd.to_numeric(
            candidate_dataframe[
                "total_reviews"
            ],
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
        .to_numpy(
            dtype=np.float32
        )
    )

    review_confidence = np.clip(
        np.log10(
            total_reviews + 1
        ) / 5,
        0,
        1,
    )

    return (
        review_percentage
        * (
            0.75
            + 0.25
            * review_confidence
        )
    )


def calculate_play_mode_preference_scores(
    candidate_dataframe: pd.DataFrame,
    requested_play_mode: str | None,
) -> np.ndarray:
    """Rank games by focus on the requested play mode."""

    if requested_play_mode is None:
        return np.zeros(
            len(candidate_dataframe),
            dtype=np.float32,
        )

    category_text = (
        candidate_dataframe["categories"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    has_single_player = (
        category_text.str.contains(
            r"single[- ]?player",
            regex=True,
        )
        .to_numpy()
    )

    has_multiplayer = (
        category_text.str.contains(
            r"multi[- ]?player",
            regex=True,
        )
        .to_numpy()
    )

    has_coop = (
        category_text.str.contains(
            r"co[- ]?op|coop|cooperative",
            regex=True,
        )
        .to_numpy()
    )

    has_pvp = (
        category_text.str.contains(
            r"\bpvp\b",
            regex=True,
        )
        .to_numpy()
    )

    if requested_play_mode == "single-player":
        preference_scores = np.ones(
            len(candidate_dataframe),
            dtype=np.float32,
        )

        preference_scores -= (
            has_multiplayer.astype(
                np.float32
            )
            * 0.20
        )

        preference_scores -= (
            has_coop.astype(
                np.float32
            )
            * 0.20
        )

        preference_scores -= (
            has_pvp.astype(
                np.float32
            )
            * 0.25
        )

        pure_single_player = (
            has_single_player
            & ~has_multiplayer
            & ~has_coop
            & ~has_pvp
        )

        preference_scores[
            pure_single_player
        ] = 1.00

    elif requested_play_mode == "co-op":
        preference_scores = np.ones(
            len(candidate_dataframe),
            dtype=np.float32,
        )

        preference_scores -= (
            has_pvp.astype(
                np.float32
            )
            * 0.25
        )

        preference_scores -= (
            has_single_player.astype(
                np.float32
            )
            * 0.05
        )

        pure_coop_without_pvp = (
            has_coop
            & ~has_pvp
        )

        preference_scores[
            pure_coop_without_pvp
        ] = np.maximum(
            preference_scores[
                pure_coop_without_pvp
            ],
            0.95,
        )

    else:
        preference_scores = np.ones(
            len(candidate_dataframe),
            dtype=np.float32,
        )

    return np.clip(
        preference_scores,
        0,
        1,
    )


def calculate_hybrid_scores(
    semantic_scores: np.ndarray,
    concept_scores: np.ndarray,
    play_mode_scores: np.ndarray,
    quality_scores: np.ndarray,
    has_requested_concepts: bool,
    has_requested_play_mode: bool,
) -> np.ndarray:
    """Combine ranking signals using context-sensitive weights."""

    normalized_semantic_scores = (
        semantic_scores + 1
    ) / 2

    if (
        has_requested_concepts
        and has_requested_play_mode
    ):
        return (
            0.55
            * normalized_semantic_scores
            + 0.20
            * concept_scores
            + 0.20
            * play_mode_scores
            + 0.05
            * quality_scores
        )

    if has_requested_play_mode:
        return (
            0.65
            * normalized_semantic_scores
            + 0.30
            * play_mode_scores
            + 0.05
            * quality_scores
        )

    if has_requested_concepts:
        return (
            0.65
            * normalized_semantic_scores
            + 0.30
            * concept_scores
            + 0.05
            * quality_scores
        )

    return (
        0.95
        * normalized_semantic_scores
        + 0.05
        * quality_scores
    )


def query_requires_clarification(
    query: str,
    filters: dict[str, object],
) -> bool:
    """Check whether a query is too broad for useful ranking."""

    requested_concepts = (
        detect_requested_concepts(
            query
        )
    )

    if requested_concepts:
        return False

    meaningful_filter_names = {
        "minimum_review_percentage",
        "is_free",
        "platform",
        "play_mode",
        "release_year_after",
        "release_year_since",
        "release_year_before",
    }

    has_meaningful_filter = any(
        filter_name
        in meaningful_filter_names
        for filter_name in filters
    )

    if has_meaningful_filter:
        return False

    return True


def search_games(
    query: str,
    top_k: int,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
    int,
    bool,
    list[str],
]:
    """Filter candidates and rank them using hybrid retrieval."""

    (
        embeddings,
        game_dataframe,
    ) = load_search_data()

    extracted_filters = (
        extract_filters(
            query
        )
    )

    requested_concepts = (
        detect_requested_concepts(
            query
        )
    )

    result_mask = apply_filters(
        game_dataframe,
        extracted_filters,
    )

    candidate_indices = np.flatnonzero(
        result_mask.to_numpy()
    )

    candidate_count = len(
        candidate_indices
    )

    clarification_required = (
        query_requires_clarification(
            query,
            extracted_filters,
        )
    )

    if (
        candidate_count == 0
        or clarification_required
    ):
        return (
            pd.DataFrame(),
            extracted_filters,
            candidate_count,
            clarification_required,
            requested_concepts,
        )

    embedding_model = SentenceTransformer(
        MODEL_NAME
    )

    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
    )

    candidate_embeddings = embeddings[
        candidate_indices
    ]

    semantic_scores = (
        candidate_embeddings
        @ query_embedding
    )

    candidate_dataframe = (
        game_dataframe
        .iloc[candidate_indices]
        .copy()
        .reset_index(drop=True)
    )

    concept_scores = (
        calculate_concept_scores(
            query,
            candidate_dataframe,
        )
    )

    quality_scores = (
        calculate_quality_scores(
            candidate_dataframe
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
            requested_play_mode,
        )
    )

    hybrid_scores = calculate_hybrid_scores(
        semantic_scores=semantic_scores,
        concept_scores=concept_scores,
        play_mode_scores=play_mode_scores,
        quality_scores=quality_scores,
        has_requested_concepts=bool(
            requested_concepts
        ),
        has_requested_play_mode=(
            requested_play_mode
            is not None
        ),
    )

    number_of_results = min(
        top_k,
        candidate_count,
    )

    ranked_positions = np.argsort(
        hybrid_scores
    )[::-1][:number_of_results]

    search_results = (
        candidate_dataframe
        .iloc[ranked_positions]
        .copy()
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
        "hybrid_score"
    ] = hybrid_scores[
        ranked_positions
    ]

    return (
        search_results,
        extracted_filters,
        candidate_count,
        clarification_required,
        requested_concepts,
    )


def format_filter_value(
    filter_name: str,
    filter_value: object,
) -> str:
    """Format one extracted filter for display."""

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
            "Released after: "
            f"{filter_value}"
        )

    if (
        filter_name
        == "release_year_since"
    ):
        return (
            "Released since: "
            f"{filter_value}"
        )

    if (
        filter_name
        == "release_year_before"
    ):
        return (
            "Released before: "
            f"{filter_value}"
        )

    return (
        f"{filter_name}: "
        f"{filter_value}"
    )


def is_free_value(
    value: object,
) -> bool:
    """Safely interpret a Boolean-looking free-game value."""

    return (
        str(value)
        .strip()
        .lower()
        == "true"
    )


def format_release_year(
    value: object,
) -> str:
    """Format a release year without a decimal point."""

    numeric_value = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(
        numeric_value
    ):
        return "Unknown"

    return str(
        int(numeric_value)
    )


def print_clarification_message(
    candidate_count: int,
) -> None:
    """Ask the user for a more specific preference."""

    print(
        f"\nThe query is too broad. "
        f"{candidate_count:,} games satisfy "
        "the current conditions."
    )

    print(
        "\nPlease add at least one preference:"
    )

    print(
        "- Genre: RPG, strategy, horror, racing"
    )

    print(
        "- Mood: relaxing, scary, story-rich"
    )

    print(
        "- Play mode: single-player, co-op, multiplayer"
    )

    print(
        "- Platform: Windows, Mac, Linux"
    )

    print(
        "- Quality: at least 80% positive reviews"
    )

    print(
        "\nExample:"
    )

    print(
        "a relaxing single-player farming game under $20"
    )


def print_results(
    query: str,
    search_results: pd.DataFrame,
    filters: dict[str, object],
    candidate_count: int,
    clarification_required: bool,
    requested_concepts: list[str],
) -> None:
    """Display hybrid retrieval results."""

    print("=" * 72)
    print("GAMEWISE HYBRID SEARCH")
    print("=" * 72)

    print(
        f"Query: {query}"
    )

    print(
        "\nEXTRACTED FILTERS"
    )

    if not filters:
        print(
            "No structured filters detected."
        )

    else:
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
            "No clear genre, mood, or gameplay concept detected."
        )

    print(
        "\nCandidates after filtering: "
        f"{candidate_count:,}"
    )

    if clarification_required:
        print_clarification_message(
            candidate_count
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

    play_mode_was_requested = (
        "play_mode" in filters
    )

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
            f"{row['hybrid_score']:.4f}"
        )

        print(
            "Semantic score: "
            f"{row['semantic_score']:.4f}"
        )

        print(
            "Concept score: "
            f"{row['concept_score']:.4f}"
        )

        if play_mode_was_requested:
            print(
                "Play-mode score: "
                f"{row['play_mode_score']:.4f}"
            )

        price = float(
            row["price_usd"]
        )

        if (
            is_free_value(
                row["is_free"]
            )
            or price == 0
        ):
            price_text = "Free"

        else:
            price_text = (
                f"${price:.2f}"
            )

        print(
            f"Price: {price_text}"
        )

        print(
            "Positive reviews: "
            f"{float(row[
                'positive_review_percentage'
            ]):.2f}% "
            f"from "
            f"{int(row['total_reviews']):,} "
            "reviews"
        )

        print(
            f"Genres: {row['genres']}"
        )

        print(
            "Categories: "
            f"{row['categories']}"
        )

        print(
            f"Tags: {row['tags']}"
        )

        print(
            "Platforms: "
            f"{row['platforms']}"
        )

        print(
            "Release year: "
            + format_release_year(
                row["release_year"]
            )
        )

        print(
            "Steam URL: "
            f"{row['steam_store_url']}"
        )


def warn_about_missing_price(
    query: str,
) -> None:
    """Warn when PowerShell may have removed a dollar amount."""

    missing_price_pattern = (
        r"\b(?:under|below|less than|up to)"
        r"\s*(?:with|for|released|$)"
    )

    if re.search(
        missing_price_pattern,
        query,
        flags=re.IGNORECASE,
    ):
        print(
            "WARNING: The price may have been "
            "removed by PowerShell."
        )

        print(
            "Use single quotes around the query:"
        )

        print(
            "python .\\scripts\\hybrid_search.py "
            "'a game under $20'\n"
        )


def main() -> None:
    """Run hybrid game search from the command line."""

    argument_parser = argparse.ArgumentParser(
        description=(
            "Search Steam games using "
            "hard metadata filters, "
            "semantic similarity, "
            "field-aware concept matching, "
            "play-mode preference, "
            "and review quality."
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
        clarification_required=clarification_required,
        requested_concepts=requested_concepts,
    )


if __name__ == "__main__":
    main()