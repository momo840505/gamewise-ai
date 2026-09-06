from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from scripts.formatting import print_results, warn_about_missing_price
from scripts.query_filters import (
    apply_filters,
    extract_filters,
    filter_invalid_game_names,
    is_query_too_broad,
)
from scripts.ranking import (
    calculate_concept_scores,
    calculate_hybrid_scores,
    calculate_play_mode_preference_scores,
    calculate_quality_scores,
    detect_requested_concepts,
)


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
