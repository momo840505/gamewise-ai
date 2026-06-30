from pathlib import Path
import argparse

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


def load_search_data() -> tuple[
    np.ndarray,
    pd.DataFrame,
]:
    """Load saved embeddings and game metadata."""

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

    embeddings = np.load(EMBEDDINGS_PATH)

    game_index_dataframe = pd.read_csv(
        EMBEDDING_INDEX_PATH
    )

    if len(embeddings) != len(
        game_index_dataframe
    ):
        raise ValueError(
            "The embedding count does not match "
            "the metadata row count."
        )

    return embeddings, game_index_dataframe


def search_games(
    query: str,
    top_k: int,
) -> pd.DataFrame:
    """Find games semantically related to a query."""

    embeddings, game_index_dataframe = (
        load_search_data()
    )

    embedding_model = SentenceTransformer(
        MODEL_NAME
    )

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    similarity_scores = (
        embeddings @ query_embedding
    )

    top_result_indices = np.argsort(
        similarity_scores
    )[::-1][:top_k]

    search_results = game_index_dataframe.iloc[
        top_result_indices
    ].copy()

    search_results["similarity_score"] = (
        similarity_scores[top_result_indices]
    )

    return search_results


def print_results(
    query: str,
    search_results: pd.DataFrame,
) -> None:
    """Display semantic game-search results."""

    print("=" * 70)
    print("GAMEWISE SEMANTIC SEARCH")
    print("=" * 70)

    print(f"Query: {query}")

    for result_number, (_, row) in enumerate(
        search_results.iterrows(),
        start=1,
    ):
        print("\n" + "-" * 70)

        print(
            f"{result_number}. {row['name']}"
        )

        print(
            f"Similarity score: "
            f"{row['similarity_score']:.4f}"
        )

        print(
            f"Price: ${row['price_usd']:.2f}"
        )

        positive_review_percentage = row[
            "positive_review_percentage"
        ]

        print(
            f"Positive reviews: "
            f"{positive_review_percentage:.2f}%"
        )

        print(f"Genres: {row['genres']}")
        print(f"Tags: {row['tags']}")
        print(f"Platforms: {row['platforms']}")

        print(
            f"Steam URL: "
            f"{row['steam_store_url']}"
        )


def main() -> None:
    """Run semantic game search from the command line."""

    argument_parser = argparse.ArgumentParser(
        description=(
            "Search Steam games using "
            "natural-language queries."
        )
    )

    argument_parser.add_argument(
        "query",
        type=str,
        help="Natural-language game request.",
    )

    argument_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return.",
    )

    arguments = argument_parser.parse_args()

    search_results = search_games(
        query=arguments.query,
        top_k=arguments.top_k,
    )

    print_results(
        query=arguments.query,
        search_results=search_results,
    )


if __name__ == "__main__":
    main()
