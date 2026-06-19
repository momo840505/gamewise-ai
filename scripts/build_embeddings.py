from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


CLEANED_DATA_PATH = Path(
    "data/processed/steam_games_cleaned.csv"
)

EMBEDDINGS_OUTPUT_PATH = Path(
    "data/processed/game_embeddings.npy"
)

EMBEDDING_INDEX_OUTPUT_PATH = Path(
    "data/processed/game_embedding_index.csv"
)

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


def load_cleaned_dataset() -> pd.DataFrame:
    """Load the cleaned Steam game dataset."""

    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: "
            f"{CLEANED_DATA_PATH}\n"
            "Run scripts/clean_dataset.py first."
        )

    dataframe = pd.read_csv(CLEANED_DATA_PATH)

    return dataframe


def safe_text(value: object) -> str:
    """Return clean text for a possibly missing value."""

    if pd.isna(value):
        return "Not specified"

    cleaned_value = str(value).strip()

    if not cleaned_value:
        return "Not specified"

    return cleaned_value


def build_embedding_text(row: pd.Series) -> str:
    """Create concise semantic text for one Steam game."""

    embedding_text = (
        f"Game: {safe_text(row['name'])}. "
        f"Description: "
        f"{safe_text(row['short_description'])} "
        f"Genres: {safe_text(row['genres'])}. "
        f"Categories: "
        f"{safe_text(row['categories'])}. "
        f"Community tags: "
        f"{safe_text(row['tags'])}. "
        f"Supported platforms: "
        f"{safe_text(row['platforms'])}. "
        f"Developer: "
        f"{safe_text(row['developer'])}."
    )

    return embedding_text


def create_embeddings(
    dataframe: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create normalized embeddings for all games."""

    dataframe = dataframe.copy()

    dataframe["embedding_text"] = dataframe.apply(
        build_embedding_text,
        axis=1,
    )

    embedding_documents = dataframe[
        "embedding_text"
    ].tolist()

    print(f"Loading model: {MODEL_NAME}")

    embedding_model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        f"Creating embeddings for "
        f"{len(embedding_documents):,} games..."
    )

    embeddings = embedding_model.encode(
        embedding_documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    index_columns = [
        "app_id",
        "name",
        "price_usd",
        "is_free",
        "genres",
        "categories",
        "tags",
        "platforms",
        "positive_review_percentage",
        "total_reviews",
        "release_year",
        "metacritic_score",
        "avg_playtime_hours",
        "required_age",
        "header_image",
        "steam_store_url",
        "short_description",
        "embedding_text",
    ]

    embedding_index_dataframe = dataframe[
        index_columns
    ].copy()

    return embeddings, embedding_index_dataframe


def save_embedding_files(
    embeddings: np.ndarray,
    embedding_index_dataframe: pd.DataFrame,
) -> None:
    """Save embeddings and their corresponding metadata."""

    EMBEDDINGS_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        EMBEDDINGS_OUTPUT_PATH,
        embeddings,
    )

    embedding_index_dataframe.to_csv(
        EMBEDDING_INDEX_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def print_summary(
    embeddings: np.ndarray,
    embedding_index_dataframe: pd.DataFrame,
) -> None:
    """Print the embedding generation results."""

    print("\n" + "=" * 70)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Games embedded: "
        f"{len(embedding_index_dataframe):,}"
    )

    print(
        f"Embedding array shape: "
        f"{embeddings.shape}"
    )

    print(
        f"Embedding data type: "
        f"{embeddings.dtype}"
    )

    print(
        f"\nEmbeddings saved to:"
        f"\n{EMBEDDINGS_OUTPUT_PATH}"
    )

    print(
        f"\nEmbedding index saved to:"
        f"\n{EMBEDDING_INDEX_OUTPUT_PATH}"
    )

    print("\nSAMPLE EMBEDDING TEXT")
    print("-" * 70)

    print(
        embedding_index_dataframe.iloc[0][
            "embedding_text"
        ]
    )


def main() -> None:
    """Run the embedding generation pipeline."""

    cleaned_dataframe = load_cleaned_dataset()

    embeddings, embedding_index_dataframe = (
        create_embeddings(cleaned_dataframe)
    )

    save_embedding_files(
        embeddings,
        embedding_index_dataframe,
    )

    print_summary(
        embeddings,
        embedding_index_dataframe,
    )


if __name__ == "__main__":
    main()