from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/raw/steam_top_games_2026.csv")


def load_dataset() -> pd.DataFrame:
    """Load the Steam game dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Place steam_top_games_2026.csv inside data/raw."
        )

    return pd.read_csv(DATASET_PATH)


def inspect_dataset(dataframe: pd.DataFrame) -> None:
    """Display basic dataset information and data-quality results."""

    print("=" * 70)
    print("STEAM DATASET OVERVIEW")
    print("=" * 70)

    print(f"Rows: {dataframe.shape[0]:,}")
    print(f"Columns: {dataframe.shape[1]:,}")

    print("\nCOLUMN NAMES")
    for column_number, column_name in enumerate(
        dataframe.columns,
        start=1,
    ):
        print(f"{column_number}. {column_name}")

    print("\nFIRST FIVE GAMES")
    selected_columns = [
        "app_id",
        "name",
        "price_usd",
        "genres",
        "tags",
    ]
    print(dataframe[selected_columns].head().to_string(index=False))

    print("\nMISSING VALUES")
    missing_values = (
        dataframe.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        print("No missing values were found.")
    else:
        print(missing_values.to_string())

    print("\nDUPLICATED COMPLETE ROWS")
    print(dataframe.duplicated().sum())

    print("\nDUPLICATED APPLICATION IDS")
    print(dataframe["app_id"].duplicated().sum())

    print("\nPRICE SUMMARY")
    print(dataframe["price_usd"].describe().to_string())

    print("\nFREE AND PAID GAMES")
    print(dataframe["is_free"].value_counts().to_string())

    print("\nPLATFORM SUPPORT")
    print(f"Windows: {dataframe['platforms_win'].sum():,}")
    print(f"Mac: {dataframe['platforms_mac'].sum():,}")
    print(f"Linux: {dataframe['platforms_linux'].sum():,}")


def main() -> None:
    """Run the dataset inspection."""

    steam_games_dataframe = load_dataset()
    inspect_dataset(steam_games_dataframe)


if __name__ == "__main__":
    main()