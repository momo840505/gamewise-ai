from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


RAW_DATA_PATH = Path("data/raw/steam_top_games_2026.csv")
PROCESSED_DATA_PATH = Path("data/processed/steam_games_cleaned.csv")
QUALITY_REPORT_PATH = Path("data/processed/data_quality_report.json")


TEXT_COLUMNS_WITH_DEFAULTS = {
    "name": "Unknown title",
    "developer": "Unknown developer",
    "publisher": "Unknown publisher",
    "genres": "Not specified",
    "categories": "Not specified",
    "tags": "Not specified",
    "estimated_owners": "Not available",
    "short_description": "No description available.",
    "header_image": "",
}


def load_dataset() -> pd.DataFrame:
    """Load the original Steam dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset was not found: {RAW_DATA_PATH}"
        )

    dataframe = pd.read_csv(RAW_DATA_PATH)

    return dataframe


def clean_text_series(
    series: pd.Series,
    default_value: str,
) -> pd.Series:
    """Remove extra spaces and replace missing text values."""

    cleaned_series = series.astype("string").str.strip()

    cleaned_series = cleaned_series.replace("", pd.NA)

    cleaned_series = cleaned_series.fillna(default_value)

    return cleaned_series


def clean_comma_separated_value(value: object) -> str:
    """Clean comma-separated genres, categories, or tags."""

    if pd.isna(value) or not str(value).strip():
        return "Not specified"

    cleaned_items = []
    seen_items = set()

    for item in str(value).split(","):
        cleaned_item = item.strip()
        normalized_item = cleaned_item.casefold()

        if (
            cleaned_item
            and normalized_item not in seen_items
        ):
            cleaned_items.append(cleaned_item)
            seen_items.add(normalized_item)

    if not cleaned_items:
        return "Not specified"

    return ", ".join(cleaned_items)


def parse_owner_range(
    owner_range: object,
) -> tuple[object, object]:
    """Extract the minimum and maximum estimated owner values."""

    if pd.isna(owner_range):
        return pd.NA, pd.NA

    match = re.search(
        r"([\d,]+)\s*\.\.\s*([\d,]+)",
        str(owner_range),
    )

    if not match:
        return pd.NA, pd.NA

    minimum_owners = int(
        match.group(1).replace(",", "")
    )

    maximum_owners = int(
        match.group(2).replace(",", "")
    )

    return minimum_owners, maximum_owners


def build_platform_text(row: pd.Series) -> str:
    """Combine supported operating systems into one text field."""

    supported_platforms = []

    if bool(row["platforms_win"]):
        supported_platforms.append("Windows")

    if bool(row["platforms_mac"]):
        supported_platforms.append("Mac")

    if bool(row["platforms_linux"]):
        supported_platforms.append("Linux")

    if not supported_platforms:
        return "Not specified"

    return ", ".join(supported_platforms)


def build_searchable_text(row: pd.Series) -> str:
    """Create the text that will later be converted into an embedding."""

    if bool(row["is_free"]):
        price_text = "Free to play"
    else:
        price_text = f"${row['price_usd']:.2f} USD"

    if pd.isna(row["metacritic_score"]):
        metacritic_text = "Not available"
    else:
        metacritic_text = str(
            int(row["metacritic_score"])
        )

    searchable_text = (
        f"Game title: {row['name']}. "
        f"Description: {row['short_description']} "
        f"Genres: {row['genres']}. "
        f"Categories: {row['categories']}. "
        f"Tags: {row['tags']}. "
        f"Developer: {row['developer']}. "
        f"Publisher: {row['publisher']}. "
        f"Release date: {row['release_date']}. "
        f"Price: {price_text}. "
        f"Platforms: {row['platforms']}. "
        f"Positive review percentage: "
        f"{row['positive_review_percentage']:.2f}% "
        f"from {row['total_reviews']:,} reviews. "
        f"Metacritic score: {metacritic_text}. "
        f"Estimated owners: "
        f"{row['estimated_owners']}. "
        f"Average all-time playtime: "
        f"{row['avg_playtime_hours']:.1f} hours. "
        f"Required age: {int(row['required_age'])}. "
        f"Achievements: "
        f"{int(row['achievements'])}."
    )

    return searchable_text


def clean_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Clean the Steam dataset and create derived fields."""

    input_row_count = len(dataframe)

    missing_values_before = (
        dataframe.isna()
        .sum()
        .astype(int)
        .to_dict()
    )

    duplicated_application_ids = int(
        dataframe["app_id"].duplicated().sum()
    )

    dataframe = dataframe.drop_duplicates(
        subset=["app_id"],
        keep="first",
    ).copy()

    for (
        column_name,
        default_value,
    ) in TEXT_COLUMNS_WITH_DEFAULTS.items():
        dataframe[column_name] = clean_text_series(
            dataframe[column_name],
            default_value,
        )

    for column_name in [
        "genres",
        "categories",
        "tags",
    ]:
        dataframe[column_name] = dataframe[
            column_name
        ].apply(clean_comma_separated_value)

    parsed_release_dates = pd.to_datetime(
        dataframe["release_date"],
        format="mixed",
        errors="coerce",
    )

    dataframe["release_date_original"] = dataframe[
        "release_date"
    ]

    dataframe["release_date"] = (
        parsed_release_dates
        .dt.strftime("%Y-%m-%d")
        .fillna("Unknown")
    )

    dataframe["release_year"] = (
        parsed_release_dates
        .dt.year
        .astype("Int64")
    )

    dataframe["metacritic_score"] = (
        pd.to_numeric(
            dataframe["metacritic_score"],
            errors="coerce",
        )
        .round()
        .astype("Int64")
    )

    numeric_columns = [
        "price_usd",
        "discount_pct",
        "recommendations",
        "positive_reviews",
        "negative_reviews",
        "avg_playtime_forever",
        "avg_playtime_2weeks",
        "median_playtime",
        "peak_ccu",
        "required_age",
        "dlc_count",
        "achievements",
    ]

    for column_name in numeric_columns:
        dataframe[column_name] = pd.to_numeric(
            dataframe[column_name],
            errors="coerce",
        ).fillna(0)

    dataframe["price_usd"] = (
        dataframe["price_usd"]
        .clip(lower=0)
        .round(2)
    )

    integer_columns = [
        column_name
        for column_name in numeric_columns
        if column_name != "price_usd"
    ]

    for column_name in integer_columns:
        dataframe[column_name] = (
            dataframe[column_name]
            .clip(lower=0)
            .round()
            .astype("int64")
        )

    dataframe["total_reviews"] = (
        dataframe["positive_reviews"]
        + dataframe["negative_reviews"]
    )

    dataframe["positive_review_percentage"] = (
        np.where(
            dataframe["total_reviews"] > 0,
            (
                dataframe["positive_reviews"]
                / dataframe["total_reviews"]
                * 100
            ),
            0,
        )
    ).round(2)

    dataframe["has_reviews"] = (
        dataframe["total_reviews"] > 0
    )

    dataframe["avg_playtime_hours"] = (
        dataframe["avg_playtime_forever"] / 60
    ).round(1)

    owner_ranges = dataframe[
        "estimated_owners"
    ].apply(parse_owner_range)

    dataframe["estimated_owners_min"] = pd.array(
        [
            owner_range[0]
            for owner_range in owner_ranges
        ],
        dtype="Int64",
    )

    dataframe["estimated_owners_max"] = pd.array(
        [
            owner_range[1]
            for owner_range in owner_ranges
        ],
        dtype="Int64",
    )

    dataframe["platforms"] = dataframe.apply(
        build_platform_text,
        axis=1,
    )

    dataframe["steam_store_url"] = (
        "https://store.steampowered.com/app/"
        + dataframe["app_id"].astype(str)
    )

    dataframe["searchable_text"] = dataframe.apply(
        build_searchable_text,
        axis=1,
    )

    quality_report = {
        "input_rows": input_row_count,
        "output_rows": len(dataframe),
        "duplicated_application_ids_removed":
            duplicated_application_ids,
        "unknown_release_dates": int(
            dataframe["release_date"]
            .eq("Unknown")
            .sum()
        ),
        "games_without_reviews": int(
            (~dataframe["has_reviews"]).sum()
        ),
        "missing_values_before_cleaning":
            missing_values_before,
        "generated_columns": [
            "release_year",
            "total_reviews",
            "positive_review_percentage",
            "has_reviews",
            "avg_playtime_hours",
            "estimated_owners_min",
            "estimated_owners_max",
            "platforms",
            "steam_store_url",
            "searchable_text",
        ],
    }

    return dataframe, quality_report


def save_results(
    dataframe: pd.DataFrame,
    quality_report: dict,
) -> None:
    """Save the cleaned CSV and data-quality report."""

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    with QUALITY_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            quality_report,
            report_file,
            indent=2,
            ensure_ascii=False,
        )


def print_summary(
    dataframe: pd.DataFrame,
    quality_report: dict,
) -> None:
    """Print the main cleaning results."""

    print("=" * 70)
    print("STEAM DATA CLEANING COMPLETE")
    print("=" * 70)

    print(
        f"Input rows: "
        f"{quality_report['input_rows']:,}"
    )

    print(
        f"Output rows: "
        f"{quality_report['output_rows']:,}"
    )

    duplicate_application_ids_removed = quality_report[
        "duplicated_application_ids_removed"
    ]
    unknown_release_dates = quality_report[
        "unknown_release_dates"
    ]
    games_without_reviews = quality_report[
        "games_without_reviews"
    ]

    print(
        "Duplicate application IDs removed: "
        f"{duplicate_application_ids_removed:,}"
    )

    print(
        "Unknown release dates: "
        f"{unknown_release_dates:,}"
    )

    print(
        "Games without reviews: "
        f"{games_without_reviews:,}"
    )

    print(
        f"Output columns: "
        f"{dataframe.shape[1]:,}"
    )

    print(
        "\nCleaned CSV:"
        f"\n{PROCESSED_DATA_PATH}"
    )

    print(
        "\nQuality report:"
        f"\n{QUALITY_REPORT_PATH}"
    )

    print("\nSAMPLE SEARCHABLE TEXT")
    print("-" * 70)
    print(dataframe.iloc[0]["searchable_text"])


def main() -> None:
    """Run the complete data-cleaning pipeline."""

    raw_dataframe = load_dataset()

    cleaned_dataframe, quality_report = (
        clean_dataset(raw_dataframe)
    )

    save_results(
        cleaned_dataframe,
        quality_report,
    )

    print_summary(
        cleaned_dataframe,
        quality_report,
    )


if __name__ == "__main__":
    main()
