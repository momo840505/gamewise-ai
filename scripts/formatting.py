from __future__ import annotations

import re
from typing import Any

import pandas as pd

from scripts.query_filters import is_free_value


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

    if is_free_value(
        row.get(
            "is_free"
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
