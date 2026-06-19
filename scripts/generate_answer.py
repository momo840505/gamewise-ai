from __future__ import annotations

import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_MODEL = "gpt-5.5"


def safe_text(
    value: Any,
    default: str = "Not available",
) -> str:
    """Convert missing values into readable text."""

    if value is None:
        return default

    if isinstance(value, float) and pd.isna(value):
        return default

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
    }:
        return default

    return text


def format_price(
    row: pd.Series,
) -> str:
    """Format one game's price."""

    price = pd.to_numeric(
        row.get("price_usd"),
        errors="coerce",
    )

    is_free = (
        str(row.get("is_free"))
        .strip()
        .lower()
        == "true"
    )

    if is_free or (
        not pd.isna(price)
        and float(price) == 0
    ):
        return "Free"

    if pd.isna(price):
        return "Unknown"

    return f"${float(price):.2f}"


def format_reviews(
    row: pd.Series,
) -> str:
    """Format review percentage and review count."""

    review_percentage = pd.to_numeric(
        row.get(
            "positive_review_percentage"
        ),
        errors="coerce",
    )

    total_reviews = pd.to_numeric(
        row.get("total_reviews"),
        errors="coerce",
    )

    if pd.isna(review_percentage):
        return "Unknown"

    if pd.isna(total_reviews):
        return (
            f"{float(review_percentage):.2f}% positive"
        )

    return (
        f"{float(review_percentage):.2f}% positive "
        f"from {int(total_reviews):,} reviews"
    )


def format_release_year(
    value: Any,
) -> str:
    """Format a release year without a decimal point."""

    release_year = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(release_year):
        return "Unknown"

    return str(
        int(release_year)
    )


def build_filter_summary(
    filters: dict[str, object],
    requested_concepts: list[str],
) -> str:
    """Convert interpreted filters into prompt context."""

    summary_parts: list[str] = []

    for (
        filter_name,
        filter_value,
    ) in filters.items():
        summary_parts.append(
            f"{filter_name}: {filter_value}"
        )

    if requested_concepts:
        summary_parts.append(
            "requested concepts: "
            + ", ".join(
                requested_concepts
            )
        )

    if not summary_parts:
        return "No explicit filters detected."

    return "\n".join(
        f"- {summary_part}"
        for summary_part in summary_parts
    )


def build_retrieval_context(
    search_results: pd.DataFrame,
) -> str:
    """Convert retrieved game rows into grounded model context."""

    context_blocks: list[str] = []

    for result_number, (_, row) in enumerate(
        search_results.iterrows(),
        start=1,
    ):
        context_block = "\n".join(
            [
                f"RESULT {result_number}",
                (
                    "Name: "
                    f"{safe_text(row.get('name'))}"
                ),
                (
                    "Price: "
                    f"{format_price(row)}"
                ),
                (
                    "Reviews: "
                    f"{format_reviews(row)}"
                ),
                (
                    "Release year: "
                    f"{format_release_year(row.get('release_year'))}"
                ),
                (
                    "Genres: "
                    f"{safe_text(row.get('genres'))}"
                ),
                (
                    "Categories: "
                    f"{safe_text(row.get('categories'))}"
                ),
                (
                    "Tags: "
                    f"{safe_text(row.get('tags'))}"
                ),
                (
                    "Platforms: "
                    f"{safe_text(row.get('platforms'))}"
                ),
                (
                    "Description: "
                    f"{safe_text(row.get('short_description'))}"
                ),
                (
                    "Steam URL: "
                    f"{safe_text(row.get('steam_store_url'))}"
                ),
            ]
        )

        context_blocks.append(
            context_block
        )

    return "\n\n".join(
        context_blocks
    )


def build_local_fallback_answer(
    search_results: pd.DataFrame,
    requested_concepts: list[str],
) -> str:
    """Create a grounded answer without an external model."""

    if search_results.empty:
        return (
            "No games satisfy all requested conditions."
        )

    answer_lines = [
        "### Recommended games",
        "",
        (
            "These recommendations are based directly "
            "on the retrieved Steam dataset records."
        ),
        "",
    ]

    for result_number, (_, row) in enumerate(
        search_results.iterrows(),
        start=1,
    ):
        game_name = safe_text(
            row.get("name")
        )

        steam_url = safe_text(
            row.get("steam_store_url"),
            default="",
        )

        price_text = format_price(
            row
        )

        review_text = format_reviews(
            row
        )

        if requested_concepts:
            reason_text = (
                "Its retrieved metadata matches "
                + ", ".join(
                    requested_concepts
                )
                + "."
            )
        else:
            reason_text = (
                "It has a strong semantic match "
                "to the request."
            )

        if steam_url:
            game_link = (
                f"[{game_name}]({steam_url})"
            )
        else:
            game_link = game_name

        answer_lines.extend(
            [
                (
                    f"{result_number}. "
                    f"**{game_link}**"
                ),
                (
                    f"   {price_text}; "
                    f"{review_text}."
                ),
                (
                    f"   {reason_text}"
                ),
                "",
            ]
        )

    answer_lines.append(
        (
            "Recommendations are limited to the "
            "information available in the local dataset."
        )
    )

    return "\n".join(
        answer_lines
    )


def generate_grounded_answer(
    query: str,
    search_results: pd.DataFrame,
    filters: dict[str, object],
    requested_concepts: list[str],
) -> tuple[str, str]:
    """
    Generate an answer grounded only in retrieved records.

    Returns:
        answer_text
        generation_mode
    """

    if search_results.empty:
        return (
            "No games satisfy all requested conditions.",
            "no_results",
        )

    load_dotenv()

    api_key = os.getenv(
        "OPENAI_API_KEY",
        "",
    ).strip()

    if not api_key:
        fallback_answer = (
            build_local_fallback_answer(
                search_results,
                requested_concepts,
            )
        )

        return (
            fallback_answer,
            "local_fallback",
        )

    model_name = os.getenv(
        "OPENAI_MODEL",
        DEFAULT_MODEL,
    ).strip()

    filter_summary = (
        build_filter_summary(
            filters,
            requested_concepts,
        )
    )

    retrieval_context = (
        build_retrieval_context(
            search_results
        )
    )

    instructions = """
You are the grounded recommendation writer for GameWise AI.

Use only the retrieved Steam game records supplied by the user.
Do not use outside knowledge.
Do not invent prices, review scores, release years, platforms,
features, descriptions, categories, tags, or Steam URLs.
Do not claim that a game satisfies a condition unless the supplied
record supports that statement.
Keep every Steam URL exactly unchanged.
If information is unavailable, omit it instead of guessing.
Recommend the games in their current retrieved order.
Write concise and readable Markdown.
Use one short introduction followed by a numbered list.
For every game, explain why it matches the user's request.
Do not mention vector, semantic, concept, hybrid, or ranking scores.
End by saying that the recommendations are based on the available
Steam dataset.
""".strip()

    user_input = f"""
USER REQUEST
{query}

INTERPRETED FILTERS
{filter_summary}

RETRIEVED GAME RECORDS
{retrieval_context}

Write a grounded recommendation answer using only these records.
""".strip()

    try:
        client = OpenAI(
            api_key=api_key
        )

        response = (
            client.responses.create(
                model=model_name,
                instructions=instructions,
                input=user_input,
            )
        )

        answer_text = (
            response.output_text.strip()
        )

        if not answer_text:
            raise ValueError(
                "The model returned an empty answer."
            )

        return (
            answer_text,
            "openai",
        )

    except Exception:
        fallback_answer = (
            build_local_fallback_answer(
                search_results,
                requested_concepts,
            )
        )

        return (
            fallback_answer,
            "local_fallback_after_error",
        )