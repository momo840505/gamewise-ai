from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.hybrid_search import search_games


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_CASES_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "evaluation_cases.json"
)

EVALUATION_RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "evaluation_results.csv"
)

EVALUATION_SUMMARY_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "evaluation_summary.md"
)


INVALID_GAME_NAMES = {
    "",
    "nan",
    "none",
    "null",
    "n/a",
    "not available",
    "unknown",
    "unknown game",
    "unknown title",
}


def safe_text(
    value: Any,
) -> str:
    """Convert a value into searchable text."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def numeric_value(
    value: Any,
) -> float | None:
    """Convert a value into a number when possible."""

    converted_value = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(converted_value):
        return None

    return float(converted_value)


def is_valid_game_name(
    value: Any,
) -> bool:
    """Check whether a game title is usable."""

    normalized_name = (
        safe_text(value)
        .casefold()
    )

    return (
        normalized_name
        not in INVALID_GAME_NAMES
    )


def is_free_game(
    row: pd.Series,
) -> bool:
    """Check whether a retrieved game is free."""

    free_value = (
        safe_text(
            row.get("is_free")
        )
        .casefold()
    )

    price = numeric_value(
        row.get("price_usd")
    )

    return (
        free_value == "true"
        or price == 0
    )


def supports_play_mode(
    categories: Any,
    requested_play_mode: str,
) -> bool:
    """Check official Steam categories for a play mode."""

    category_text = safe_text(
        categories
    )

    if requested_play_mode == "single-player":
        pattern = r"single[- ]?player"

    elif requested_play_mode == "co-op":
        pattern = (
            r"co[- ]?op|coop|cooperative"
        )

    else:
        pattern = r"multi[- ]?player"

    return bool(
        re.search(
            pattern,
            category_text,
            flags=re.IGNORECASE,
        )
    )


def evaluate_hard_filters(
    row: pd.Series,
    filters: dict[str, object],
) -> tuple[int, int, list[str]]:
    """Check every extracted hard filter for one result."""

    passed_checks = 0
    total_checks = 0
    errors: list[str] = []

    if "maximum_price" in filters:
        total_checks += 1

        price = numeric_value(
            row.get("price_usd")
        )

        maximum_price = float(
            filters["maximum_price"]
        )

        if (
            price is not None
            and price <= maximum_price
        ):
            passed_checks += 1

        else:
            errors.append(
                "price exceeds maximum"
            )

    if (
        "minimum_review_percentage"
        in filters
    ):
        total_checks += 1

        review_percentage = numeric_value(
            row.get(
                "positive_review_percentage"
            )
        )

        minimum_percentage = float(
            filters[
                "minimum_review_percentage"
            ]
        )

        if (
            review_percentage is not None
            and review_percentage
            >= minimum_percentage
        ):
            passed_checks += 1

        else:
            errors.append(
                "review percentage is too low"
            )

    if filters.get("is_free") is True:
        total_checks += 1

        if is_free_game(row):
            passed_checks += 1

        else:
            errors.append(
                "game is not free"
            )

    if "platform" in filters:
        total_checks += 1

        requested_platform = (
            str(filters["platform"])
            .casefold()
        )

        platforms = (
            safe_text(
                row.get("platforms")
            )
            .casefold()
        )

        if requested_platform in platforms:
            passed_checks += 1

        else:
            errors.append(
                "platform is unsupported"
            )

    release_year = numeric_value(
        row.get("release_year")
    )

    if "release_year_after" in filters:
        total_checks += 1

        requested_year = int(
            filters[
                "release_year_after"
            ]
        )

        if (
            release_year is not None
            and release_year
            > requested_year
        ):
            passed_checks += 1

        else:
            errors.append(
                "release year is not after cutoff"
            )

    if "release_year_since" in filters:
        total_checks += 1

        requested_year = int(
            filters[
                "release_year_since"
            ]
        )

        if (
            release_year is not None
            and release_year
            >= requested_year
        ):
            passed_checks += 1

        else:
            errors.append(
                "release year is before requested period"
            )

    if "release_year_before" in filters:
        total_checks += 1

        requested_year = int(
            filters[
                "release_year_before"
            ]
        )

        if (
            release_year is not None
            and release_year
            < requested_year
        ):
            passed_checks += 1

        else:
            errors.append(
                "release year is not before cutoff"
            )

    if "play_mode" in filters:
        total_checks += 1

        requested_play_mode = str(
            filters["play_mode"]
        )

        if supports_play_mode(
            row.get("categories"),
            requested_play_mode,
        ):
            passed_checks += 1

        else:
            errors.append(
                "official play mode is unsupported"
            )

    return (
        passed_checks,
        total_checks,
        errors,
    )


def is_concept_relevant(
    row: pd.Series,
    relevance_terms: list[str],
) -> bool:
    """
    Check relevance using independent evaluation terms.

    This does not use the model's concept score.
    """

    if not relevance_terms:
        return False

    metadata_text = " ".join(
        [
            safe_text(
                row.get("genres")
            ),
            safe_text(
                row.get("tags")
            ),
            safe_text(
                row.get(
                    "short_description"
                )
            ),
        ]
    ).casefold()

    return any(
        relevance_term.casefold()
        in metadata_text
        for relevance_term
        in relevance_terms
    )


def expected_behavior_passed(
    expected_behavior: str,
    search_results: pd.DataFrame,
    candidate_count: int,
    clarification_required: bool,
) -> bool:
    """Check clarification, no-result, or result behavior."""

    if expected_behavior == "clarification":
        return (
            clarification_required
            and search_results.empty
        )

    if expected_behavior == "no_results":
        return (
            candidate_count == 0
            and search_results.empty
            and not clarification_required
        )

    return (
        not search_results.empty
        and not clarification_required
    )


def evaluate_case(
    evaluation_case: dict[str, Any],
) -> dict[str, Any]:
    """Run and evaluate one retrieval case."""

    case_id = str(
        evaluation_case["case_id"]
    )

    query = str(
        evaluation_case["query"]
    )

    top_k = int(
        evaluation_case.get(
            "top_k",
            5,
        )
    )

    expected_behavior = str(
        evaluation_case[
            "expected_behavior"
        ]
    )

    relevance_terms = list(
        evaluation_case.get(
            "relevance_terms",
            [],
        )
    )

    start_time = time.perf_counter()

    (
        search_results,
        extracted_filters,
        candidate_count,
        clarification_required,
        requested_concepts,
    ) = search_games(
        query=query,
        top_k=top_k,
    )

    latency_seconds = (
        time.perf_counter()
        - start_time
    )

    behavior_passed = (
        expected_behavior_passed(
            expected_behavior=expected_behavior,
            search_results=search_results,
            candidate_count=candidate_count,
            clarification_required=(
                clarification_required
            ),
        )
    )

    hard_filter_passed = 0
    hard_filter_checks = 0
    hard_filter_errors: list[str] = []

    valid_title_count = 0
    result_names: list[str] = []

    concept_relevant_count = 0

    concept_evaluated_count = min(
        len(search_results),
        5,
    )

    for result_position, (_, row) in enumerate(
        search_results.iterrows(),
        start=1,
    ):
        (
            row_passed_checks,
            row_total_checks,
            row_errors,
        ) = evaluate_hard_filters(
            row=row,
            filters=extracted_filters,
        )

        hard_filter_passed += (
            row_passed_checks
        )

        hard_filter_checks += (
            row_total_checks
        )

        for row_error in row_errors:
            hard_filter_errors.append(
                f"result {result_position}: "
                f"{row_error}"
            )

        game_name = safe_text(
            row.get("name")
        )

        result_names.append(
            game_name.casefold()
        )

        if is_valid_game_name(
            game_name
        ):
            valid_title_count += 1

        if (
            result_position <= 5
            and is_concept_relevant(
                row=row,
                relevance_terms=(
                    relevance_terms
                ),
            )
        ):
            concept_relevant_count += 1

    result_count = len(
        search_results
    )

    unique_name_count = len(
        set(result_names)
    )

    if hard_filter_checks:
        hard_filter_accuracy = (
            hard_filter_passed
            / hard_filter_checks
        )

    else:
        hard_filter_accuracy = 1.0

    if result_count:
        valid_title_rate = (
            valid_title_count
            / result_count
        )

        duplicate_free_rate = (
            unique_name_count
            / result_count
        )

    else:
        valid_title_rate = 1.0
        duplicate_free_rate = 1.0

    if concept_evaluated_count:
        concept_relevance_at_5 = (
            concept_relevant_count
            / concept_evaluated_count
        )

    else:
        concept_relevance_at_5 = None

    notes: list[str] = []

    if not behavior_passed:
        notes.append(
            "unexpected search behavior"
        )

    notes.extend(
        hard_filter_errors
    )

    if valid_title_rate < 1:
        notes.append(
            "invalid title detected"
        )

    if duplicate_free_rate < 1:
        notes.append(
            "duplicate recommendation detected"
        )

    if not notes:
        notes.append(
            "passed"
        )

    return {
        "case_id": case_id,
        "query": query,
        "expected_behavior": (
            expected_behavior
        ),
        "behavior_passed": (
            behavior_passed
        ),
        "candidate_count": (
            candidate_count
        ),
        "result_count": result_count,
        "hard_filter_passed": (
            hard_filter_passed
        ),
        "hard_filter_checks": (
            hard_filter_checks
        ),
        "hard_filter_accuracy": (
            hard_filter_accuracy
        ),
        "valid_title_count": (
            valid_title_count
        ),
        "valid_title_rate": (
            valid_title_rate
        ),
        "unique_name_count": (
            unique_name_count
        ),
        "duplicate_free_rate": (
            duplicate_free_rate
        ),
        "concept_relevant_count": (
            concept_relevant_count
        ),
        "concept_evaluated_count": (
            concept_evaluated_count
        ),
        "concept_relevance_at_5": (
            concept_relevance_at_5
        ),
        "latency_seconds": (
            latency_seconds
        ),
        "extracted_filters": json.dumps(
            extracted_filters,
            ensure_ascii=False,
        ),
        "requested_concepts": ", ".join(
            requested_concepts
        ),
        "notes": "; ".join(
            notes
        ),
    }


def percentage_text(
    numerator: int,
    denominator: int,
) -> str:
    """Format an aggregate percentage."""

    if denominator == 0:
        return "Not applicable"

    return (
        f"{numerator / denominator:.2%}"
    )


def write_summary(
    results_dataframe: pd.DataFrame,
) -> None:
    """Create a Markdown evaluation summary."""

    total_cases = len(
        results_dataframe
    )

    behavior_passed_count = int(
        results_dataframe[
            "behavior_passed"
        ].sum()
    )

    hard_filter_passed = int(
        results_dataframe[
            "hard_filter_passed"
        ].sum()
    )

    hard_filter_checks = int(
        results_dataframe[
            "hard_filter_checks"
        ].sum()
    )

    valid_title_count = int(
        results_dataframe[
            "valid_title_count"
        ].sum()
    )

    result_count = int(
        results_dataframe[
            "result_count"
        ].sum()
    )

    unique_name_count = int(
        results_dataframe[
            "unique_name_count"
        ].sum()
    )

    concept_relevant_count = int(
        results_dataframe[
            "concept_relevant_count"
        ].sum()
    )

    concept_evaluated_count = int(
        results_dataframe[
            "concept_evaluated_count"
        ].sum()
    )

    average_latency = float(
        results_dataframe[
            "latency_seconds"
        ].mean()
    )

    summary_lines = [
        "# GameWise Retrieval Evaluation",
        "",
        "## Overall results",
        "",
        (
            "- Search behavior pass rate: "
            f"{behavior_passed_count}/{total_cases} "
            f"({behavior_passed_count / total_cases:.2%})"
        ),
        (
            "- Hard filter accuracy: "
            + percentage_text(
                hard_filter_passed,
                hard_filter_checks,
            )
        ),
        (
            "- Valid title rate: "
            + percentage_text(
                valid_title_count,
                result_count,
            )
        ),
        (
            "- Duplicate-free rate: "
            + percentage_text(
                unique_name_count,
                result_count,
            )
        ),
        (
            "- Independent concept relevance@5: "
            + percentage_text(
                concept_relevant_count,
                concept_evaluated_count,
            )
        ),
        (
            "- Average retrieval latency: "
            f"{average_latency:.3f} seconds"
        ),
        "",
        "## Case results",
        "",
    ]

    for _, row in (
        results_dataframe.iterrows()
    ):
        concept_value = row[
            "concept_relevance_at_5"
        ]

        if pd.isna(concept_value):
            concept_text = (
                "Not applicable"
            )

        else:
            concept_text = (
                f"{float(concept_value):.2%}"
            )

        summary_lines.extend(
            [
                (
                    f"### {row['case_id']}"
                ),
                "",
                (
                    f"- Query: `{row['query']}`"
                ),
                (
                    "- Behavior passed: "
                    f"{row['behavior_passed']}"
                ),
                (
                    "- Results returned: "
                    f"{row['result_count']}"
                ),
                (
                    "- Hard filter accuracy: "
                    f"{float(row['hard_filter_accuracy']):.2%}"
                ),
                (
                    "- Valid title rate: "
                    f"{float(row['valid_title_rate']):.2%}"
                ),
                (
                    "- Duplicate-free rate: "
                    f"{float(row['duplicate_free_rate']):.2%}"
                ),
                (
                    "- Concept relevance@5: "
                    f"{concept_text}"
                ),
                (
                    "- Latency: "
                    f"{float(row['latency_seconds']):.3f} seconds"
                ),
                (
                    f"- Notes: {row['notes']}"
                ),
                "",
            ]
        )

    EVALUATION_SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Run all evaluation cases."""

    evaluation_cases = json.loads(
        EVALUATION_CASES_PATH.read_text(
            encoding="utf-8",
        )
    )

    evaluation_results: list[
        dict[str, Any]
    ] = []

    total_cases = len(
        evaluation_cases
    )

    for case_number, evaluation_case in enumerate(
        evaluation_cases,
        start=1,
    ):
        print(
            f"[{case_number}/{total_cases}] "
            f"Evaluating "
            f"{evaluation_case['case_id']}..."
        )

        case_result = evaluate_case(
            evaluation_case
        )

        evaluation_results.append(
            case_result
        )

    results_dataframe = pd.DataFrame(
        evaluation_results
    )

    results_dataframe.to_csv(
        EVALUATION_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    write_summary(
        results_dataframe
    )

    print()
    print(
        "Evaluation completed."
    )

    print(
        "CSV results: "
        f"{EVALUATION_RESULTS_PATH}"
    )

    print(
        "Markdown summary: "
        f"{EVALUATION_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()