import pandas as pd

from scripts.generate_answer import (
    build_retrieval_context,
    generate_grounded_answer,
)


def create_search_results() -> pd.DataFrame:
    """Create one controlled retrieved game record."""

    return pd.DataFrame(
        {
            "name": [
                "Example Game",
            ],
            "price_usd": [
                9.99,
            ],
            "is_free": [
                False,
            ],
            "positive_review_percentage": [
                92.5,
            ],
            "total_reviews": [
                1000,
            ],
            "release_year": [
                2024,
            ],
            "genres": [
                "Adventure",
            ],
            "categories": [
                "Single-player",
            ],
            "tags": [
                "Relaxing, Casual",
            ],
            "platforms": [
                "Windows, Linux",
            ],
            "short_description": [
                "A relaxing adventure game.",
            ],
            "steam_store_url": [
                (
                    "https://store.steampowered.com/"
                    "app/123456"
                ),
            ],
        }
    )


def test_retrieval_context_contains_grounded_fields() -> None:
    """Context must contain retrieved data and the exact URL."""

    search_results = (
        create_search_results()
    )

    context = build_retrieval_context(
        search_results
    )

    assert "Example Game" in context
    assert "$9.99" in context
    assert "92.50% positive" in context

    assert (
        "https://store.steampowered.com/app/123456"
        in context
    )


def test_missing_api_key_uses_local_fallback(
    monkeypatch,
) -> None:
    """The application must work without an API key."""

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "",
    )

    answer_text, generation_mode = (
        generate_grounded_answer(
            query=(
                "a relaxing single-player game"
            ),
            search_results=(
                create_search_results()
            ),
            filters={
                "play_mode": "single-player",
            },
            requested_concepts=[
                "relaxing",
            ],
        )
    )

    assert (
        generation_mode
        == "local_fallback"
    )

    assert (
        "Example Game"
        in answer_text
    )

    assert (
        "https://store.steampowered.com/app/123456"
        in answer_text
    )


def test_empty_results_do_not_call_model() -> None:
    """Empty retrieval results must return a no-results answer."""

    answer_text, generation_mode = (
        generate_grounded_answer(
            query="an impossible request",
            search_results=pd.DataFrame(),
            filters={},
            requested_concepts=[],
        )
    )

    assert (
        generation_mode
        == "no_results"
    )

    assert (
        "No games satisfy"
        in answer_text
    )

def test_zero_price_paid_record_is_not_labeled_free() -> None:
    search_results = create_search_results()
    search_results.loc[0, "price_usd"] = 0.0
    search_results.loc[0, "is_free"] = False

    context = build_retrieval_context(
        search_results
    )

    assert "Price: $0.00" in context
    assert "Price: Free" not in context
