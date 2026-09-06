import numpy as np
import pandas as pd

from scripts.formatting import print_results
from scripts.query_filters import (
    apply_filters,
    extract_filters,
    filter_invalid_game_names,
    query_requires_clarification,
)
from scripts.ranking import (
    calculate_concept_scores,
    calculate_hybrid_scores,
    calculate_play_mode_preference_scores,
    detect_requested_concepts,
)


def create_test_dataframe() -> pd.DataFrame:
    """Create a small controlled dataset for retrieval tests."""

    return pd.DataFrame(
        {
            "name": [
                "Pure Single Game",
                "Mixed Party Game",
                "Tags Only Co-op Game",
                "Official Co-op Game",
            ],
            "price_usd": [
                9.99,
                9.99,
                9.99,
                9.99,
            ],
            "positive_review_percentage": [
                90.0,
                90.0,
                90.0,
                90.0,
            ],
            "total_reviews": [
                1000,
                1000,
                1000,
                1000,
            ],
            "release_year": [
                2022,
                2022,
                2022,
                2022,
            ],
            "is_free": [
                False,
                False,
                False,
                False,
            ],
            "platforms": [
                "Windows, Linux",
                "Windows, Linux",
                "Windows",
                "Windows",
            ],
            "genres": [
                "Casual",
                "Casual",
                "Action",
                "Adventure",
            ],
            "categories": [
                "Single-player",
                (
                    "Single-player, Multi-player, "
                    "PvP, Online PvP, Co-op, Online Co-op"
                ),
                "Multi-player, PvP, Online PvP",
                "Multi-player, Co-op, Online Co-op",
            ],
            "tags": [
                "Relaxing, Cozy, Casual",
                "Relaxing, Casual, Party, Multiplayer",
                "Survival, Co-op",
                "Survival, Co-op",
            ],
            "short_description": [
                "A peaceful and relaxing single-player experience.",
                "An online multiplayer party game.",
                "A competitive survival game.",
                "A cooperative survival adventure.",
            ],
        }
    )


def create_print_result_dataframe() -> pd.DataFrame:
    """Create one completed search result for display tests."""

    return pd.DataFrame(
        {
            "name": [
                "Example Game"
            ],
            "hybrid_score": [
                0.80
            ],
            "semantic_score": [
                0.60
            ],
            "concept_score": [
                1.00
            ],
            "play_mode_score": [
                0.95
            ],
            "price_usd": [
                0.00
            ],
            "is_free": [
                True
            ],
            "positive_review_percentage": [
                92.00
            ],
            "total_reviews": [
                1000
            ],
            "genres": [
                "Adventure"
            ],
            "categories": [
                "Single-player"
            ],
            "tags": [
                "Psychological Horror"
            ],
            "platforms": [
                "Windows"
            ],
            "release_year": [
                2022
            ],
            "steam_store_url": [
                "https://store.steampowered.com/app/1"
            ],
        }
    )


def test_extract_filters_detects_all_constraints() -> None:
    """All requested structured filters must be detected."""

    filters = extract_filters(
        (
            "a cooperative game under $20 "
            "with at least 80% positive reviews "
            "for Linux released after 2020"
        )
    )

    assert filters["maximum_price"] == 20.0
    assert filters["minimum_review_percentage"] == 80.0
    assert filters["platform"] == "Linux"
    assert filters["release_year_after"] == 2020
    assert filters["play_mode"] == "co-op"


def test_platform_filter_ignores_substrings() -> None:
    """
    Regression test for a bug where the platform regex had no \\b word
    boundaries, so any word merely CONTAINING "win", "mac", or "linux"
    as a substring incorrectly set platform: Windows/Mac/Linux even
    though the user never mentioned a platform.
    """

    assert "platform" not in extract_filters("a twin stick shooter")
    assert "platform" not in extract_filters("games about winning")
    assert "platform" not in extract_filters("unwind after a long day")

    # Whole-word mentions must still work correctly.
    assert extract_filters("a game for windows")["platform"] == "Windows"
    assert extract_filters("a game for win")["platform"] == "Windows"
    assert extract_filters("a game for mac")["platform"] == "Mac"
    assert extract_filters("a game for linux")["platform"] == "Linux"


def test_play_mode_ignores_known_ambiguous_substrings() -> None:
    """
    Regression test for the same class of bug as
    test_platform_filter_ignores_substrings, but for keyword ambiguity
    rather than a missing \\b boundary: "coop" is also the English word
    for a chicken enclosure, and the Chinese multiplayer keyword "多人"
    is also the tail of "很多人" ("a lot of people"). Neither query below
    is asking for a play mode, so play_mode must not be set.
    """

    assert "play_mode" not in extract_filters(
        "a relaxing farming game with a chicken coop"
    )
    assert "play_mode" not in extract_filters(
        "很多人推薦這款遊戲，畫面很漂亮"
    )

    # Whole-word / genuine mentions must still work correctly.
    assert extract_filters("a coop game with friends")["play_mode"] == "co-op"
    assert extract_filters("多人遊戲")["play_mode"] == "multiplayer"


def test_extract_filters_supports_traditional_chinese_query() -> None:
    """Traditional Chinese portfolio queries should map to structured filters."""

    filters = extract_filters(
        (
            "我想找一款 20 美元以下、"
            "至少 80% 好評、支援 Mac、"
            "可以合作、2020 年之後推出的生存遊戲"
        )
    )

    assert filters["maximum_price"] == 20.0
    assert filters["minimum_review_percentage"] == 80.0
    assert filters["platform"] == "Mac"
    assert filters["release_year_after"] == 2020
    assert filters["play_mode"] == "co-op"


def test_extract_filters_supports_compact_chinese_query() -> None:
    """Chinese constraints should work without spaces around the terms."""

    filters = extract_filters(
        "超級恐怖的且免費且至少80%正面評價"
    )

    assert filters["is_free"] is True
    assert filters["minimum_review_percentage"] == 80.0


def test_extract_filters_supports_more_chinese_phrases() -> None:
    """Common Chinese alternatives should map to the same hard filters."""

    filters = extract_filters(
        (
            "預算20以內，正評率80以上，"
            "支援Mac，跟朋友連線合作，2021年後"
        )
    )

    assert filters["maximum_price"] == 20.0
    assert filters["minimum_review_percentage"] == 80.0
    assert filters["platform"] == "Mac"
    assert filters["release_year_after"] == 2021
    assert filters["play_mode"] == "co-op"


def test_extract_filters_supports_colloquial_chinese_phrases() -> None:
    """Colloquial Chinese search phrases should map to structured filters."""

    filters = extract_filters(
        "20塊以下，好評80以上，MacBook，朋友一起玩"
    )

    assert filters["maximum_price"] == 20.0
    assert filters["minimum_review_percentage"] == 80.0
    assert filters["platform"] == "Mac"
    assert filters["play_mode"] == "co-op"


def test_detect_requested_concepts_supports_traditional_chinese() -> None:
    """Traditional Chinese concept terms should trigger the ranking signals."""

    concepts = detect_requested_concepts(
        "推薦一款放鬆、單人、農場模擬遊戲"
    )

    assert "relaxing" in concepts
    assert "farming" in concepts
    assert "simulation" in concepts


def test_detect_requested_concepts_supports_chinese_horror() -> None:
    """General Chinese horror terms should trigger concept ranking."""

    concepts = detect_requested_concepts(
        "超級恐怖的生存遊戲"
    )

    assert "horror" in concepts
    assert "survival" in concepts


def test_detect_requested_concepts_supports_expanded_chinese_terms() -> None:
    """Expanded Chinese genre terms should trigger existing concepts."""

    concepts = detect_requested_concepts(
        "想玩喪屍開放世界求生，也可以是日式RPG劇情多結局"
    )

    assert "survival" in concepts
    assert "open world" in concepts
    assert "rpg" in concepts
    assert "story rich" in concepts


def test_free_filter_detects_single_free_word() -> None:
    """Free must work even when other words occur before game."""

    filters = extract_filters(
        "a free psychological horror game"
    )

    assert filters["is_free"] is True


def test_coop_filter_uses_categories_not_tags() -> None:
    """Tags-only Co-op records must fail the hard filter."""

    dataframe = create_test_dataframe()

    result_mask = apply_filters(
        dataframe,
        {
            "play_mode": "co-op",
        },
    )

    assert bool(
        result_mask.iloc[2]
    ) is False

    assert bool(
        result_mask.iloc[3]
    ) is True


def test_after_year_is_strict() -> None:
    """After 2020 must exclude games from 2020."""

    dataframe = (
        create_test_dataframe()
        .iloc[:3]
        .copy()
    )

    dataframe["release_year"] = [
        2020,
        2021,
        2022,
    ]

    result_mask = apply_filters(
        dataframe,
        {
            "release_year_after": 2020,
        },
    )

    assert result_mask.tolist() == [
        False,
        True,
        True,
    ]


def test_psychological_horror_exact_match_scores_higher() -> None:
    """Exact Psychological Horror must beat generic Horror."""

    dataframe = pd.DataFrame(
        {
            "genres": [
                "Adventure",
                "Adventure",
            ],
            "tags": [
                (
                    "Psychological Horror, "
                    "Horror, Atmospheric"
                ),
                "Horror, Dark, Atmospheric",
            ],
            "short_description": [
                (
                    "A psychological horror "
                    "experience."
                ),
                "A general horror experience.",
            ],
        }
    )

    concept_scores = (
        calculate_concept_scores(
            "a psychological horror game",
            dataframe,
        )
    )

    assert concept_scores[0] == 1.0
    assert (
        concept_scores[0]
        > concept_scores[1]
    )


def test_description_only_survival_scores_lower_than_survival_tag() -> None:
    """
    Incidental survival wording must not equal a real
    Survival tag.
    """

    dataframe = pd.DataFrame(
        {
            "genres": [
                "Adventure",
                "Action",
            ],
            "tags": [
                (
                    "Survival, Crafting, "
                    "Base-Building"
                ),
                "Tactical, Shooter, Co-op",
            ],
            "short_description": [
                (
                    "Build a shelter and explore "
                    "a dangerous world."
                ),
                (
                    "Fight for survival in a "
                    "realistic military battle."
                ),
            ],
        }
    )

    concept_scores = (
        calculate_concept_scores(
            "a cooperative survival game",
            dataframe,
        )
    )

    assert concept_scores[0] == 1.0
    assert concept_scores[1] <= 0.25

    assert (
        concept_scores[0]
        > concept_scores[1]
    )


def test_invalid_names_are_removed_without_losing_alignment() -> None:
    """
    Invalid names and their corresponding embeddings
    must be removed together.
    """

    embeddings = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ],
        dtype=np.float32,
    )

    dataframe = pd.DataFrame(
        {
            "name": [
                "Valid Game",
                "not available",
                "",
                None,
            ],
        }
    )

    (
        filtered_embeddings,
        filtered_dataframe,
    ) = filter_invalid_game_names(
        embeddings,
        dataframe,
    )

    assert (
        filtered_dataframe[
            "name"
        ].tolist()
        == [
            "Valid Game"
        ]
    )

    assert np.array_equal(
        filtered_embeddings,
        np.array(
            [
                [1.0, 1.0]
            ],
            dtype=np.float32,
        ),
    )


def test_pure_single_player_scores_higher_than_mixed_game() -> None:
    """Pure Single-player must beat a mixed party game."""

    dataframe = (
        create_test_dataframe()
        .iloc[:2]
        .copy()
    )

    play_mode_scores = (
        calculate_play_mode_preference_scores(
            dataframe,
            "single-player",
        )
    )

    assert play_mode_scores[0] == 1.0

    assert (
        play_mode_scores[0]
        > play_mode_scores[1]
    )


def test_play_mode_score_fixes_original_ranking_problem() -> None:
    """
    Play-mode focus must overcome a small semantic
    disadvantage.
    """

    semantic_scores = np.array(
        [
            0.49,
            0.54,
        ],
        dtype=np.float32,
    )

    concept_scores = np.array(
        [
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )

    play_mode_scores = np.array(
        [
            1.0,
            0.35,
        ],
        dtype=np.float32,
    )

    quality_scores = np.array(
        [
            0.95,
            0.90,
        ],
        dtype=np.float32,
    )

    hybrid_scores = (
        calculate_hybrid_scores(
            semantic_scores=semantic_scores,
            concept_scores=concept_scores,
            play_mode_scores=play_mode_scores,
            quality_scores=quality_scores,
            has_requested_concepts=True,
            has_requested_play_mode=True,
        )
    )

    assert (
        hybrid_scores[0]
        > hybrid_scores[1]
    )


def test_price_only_query_requires_clarification() -> None:
    """A query containing only price must ask for details."""

    query = "a game under $20"

    filters = extract_filters(
        query
    )

    assert query_requires_clarification(
        query,
        filters,
    ) is True


def test_detailed_query_does_not_require_clarification() -> None:
    """A query with mood and play mode can be ranked."""

    query = (
        "a relaxing single-player "
        "casual game under $15"
    )

    filters = extract_filters(
        query
    )

    assert query_requires_clarification(
        query,
        filters,
    ) is False


def test_play_mode_score_is_hidden_when_not_requested(
    capsys,
) -> None:
    """
    Output must not show Play-mode score when the user
    did not request a play mode.
    """

    print_results(
        query=(
            "a free psychological "
            "horror game"
        ),
        search_results=(
            create_print_result_dataframe()
        ),
        filters={
            "is_free": True,
        },
        candidate_count=1,
        clarification_required=False,
        requested_concepts=[
            "psychological horror",
        ],
    )

    captured_output = (
        capsys.readouterr().out
    )

    assert (
        "Play-mode score:"
        not in captured_output
    )


def test_play_mode_score_is_displayed_when_requested(
    capsys,
) -> None:
    """
    Output must show Play-mode score when the user
    requested a play mode.
    """

    print_results(
        query=(
            "a relaxing "
            "single-player game"
        ),
        search_results=(
            create_print_result_dataframe()
        ),
        filters={
            "play_mode": "single-player",
        },
        candidate_count=1,
        clarification_required=False,
        requested_concepts=[
            "relaxing",
        ],
    )

    captured_output = (
        capsys.readouterr().out
    )

    assert (
        "Play-mode score:"
        in captured_output
    )
