import math
from typing import Any

import pandas as pd
import streamlit as st

from scripts.generate_answer import (
    generate_grounded_answer,
)
from scripts.hybrid_search import (
    format_filter_value,
    format_release_year,
    is_free_value,
    search_games,
)


st.set_page_config(
    page_title="GameWise AI",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_page_style() -> None:
    """Apply custom styling to the Streamlit application."""

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        .gamewise-title {
            font-size: 2.8rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.35rem;
        }

        .gamewise-subtitle {
            font-size: 1.05rem;
            color: #6b7280;
            margin-bottom: 1.8rem;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 750;
            margin-top: 1rem;
            margin-bottom: 0.75rem;
        }

        .result-count {
            color: #6b7280;
            margin-bottom: 1rem;
        }

        .score-caption {
            color: #6b7280;
            font-size: 0.88rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 0.8rem;
            padding: 0.8rem;
        }

        div[data-testid="stForm"] {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 1rem;
            padding: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def run_cached_search(
    query: str,
    top_k: int,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
    int,
    bool,
    list[str],
]:
    """Run and cache one hybrid retrieval request."""

    return search_games(
        query=query,
        top_k=top_k,
    )


@st.cache_data(show_spinner=False)
def run_cached_generation(
    query: str,
    search_results: pd.DataFrame,
    filters: dict[str, object],
    requested_concepts: list[str],
) -> tuple[str, str]:
    """Generate and cache a grounded recommendation answer."""

    return generate_grounded_answer(
        query=query,
        search_results=search_results,
        filters=filters,
        requested_concepts=requested_concepts,
    )


def safe_text(
    value: Any,
    default: str = "Not available",
) -> str:
    """Convert missing values into readable text."""

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text.casefold() in {
        "",
        "nan",
        "none",
        "null",
        "not available",
    }:
        return default

    return text


def shorten_text(
    value: Any,
    maximum_length: int = 320,
) -> str:
    """Shorten long game descriptions."""

    text = safe_text(
        value,
        default="No description is available.",
    )

    if len(text) <= maximum_length:
        return text

    return (
        text[: maximum_length - 3]
        .rstrip()
        + "..."
    )


def format_price(
    row: pd.Series,
) -> str:
    """Format a retrieved game's price."""

    price = pd.to_numeric(
        row.get("price_usd"),
        errors="coerce",
    )

    if (
        is_free_value(
            row.get("is_free")
        )
        or (
            not pd.isna(price)
            and float(price) == 0
        )
    ):
        return "Free"

    if pd.isna(price):
        return "Unknown"

    return f"${float(price):.2f}"


def format_review_summary(
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
        return "Review information unavailable"

    if pd.isna(total_reviews):
        return (
            f"{float(review_percentage):.2f}% positive"
        )

    return (
        f"{float(review_percentage):.2f}% positive "
        f"from {int(total_reviews):,} reviews"
    )


def format_tags(
    value: Any,
    maximum_tags: int = 8,
) -> str:
    """Return a compact list of tags."""

    tag_text = safe_text(
        value,
        default="",
    )

    if not tag_text:
        return "No tags available"

    tags = [
        tag.strip()
        for tag in tag_text.split(",")
        if tag.strip()
    ]

    selected_tags = tags[
        :maximum_tags
    ]

    return " · ".join(
        selected_tags
    )


def build_match_reasons(
    row: pd.Series,
    filters: dict[str, object],
    requested_concepts: list[str],
) -> list[str]:
    """Build grounded reasons using only retrieved metadata."""

    match_reasons: list[str] = []

    price = pd.to_numeric(
        row.get("price_usd"),
        errors="coerce",
    )

    review_percentage = pd.to_numeric(
        row.get(
            "positive_review_percentage"
        ),
        errors="coerce",
    )

    release_year = pd.to_numeric(
        row.get("release_year"),
        errors="coerce",
    )

    platforms = safe_text(
        row.get("platforms"),
        default="",
    )

    categories = safe_text(
        row.get("categories"),
        default="",
    )

    if filters.get("is_free") is True:
        match_reasons.append(
            "It is available for free."
        )

    if (
        "maximum_price" in filters
        and not pd.isna(price)
    ):
        maximum_price = float(
            filters["maximum_price"]
        )

        if float(price) == 0:
            match_reasons.append(
                f"It is free, so it is within your "
                f"${maximum_price:.2f} budget."
            )
        else:
            match_reasons.append(
                f"Its ${float(price):.2f} price is within "
                f"your ${maximum_price:.2f} budget."
            )

    if (
        "minimum_review_percentage"
        in filters
        and not pd.isna(
            review_percentage
        )
    ):
        minimum_review_percentage = float(
            filters[
                "minimum_review_percentage"
            ]
        )

        match_reasons.append(
            f"Its {float(review_percentage):.2f}% positive "
            f"review score meets your "
            f"{minimum_review_percentage:.2f}% requirement."
        )

    if "platform" in filters:
        requested_platform = str(
            filters["platform"]
        )

        if requested_platform.casefold() in (
            platforms.casefold()
        ):
            match_reasons.append(
                f"It supports {requested_platform}."
            )

    if "play_mode" in filters:
        requested_play_mode = str(
            filters["play_mode"]
        )

        match_reasons.append(
            f"Its official Steam categories support "
            f"{requested_play_mode}."
        )

    if (
        "release_year_after" in filters
        and not pd.isna(release_year)
    ):
        requested_year = int(
            filters[
                "release_year_after"
            ]
        )

        match_reasons.append(
            f"It was released in {int(release_year)}, "
            f"which is after {requested_year}."
        )

    if (
        "release_year_since" in filters
        and not pd.isna(release_year)
    ):
        requested_year = int(
            filters[
                "release_year_since"
            ]
        )

        match_reasons.append(
            f"It was released in {int(release_year)}, "
            f"which satisfies your request for games "
            f"released since {requested_year}."
        )

    if (
        "release_year_before" in filters
        and not pd.isna(release_year)
    ):
        requested_year = int(
            filters[
                "release_year_before"
            ]
        )

        match_reasons.append(
            f"It was released in {int(release_year)}, "
            f"which is before {requested_year}."
        )

    concept_score = pd.to_numeric(
        row.get("concept_score"),
        errors="coerce",
    )

    if requested_concepts:
        concept_text = ", ".join(
            requested_concepts
        )

        if (
            not pd.isna(concept_score)
            and float(concept_score) >= 0.80
        ):
            match_reasons.append(
                f"Its genres, tags, and description strongly "
                f"match: {concept_text}."
            )

        elif (
            not pd.isna(concept_score)
            and float(concept_score) >= 0.40
        ):
            match_reasons.append(
                f"Its metadata partially matches: "
                f"{concept_text}."
            )

    if not match_reasons:
        match_reasons.append(
            "Its retrieved metadata has strong semantic "
            "similarity to your request."
        )

    return match_reasons


def display_sidebar() -> bool:
    """Display application details and return generation preference."""

    with st.sidebar:
        st.header("🎮 GameWise AI")

        st.write(
            "GameWise combines hard metadata filters, "
            "semantic retrieval, concept matching, "
            "play-mode preferences, and review quality."
        )

        st.divider()

        use_grounded_answer = st.toggle(
            "Generate recommendation summary",
            value=True,
        )

        st.caption(
            "When an OpenAI API key is available, "
            "GameWise uses the configured model. "
            "Otherwise, it creates a local grounded summary."
        )

        st.divider()

        st.subheader(
            "Example searches"
        )

        st.code(
            "a relaxing single-player casual game under $15",
            language=None,
        )

        st.code(
            (
                "a cooperative survival game under $20 "
                "with at least 80% positive reviews"
            ),
            language=None,
        )

        st.code(
            "a free psychological horror game",
            language=None,
        )

        st.code(
            (
                "a turn-based tactical strategy game "
                "under $20 for Linux"
            ),
            language=None,
        )

        st.divider()

        st.caption(
            "All recommendations are grounded in the "
            "local Steam game dataset."
        )

    return use_grounded_answer


def display_filter_summary(
    filters: dict[str, object],
    requested_concepts: list[str],
    candidate_count: int,
) -> None:
    """Display interpreted hard filters and concepts."""

    st.markdown(
        '<div class="section-title">'
        "How GameWise understood your request"
        "</div>",
        unsafe_allow_html=True,
    )

    filter_column, concept_column = (
        st.columns(2)
    )

    with filter_column:
        st.markdown(
            "**Structured filters**"
        )

        if filters:
            for (
                filter_name,
                filter_value,
            ) in filters.items():
                st.write(
                    "✓ "
                    + format_filter_value(
                        filter_name,
                        filter_value,
                    )
                )
        else:
            st.caption(
                "No structured filters detected."
            )

    with concept_column:
        st.markdown(
            "**Detected concepts**"
        )

        if requested_concepts:
            for concept_name in (
                requested_concepts
            ):
                st.write(
                    f"✓ {concept_name}"
                )
        else:
            st.caption(
                "No clear genre, mood, or gameplay "
                "concept was detected."
            )

    st.markdown(
        (
            '<div class="result-count">'
            f"{candidate_count:,} games remained after "
            "applying the hard filters."
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def display_clarification_message(
    candidate_count: int,
) -> None:
    """Ask the user to provide more useful preferences."""

    st.info(
        f"Your request is too broad. "
        f"{candidate_count:,} games satisfy "
        "the current conditions."
    )

    st.markdown(
        """
Please add at least one preference:

- **Genre:** RPG, strategy, horror, racing
- **Mood:** relaxing, scary, story-rich
- **Play mode:** single-player, co-op, multiplayer
- **Platform:** Windows, Mac, Linux
- **Quality:** at least 80% positive reviews
        """
    )

    st.markdown(
        "**Example**"
    )

    st.code(
        "a relaxing single-player farming game under $20",
        language=None,
    )


def display_generation_status(
    generation_mode: str,
) -> None:
    """Display which answer-generation mode was used."""

    if generation_mode == "openai":
        st.caption(
            "The recommendation summary was generated "
            "from the retrieved Steam records."
        )

    elif generation_mode == "local_fallback":
        st.caption(
            "No API key was detected. GameWise used "
            "a local grounded recommendation summary."
        )

    elif (
        generation_mode
        == "local_fallback_after_error"
    ):
        st.caption(
            "The external model request was unavailable. "
            "GameWise used a local grounded summary."
        )


def display_game_result(
    result_number: int,
    row: pd.Series,
    filters: dict[str, object],
    requested_concepts: list[str],
) -> None:
    """Display one retrieved game recommendation."""

    game_name = safe_text(
        row.get("name"),
        default="Unknown game",
    )

    steam_url = safe_text(
        row.get("steam_store_url"),
        default="",
    )

    description = shorten_text(
        row.get("short_description")
    )

    with st.container(
        border=True
    ):
        title_column, metric_column = (
            st.columns(
                [3, 1]
            )
        )

        with title_column:
            st.caption(
                f"Recommendation {result_number}"
            )

            st.markdown(
                f"### {game_name}"
            )

            st.write(
                description
            )

        with metric_column:
            st.metric(
                "Price",
                format_price(row),
            )

            st.metric(
                "Release year",
                format_release_year(
                    row.get("release_year")
                ),
            )

        st.markdown(
            f"**Reviews:** "
            f"{format_review_summary(row)}"
        )

        st.markdown(
            f"**Genres:** "
            f"{safe_text(row.get('genres'))}"
        )

        st.markdown(
            f"**Platforms:** "
            f"{safe_text(row.get('platforms'))}"
        )

        st.markdown(
            f"**Tags:** "
            f"{format_tags(row.get('tags'))}"
        )

        st.markdown(
            "**Why it matches your request**"
        )

        match_reasons = build_match_reasons(
            row=row,
            filters=filters,
            requested_concepts=requested_concepts,
        )

        for match_reason in match_reasons:
            st.write(
                f"✓ {match_reason}"
            )

        if steam_url:
            st.link_button(
                "Open on Steam",
                steam_url,
                use_container_width=False,
            )

        with st.expander(
            "View ranking details"
        ):
            hybrid_score = pd.to_numeric(
                row.get("hybrid_score"),
                errors="coerce",
            )

            semantic_score = pd.to_numeric(
                row.get("semantic_score"),
                errors="coerce",
            )

            concept_score = pd.to_numeric(
                row.get("concept_score"),
                errors="coerce",
            )

            if not pd.isna(
                hybrid_score
            ):
                st.write(
                    "Hybrid score:",
                    f"{float(hybrid_score):.4f}",
                )

            if not pd.isna(
                semantic_score
            ):
                st.write(
                    "Semantic score:",
                    f"{float(semantic_score):.4f}",
                )

            if not pd.isna(
                concept_score
            ):
                st.write(
                    "Concept score:",
                    f"{float(concept_score):.4f}",
                )

            if "play_mode" in filters:
                play_mode_score = (
                    pd.to_numeric(
                        row.get(
                            "play_mode_score"
                        ),
                        errors="coerce",
                    )
                )

                if not pd.isna(
                    play_mode_score
                ):
                    st.write(
                        "Play-mode score:",
                        f"{float(play_mode_score):.4f}",
                    )

            st.write(
                "Official categories:",
                safe_text(
                    row.get("categories")
                ),
            )


def clear_previous_search() -> None:
    """Remove the previous search result from session state."""

    st.session_state.pop(
        "search_payload",
        None,
    )

    st.session_state.pop(
        "submitted_query",
        None,
    )


def main() -> None:
    """Run the GameWise Streamlit application."""

    apply_page_style()

    use_grounded_answer = (
        display_sidebar()
    )

    st.markdown(
        '<div class="gamewise-title">'
        "🎮 GameWise AI"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="gamewise-subtitle">'
        "Describe the kind of game you want. "
        "GameWise will apply your requirements, "
        "retrieve matching Steam games, and explain "
        "why each recommendation fits."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.form(
        "game_search_form"
    ):
        query = st.text_input(
            "What kind of game are you looking for?",
            placeholder=(
                "Example: a relaxing single-player "
                "farming game under $20"
            ),
        )

        top_k = st.slider(
            "Number of recommendations",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
        )

        search_submitted = (
            st.form_submit_button(
                "Find games",
                use_container_width=True,
            )
        )

    if search_submitted:
        cleaned_query = (
            query.strip()
        )

        if not cleaned_query:
            st.warning(
                "Please enter a game request."
            )

            clear_previous_search()

        else:
            with st.spinner(
                "Searching the Steam game collection..."
            ):
                search_payload = (
                    run_cached_search(
                        query=cleaned_query,
                        top_k=top_k,
                    )
                )

            st.session_state[
                "search_payload"
            ] = search_payload

            st.session_state[
                "submitted_query"
            ] = cleaned_query

    if (
        "search_payload"
        not in st.session_state
    ):
        st.info(
            "Enter a request above to begin."
        )
        return

    (
        search_results,
        extracted_filters,
        candidate_count,
        clarification_required,
        requested_concepts,
    ) = st.session_state[
        "search_payload"
    ]

    submitted_query = (
        st.session_state.get(
            "submitted_query",
            "",
        )
    )

    st.divider()

    st.markdown(
        f"## Results for `{submitted_query}`"
    )

    display_filter_summary(
        filters=extracted_filters,
        requested_concepts=requested_concepts,
        candidate_count=candidate_count,
    )

    if clarification_required:
        display_clarification_message(
            candidate_count
        )
        return

    if search_results.empty:
        st.warning(
            "No games satisfy all requested conditions. "
            "Try increasing the budget, lowering the "
            "review requirement, changing the platform, "
            "or removing one condition."
        )
        return

    if use_grounded_answer:
        with st.spinner(
            "Writing a grounded recommendation..."
        ):
            (
                generated_answer,
                generation_mode,
            ) = run_cached_generation(
                query=submitted_query,
                search_results=search_results,
                filters=extracted_filters,
                requested_concepts=requested_concepts,
            )

        st.markdown(
            '<div class="section-title">'
            "GameWise recommendation"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            generated_answer
        )

        display_generation_status(
            generation_mode
        )

        st.divider()

    st.markdown(
        '<div class="section-title">'
        "Recommended games"
        "</div>",
        unsafe_allow_html=True,
    )

    for (
        result_number,
        (_, result_row),
    ) in enumerate(
        search_results.iterrows(),
        start=1,
    ):
        display_game_result(
            result_number=result_number,
            row=result_row,
            filters=extracted_filters,
            requested_concepts=requested_concepts,
        )


if __name__ == "__main__":
    main()