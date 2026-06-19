from __future__ import annotations

import html
import os
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from scripts.generate_answer import generate_grounded_answer
from scripts.hybrid_search import (
    format_filter_value,
    format_release_year,
    is_free_value,
    search_games,
)


load_dotenv()

EXAMPLE_SEARCHES = [
    (
        "🌿 Cozy single-player",
        "a relaxing single-player casual game under $15",
    ),
    (
        "🛡️ Co-op survival",
        (
            "a cooperative survival game under $20 "
            "with at least 80% positive reviews"
        ),
    ),
    (
        "👻 Free psychological horror",
        "a free psychological horror game",
    ),
    (
        "♟️ Tactical strategy",
        (
            "a turn-based tactical strategy game "
            "under $20 for Linux"
        ),
    ),
]

st.set_page_config(
    page_title="GameWise AI",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_page_style() -> None:
    """Apply a polished visual style."""

    st.markdown(
        """
        <style>
        :root {
            --purple: #6f52ed;
            --purple-dark: #4e36c4;
            --pink: #eb72ad;
            --text: #222238;
            --muted: #6e6e82;
            --border: rgba(111, 82, 237, 0.16);
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 8% 0%,
                    rgba(111, 82, 237, 0.10),
                    transparent 32%
                ),
                radial-gradient(
                    circle at 94% 8%,
                    rgba(235, 114, 173, 0.08),
                    transparent 28%
                ),
                #fbfbff;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.8rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    #f5f2ff 0%,
                    #fcfbff 60%,
                    #ffffff 100%
                );
            border-right: 1px solid var(--border);
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.2rem 2.4rem;
            margin-bottom: 1.25rem;
            border: 1px solid rgba(111, 82, 237, 0.18);
            border-radius: 24px;
            background:
                linear-gradient(
                    120deg,
                    rgba(111, 82, 237, 0.13),
                    rgba(235, 114, 173, 0.08)
                );
            box-shadow:
                0 18px 50px rgba(83, 66, 150, 0.08);
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 210px;
            height: 210px;
            right: -55px;
            top: -100px;
            border-radius: 50%;
            background:
                linear-gradient(
                    135deg,
                    rgba(111, 82, 237, 0.22),
                    rgba(235, 114, 173, 0.14)
                );
        }

        .hero-title {
            position: relative;
            z-index: 1;
            margin-bottom: 0.65rem;
            color: var(--text);
            font-size: 3rem;
            font-weight: 850;
            line-height: 1.05;
        }

        .hero-subtitle {
            position: relative;
            z-index: 1;
            max-width: 800px;
            color: var(--muted);
            font-size: 1.08rem;
            line-height: 1.65;
        }

        .section-title {
            margin-top: 0.7rem;
            margin-bottom: 0.8rem;
            color: var(--text);
            font-size: 1.35rem;
            font-weight: 780;
        }

        .filter-chip,
        .concept-chip,
        .tag-chip {
            display: inline-block;
            margin: 0.16rem 0.28rem 0.16rem 0;
            border-radius: 999px;
            line-height: 1.2;
        }

        .filter-chip {
            padding: 0.45rem 0.72rem;
            border: 1px solid rgba(111, 82, 237, 0.18);
            background: rgba(111, 82, 237, 0.10);
            color: #4f39ba;
            font-size: 0.87rem;
            font-weight: 650;
        }

        .concept-chip {
            padding: 0.45rem 0.72rem;
            border: 1px solid rgba(235, 114, 173, 0.20);
            background: rgba(235, 114, 173, 0.10);
            color: #a74273;
            font-size: 0.87rem;
            font-weight: 650;
        }

        .tag-chip {
            padding: 0.30rem 0.56rem;
            border: 1px solid #e6e6ef;
            background: #f4f4fa;
            color: #5d5d70;
            font-size: 0.78rem;
        }

        .match-pill {
            display: inline-block;
            padding: 0.40rem 0.70rem;
            border-radius: 999px;
            white-space: nowrap;
            font-size: 0.82rem;
            font-weight: 750;
        }

        .match-excellent {
            border: 1px solid #bdebd2;
            background: #e7f8ef;
            color: #176b43;
        }

        .match-strong {
            border: 1px solid #d8d0ff;
            background: #f0edff;
            color: #4e38b8;
        }

        .match-good {
            border: 1px solid #f4dfaa;
            background: #fff7df;
            color: #8b5c16;
        }

        .match-partial {
            border: 1px solid #f2cfdb;
            background: #fff0f5;
            color: #7a5162;
        }

        .feature-card {
            min-height: 165px;
            padding: 1.2rem;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.80);
        }

        .feature-icon {
            margin-bottom: 0.5rem;
            font-size: 1.8rem;
        }

        .feature-title {
            margin-bottom: 0.35rem;
            color: var(--text);
            font-size: 1rem;
            font-weight: 750;
        }

        .feature-text {
            color: var(--muted);
            font-size: 0.90rem;
            line-height: 1.5;
        }

        .summary-mode {
            display: inline-block;
            margin-bottom: 0.65rem;
            padding: 0.28rem 0.58rem;
            border-radius: 999px;
            background: rgba(111, 82, 237, 0.10);
            color: var(--purple-dark);
            font-size: 0.78rem;
            font-weight: 700;
        }

        div[data-testid="stForm"] {
            padding: 1.25rem;
            border: 1px solid var(--border);
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.86);
            box-shadow:
                0 12px 35px rgba(83, 66, 150, 0.05);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--border);
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow:
                0 10px 32px rgba(83, 66, 150, 0.055);
        }

        div[data-testid="stMetric"] {
            padding: 0.72rem 0.82rem;
            border: 1px solid rgba(111, 82, 237, 0.12);
            border-radius: 14px;
            background: rgba(248, 247, 255, 0.80);
        }

        div[data-testid="stButton"] button,
        div[data-testid="stFormSubmitButton"] button,
        div[data-testid="stDownloadButton"] button,
        div[data-testid="stLinkButton"] a {
            min-height: 2.7rem;
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    """Create session values used by the interface."""

    defaults = {
        "query_input": "",
        "search_history": [],
        "shortlist": [],
        "auto_submit": False,
        "pending_toast": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def display_pending_toast() -> None:
    """Display a saved notification after Streamlit reruns."""

    message = st.session_state.pop(
        "pending_toast",
        "",
    )

    if message:
        st.toast(
            message,
            icon="💜",
        )

def select_example_query(query: str) -> None:
    """Insert an example query and run it automatically."""

    st.session_state["query_input"] = query
    st.session_state["auto_submit"] = True


def clear_current_search() -> None:
    """Clear the current search."""

    st.session_state["query_input"] = ""
    st.session_state["auto_submit"] = False
    st.session_state.pop(
        "search_payload",
        None,
    )
    st.session_state.pop(
        "submitted_query",
        None,
    )


def clear_search_history() -> None:
    """Remove all recent searches."""

    st.session_state["search_history"] = []


def clear_shortlist() -> None:
    """Remove every saved game."""

    st.session_state["shortlist"] = []


def add_to_search_history(query: str) -> None:
    """Save the newest query without duplicates."""

    history = list(
        st.session_state.get(
            "search_history",
            [],
        )
    )

    history = [
        saved_query
        for saved_query in history
        if saved_query != query
    ]

    history.insert(
        0,
        query,
    )

    st.session_state["search_history"] = history[:5]


def add_to_shortlist(
    game_name: str,
    steam_url: str,
) -> None:
    """Save one recommendation and refresh the interface."""

    shortlist = list(
        st.session_state.get(
            "shortlist",
            [],
        )
    )

    already_saved = any(
        item["name"] == game_name
        for item in shortlist
    )

    if already_saved:
        st.session_state["pending_toast"] = (
            f"{game_name} is already in your shortlist."
        )

        st.rerun()

    shortlist.append(
        {
            "name": game_name,
            "url": steam_url,
        }
    )

    st.session_state["shortlist"] = shortlist

    st.session_state["pending_toast"] = (
        f"{game_name} was added to your shortlist."
    )

    st.rerun()

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
    """Run and cache one hybrid search."""

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
    """Generate and cache one grounded summary."""

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
    maximum_length: int = 330,
) -> str:
    """Shorten a long description."""

    text = safe_text(
        value,
        default="No description is available.",
    )

    if len(text) <= maximum_length:
        return text

    return (
        text[: maximum_length - 3].rstrip()
        + "..."
    )


def format_price(
    row: pd.Series,
) -> str:
    """Format one game's price."""

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


def format_review_percentage(
    row: pd.Series,
) -> str:
    """Format the positive-review percentage."""

    percentage = pd.to_numeric(
        row.get(
            "positive_review_percentage"
        ),
        errors="coerce",
    )

    if pd.isna(percentage):
        return "Unknown"

    return f"{float(percentage):.1f}%"


def format_review_summary(
    row: pd.Series,
) -> str:
    """Format review percentage and count."""

    percentage = pd.to_numeric(
        row.get(
            "positive_review_percentage"
        ),
        errors="coerce",
    )

    total_reviews = pd.to_numeric(
        row.get("total_reviews"),
        errors="coerce",
    )

    if pd.isna(percentage):
        return "Review information unavailable"

    if pd.isna(total_reviews):
        return (
            f"{float(percentage):.2f}% positive"
        )

    return (
        f"{float(percentage):.2f}% positive "
        f"from {int(total_reviews):,} reviews"
    )


def get_tags(
    value: Any,
    maximum_tags: int = 8,
) -> list[str]:
    """Return the first useful tags."""

    tag_text = safe_text(
        value,
        default="",
    )

    if not tag_text:
        return []

    tags = [
        tag.strip()
        for tag in tag_text.split(",")
        if tag.strip()
    ]

    return tags[:maximum_tags]


def render_chips(
    values: list[str],
    css_class: str,
) -> None:
    """Display filters, concepts, or tags as chips."""

    if not values:
        return

    chip_html = "".join(
        (
            f'<span class="{css_class}">'
            f"{html.escape(value)}"
            "</span>"
        )
        for value in values
    )

    st.markdown(
        chip_html,
        unsafe_allow_html=True,
    )


def get_match_label(
    row: pd.Series,
) -> tuple[str, str]:
    """Convert the hybrid score into a friendly label."""

    score_value = pd.to_numeric(
        row.get("hybrid_score"),
        errors="coerce",
    )

    if pd.isna(score_value):
        return (
            "Relevant match",
            "match-good",
        )

    score = float(
        score_value
    )

    if score >= 0.84:
        return (
            "Excellent match",
            "match-excellent",
        )

    if score >= 0.79:
        return (
            "Strong match",
            "match-strong",
        )

    if score >= 0.72:
        return (
            "Good match",
            "match-good",
        )

    return (
        "Partial match",
        "match-partial",
    )


def build_match_reasons(
    row: pd.Series,
    filters: dict[str, object],
    requested_concepts: list[str],
) -> list[str]:
    """Build clear reasons from retrieved metadata."""

    reasons: list[str] = []

    price = pd.to_numeric(
        row.get("price_usd"),
        errors="coerce",
    )

    percentage = pd.to_numeric(
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

    if filters.get("is_free") is True:
        reasons.append(
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
            reasons.append(
                f"It is free and fits your "
                f"${maximum_price:.2f} budget."
            )

        else:
            reasons.append(
                f"Its ${float(price):.2f} price fits "
                f"your ${maximum_price:.2f} budget."
            )

    if (
        "minimum_review_percentage"
        in filters
        and not pd.isna(percentage)
    ):
        minimum_percentage = float(
            filters[
                "minimum_review_percentage"
            ]
        )

        reasons.append(
            f"Its {float(percentage):.2f}% positive "
            f"rating meets your "
            f"{minimum_percentage:.2f}% requirement."
        )

    if "platform" in filters:
        requested_platform = str(
            filters["platform"]
        )

        if (
            requested_platform.casefold()
            in platforms.casefold()
        ):
            reasons.append(
                f"It supports {requested_platform}."
            )

    if "play_mode" in filters:
        requested_play_mode = str(
            filters["play_mode"]
        )

        reasons.append(
            f"Its official Steam features support "
            f"{requested_play_mode} play."
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

        reasons.append(
            f"It was released in {int(release_year)}, "
            f"after your {requested_year} cutoff."
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

        reasons.append(
            f"Its {int(release_year)} release fits "
            f"your request for games since "
            f"{requested_year}."
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

        reasons.append(
            f"It was released in {int(release_year)}, "
            f"before {requested_year}."
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
            reasons.append(
                f"Its genres and tags strongly match "
                f"your interest in {concept_text}."
            )

        elif (
            not pd.isna(concept_score)
            and float(concept_score) >= 0.40
        ):
            reasons.append(
                f"Its metadata partially matches "
                f"{concept_text}."
            )

    if not reasons:
        reasons.append(
            "Its overall metadata is highly relevant "
            "to your request."
        )

    return reasons


def display_sidebar() -> tuple[
    bool,
    bool,
]:
    """Display settings, examples, history, and shortlist."""

    with st.sidebar:
        st.markdown(
            "## 🎮 GameWise AI"
        )

        st.caption(
            "Natural-language Steam discovery using "
            "filters, semantic retrieval, and grounded "
            "recommendations."
        )

        st.divider()

        generate_summary = st.toggle(
            "Generate recommendation summary",
            value=True,
        )

        developer_mode = st.toggle(
            "Developer details",
            value=False,
        )

        api_key_available = bool(
            os.getenv(
                "OPENAI_API_KEY",
                "",
            ).strip()
        )

        if api_key_available:
            st.success(
                "AI summary is ready.",
                icon="✨",
            )

        else:
            st.info(
                "Local summary mode is active.",
                icon="🧩",
            )

        st.divider()

        st.markdown(
            "### Try an example"
        )

        for (
            button_label,
            query,
        ) in EXAMPLE_SEARCHES:
            st.button(
                button_label,
                key=f"example_{button_label}",
                use_container_width=True,
                on_click=select_example_query,
                args=(query,),
            )

        history = st.session_state.get(
            "search_history",
            [],
        )

        if history:
            st.divider()

            title_column, clear_column = (
                st.columns(
                    [3, 1]
                )
            )

            with title_column:
                st.markdown(
                    "### Recent searches"
                )

            with clear_column:
                st.button(
                    "Clear",
                    key="clear_history",
                    on_click=clear_search_history,
                )

            for index, query in enumerate(
                history
            ):
                label = (
                    query
                    if len(query) <= 42
                    else query[:39] + "..."
                )

                st.button(
                    label,
                    key=f"history_{index}",
                    use_container_width=True,
                    on_click=select_example_query,
                    args=(query,),
                )

        shortlist = st.session_state.get(
            "shortlist",
            [],
        )

        if shortlist:
            st.divider()

            title_column, clear_column = (
                st.columns(
                    [3, 1]
                )
            )

            with title_column:
                st.markdown(
                    "### 💜 Shortlist"
                )

            with clear_column:
                st.button(
                    "Clear",
                    key="clear_shortlist",
                    on_click=clear_shortlist,
                )

            for saved_game in shortlist:
                game_name = saved_game[
                    "name"
                ]

                steam_url = saved_game[
                    "url"
                ]

                if steam_url:
                    st.markdown(
                        f"- [{game_name}]({steam_url})"
                    )

                else:
                    st.markdown(
                        f"- {game_name}"
                    )

        st.divider()

        st.caption(
            "Prices and metadata come from the "
            "local dataset snapshot and may differ "
            "from Steam."
        )

    return (
        generate_summary,
        developer_mode,
    )


def display_hero() -> None:
    """Display the main introduction."""

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                🎮 GameWise AI
            </div>
            <div class="hero-subtitle">
                Tell GameWise what you feel like playing.
                It will understand your budget, platform,
                preferred play mode, review expectations,
                and game style before recommending the
                strongest matches.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_search_form() -> tuple[
    bool,
    int,
]:
    """Display the natural-language search controls."""

    with st.form(
        "game_search_form"
    ):
        st.markdown(
            "### What would you like to play?"
        )

        st.text_area(
            "Describe your ideal game",
            key="query_input",
            placeholder=(
                "Example: a relaxing single-player "
                "farming game under $20 for Mac"
            ),
            height=92,
            label_visibility="collapsed",
        )

        slider_column, button_column = (
            st.columns(
                [2, 1]
            )
        )

        with slider_column:
            top_k = st.slider(
                "Number of recommendations",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
            )

        with button_column:
            st.write("")
            st.write("")

            submitted = (
                st.form_submit_button(
                    "✨ Find my games",
                    use_container_width=True,
                )
            )

    st.caption(
        "Tip: Include a genre, mood, platform, "
        "budget, play mode, or review requirement."
    )

    return (
        submitted,
        top_k,
    )


def display_empty_home() -> None:
    """Explain how GameWise works before the first search."""

    st.markdown(
        '<div class="section-title">'
        "How it works"
        "</div>",
        unsafe_allow_html=True,
    )

    columns = st.columns(3)

    cards = [
        (
            columns[0],
            "💬",
            "Describe what you want",
            (
                "Write naturally instead of "
                "completing a long filter form."
            ),
        ),
        (
            columns[1],
            "🔎",
            "GameWise understands",
            (
                "Budget, platform, year, reviews, "
                "concepts, and play mode are "
                "interpreted automatically."
            ),
        ),
        (
            columns[2],
            "✨",
            "Compare grounded matches",
            (
                "Read clear reasons, open Steam "
                "pages, and save favourites."
            ),
        ),
    ]

    for (
        column,
        icon,
        title,
        description,
    ) in cards:
        with column:
            st.markdown(
                (
                    '<div class="feature-card">'
                    f'<div class="feature-icon">'
                    f"{icon}"
                    "</div>"
                    f'<div class="feature-title">'
                    f"{title}"
                    "</div>"
                    f'<div class="feature-text">'
                    f"{description}"
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def display_filter_summary(
    filters: dict[str, object],
    requested_concepts: list[str],
    candidate_count: int,
    result_count: int,
) -> None:
    """Display how the request was interpreted."""

    st.markdown(
        '<div class="section-title">'
        "How GameWise understood your request"
        "</div>",
        unsafe_allow_html=True,
    )

    filter_values = [
        format_filter_value(
            name,
            value,
        )
        for (
            name,
            value,
        ) in filters.items()
    ]

    filter_column, concept_column = (
        st.columns(2)
    )

    with filter_column:
        st.markdown(
            "**Requirements**"
        )

        if filter_values:
            render_chips(
                filter_values,
                "filter-chip",
            )

        else:
            st.caption(
                "No structured requirements were detected."
            )

    with concept_column:
        st.markdown(
            "**Game style and concepts**"
        )

        if requested_concepts:
            render_chips(
                requested_concepts,
                "concept-chip",
            )

        else:
            st.caption(
                "No clear style or genre was detected."
            )

    candidate_column, result_column = (
        st.columns(2)
    )

    with candidate_column:
        st.metric(
            "Matching candidates",
            f"{candidate_count:,}",
        )

    with result_column:
        st.metric(
            "Recommendations returned",
            f"{result_count:,}",
        )


def display_clarification_message(
    candidate_count: int,
) -> None:
    """Ask the user to make a broad request more specific."""

    st.info(
        (
            f"Your request matches "
            f"{candidate_count:,} games, "
            "so more detail is needed for a "
            "useful recommendation."
        ),
        icon="💡",
    )

    st.markdown(
        """
Add one or more details:

- **Genre:** RPG, strategy, horror, racing
- **Mood:** relaxing, scary, emotional, story-rich
- **Play mode:** single-player, co-op, multiplayer
- **Platform:** Windows, Mac, Linux
- **Quality:** at least 80% positive reviews
        """
    )

    st.code(
        "a relaxing single-player farming game under $20",
        language=None,
    )


def display_no_results_message() -> None:
    """Explain that no game satisfies every condition."""

    st.warning(
        (
            "No games satisfy every requested condition. "
            "GameWise did not silently weaken or remove "
            "any requirement."
        ),
        icon="🔍",
    )

    st.markdown(
        """
Try one small change:

- Increase the maximum price
- Lower the review requirement
- Remove the release-year restriction
- Try another platform
- Change co-op to multiplayer
        """
    )


def sort_results(
    search_results: pd.DataFrame,
    sort_option: str,
) -> pd.DataFrame:
    """Sort retrieved games for easier comparison."""

    sorted_results = (
        search_results.copy()
    )

    if sort_option == "Highest reviews":
        sorted_results = (
            sorted_results.sort_values(
                by=[
                    "positive_review_percentage",
                    "total_reviews",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
        )

    elif sort_option == "Lowest price":
        sorted_results = (
            sorted_results.sort_values(
                by=[
                    "price_usd",
                    "hybrid_score",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
        )

    elif sort_option == "Newest releases":
        sorted_results = (
            sorted_results.sort_values(
                by=[
                    "release_year",
                    "hybrid_score",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
        )

    return sorted_results.reset_index(
        drop=True
    )


def display_result_toolbar(
    search_results: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    str,
]:
    """Display sorting and CSV download controls."""

    sort_column, download_column = (
        st.columns(
            [2, 1]
        )
    )

    with sort_column:
        sort_option = st.selectbox(
            "Sort recommendations",
            options=[
                "Best match",
                "Highest reviews",
                "Lowest price",
                "Newest releases",
            ],
        )

    sorted_results = sort_results(
        search_results,
        sort_option,
    )

    export_columns = [
        column_name
        for column_name in [
            "name",
            "price_usd",
            "positive_review_percentage",
            "total_reviews",
            "release_year",
            "genres",
            "categories",
            "tags",
            "platforms",
            "steam_store_url",
        ]
        if column_name
        in sorted_results.columns
    ]

    csv_data = (
        sorted_results[
            export_columns
        ]
        .to_csv(
            index=False
        )
        .encode("utf-8-sig")
    )

    with download_column:
        st.write("")
        st.write("")

        st.download_button(
            "⬇ Download results",
            data=csv_data,
            file_name=(
                "gamewise_recommendations.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    return (
        sorted_results,
        sort_option,
    )


def generation_label(
    mode: str,
) -> str:
    """Return a readable summary-mode label."""

    if mode == "openai":
        return (
            "✨ AI-generated grounded summary"
        )

    if mode == "local_fallback":
        return (
            "🧩 Local grounded summary"
        )

    if (
        mode
        == "local_fallback_after_error"
    ):
        return (
            "🧩 Local fallback after model error"
        )

    return (
        "Grounded recommendation summary"
    )


def display_generated_summary(
    query: str,
    search_results: pd.DataFrame,
    filters: dict[str, object],
    requested_concepts: list[str],
) -> None:
    """Generate and display a grounded recommendation summary."""

    with st.spinner(
        "Writing a grounded recommendation..."
    ):
        answer, mode = (
            run_cached_generation(
                query=query,
                search_results=search_results,
                filters=filters,
                requested_concepts=(
                    requested_concepts
                ),
            )
        )

    st.markdown(
        '<div class="section-title">'
        "GameWise summary"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.container(
        border=True
    ):
        st.markdown(
            (
                '<span class="summary-mode">'
                f"{html.escape(generation_label(mode))}"
                "</span>"
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            answer
        )


def display_game_card(
    result_number: int,
    row: pd.Series,
    filters: dict[str, object],
    requested_concepts: list[str],
    developer_mode: bool,
) -> None:
    """Display one polished recommendation card."""

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

    match_label, match_class = (
        get_match_label(
            row
        )
    )

    with st.container(
        border=True
    ):
        title_column, match_column = (
            st.columns(
                [4, 1]
            )
        )

        with title_column:
            st.caption(
                "🏆 Best overall match"
                if result_number == 1
                else (
                    f"Recommendation "
                    f"{result_number}"
                )
            )

            st.markdown(
                f"### {game_name}"
            )

        with match_column:
            st.markdown(
                (
                    f'<span class="match-pill '
                    f'{match_class}">'
                    f"{html.escape(match_label)}"
                    "</span>"
                ),
                unsafe_allow_html=True,
            )

        st.write(
            description
        )

        price_column, review_column, year_column = (
            st.columns(3)
        )

        with price_column:
            st.metric(
                "Price",
                format_price(
                    row
                ),
            )

        with review_column:
            st.metric(
                "Positive reviews",
                format_review_percentage(
                    row
                ),
            )

        with year_column:
            st.metric(
                "Release year",
                format_release_year(
                    row.get(
                        "release_year"
                    )
                ),
            )

        st.markdown(
            f"**Genres:** "
            f"{safe_text(row.get('genres'))}"
        )

        st.markdown(
            f"**Platforms:** "
            f"{safe_text(row.get('platforms'))}"
        )

        tags = get_tags(
            row.get("tags")
        )

        if tags:
            render_chips(
                tags,
                "tag-chip",
            )

        st.markdown(
            "#### Why it fits"
        )

        reasons = build_match_reasons(
            row=row,
            filters=filters,
            requested_concepts=requested_concepts,
        )

        for reason in reasons:
            st.write(
                f"✅ {reason}"
            )

        st.caption(
            format_review_summary(
                row
            )
        )

        steam_column, save_column = (
            st.columns(2)
        )

        with steam_column:
            if steam_url:
                st.link_button(
                    "Open on Steam ↗",
                    steam_url,
                    use_container_width=True,
                )

        with save_column:
            if st.button(
                "♡ Save to shortlist",
                key=(
                    f"save_{result_number}_"
                    f"{game_name}"
                ),
                use_container_width=True,
            ):
                add_to_shortlist(
                    game_name,
                    steam_url,
                )

        if developer_mode:
            with st.expander(
                "🛠 Developer ranking details"
            ):
                fields = [
                    (
                        "Hybrid score",
                        "hybrid_score",
                    ),
                    (
                        "Semantic score",
                        "semantic_score",
                    ),
                    (
                        "Concept score",
                        "concept_score",
                    ),
                ]

                if "play_mode" in filters:
                    fields.append(
                        (
                            "Play-mode score",
                            "play_mode_score",
                        )
                    )

                for (
                    label,
                    column_name,
                ) in fields:
                    value = pd.to_numeric(
                        row.get(
                            column_name
                        ),
                        errors="coerce",
                    )

                    if not pd.isna(
                        value
                    ):
                        st.write(
                            f"{label}:",
                            f"{float(value):.4f}",
                        )

                st.write(
                    "Official categories:",
                    safe_text(
                        row.get(
                            "categories"
                        )
                    ),
                )

                st.caption(
                    "These are internal ranking "
                    "signals, not user ratings."
                )


def display_footer() -> None:
    """Display the dataset disclaimer."""

    st.divider()

    st.caption(
        "GameWise AI uses a local Steam dataset "
        "snapshot. Prices, availability, and store "
        "information may change."
    )


def main() -> None:
    """Run the GameWise application."""

    initialize_session_state()
    apply_page_style()
    display_pending_toast()

    generate_summary, developer_mode = (
        display_sidebar()
    )

    display_hero()

    form_submitted, top_k = (
        display_search_form()
    )

    auto_submit = bool(
        st.session_state.pop(
            "auto_submit",
            False,
        )
    )

    if (
        form_submitted
        or auto_submit
    ):
        query = (
            st.session_state[
                "query_input"
            ].strip()
        )

        if not query:
            st.warning(
                (
                    "Please describe the kind of "
                    "game you are looking for."
                ),
                icon="✍️",
            )

            st.session_state.pop(
                "search_payload",
                None,
            )

        else:
            with st.spinner(
                (
                    "Understanding your request and "
                    "searching the Steam collection..."
                )
            ):
                payload = (
                    run_cached_search(
                        query=query,
                        top_k=top_k,
                    )
                )

            st.session_state[
                "search_payload"
            ] = payload

            st.session_state[
                "submitted_query"
            ] = query

            add_to_search_history(
                query
            )
            st.rerun()

    if (
        "search_payload"
        not in st.session_state
    ):
        display_empty_home()
        display_footer()
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

    title_column, clear_column = (
        st.columns(
            [5, 1]
        )
    )

    with title_column:
        st.markdown(
            "## Your recommendations"
        )

        st.caption(
            f'Based on: "{submitted_query}"'
        )

    with clear_column:
        st.button(
            "Clear search",
            key="clear_current_search",
            use_container_width=True,
            on_click=clear_current_search,
        )

    display_filter_summary(
        filters=extracted_filters,
        requested_concepts=requested_concepts,
        candidate_count=candidate_count,
        result_count=len(
            search_results
        ),
    )

    if clarification_required:
        display_clarification_message(
            candidate_count
        )

        display_footer()
        return

    if search_results.empty:
        display_no_results_message()
        display_footer()
        return

    displayed_results, sort_option = (
        display_result_toolbar(
            search_results
        )
    )

    if generate_summary:
        display_generated_summary(
            query=submitted_query,
            search_results=displayed_results,
            filters=extracted_filters,
            requested_concepts=requested_concepts,
        )

    st.markdown(
        '<div class="section-title">'
        "Recommended games"
        "</div>",
        unsafe_allow_html=True,
    )

    if sort_option != "Best match":
        st.caption(
            f"Results are sorted by "
            f"{sort_option.lower()}."
        )

    for (
        result_number,
        (_, row),
    ) in enumerate(
        displayed_results.iterrows(),
        start=1,
    ):
        display_game_card(
            result_number=result_number,
            row=row,
            filters=extracted_filters,
            requested_concepts=requested_concepts,
            developer_mode=developer_mode,
        )

    display_footer()


if __name__ == "__main__":
    main()