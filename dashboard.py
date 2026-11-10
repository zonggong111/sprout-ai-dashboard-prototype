"""Business-facing Streamlit prototype over Phase 2 / 2.5 CSV outputs."""

from __future__ import annotations

import html
import hashlib
import os
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.dashboard_extensions import (
    aggregate_engagement,
    aggregate_portal_usage,
    aggregate_topic_trend,
    conversation_intensity,
    conversation_quality_trend,
    engagement_funnel,
    entry_point_performance,
    filter_rows as filter_extension_rows,
    language_distribution,
    operational_demo,
    operational_latency_samples,
    portal_activity_heatmap,
    portal_summary,
    portal_user_mix,
    question_input_types,
    time_bucket,
)
from src.feedback_analysis import (parse_feedback_reasons, normalise_feedback_submissions,
                                   feedback_reason_summary, feedback_reason_matrix)
from src.metric_traceability import metric_traceability


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "outputs" / "tables"
EXPECTED_ENTRY_POINTS = [
    "bottom_nav", "ws_quick_replies", "lp_quick_replies", "about_us",
]
OFFICIAL_FEEDBACK_REASONS = [
    "Wrong information", "Missing details", "Not relevant", "Hard to understand",
    "Slow or not working", "Something else",
]
FEEDBACK_SMALL_SAMPLE_THRESHOLD = 5
FEEDBACK_REASON_GUIDANCE = {
    "Wrong information": (
        "Some parents indicated that an answer did not match the information they expected.",
        "Review the linked cases for possible accuracy, context, or expectation gaps.",
    ),
    "Missing details": (
        "This may indicate that some answers were relevant but did not provide enough useful detail.",
        "Check the affected question categories, then inspect whether linked responses need clearer next steps or fuller coverage.",
    ),
    "Not relevant": (
        "This may indicate a mismatch between the parent’s question and the answer provided.",
        "Review the linked questions and responses for intent recognition or relevance issues.",
    ),
    "Hard to understand": (
        "This may indicate that some answers were difficult to follow.",
        "Review the linked responses for language, structure, and clarity.",
    ),
    "Slow or not working": (
        "This may point to an interaction or performance issue rather than response content.",
        "Review the linked cases alongside available technical or performance information.",
    ),
    "Something else": (
        "The selected option does not identify a specific issue on its own.",
        "Prioritise cases with written feedback to understand what the parent experienced.",
    ),
}
FEATURE_INTEREST_MIN_QUERIES = 30
FEATURE_INTEREST_MIN_ESTABLISHED_FEATURES = 3
FEATURE_INTEREST_MIN_QUERIES_PER_FEATURE = 3

PAGES = {
    "Parent Portal Reach": {
        "icon": "◫",
        "group": "OVERVIEW & USAGE",
        "purpose": "Understand synthetic Parent Portal reach, activity patterns, and tracked portal interactions.",
    },
    "Sprout Usage": {
        "icon": "◉",
        "group": "OVERVIEW & USAGE",
        "purpose": "Understand how the synthetic prototype is used, including input method and conversation depth.",
    },
    "Entry & Journey": {
        "icon": "↳",
        "group": "OVERVIEW & USAGE",
        "purpose": "Follow the synthetic journey from Parent Portal entry to an active Sprout conversation.",
    },
    "Topics & Trends": {
        "icon": "◎",
        "group": "PARENT NEEDS",
        "purpose": "Understand which parent needs appear most often and how question volume changes over time.",
    },
    "Repeated Questions": {
        "icon": "≋",
        "group": "PARENT NEEDS",
        "purpose": "Review recurring or rephrased questions that may warrant clearer content or support.",
    },
    "Feedback Overview": {
        "icon": "◇",
        "group": "FEEDBACK & QUALITY",
        "purpose": "Compare positive and negative Sprout feedback, then inspect the reasons behind negative ratings.",
    },
    "Explore Feedback Cases": {
        "icon": "⌕",
        "group": "FEEDBACK & QUALITY",
        "purpose": "Inspect complete synthetic feedback cases while preserving the current filtering workflow.",
    },
    "Operations & AI Quality": {
        "icon": "△",
        "group": "OPERATIONS",
        "purpose": "Explore the proposed service, review, and cost measures needed for operational oversight.",
    },
}

OUTPUT_FILES = {
    "engagement_kpis": "engagement/engagement_kpis.csv",
    "engagement_daily": "engagement/engagement_daily.csv",
    "engagement_by_user_type": "engagement/engagement_by_user_type.csv",
    "entry_point_effectiveness": "engagement/entry_point_effectiveness.csv",
    "feedback_summary": "engagement/feedback_summary.csv",
    "topic_summary": "parent_needs/topic_summary.csv",
    "topic_weekly_trend": "parent_needs/topic_weekly_trend.csv",
    "repeated_question_clusters": "parent_needs/repeated_question_clusters.csv",
    "feature_interest_summary": "parent_needs/feature_interest_summary.csv",
    "priority_topics": "quality/priority_topics.csv",
    "cleaned_ga_events": "cleaned/cleaned_ga_events.csv",
    "cleaned_conversations": "cleaned/cleaned_conversations.csv",
    **{name: f"feedback/{name}.csv" for name in [
        'feedback_overview', 'feedback_reason_summary', 'feedback_reason_by_category',
        'feedback_reason_matrix', 'feedback_cases', 'feedback_case_reasons']},
}


def dashboard_data_root() -> Path:
    """Use project-relative exports; the override supports deployment/testing."""
    configured = os.environ.get("SPROUT_DASHBOARD_DATA_ROOT")
    return Path(configured).expanduser() if configured else DEFAULT_DATA_ROOT


@st.cache_data(show_spinner=False)
def load_output(data_root: str, relative_path: str, file_version: tuple = ()) -> pd.DataFrame:
    """Read an output safely; optional missing or malformed files become empty."""
    path = Path(data_root) / relative_path
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def load_dashboard_tables(data_root: Path) -> dict[str, pd.DataFrame]:
    return {
        name: load_output(str(data_root), relative_path, (
            (data_root / relative_path).stat().st_mtime_ns,
            (data_root / relative_path).stat().st_size,
        ) if (data_root / relative_path).is_file() else ())
        for name, relative_path in OUTPUT_FILES.items()
    }


def _show_missing(message: str) -> None:
    st.info(f"{message} This optional output is not currently available.")


def _values(tables: list[pd.DataFrame], column: str) -> list[str]:
    found: set[str] = set()
    for table in tables:
        if column in table:
            found.update(table[column].dropna().astype(str).str.strip())
    return sorted(value for value in found if value)


def _human_label(value: str) -> str:
    labels = {
        "bottom_nav": "Bottom navigation",
        "ws_quick_replies": "Wellbeing Snapshot quick replies",
        "lp_quick_replies": "Lifestyle Prescription quick replies",
        "about_us": "About Us",
        "pre_lq": "Before Lifestyle Questionnaire",
        "post_lq": "After Lifestyle Questionnaire",
        "post_screening": "After health screening",
        "entry_sessions": "Entered Sprout AI",
        "active_chat_sessions": "Started a conversation",
        "chat_messages_sent": "User messages sent",
    }
    return labels.get(value, value.replace("_", " ").title())


def _filter(
    table: pd.DataFrame,
    *,
    environment: str = "All",
    topics: list[str] | None = None,
    stages: list[str] | None = None,
    entry_points: list[str] | None = None,
    date_range: tuple[date, date] | None = None,
    date_column: str | None = None,
) -> pd.DataFrame:
    result = table.copy()
    if environment != "All" and "beta_or_production" in result:
        result = result.loc[result["beta_or_production"].astype(str).eq(environment)]
    if topics and "primary_topic" in result:
        result = result.loc[result["primary_topic"].isin(topics)]
    if stages and "user_type" in result:
        result = result.loc[result["user_type"].isin(stages)]
    if entry_points and "entry_point" in result:
        result = result.loc[result["entry_point"].isin(entry_points)]
    if date_range and date_column and date_column in result:
        dates = pd.to_datetime(result[date_column], errors="coerce").dt.date
        result = result.loc[dates.between(*date_range)]
    return result.reset_index(drop=True)


def _numeric(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table:
        return pd.Series(0.0, index=table.index, dtype=float)
    return pd.to_numeric(table[column], errors="coerce").fillna(0)


def _safe_rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _format_count(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.0f}"


def _format_number(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.2f}"


def _format_rate(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.1%}"


def _metric_value(kpis: pd.DataFrame, metric: str) -> float | None:
    if kpis.empty or not {"metric", "value"}.issubset(kpis.columns):
        return None
    values = pd.to_numeric(
        kpis.loc[kpis["metric"].eq(metric), "value"], errors="coerce"
    ).dropna()
    return float(values.iloc[0]) if len(values) else None


def _date_control(
    table: pd.DataFrame,
    column: str,
    *,
    key: str,
) -> tuple[date, date] | None:
    if column not in table:
        return None
    dates = pd.to_datetime(table[column], errors="coerce").dropna().dt.date
    if dates.empty:
        return None
    bounds = (dates.min(), dates.max())
    selected = st.date_input(
        "Date range", value=bounds, min_value=bounds[0], max_value=bounds[1], key=key,
    )
    if isinstance(selected, (tuple, list)) and len(selected) == 2:
        return selected[0], selected[1]
    if isinstance(selected, date):
        return selected, selected
    return bounds


def _environment_control(
    tables: list[pd.DataFrame],
    *,
    key: str,
) -> str:
    environments = _values(tables, "beta_or_production")
    if len(environments) <= 1:
        return environments[0] if environments else "All"
    return st.selectbox("Environment", ["All", *environments], key=key)


def _display_table(
    table: pd.DataFrame,
    columns: list[str],
    labels: dict[str, str],
    rate_columns: set[str] | None = None,
) -> None:
    available = [column for column in columns if column in table]
    if not available:
        _show_missing("The requested fields could not be found.")
        return
    view = table[available].copy()
    for column in rate_columns or set():
        if column in view:
            view[column] = pd.to_numeric(view[column], errors="coerce").map(_format_rate)
    st.dataframe(
        view.rename(columns=labels), hide_index=True, width="stretch",
    )


def _story(what_happened: str, why_it_matters: str, what_to_review: str) -> None:
    """Explain a finding without asking the reader to interpret the visual."""
    first, second, third = st.columns(3)
    with first:
        st.markdown("**What happened?**")
        st.write(what_happened)
    with second:
        st.markdown("**Why does it matter?**")
        st.write(why_it_matters)
    with third:
        st.markdown("**What should we review?**")
        st.write(what_to_review)


def _engagement_values(
    filtered_daily: pd.DataFrame,
    source_daily: pd.DataFrame,
    kpis: pd.DataFrame,
) -> dict[str, float | None]:
    if source_daily.empty:
        return {
            "entries": _metric_value(kpis, "chat_sessions"),
            "active": _metric_value(kpis, "active_chat_sessions"),
            "activation": _metric_value(kpis, "chat_activation_rate"),
            "messages": _metric_value(kpis, "chat_messages"),
            "depth": _metric_value(kpis, "messages_per_active_session"),
        }
    if filtered_daily.empty:
        return {name: None for name in ["entries", "active", "activation", "messages", "depth"]}
    entries = float(_numeric(filtered_daily, "entry_sessions").sum())
    entry_sessions = float(_numeric(filtered_daily, "entry_sessions").sum())
    active = float(_numeric(filtered_daily, "active_chat_sessions").sum())
    messages = float(_numeric(filtered_daily, "chat_messages_sent").sum())
    return {
        "entries": entries,
        "active": active,
        "activation": _safe_rate(active, entry_sessions),
        "messages": messages,
        "depth": _safe_rate(messages, active),
    }


def engagement_trend_figure(daily: pd.DataFrame, mode: str = "Both") -> go.Figure:
    measures = {"Started conversations": "active_chat_sessions", "User messages sent": "chat_messages_sent"}
    figure = go.Figure()
    for label, column in measures.items():
        if column in daily and (mode == "Both" or mode == label):
            figure.add_scatter(
                x=daily["event_date"], y=daily[column], name=label,
                mode="lines+markers",
                line=dict(color="#167C80" if label == "Started conversations" else "#B66832", width=2.5),
                marker=dict(size=5),
            )
    figure.update_layout(
        title="Engagement over time" if mode == "Both" else f"{mode} over time",
        height=310, hovermode="x unified", yaxis_title="Count", xaxis_title="Date",
        legend=dict(orientation="h", y=1.12, x=0), margin=dict(l=10, r=10, t=65, b=10),
    )
    figure.update_yaxes(gridcolor="#E9EEEE", rangemode="tozero")
    return figure


def _grouped_item(title: object, metadata: object, *, reasons=(), note: str = "") -> None:
    """Render related text as one unit; escape all data-driven content."""
    chips = "".join(
        f'<span class="reason-chip">{html.escape(str(reason))}</span>'
        for reason in reasons
    )
    st.markdown(
        '<div class="grouped-item">'
        f'<div class="grouped-item-title">{html.escape(str(title))}</div>'
        f'<div class="grouped-item-meta">{html.escape(str(metadata))}</div>'
        + (f'<div class="grouped-item-reasons">{chips}</div>' if chips else "")
        + (f'<div class="grouped-item-meta">{html.escape(note)}</div>' if note else "")
        + '</div>',
        unsafe_allow_html=True,
    )


def _summary_item(label: str, value: object, *, note: str = "") -> None:
    st.markdown(
        '<div class="summary-insight">'
        f'<div class="summary-value">{html.escape(str(value))}</div>'
        f'<div class="summary-descriptor">{html.escape(label)}</div>'
        + (f'<div class="summary-note">{html.escape(note)}</div>' if note else "")
        + '</div>', unsafe_allow_html=True,
    )


def _summary_heading(title: str) -> None:
    st.markdown(f'<h4 class="summary-heading">{html.escape(title)}</h4>', unsafe_allow_html=True)


def _metric_status(text: str = "Illustrative data · pending technical validation") -> None:
    st.markdown(
        f'<span class="metric-status">{html.escape(text)}</span>',
        unsafe_allow_html=True,
    )


PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "zoomIn2d", "zoomOut2d", "autoScale2d",
    ],
}


def _plotly_chart(figure: go.Figure) -> None:
    """Render charts with one unobtrusive, consistent interaction toolbar."""
    figure.update_layout(autosize=True)
    figure.update_xaxes(automargin=True)
    figure.update_yaxes(automargin=True)
    st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG)


def _chart_heading(title: str, subtitle: str = "") -> None:
    """Keep a chart title and its explanation visually attached."""
    st.markdown(f'<h4 class="chart-title">{html.escape(title)}</h4>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="chart-subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def _chart_caption(text: str) -> None:
    """Render a readable card footer that remains inside its card boundary."""
    st.caption(text)


@contextmanager
def _chart_card(title: str, subtitle: str = "", *, key: str):
    """Provide one flexible title / visual / caption card structure."""
    with st.container(border=True, key=f"chart_card_{key}"):
        _chart_heading(title, subtitle)
        yield


@contextmanager
def _summary_card(title: str, *, key: str):
    """Use the shared card shell while preserving summary-card hierarchy."""
    with st.container(border=True, key=f"chart_card_{key}"):
        _summary_heading(title)
        yield


def _render_audience_mix(mix: pd.DataFrame) -> None:
    """Render a compact, accessible comparison without a large label gutter."""
    if mix.empty or not {"user_type", "user_count"}.issubset(mix):
        st.info("No audience mix is available for the selected period.")
        return
    counts: dict[str, float] = {}
    for _, row in mix.iterrows():
        value = pd.to_numeric(row["user_count"], errors="coerce")
        counts[str(row["user_type"]).strip()] = 0 if pd.isna(value) else float(value)
    categories = [("Returning", "Returning users", "#167C80"),
                  ("First-time", "First-time users", "#4F7FA3")]
    total = sum(max(counts.get(source, 0), 0) for _, source, _ in categories)
    definitions_complete = all(source in counts for _, source, _ in categories) and total > 0
    if total <= 0:
        st.info("No users were recorded for the selected period.")
        return
    rows = []
    for label, source, color in categories:
        count = max(counts.get(source, 0), 0)
        share = count / total
        share_text = f'<span class="audience-share">{share:.1%}</span>' if definitions_complete else ""
        rows.append(
            '<div class="audience-row">'
            '<div class="audience-row-header">'
            f'<span class="audience-label">{label}</span>'
            f'<span class="audience-count">{count:,.0f}</span>{share_text}'
            '</div>'
            '<div class="audience-track" aria-hidden="true">'
            f'<span class="audience-fill" style="width:{share:.3%};background:{color}"></span>'
            '</div></div>'
        )
    st.markdown(
        '<div class="audience-mix" role="img" '
        f'aria-label="Returning users {counts.get("Returning users", 0):,.0f}; '
        f'First-time users {counts.get("First-time users", 0):,.0f}">'
        + "".join(rows) + '</div>',
        unsafe_allow_html=True,
    )


def _section_intro(title: str, description: str = "") -> None:
    st.markdown(f"### {title}")
    if description:
        st.markdown(f'<div class="section-description">{html.escape(description)}</div>', unsafe_allow_html=True)


def render_metric_traceability(page: str) -> None:
    """Expose a compact audit trail from business metric to dashboard visual."""
    mapping = metric_traceability(page)
    if mapping.empty:
        return
    with st.expander("Metric traceability", expanded=False):
        st.caption(
            "Stable prototype mapping aligned to the available reviewed metrics crosswalk. "
            "Shown and Partial use bundled synthetic data; Proposed and Requires Validation are not production claims."
        )
        st.dataframe(mapping, hide_index=True, width="stretch")


def _available_date_bounds(tables: dict[str, pd.DataFrame]) -> tuple[date, date] | None:
    """Find the usable synthetic date range across the report's record-level tables."""
    dates: list[pd.Series] = []
    for name, column in [
        ("cleaned_ga_events", "event_timestamp"),
        ("cleaned_conversations", "timestamp"),
        ("engagement_daily", "event_date"),
    ]:
        table = tables.get(name, pd.DataFrame())
        if column in table:
            parsed = pd.to_datetime(table[column], errors="coerce", utc=True).dropna().dt.date
            if not parsed.empty:
                dates.append(parsed)
    if not dates:
        return None
    combined = pd.concat(dates, ignore_index=True)
    return combined.min(), combined.max()


def _reset_global_filters() -> None:
    # Assign widget defaults explicitly. Removing a segmented-control key in a
    # callback can let the browser resubmit its previous selection on the next
    # interaction, leaving the visible control and calculated grain out of sync.
    st.session_state["global_period"] = "Last 90 available days"
    st.session_state["global_granularity"] = "Week"
    for key in ["global_custom_dates", "global_environment"]:
        st.session_state.pop(key, None)


def render_report_header(
    page: str, tables: dict[str, pd.DataFrame], subpage: str | None = None,
) -> dict[str, object]:
    """Render the shared report heading and return one consistent filter contract."""
    st.title(page)
    st.caption(PAGES[page]["purpose"])
    st.markdown(
        '<span class="demo-indicator">SYNTHETIC / PROTOTYPE DATA — DEMONSTRATION ONLY</span>',
        unsafe_allow_html=True,
    )

    bounds = _available_date_bounds(tables)
    environments = _values(
        [tables.get("cleaned_ga_events", pd.DataFrame()),
         tables.get("cleaned_conversations", pd.DataFrame())],
        "beta_or_production",
    )
    if "global_period" not in st.session_state:
        st.session_state["global_period"] = "Last 90 available days"
    if "global_granularity" not in st.session_state:
        st.session_state["global_granularity"] = "Week"
    with st.container(key="global_filter_toolbar"):
        period_col, date_col, grain_col, reset_col = st.columns(
            [1.45, 1.45, 1.35, .72], vertical_alignment="bottom"
        )
        with period_col:
            st.markdown('<div class="filter-label">Period</div>', unsafe_allow_html=True)
            preset = st.selectbox(
                "Period",
                ["Last 7 available days", "Last 30 available days", "Last 90 available days", "Custom"],
                key="global_period",
                label_visibility="collapsed",
            )
        with grain_col:
            st.markdown('<div class="filter-label">Time granularity</div>', unsafe_allow_html=True)
            granularity = st.segmented_control(
                "Time granularity", ["Day", "Week", "Month"],
                key="global_granularity",
                label_visibility="collapsed",
            )
        with reset_col:
            st.markdown(
                '<div class="filter-label filter-label-spacer" aria-hidden="true">Action</div>',
                unsafe_allow_html=True,
            )
            st.button(
                "Reset", key="reset_global_filters", width="stretch",
                on_click=_reset_global_filters,
            )

        if bounds is None:
            date_range = None
            with date_col:
                st.markdown('<div class="filter-label">Selected dates</div>', unsafe_allow_html=True)
                st.markdown('<div class="filter-date">No dated records</div>', unsafe_allow_html=True)
        elif preset == "Custom":
            with date_col:
                st.markdown('<div class="filter-label">Selected dates</div>', unsafe_allow_html=True)
                selected = st.date_input(
                    "Custom date range", value=bounds, min_value=bounds[0],
                    max_value=bounds[1], key="global_custom_dates",
                    label_visibility="collapsed",
                )
            date_range = (
                (selected[0], selected[1])
                if isinstance(selected, (tuple, list)) and len(selected) == 2
                else (selected, selected) if isinstance(selected, date) else bounds
            )
        else:
            days = {"Last 7 available days": 7, "Last 30 available days": 30,
                    "Last 90 available days": 90}[preset]
            date_range = (max(bounds[0], bounds[1] - pd.Timedelta(days=days - 1)), bounds[1])
            with date_col:
                st.markdown('<div class="filter-label">Selected dates</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="filter-date">{date_range[0]:%d %b} – {date_range[1]:%d %b %Y}</div>',
                    unsafe_allow_html=True,
                )
    if len(environments) > 1:
        with st.expander("Advanced filters", expanded=False):
            environment = st.selectbox(
                "Environment", ["All", *environments], key="global_environment",
            )
    else:
        environment = environments[0] if environments else "All"
    return {
        "date_range": date_range,
        "granularity": granularity,
        "environment": environment,
    }


def _set_active_page(page: str) -> None:
    st.session_state["active_report_page"] = page


def render_sidebar_navigation() -> tuple[str, str | None]:
    """Render one compact primary navigation that persists through reruns."""
    if st.session_state.get("active_report_page") not in PAGES:
        st.session_state["active_report_page"] = next(iter(PAGES))
    active = st.session_state["active_report_page"]
    with st.sidebar:
        st.markdown('<div class="workspace-title">PARENT PORTAL &amp;<br>SPROUT AI</div>', unsafe_allow_html=True)
        current_group = None
        for page, config in PAGES.items():
            if config["group"] != current_group:
                current_group = config["group"]
                st.markdown(f'<div class="nav-group">{current_group}</div>', unsafe_allow_html=True)
            st.button(
                f'{config["icon"]}  {page}', key=f"nav_{page}", width="stretch",
                type="primary" if page == active else "secondary",
                on_click=_set_active_page, args=(page,),
            )
        st.divider()
        st.caption("Synthetic prototype · No production data")
    page = st.session_state["active_report_page"]
    return page, None


def render_portal_usage_section(
    events: pd.DataFrame, granularity: str, *, show_funnel: bool = False,
) -> None:
    """Render proposed Parent Portal usage views from the bundled demo GA export."""
    _section_intro("Reach Summary", "How many synthetic users reached the portal and how engagement changed over time.")
    if events.empty:
        _show_missing("Parent Portal event data could not be loaded.")
        return

    values = portal_summary(events)
    cards = st.container(key="portal_metrics").columns(5)
    items = [
        ("Unique PP users", _format_count(values["unique_users"]), "Distinct anonymised demo users in the selected period."),
        ("First-time users", _format_count(values["first_time_users"]), "Users with a recorded first-login flag in the selected period."),
        ("Returning users", _format_count(values["returning_users"]), "Selected-period users without a recorded first-login flag."),
        ("Page views", _format_count(values["page_views"]), "Tracked Parent Portal and Sprout page-view events."),
        ("Average engagement per session", f"{values['average_engagement_seconds']:.1f}s", "Total tracked engagement seconds divided by distinct GA sessions; validate against the final GA definition."),
    ]
    for card, (label, value, help_text) in zip(cards, items):
        card.metric(label, value, help=help_text)

    trend = aggregate_portal_usage(events, granularity)
    mix = portal_user_mix(events)
    trend_col, mix_col = st.container(key="portal_trends").columns([7, 3], gap="medium")
    with trend_col:
        with _chart_card(
            f"{granularity} usage trend",
            "Compare reach and page-view activity over the selected period.",
            key="portal_usage_trend",
        ):
            mode = st.segmented_control(
                "Usage measure",
                ["Both", "Unique users", "Page views"],
                default="Both", key="portal_usage_mode", label_visibility="collapsed",
            )
            if not trend.empty:
                figure = go.Figure()
                for label, column, color in [
                    ("Unique users", "unique_users", "#167C80"),
                    ("Page views", "page_views", "#4F7FA3"),
                ]:
                    if mode in ("Both", label):
                        figure.add_scatter(
                            x=trend["period"], y=trend[column], name=label,
                            mode="lines+markers", line=dict(width=2.5, color=color), marker=dict(size=6),
                        )
                figure.update_layout(
                    height=270, hovermode="x unified", xaxis_title=granularity, yaxis_title="Count",
                    legend=dict(orientation="h", y=1.12, x=0), margin=dict(l=5, r=10, t=40, b=5),
                )
                figure.update_yaxes(gridcolor="#E9EEEE", rangemode="tozero")
                _plotly_chart(figure)
                _chart_caption(f"Unique users are deduplicated within each {granularity.lower()} and must not be added across periods.")
            else:
                _show_missing("Portal usage trend could not be calculated.")
    with mix_col:
        with _chart_card(
            "New vs returning users", "Selected-period audience mix.",
            key="audience_mix",
        ):
            _render_audience_mix(mix)
            _chart_caption("Uses the synthetic first-login flag; the production GA definition remains to be validated.")

    _section_intro("Engagement & Access Patterns", "When parents use the portal and how they progress into Sprout AI.")
    pattern_container = st.container(key="portal_patterns")
    if show_funnel:
        heatmap_col, funnel_col = pattern_container.columns([6.5, 3.5], gap="medium")
    else:
        heatmap_col, funnel_col = pattern_container, None
    with heatmap_col:
        with _chart_card(
            "Parent Portal page views by weekday and hour",
            "Recorded page-view events are converted from UTC to Singapore Time before grouping.",
            key="portal_activity_pattern",
        ):
            activity_view = st.segmented_control(
                "Activity view", ["Heatmap", "Hourly trend"], default="Heatmap",
                key="portal_activity_view", label_visibility="collapsed",
            )
            heatmap = portal_activity_heatmap(events)
            if activity_view == "Hourly trend":
                hourly = heatmap.sum(axis=0, min_count=1).rename_axis("hour").reset_index(name="page_views")
                figure = px.bar(
                    hourly, x="hour", y="page_views", text="page_views",
                    labels={"hour": "Hour of day (SGT)", "page_views": "Page views"},
                    color_discrete_sequence=["#167C80"],
                )
                figure.update_traces(texttemplate="%{text:.0f}", textposition="outside", cliponaxis=False)
                figure.update_xaxes(tickmode="array", tickvals=list(range(24)), ticktext=[f"{h:02d}" for h in range(24)])
                figure.update_yaxes(gridcolor="#E9EEEE", rangemode="tozero")
                figure.update_layout(height=300, margin=dict(l=5, r=10, t=20, b=5))
            else:
                recorded_values = heatmap.stack()
                maximum = float(recorded_values.max()) if not recorded_values.empty else 0
                figure = go.Figure(go.Heatmap(
                    z=heatmap.values, x=list(range(24)), y=heatmap.index.tolist(),
                    zmin=0, zmax=max(maximum, 1),
                    colorscale=[[0, "#F1F3F4"], [.01, "#DCEDEE"], [.4, "#79AFB2"], [1, "#0D666B"]],
                    colorbar=dict(title="Page views", thickness=12, len=.82),
                    customdata=heatmap.values,
                    hovertemplate="%{y}, %{x}:00 SGT<br>%{customdata:.0f} page views<extra></extra>",
                    xgap=1, ygap=1,
                ))
                if heatmap.isna().any().any():
                    missing = heatmap.isna().astype(float).where(heatmap.isna())
                    figure.add_trace(go.Heatmap(
                        z=missing.values, x=list(range(24)), y=heatmap.index.tolist(),
                        zmin=0, zmax=1, colorscale=[[0, "#F1DDC7"], [1, "#F1DDC7"]],
                        showscale=False, xgap=1, ygap=1,
                        hovertemplate="%{y}, %{x}:00 SGT<br>Missing or unavailable data<extra></extra>",
                    ))
                annotations = []
                for row_index, weekday in enumerate(heatmap.index):
                    for hour, value in enumerate(heatmap.loc[weekday]):
                        if pd.notna(value) and value > 0:
                            annotations.append(dict(
                                x=hour, y=weekday, text=f"<b>{int(value)}</b>" if value == maximum else str(int(value)),
                                showarrow=False, font=dict(size=10, color="white" if value >= maximum * .5 else "#234247"),
                            ))
                figure.update_layout(
                    height=300, annotations=annotations,
                    xaxis=dict(title="Hour of day (SGT)", tickmode="array", tickvals=list(range(24)),
                               ticktext=[f"{h:02d}" for h in range(24)], tickfont=dict(size=10)),
                    yaxis=dict(title="", autorange="reversed"),
                    margin=dict(l=5, r=10, t=20, b=5),
                    shapes=[dict(type="line", x0=x, x1=x, y0=-.5, y1=6.5, line=dict(color="#AAB6B8", width=1.2)) for x in [5.5, 11.5, 17.5]],
                )
            _plotly_chart(figure)
            invalid_timestamps = int(heatmap.attrs.get("invalid_timestamp_count", 0))
            missing_note = (
                "Amber cells indicate missing or unavailable data. "
                if heatmap.isna().any().any() else
                "No weekday-hour cells are missing in this selection. "
            )
            invalid_note = (
                f"{invalid_timestamps} page-view record(s) with invalid timestamps were excluded. "
                if invalid_timestamps else ""
            )
            _chart_caption(
                "Neutral pale cells are recorded zero page views; they are not login counts or unique users. "
                + missing_note + invalid_note + "Hover for exact SGT values."
            )
    if funnel_col is not None:
      with funnel_col:
        with _chart_card(
            "User engagement funnel",
            "How many distinct users progress from portal use to a message.",
            key="portal_funnel",
        ):
            funnel = engagement_funnel(events)
            if not funnel.empty:
                figure = go.Figure(go.Funnel(
                    y=funnel["stage"], x=funnel["users"],
                    textinfo="value+percent initial", marker=dict(color=["#B8D5D7", "#5F9EA2", "#167C80"]),
                ))
                figure.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=5))
                _plotly_chart(figure)
                _chart_caption("All stages use distinct synthetic users; percentages use Parent Portal users as the denominator.")
            else:
                _show_missing("The proposed engagement funnel could not be calculated.")

    _section_intro(
        "Tracked Portal Interactions",
        "Which synthetic Parent Portal actions were recorded in the bundled event export.",
    )
    action_col, pillar_col = st.container(key="portal_actions").columns([6, 4], gap="medium")
    with action_col:
        with _chart_card(
            "CTA and journey actions", "Tracked action events in the current selection.",
            key="portal_actions",
        ):
            action_labels = {
                "child_card_select": "Select child profile",
                "view_health_plan": "View Health Plan",
                "view_personalised_recommendations": "View recommendations",
                "complete_lq": "Complete lifestyle questionnaire",
                "resource_card_click": "Open resource card",
            }
            actions = (
                events.loc[events.get("event_name", pd.Series(index=events.index)).isin(action_labels)]
                .groupby("event_name").size().rename("clicks").reset_index()
            )
            if not actions.empty:
                actions["action"] = actions["event_name"].map(action_labels)
                actions = actions.sort_values("clicks")
                figure = px.bar(
                    actions, x="clicks", y="action", orientation="h", text="clicks",
                    labels={"clicks": "Tracked actions", "action": ""},
                    color_discrete_sequence=["#167C80"],
                )
                figure.update_traces(textposition="outside", cliponaxis=False)
                figure.update_xaxes(range=[0, max(actions["clicks"].max() * 1.2, 1)], gridcolor="#E9EEEE")
                figure.update_layout(height=275, margin=dict(l=5, r=35, t=10, b=5))
                _plotly_chart(figure)
                _chart_caption("Actions use explicit synthetic event names; they are not inferred from page views.")
            else:
                _show_missing("Tracked portal actions could not be calculated.")
    with pillar_col:
        with _chart_card(
            "Resource-card clicks by pillar", "Recorded resource clicks grouped by synthetic content pillar.",
            key="portal_resource_pillars",
        ):
            resource_mask = events.get("event_name", pd.Series(index=events.index)).eq("resource_card_click")
            resource = events.loc[resource_mask].copy()
            if not resource.empty and "content_pillar" in resource:
                pillars = (
                    resource["content_pillar"].fillna("Unclassified").value_counts()
                    .rename_axis("pillar").reset_index(name="clicks").sort_values("clicks")
                )
                figure = px.bar(
                    pillars, x="clicks", y="pillar", orientation="h", text="clicks",
                    labels={"clicks": "Clicks", "pillar": ""}, color_discrete_sequence=["#4F7FA3"],
                )
                figure.update_traces(textposition="outside", cliponaxis=False)
                figure.update_layout(height=275, margin=dict(l=5, r=35, t=10, b=5))
                figure.update_xaxes(gridcolor="#E9EEEE", rangemode="tozero")
                _plotly_chart(figure)
            else:
                _show_missing("Resource-pillar data could not be loaded.")

    with st.expander("Metrics awaiting source validation", expanded=False):
        st.dataframe(pd.DataFrame([
            {"Metric": "Usage by access channel", "Status": "Requires Validation", "Reason": "No verified access-channel field is present; device category is not used as a substitute."},
            {"Metric": "Scroll depth", "Status": "Requires Validation", "Reason": "No eligible scroll event or threshold is present in the bundled synthetic export."},
        ]), hide_index=True, width="stretch")


def render_engagement_page(
    tables: dict[str, pd.DataFrame], filters: dict[str, object], *, view: str = "journey",
) -> None:
    daily = tables["engagement_daily"]
    ga_events = tables["cleaned_ga_events"]
    environment = str(filters["environment"])
    dates = filters["date_range"]
    granularity = str(filters["granularity"])
    filtered = _filter(daily, environment=environment, date_range=dates, date_column="event_date")
    filtered_events = filter_extension_rows(
        ga_events, environment=environment, date_range=dates, date_column="event_timestamp",
    )
    if view == "portal":
        render_portal_usage_section(filtered_events, granularity)
        return
    _section_intro("Parent Portal to Sprout Funnel", "How distinct synthetic users progress from portal use to a user message.")
    funnel = engagement_funnel(filtered_events)
    with _chart_card("User engagement funnel", "All stages use distinct synthetic users.", key="journey_funnel"):
        if not funnel.empty:
            figure = go.Figure(go.Funnel(
                y=funnel["stage"], x=funnel["users"], textinfo="value+percent initial",
                marker=dict(color=["#B8D5D7", "#5F9EA2", "#167C80"]),
            ))
            figure.update_layout(height=255, margin=dict(l=10, r=10, t=15, b=5))
            _plotly_chart(figure)
            _chart_caption("Percentages use synthetic Parent Portal users as the denominator; identity linkage requires production validation.")
        else:
            _show_missing("The proposed engagement funnel could not be calculated.")
    values = _engagement_values(filtered, daily, tables["engagement_kpis"])
    _section_intro("Sprout Conversation Journey", "How portal sessions progress from entry to active conversation.")
    journey = st.container(key="journey_metrics")
    conversion_group, activity_group = journey.columns([3, 2], gap="medium")
    with conversion_group, st.container(key="journey_conversion"):
        st.markdown('<div class="metric-group-label">Conversion journey</div>', unsafe_allow_html=True)
        conversion_cards = st.columns(3)
        for card, label, value, help_text in zip(conversion_cards, [
            "Entered Sprout AI", "Conversations Started", "Conversion Rate",
        ], [
            _format_count(values["entries"]), _format_count(values["active"]),
            _format_rate(values["activation"]),
        ], [
            "Distinct sessions reaching the chat experience.",
            "Distinct entered sessions sending at least one user message.",
            "Conversations started divided by sessions that entered Sprout AI.",
        ]):
            card.metric(label, value, help=help_text)
    with activity_group, st.container(key="journey_activity"):
        st.markdown('<div class="metric-group-label">Conversation activity</div>', unsafe_allow_html=True)
        activity_cards = st.columns(2)
        for card, label, value, help_text in zip(activity_cards, [
            "Messages Sent", "Messages per Conversation",
        ], [
            _format_count(values["messages"]), _format_number(values["depth"]),
        ], [
            "Tracked user-message events; this is activity volume, not another funnel stage.",
            "User messages divided by active conversation sessions; this is an average, not a funnel stage.",
        ]):
            card.metric(label, value, help=help_text)
    _section_intro(f"Conversation Activity by {granularity}", "Compare conversation starts with user-message volume.")
    chart, notes = st.container(key="activity_section").columns([7, 3], gap="medium")
    with chart:
        with _chart_card(
            "Conversation activity trend",
            "Toggle between both measures or focus on a single series.",
            key="conversation_activity",
        ):
            mode = st.segmented_control("Trend measure", ["Both", "Started conversations", "User messages sent"],
                                        default="Both", key="engagement_trend_mode", label_visibility="collapsed")
            if not filtered.empty and "event_date" in filtered:
                trend = aggregate_engagement(filtered, granularity).rename(columns={"period": "event_date"})
                _plotly_chart(engagement_trend_figure(trend, mode or "Both"))
            else:
                trend = pd.DataFrame()
                _show_missing("Activity data could not be loaded.")
    with notes:
        with _summary_card("Activity summary", key="conversation_activity_summary"):
            if not trend.empty:
                _summary_item("Recorded period", f"{len(trend)} {granularity.lower()} periods")
                for field, label in [("active_chat_sessions", "Peak started conversations"), ("chat_messages_sent", "Peak user messages")]:
                    if field in trend:
                        _summary_item(label, _format_count(trend[field].max()))
                if "active_chat_sessions" in trend:
                    peak = trend.sort_values(["active_chat_sessions", "event_date"], ascending=[False, True]).iloc[0]
                    _summary_item("Most active date · conversation starts", str(peak["event_date"])[:10])
                _chart_caption("Peaks describe recorded usage; release timing and entry routes may help explain changes.")

    _section_intro("How Parents Enter Sprout AI", "Compare usage and entry-to-conversation conversion across access routes.")
    entry = entry_point_performance(filtered_events)
    if entry.empty:
        entry = tables["entry_point_effectiveness"]
    chart, notes = st.container(key="entry_section").columns([7, 3], gap="medium")
    with chart:
        with _chart_card(
            "Entry-point performance",
            "Entered sessions and started conversations by access route.",
            key="entry_performance",
        ):
            with st.expander("Entry-point filter", expanded=False):
                selected = st.multiselect("Entry point", _values([entry], "entry_point"), format_func=_human_label, key="engagement_entry_points")
            entry = _filter(entry, entry_points=selected)
            required = {"entry_point", "entry_sessions", "active_chat_sessions", "activation_rate"}
            if not entry.empty and required.issubset(entry):
                entry = entry.sort_values("entry_sessions").copy()
                entry["Entry point"] = entry["entry_point"].map(_human_label)
                fig = go.Figure()
                for field, label, color in [("entry_sessions", "Entered Sprout AI", "#A7C8CA"),
                                            ("active_chat_sessions", "Started a conversation", "#167C80")]:
                    fig.add_bar(y=entry["Entry point"], x=entry[field], orientation="h", name=label,
                                marker_color=color, text=entry[field], textposition="auto", textangle=0,
                                customdata=entry["activation_rate"],
                                hovertemplate="%{y}<br>" + label + ": %{x}<br>Entry-to-conversation conversion: %{customdata:.1%}<extra></extra>")
                fig.update_layout(height=300, barmode="group", xaxis_title="Sessions",
                                  legend=dict(orientation="h", y=1.15), margin=dict(l=5,r=10,t=40,b=10))
                fig.update_xaxes(gridcolor="#E9EEEE")
                _plotly_chart(fig)
            else:
                _show_missing("Entry-point data could not be loaded.")
    with notes:
        with _summary_card("Entry-point summary", key="entry_summary"):
            if not entry.empty and required.issubset(entry):
                for label, field, highest in [("Highest entry volume","entry_sessions",True),
                                              ("Highest conversion","activation_rate",True),
                                              ("Lowest conversion","activation_rate",False)]:
                    ordered = entry.dropna(subset=[field]).sort_values([field,"entry_point"], ascending=[not highest,True])
                    if ordered.empty:
                        continue
                    row = ordered.iloc[0]
                    suffix = f" · {row[field]:.1%}" if field == "activation_rate" else f" · {int(row[field])} sessions"
                    note = (f"{int(row['active_chat_sessions'])} of {int(row['entry_sessions'])} entered sessions started a conversation."
                            if label == "Lowest conversion" else "")
                    _summary_item(label, _human_label(row["entry_point"]) + suffix, note=note)
                _chart_caption("Different routes can reflect different visitor intentions. Ties use alphabetical order.")


def render_parent_needs_page(tables: dict[str, pd.DataFrame], filters: dict[str, object]) -> None:
    environment = str(filters["environment"])
    dates = filters["date_range"]
    granularity = str(filters["granularity"])
    conversations = filter_extension_rows(
        tables["cleaned_conversations"], environment=environment,
        date_range=dates, date_column="timestamp",
    )
    if not conversations.empty and "primary_topic" in conversations:
        summary = (
            conversations["primary_topic"].fillna("Unclassified").value_counts()
            .rename_axis("primary_topic").reset_index(name="query_count")
        )
    else:
        summary = _filter(tables["topic_summary"], environment=environment)
    repeated = _filter(tables["repeated_question_clusters"], environment=environment)
    valid_repeated = not repeated.empty and {"representative_question","number_of_similar_questions"}.issubset(repeated)
    if valid_repeated:
        repeated = repeated.sort_values("number_of_similar_questions", ascending=False)
    _section_intro("Main Topics & Parent Needs", "What parents ask about most in the current synthetic selection.")
    chart = st.container(key="needs_top")
    with chart:
        with _chart_card(
            "Topic distribution",
            "Questions ranked by volume in the current filter selection.",
            key="topic_distribution",
        ):
            if not summary.empty and {"primary_topic","query_count"}.issubset(summary):
                # Aggregate environments before ranking the complete current selection.
                summary = summary.groupby("primary_topic", as_index=False)["query_count"].sum()
                summary["percentage_of_queries"] = summary["query_count"] / summary["query_count"].sum()
                top = summary.sort_values("query_count").tail(10).copy()
                top["Share label"] = top.apply(lambda r: f"{int(r['query_count'])} · {r['percentage_of_queries']:.1%}", axis=1)
                figure = px.bar(top, x="query_count", y="primary_topic", orientation="h", text="Share label",
                    labels={"query_count":"Question count", "primary_topic":"Question Category"}, color_discrete_sequence=["#167C80"])
                figure.update_traces(textposition="outside", cliponaxis=False)
                figure.update_xaxes(range=[0, top["query_count"].max()*1.28], gridcolor="#E9EEEE")
                figure.update_layout(height=max(260, min(360, 28*len(top)+60)), margin=dict(l=5,r=35,t=10,b=10))
                _plotly_chart(figure)
                _chart_caption("Counts and shares use questions in the current global filter selection.")
            else:
                _show_missing("Question Category data could not be loaded.")
    _section_intro(f"Parent Needs Trend by {granularity}", "How selected question categories change over time.")
    chart, notes = st.container(key="needs_trend").columns([7, 3], gap="medium")
    with chart:
        with _chart_card(
            "Question volume trend",
            "Choose the leading topics or one category for a focused view.",
            key="question_volume_trend",
        ):
            choice = st.selectbox("Topic trend", ["Top 3 topics", *_values([summary], "primary_topic")], key="needs_trend_topic", label_visibility="collapsed")
            selected = summary.nlargest(3, "query_count")["primary_topic"].tolist() if choice == "Top 3 topics" and "query_count" in summary else [choice]
            if not conversations.empty and {"timestamp", "primary_topic"}.issubset(conversations):
                trend = aggregate_topic_trend(conversations, selected, granularity)
                figure = go.Figure()
                colors = ["#167C80", "#B66832", "#6653A1"]
                date_hover = "%b %Y" if granularity == "Month" else "%d %b %Y"
                for topic, color in zip(selected, colors):
                    topic_trend = trend.loc[trend["primary_topic"].eq(topic)]
                    figure.add_scatter(
                        x=topic_trend["period"], y=topic_trend["query_count"],
                        name=topic, mode="lines+markers", connectgaps=False,
                        line=dict(width=2.3, color=color), marker=dict(size=6),
                        hovertemplate=(
                            f"{html.escape(topic)}<br>%{{x|{date_hover}}}<br>"
                            "%{y:.0f} questions<extra></extra>"
                        ),
                    )
                missing_periods = (
                    trend.loc[trend["period_status"].eq("No records"), "period"]
                    .drop_duplicates().sort_values()
                )
                if not missing_periods.empty:
                    figure.add_scatter(
                        x=missing_periods, y=[0] * len(missing_periods), name="No records",
                        mode="markers", marker=dict(symbol="x", size=8, color="#929A9E"),
                        hovertemplate=f"%{{x|{date_hover}}}<br>No conversation records<extra></extra>",
                    )
                periods = sorted(pd.to_datetime(trend["period"].dropna().unique()))
                if granularity == "Month":
                    figure.update_xaxes(
                        tickmode="array", tickvals=periods,
                        ticktext=[pd.Timestamp(period).strftime("%b %Y") for period in periods],
                        title="Month",
                    )
                elif granularity == "Week":
                    figure.update_xaxes(
                        tickmode="array", tickvals=periods,
                        ticktext=[pd.Timestamp(period).strftime("%d %b") for period in periods],
                        title="Week starting Monday",
                    )
                else:
                    figure.update_xaxes(tickformat="%d %b", title="Date")
                figure.update_yaxes(title="Questions", gridcolor="#E9EEEE", rangemode="tozero", dtick=1)
                figure.update_layout(
                    height=320, legend=dict(orientation="h", y=1.22), legend_title_text="",
                    margin=dict(l=5, r=10, t=65, b=10), hovermode="x unified",
                )
                _plotly_chart(figure)
                _chart_caption(
                    "A zero means the category had no questions in a period covered by other conversation records. "
                    "A gap and × marker mean the entire period has no conversation records; values are not interpolated."
                )
            else:
                trend = pd.DataFrame()
                _show_missing("Question trend data could not be loaded.")
    with notes:
        with _summary_card("Trend summary", key="question_trend_summary"):
            _summary_item("Current selection", choice)
            if not trend.empty:
                recorded_trend = trend.loc[trend["period_status"].eq("Recorded")].copy()
                periods = sorted(recorded_trend["period"].unique())
                latest = recorded_trend.loc[recorded_trend["period"].eq(periods[-1])]
                leader = latest.sort_values(["query_count","primary_topic"], ascending=[False,True]).iloc[0]
                _summary_item("Highest recent volume", f"{leader['primary_topic']} · {int(leader['query_count'])}",
                              note=f"Latest recorded {granularity.lower()}: {str(periods[-1])[:10]}")
                if len(periods)>1:
                    pivot = recorded_trend.pivot(index="period", columns="primary_topic", values="query_count")
                    delta = (pivot.loc[periods[-1]] - pivot.loc[periods[-2]]).sort_values(ascending=False)
                    if delta.iloc[0]>0:
                        _summary_item("Largest recent increase", f"{delta.index[0]} · +{int(delta.iloc[0])} questions")
                    else:
                        _summary_item("Recent change", "No selected category increased")
                    _chart_caption(f"Comparison uses the last two recorded {granularity.lower()} periods; these are descriptive counts, not a significance test.")
    matrix = tables["feedback_reason_matrix"]
    if not matrix.empty:
        with st.expander("Illustrative reviewed issues by topic", expanded=False):
            st.caption(
                "Synthetic negative-feedback cases by parent-need topic and selected issue type. "
                "This is a reviewed-case example, not a population failure rate."
            )
            _render_feedback_reason_matrix(matrix)
    features = _filter(tables["feature_interest_summary"], environment=environment)
    if not features.empty and {"related_feature","query_count"}.issubset(features):
        with st.expander("Additional analysis", expanded=False):
            st.markdown("#### Related Parent Portal feature interest")
            st.caption("Small feature counts are exploratory and do not establish importance.")
            _display_table(features, ["related_feature","query_count"], {"related_feature":"Related feature","query_count":"Questions"})


def render_repeated_questions_page(
    tables: dict[str, pd.DataFrame], filters: dict[str, object],
) -> None:
    """Render recurring-question clusters as a focused, non-duplicated page."""
    repeated = _filter(
        tables["repeated_question_clusters"], environment=str(filters["environment"]),
    )
    required = {"representative_question", "number_of_similar_questions"}
    if repeated.empty or not required.issubset(repeated):
        _show_missing("Repeated-question data could not be loaded.")
        return
    repeated = repeated.sort_values("number_of_similar_questions", ascending=False).reset_index(drop=True)
    total_variants = int(pd.to_numeric(repeated["number_of_similar_questions"], errors="coerce").fillna(0).sum())
    cards = st.container(key="repeated_metrics").columns(3)
    cards[0].metric("Repeated-question clusters", f"{len(repeated):,}", help="Groups of similar synthetic questions identified by the existing analysis logic.")
    cards[1].metric("Similar question variants", f"{total_variants:,}", help="Total questions represented across the displayed clusters.")
    cards[2].metric("Largest cluster", f"{int(repeated['number_of_similar_questions'].max()):,}", help="Questions in the largest synthetic similarity cluster.")
    _section_intro("Repeated or Rephrased Questions", "Recurring needs that may warrant clearer content, prompts, or support.")
    chart, detail = st.container(key="repeated_analysis").columns([6, 4], gap="medium")
    with chart:
        with _chart_card("Largest repeated-question clusters", "Ranked by the number of similar synthetic questions.", key="repeated_clusters"):
            top = repeated.head(10).sort_values("number_of_similar_questions")
            figure = px.bar(
                top, x="number_of_similar_questions", y="representative_question",
                orientation="h", text="number_of_similar_questions",
                labels={"number_of_similar_questions": "Similar questions", "representative_question": ""},
                color_discrete_sequence=["#167C80"],
            )
            figure.update_traces(textposition="outside", cliponaxis=False)
            figure.update_layout(height=max(300, 42 * len(top)), margin=dict(l=5, r=35, t=10, b=5))
            figure.update_xaxes(gridcolor="#E9EEEE", rangemode="tozero")
            _plotly_chart(figure)
    with detail:
        with _summary_card("Cluster review", key="repeated_cluster_review"):
            for _, row in repeated.head(4).iterrows():
                _grouped_item(
                    row["representative_question"],
                    f"{row.get('primary_topic', 'Unclassified')} · {int(row['number_of_similar_questions'])} similar questions",
                )
    with st.expander("View all clusters and example phrasings", expanded=False):
        _display_table(repeated, ["representative_question", "primary_topic", "number_of_similar_questions"], {
            "representative_question": "Representative question", "primary_topic": "Question Category",
            "number_of_similar_questions": "Similar questions",
        })
        example_columns = [column for column in repeated if column.startswith("example_question_")]
        if example_columns:
            st.markdown("#### Example phrasings")
            for _, row in repeated.head(8).iterrows():
                phrases = dict.fromkeys(str(row[column]) for column in example_columns if pd.notna(row[column]))
                _grouped_item(row["representative_question"], "\n".join(phrases))


def _feature_interest_is_sufficient(features: pd.DataFrame) -> bool:
    if features.empty or not {"query_count", "related_feature"}.issubset(features.columns):
        return False
    counts = pd.to_numeric(features["query_count"], errors="coerce").fillna(0)
    return bool(
        counts.sum() >= FEATURE_INTEREST_MIN_QUERIES
        and counts.ge(FEATURE_INTEREST_MIN_QUERIES_PER_FEATURE).sum()
        >= FEATURE_INTEREST_MIN_ESTABLISHED_FEATURES
    )


def _quality_filters(priority: pd.DataFrame) -> dict:
    environments = _values([priority], "beta_or_production")
    if len(environments) <= 1:
        return {"environment": environments[0] if environments else "All"}
    with st.expander("Filter environment", expanded=False):
        environment = st.selectbox(
            "Environment", ["All", *environments], key="quality_environment",
        )
    return {"environment": environment}


def feedback_overview_counts(feedback: pd.DataFrame) -> dict[str, float]:
    """Calculate the GA feedback mix with all submissions as denominator."""
    if feedback.empty or not {"feedback_type", "feedback_count"}.issubset(feedback.columns):
        return {"total": 0, "positive": 0, "negative": 0, "general": 0, "negative_share": 0}
    labels = feedback["feedback_type"].fillna("unknown").astype(str).str.strip().str.lower()
    counts = pd.to_numeric(feedback["feedback_count"], errors="coerce").fillna(0)
    grouped = counts.groupby(labels).sum()
    total = float(counts.sum())
    negative = float(grouped.get("negative", 0))
    return {
        "total": total,
        "positive": float(grouped.get("positive", 0)),
        "negative": negative,
        "general": float(grouped.get("general", 0)),
        "negative_share": negative / total if total else 0,
    }


def negative_feedback_by_topic(priority: pd.DataFrame) -> pd.DataFrame:
    """Return visible topic feedback counts, denominators, rates, and evidence status."""
    required = {"primary_topic", "feedback_observations", "negative_feedback_rate"}
    if priority.empty or not required.issubset(priority.columns):
        return pd.DataFrame()
    output = priority.copy()
    output["total_feedback_count"] = pd.to_numeric(
        output["feedback_observations"], errors="coerce"
    ).fillna(0).astype(int)
    rates = pd.to_numeric(output["negative_feedback_rate"], errors="coerce")
    output["negative_feedback_count"] = (
        rates.fillna(0) * output["total_feedback_count"]
    ).round().astype(int)
    output["negative_feedback_share"] = rates
    minimum = pd.to_numeric(
        output.get("minimum_quality_observations", 5), errors="coerce"
    )
    if not isinstance(minimum, pd.Series):
        minimum = pd.Series(float(minimum), index=output.index)
    output["evidence_note"] = "Sufficient for configured rate comparison"
    output.loc[
        output["total_feedback_count"].lt(minimum.fillna(5)), "evidence_note"
    ] = "Insufficient evidence for a conclusion — continue monitoring."
    output = output.loc[output["total_feedback_count"].gt(0)].copy()
    output["negative_feedback_evidence"] = output.apply(
        lambda row: (
            f"{row['negative_feedback_count']} of {row['total_feedback_count']} negative "
            f"({_format_rate(row['negative_feedback_share'])})"
        ),
        axis=1,
    )
    return output


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _count_from_rate(row: pd.Series, rate: str, denominator: str) -> tuple[int, int]:
    total = pd.to_numeric(pd.Series([row.get(denominator)]), errors="coerce").fillna(0).iloc[0]
    value = pd.to_numeric(pd.Series([row.get(rate)]), errors="coerce").fillna(0).iloc[0]
    return int(round(value * total)), int(total)


def _priority_explanation(row: pd.Series) -> tuple[str, str, str]:
    unsuccessful, success_total = _count_from_rate(row, "unsuccessful_rate", "success_observations")
    negative, feedback_total = _count_from_rate(row, "negative_feedback_rate", "feedback_observations")
    evidence = (
        f"{int(row.get('query_count', 0))} questions; "
        f"{unsuccessful} of {success_total} evaluated responses unsuccessful; "
        f"{negative} of {feedback_total} feedback submissions negative."
    )
    if _is_true(row.get("insufficient_quality_evidence_flag")):
        reason = (
            "Meaningful question demand, but insufficient evidence for a conclusion — continue monitoring."
            if _is_true(row.get("high_demand_flag"))
            else "Insufficient evidence for a conclusion — continue monitoring."
        )
        next_step = "Collect more evaluated responses and feedback before drawing a conclusion."
    else:
        reasons = []
        if _is_true(row.get("high_demand_flag")):
            reasons.append("meaningful question demand")
        if _is_true(row.get("unsuccessful_review_flag")):
            reasons.append(f"{unsuccessful} of {success_total} evaluated responses were unsuccessful")
        if _is_true(row.get("negative_feedback_review_flag")):
            reasons.append(f"{negative} of {feedback_total} feedback submissions were negative")
        reason = "; combined with ".join(reasons) if reasons else "No configured demand or response-quality review threshold was met."
        if unsuccessful and negative:
            next_step = "Review both unsuccessful-response and negative-feedback cases to identify the underlying issue."
        elif unsuccessful:
            next_step = "Review unsuccessful cases for knowledge coverage, retrieval, clarity, or another cause."
        elif negative:
            next_step = "Review the negative-feedback cases to understand the reason for dissatisfaction."
        elif _is_true(row.get("high_demand_flag")):
            next_step = "Maintain coverage for this common need and continue monitoring response evidence."
        else:
            next_step = "Continue monitoring."
    return evidence, reason[0].upper() + reason[1:], next_step


def _negative_feedback_cases(
    conversations: pd.DataFrame,
    *,
    environment: str,
    topics: list[str] | None = None,
) -> pd.DataFrame:
    cases = _filter(conversations, environment=environment, topics=topics)
    feedback_column = (
        "user_feedback" if "user_feedback" in cases
        else "feedback_type" if "feedback_type" in cases else None
    )
    if feedback_column is None:
        return pd.DataFrame()
    return cases.loc[
        cases[feedback_column].astype(str).str.strip().str.lower().eq("negative")
    ].reset_index(drop=True)


def affected_question_categories(
    submissions: pd.DataFrame, reason: str,
) -> pd.DataFrame:
    """Rank question categories by negative submissions selecting a reason."""
    columns = [
        "Question Category", "Cases", "Category Negative Cases", "Share",
        "Count / denominator", "Sample Guidance",
    ]
    if submissions.empty or "primary_topic" not in submissions or "_feedback_reasons" not in submissions:
        return pd.DataFrame(columns=columns)
    working = submissions.copy()
    working["Question Category"] = (
        working["primary_topic"].fillna("Unclassified").astype(str).str.strip()
        .replace("", "Unclassified")
    )
    working["_selected_reason"] = working["_feedback_reasons"].map(
        lambda values: reason in values
    )
    grouped = working.groupby("Question Category", as_index=False).agg(
        Cases=("_selected_reason", "sum"),
        **{"Category Negative Cases": ("_submission_key", "nunique")},
    )
    grouped = grouped.loc[grouped["Cases"].gt(0)].copy()
    grouped["Cases"] = grouped["Cases"].astype(int)
    grouped["Category Negative Cases"] = grouped["Category Negative Cases"].astype(int)
    grouped["Share"] = grouped["Cases"] / grouped["Category Negative Cases"]
    grouped["Count / denominator"] = grouped.apply(
        lambda row: (
            f"{row['Cases']} of {row['Category Negative Cases']} negative · "
            f"{row['Share']:.1%}"
        ),
        axis=1,
    )
    grouped["Sample Guidance"] = grouped["Category Negative Cases"].map(
        lambda count: "Small sample — interpret cautiously."
        if count < FEEDBACK_SMALL_SAMPLE_THRESHOLD else ""
    )
    return grouped[columns].sort_values(
        ["Cases", "Category Negative Cases", "Question Category"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def response_issues_to_review(priority: pd.DataFrame) -> pd.DataFrame:
    """Translate visible response evidence into a plain-language review table."""
    required = {
        "primary_topic", "query_count", "unsuccessful_rate", "success_observations",
        "negative_feedback_rate", "feedback_observations",
    }
    columns = [
        "Question Category", "Total Questions", "Unsuccessful Responses",
        "Negative Feedback", "Main Signal",
    ]
    if priority.empty or not required.issubset(priority.columns):
        return pd.DataFrame(columns=columns)
    rows: list[dict] = []
    for _, row in priority.iterrows():
        unsuccessful, evaluated = _count_from_rate(
            row, "unsuccessful_rate", "success_observations"
        )
        negative, feedback_total = _count_from_rate(
            row, "negative_feedback_rate", "feedback_observations"
        )
        if not unsuccessful and not negative:
            continue
        total_questions = int(
            pd.to_numeric(pd.Series([row.get("query_count")]), errors="coerce")
            .fillna(0).iloc[0]
        )
        unsuccessful_rate = unsuccessful / evaluated if evaluated else 0
        negative_rate = negative / feedback_total if feedback_total else 0
        unsuccessful_text = f"{unsuccessful} of {evaluated} unsuccessful"
        if unsuccessful and evaluated:
            unsuccessful_text += f" · {unsuccessful_rate:.1%}"
        negative_text = f"{negative} of {feedback_total} feedback negative"
        if negative and feedback_total:
            negative_text += f" · {negative_rate:.1%}"
        if unsuccessful and negative:
            signal = "Unsuccessful responses + negative feedback"
        elif unsuccessful and _is_true(row.get("high_demand_flag")):
            signal = "High volume + some unsuccessful responses"
        elif unsuccessful:
            signal = "Some unsuccessful responses"
        elif negative and _is_true(row.get("high_demand_flag")):
            signal = "High volume + negative feedback"
        else:
            signal = "Negative feedback"
        if (evaluated and evaluated < FEEDBACK_SMALL_SAMPLE_THRESHOLD) or (
            feedback_total and feedback_total < FEEDBACK_SMALL_SAMPLE_THRESHOLD
        ):
            signal += " — small sample"
        rows.append({
            "Question Category": row["primary_topic"],
            "Total Questions": total_questions,
            "Unsuccessful Responses": unsuccessful_text,
            "Negative Feedback": negative_text,
            "Main Signal": signal,
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["Total Questions", "Question Category"], ascending=[False, True]
    ).reset_index(drop=True)


def filter_feedback_submissions(
    submissions: pd.DataFrame,
    *,
    topics: list[str] | None = None,
    reasons: list[str] | None = None,
    date_range: tuple[date, date] | None = None,
    has_written_feedback: bool = False,
) -> pd.DataFrame:
    """Apply the business-facing case filters without changing stored data."""
    filtered = submissions.copy()
    if topics and "primary_topic" in filtered:
        filtered = filtered.loc[filtered["primary_topic"].isin(topics)]
    if reasons and "_feedback_reasons" in filtered:
        selected = set(reasons)
        filtered = filtered.loc[
            filtered["_feedback_reasons"].map(lambda values: bool(selected.intersection(values)))
        ]
    if date_range and "timestamp" in filtered:
        dates = pd.to_datetime(filtered["timestamp"], errors="coerce").dt.date
        filtered = filtered.loc[dates.between(*date_range)]
    if has_written_feedback and "_has_written_feedback" in filtered:
        filtered = filtered.loc[filtered["_has_written_feedback"]]
    return filtered.reset_index(drop=True)


def _case_has_response_linkage(cases: pd.DataFrame) -> bool:
    if cases.empty or "ai_response" not in cases:
        return False
    response_present = cases["ai_response"].fillna("").astype(str).str.strip().ne("")
    identifiers = [
        column for column in ["feedback_id", "response_id", "message_id", "conversation_id", "query_id"]
        if column in cases
    ]
    identifier_present = (
        cases[identifiers].fillna("").astype(str).apply(lambda column: column.str.strip().ne("")).any(axis=1)
        if identifiers else pd.Series(False, index=cases.index)
    )
    return bool((response_present & identifier_present).any())


def _render_feedback_overview(feedback: pd.DataFrame) -> None:
    if feedback.empty or "total_feedback_submissions" not in feedback:
        _show_missing("Feedback overview could not be loaded.")
        return
    row = feedback.iloc[0]
    counts = {"total": row["total_feedback_submissions"], "positive": row["positive_count"],
              "negative": row["negative_count"], "general": row["general_count"],
              "negative_share": row["negative_feedback_share"]}
    if not counts["total"]:
        _show_missing("Overall GA feedback data could not be loaded.")
        return
    cards = st.container(key="feedback_metrics").columns(5)
    for card, label, value in zip(
        cards,
        ["Total feedback submissions", "Positive", "Negative", "General", "Negative share"],
        [
            _format_count(counts["total"]), _format_count(counts["positive"]),
            _format_count(counts["negative"]), _format_count(counts["general"]),
            _format_rate(counts["negative_share"]),
        ],
    ):
        card.metric(label, value)
    _section_intro("Feedback Balance", "Positive, negative, and general feedback shown together before issue detail.")
    balance = pd.DataFrame({
        "feedback": ["Positive", "Negative", "General"],
        "submissions": [counts["positive"], counts["negative"], counts["general"]],
    })
    with _chart_card(
        "Submitted feedback mix", "All recorded feedback types in the current synthetic selection.",
        key="feedback_balance",
    ):
        figure = px.bar(
            balance, x="feedback", y="submissions", text="submissions", color="feedback",
            labels={"feedback": "Feedback type", "submissions": "Submissions"},
            color_discrete_map={"Positive": "#167C80", "Negative": "#B66832", "General": "#7A8790"},
        )
        figure.update_layout(height=245, showlegend=False, margin=dict(l=5, r=10, t=10, b=5))
        figure.update_yaxes(gridcolor="#E9EEEE", rangemode="tozero", nticks=6)
        _plotly_chart(figure)
        _chart_caption(
            f"Negative feedback accounts for {counts['negative_share']:.1%} of all recorded feedback; "
            "the reason analysis below uses negative submissions only."
        )


def _render_feedback_reasons(exported: pd.DataFrame) -> None:
    summary = exported.rename(columns={"feedback_reason": "Reason", "negative_case_count": "Count",
                                        "percentage_of_negative_cases": "Percentage of negative feedback"}).copy()
    if summary.empty:
        st.info("Detailed dissatisfaction reason data is not available in the current dataset.")
        return
    with _chart_card(
        "Most common reasons for negative feedback",
        "Reasons selected by parents on negative-feedback submissions.",
        key="feedback_reasons",
    ):
        summary = summary.sort_values(["Count", "Reason"], ascending=[False, True])
        denominator = int(summary["negative_feedback_submissions"].iloc[0])
        summary["Count / denominator"] = summary["Count"].map(lambda n: f"{n} of {denominator} negative feedback submissions")
        summary["Display label"] = summary.apply(
            lambda row: f"{int(row['Count'])} cases · {float(row['Percentage of negative feedback']):.1%}",
            axis=1,
        )
        figure = px.bar(
            summary.sort_values("Count"), x="Count", y="Reason", orientation="h",
            text="Display label", custom_data=["Percentage of negative feedback", "Count / denominator"],
            labels={"Count": "Negative feedback submissions", "Reason": "Feedback reason"},
            color_discrete_sequence=["#356F76"],
        )
        figure.update_traces(
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>%{customdata[1]}<br>Share of negative submissions: %{customdata[0]:.1%}<extra></extra>",
        )
        figure.update_layout(
            height=max(250, min(320, 34 * len(summary) + 80)), showlegend=False,
            margin=dict(l=10, r=100, t=10, b=5),
        )
        figure.update_xaxes(rangemode="tozero", dtick=1)
        _plotly_chart(figure)
        _chart_caption("Parents can select more than one reason, so percentages may not sum to 100%.")
def _render_feedback_reason_matrix(exported: pd.DataFrame) -> None:
    st.markdown("### Question Category × Feedback Reason")
    if exported.empty or not {"question_category", "feedback_reason", "case_count"}.issubset(exported):
        st.info("Question Category × Feedback Reason data is not available in the current dataset.")
        return
    matrix = exported.pivot(index="question_category", columns="feedback_reason", values="case_count").fillna(0)
    matrix = matrix.reindex(columns=OFFICIAL_FEEDBACK_REASONS, fill_value=0)
    category_totals = exported.groupby("question_category")["category_negative_cases"].first()
    customdata = [
        [
            [int(category_totals.get(category, 0)), value / category_totals.get(category, 1)]
            for value in matrix.loc[category].tolist()
        ]
        for category in matrix.index
    ]
    figure = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        text=matrix.values,
        customdata=customdata,
        texttemplate="%{text}",
        hovertemplate=(
            "Question Category: %{y}<br>Feedback Reason: %{x}<br>"
            "Cases selecting reason: %{z}<br>Total negative cases in category: "
            "%{customdata[0]}<br>Share: %{customdata[1]:.1%}<extra></extra>"
        ),
        colorscale=[[0, "#F2F7F7"], [1, "#2F6B73"]],
        colorbar=dict(title="Submissions"),
    ))
    figure.update_layout(
        height=max(320, 42 * len(matrix.index) + 150),
        xaxis_title="Feedback Reason", yaxis_title="Question Category",
        margin=dict(l=10, r=20, t=20, b=10),
    )
    _plotly_chart(figure)
    st.caption(
        "Each cell counts negative feedback submissions that selected that reason. "
        "A submission can contribute to more than one reason column."
    )


def _written_feedback_text(value: object) -> str:
    if value is not None and pd.notna(value) and str(value).strip():
        return str(value).strip()
    return "No additional written feedback provided."


def _case_navigation_label(row: pd.Series) -> str:
    topic = str(row.get("primary_topic", "Unclassified")).strip() or "Unclassified"
    question = str(row.get("user_query", "Question unavailable")).strip() or "Question unavailable"
    if len(question) > 72:
        question = question[:69].rstrip() + "…"
    reasons = row.get("_feedback_reasons", ())
    reason_text = " + ".join(reasons) if reasons else "Reason not recorded"
    return f"{topic} · {reason_text} — “{question}”"


def _render_full_feedback_case(row: pd.Series) -> None:
    with st.container(border=True):
        question = str(row.get("user_query", "Question text is not available."))
        _grouped_item(
            f"{question[:150]}{'…' if len(question)>150 else ''}",
            row.get("primary_topic", "Unclassified"),
            reasons=row.get("_feedback_reasons", ()),
            note="Has written feedback" if str(row.get("feedback_comment", "")).strip() and pd.notna(row.get("feedback_comment")) else "No written comment",
        )
        with st.expander("View full case", expanded=False):
            st.markdown("#### Parent asked")
            st.write(f'“{row.get("user_query", "Question text is not available.")}”')
            st.markdown("#### Sprout answered")
            if "SYNTHETIC" in str(row.get("data_notice", "")).upper() or "SYNTHETIC" in str(row.get("ai_response", "")).upper():
                st.caption("SYNTHETIC RESPONSE — DEMONSTRATION ONLY")
            with st.expander("Show full response", expanded=True):
                response = row.get("ai_response")
                if pd.notna(response) and str(response).strip():
                    st.write(response)
                else:
                    st.info("The full response is unavailable in this source.")
            st.markdown("#### Parent feedback")
            st.markdown("**Parent selected**")
            reasons = row.get("_feedback_reasons", ())
            if reasons:
                chips = "".join(
                    f'<span class="reason-chip">{html.escape(reason)}</span>' for reason in reasons
                )
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.caption("No feedback reason was recorded.")
            st.markdown("**Tell us more (Optional)**")
            st.write(f'“{_written_feedback_text(row.get("feedback_comment"))}”')
            metadata = [
                ("Timestamp", "timestamp"), ("Journey stage", "user_type"),
                ("Entry point", "entry_point"), ("Anonymised session", "session_id"),
                ("Conversation ID", "conversation_id"), ("Response ID", "response_id"),
                ("Message ID", "message_id"), ("Query ID", "query_id"),
                ("Feedback ID", "feedback_id"),
            ]
            with st.expander("Technical details", expanded=False):
                shown = False
                for label, column in metadata:
                    value = row.get(column)
                    if column in row and pd.notna(value) and str(value).strip():
                        st.markdown(f"**{label}:** {html.escape(str(value))}")
                        shown = True
                if not shown:
                    st.caption("No additional context or identifiers are available.")


def _select_feedback_reason(reason: str) -> None:
    st.session_state["explorer_reason"] = reason
    st.session_state["explorer_category"] = None


def _select_feedback_category(category: str | None) -> None:
    st.session_state["explorer_category"] = category


def _render_feedback_explorer(tables: dict[str, pd.DataFrame], cases: pd.DataFrame) -> None:
    summary = tables["feedback_reason_summary"]
    categories = tables["feedback_reason_by_category"]
    if cases.empty or not {"feedback_reason", "negative_case_count"}.issubset(summary):
        st.info("Feedback cases and reasons are unavailable in this source.")
        return
    if not {"feedback_reason", "question_category", "case_count"}.issubset(categories):
        st.info("Question Category detail is unavailable in this source.")
        return

    # Reset defaults on source changes, while preserving selections during normal clicks.
    fingerprint = hashlib.sha256(
        (summary.to_json() + categories.to_json() + cases.to_json()).encode()
    ).hexdigest()
    counts = summary.groupby("feedback_reason")["negative_case_count"].sum()
    ranked_reasons = sorted(OFFICIAL_FEEDBACK_REASONS, key=lambda reason: (-counts.get(reason, 0), reason))
    if st.session_state.get("explorer_data_version") != fingerprint:
        st.session_state["explorer_data_version"] = fingerprint
        _select_feedback_reason(ranked_reasons[0])
    reason = st.session_state.get("explorer_reason", ranked_reasons[0])
    if reason not in ranked_reasons:
        reason = ranked_reasons[0]
        _select_feedback_reason(reason)

    ranked_categories = categories.loc[categories["feedback_reason"].eq(reason)].copy()
    ranked_categories["question_category"] = ranked_categories["question_category"].fillna("Unclassified")
    ranked_categories = ranked_categories.groupby("question_category", as_index=False)["case_count"].sum()
    ranked_categories = ranked_categories.loc[ranked_categories["case_count"].gt(0)].sort_values(
        ["case_count", "question_category"], ascending=[False, True])
    available_categories = ranked_categories["question_category"].tolist()
    category = st.session_state.get("explorer_category")
    # Empty string means the deliberate All categories choice; None requests a default.
    if category is None or (category != "" and category not in available_categories):
        category = available_categories[0] if available_categories else ""
        st.session_state["explorer_category"] = category

    panel = st.container(key="feedback_explorer")
    navigation, content = panel.columns([1, 2.5], gap="large")
    with navigation:
        st.markdown("**Feedback Reasons**")
        for option in ranked_reasons:
            count = int(counts.get(option, 0))
            st.button(
                f"{option} · {count}", key=f"explorer_reason_{option}",
                type="primary" if option == reason else "secondary", width="stretch",
                on_click=_select_feedback_reason, args=(option,),
            )
        st.markdown("**Question Categories**")
        st.button(
            f"All categories · {int(ranked_categories['case_count'].sum())}",
            key="explorer_category_all", width="stretch",
            type="primary" if category == "" else "secondary",
            on_click=_select_feedback_category, args=("",),
        )
        for _, item in ranked_categories.iterrows():
            label = item["question_category"]
            st.button(
                f"{label} · {int(item['case_count'])}", key=f"explorer_category_{label}",
                width="stretch", type="primary" if label == category else "secondary",
                on_click=_select_feedback_category, args=(label,),
            )
    with content:
        st.markdown(f"#### {reason} → {category or 'All categories'}")
        matching = filter_feedback_submissions(
            cases, reasons=[reason], topics=[category] if category else None,
        )
        st.markdown(f"**{len(matching)} matching cases**")
        if matching.empty:
            st.info("No cases match this selection.")
        else:
            if len(matching) < FEEDBACK_SMALL_SAMPLE_THRESHOLD:
                st.caption(f"Based on {len(matching)} matching cases. Small sample — interpret cautiously.")
            for _, row in matching.iterrows():
                _render_full_feedback_case(row)
    if not tables["feedback_reason_matrix"].empty:
        with st.expander("Advanced matrix view", expanded=False):
            _render_feedback_reason_matrix(tables["feedback_reason_matrix"])


def render_conversation_signals(tables: dict[str, pd.DataFrame], filters: dict[str, object]) -> None:
    """Extend conversation monitoring without changing the case-review workflow."""
    conversations = tables["cleaned_conversations"]
    ga_events = tables["cleaned_ga_events"]
    environment = str(filters["environment"])
    dates = filters["date_range"]
    granularity = str(filters["granularity"])
    selected = filter_extension_rows(
        conversations, environment=environment, date_range=dates, date_column="timestamp",
    )
    selected_events = filter_extension_rows(
        ga_events, environment=environment, date_range=dates, date_column="event_timestamp",
    )
    if selected.empty:
        _show_missing("Conversation signal data could not be loaded.")
        return

    session_key = "conversation_id" if "conversation_id" in selected else "session_id"
    sessions = selected[session_key].dropna().nunique() if session_key in selected else 0
    users = selected["user_id_anonymised"].dropna().nunique() if "user_id_anonymised" in selected else 0
    questions = len(selected)
    avg_questions = questions / sessions if sessions else None
    avg_per_user = questions / users if users else None
    grouped = selected.dropna(subset=[session_key]).groupby(session_key).size() if session_key in selected else pd.Series(dtype=int)
    follow_up_sessions = int(grouped.gt(1).sum())
    cards = st.container(key="conversation_signal_metrics").columns(5)
    for card, label, value, help_text in [
        (cards[0], "Unique Sprout users", _format_count(float(users)), "Distinct anonymised synthetic users with a recorded Sprout question."),
        (cards[1], "Conversation sessions", _format_count(float(sessions)), "Distinct synthetic conversation identifiers."),
        (cards[2], "User messages", _format_count(float(questions)), "Rows in the synthetic conversation log; not GA message events."),
        (cards[3], "Messages per user", _format_number(avg_per_user), "Recorded user messages divided by distinct active Sprout users."),
        (cards[4], "Messages per session", _format_number(avg_questions), "Recorded user messages divided by identified conversation sessions."),
    ]:
        card.metric(label, value, help=help_text)

    input_types = question_input_types(selected_events)
    intensity = conversation_intensity(selected)
    _section_intro("Conversation Characteristics", "How parents submit questions and how much interaction each conversation contains.")
    input_col, intensity_col = st.container(key="conversation_characteristics").columns(2, gap="medium")
    with input_col:
        with _chart_card(
            "User question input types",
            "Quick replies and free-text messages in the selected synthetic GA events.",
            key="question_input_types",
        ):
            if not input_types.empty:
                figure = px.bar(
                    input_types, x="input_type", y="message_count", text="message_count",
                    labels={"input_type": "Input type", "message_count": "User messages"},
                    color_discrete_sequence=["#167C80"],
                )
                figure.update_layout(height=300, margin=dict(l=5, r=10, t=10, b=10))
                figure.update_yaxes(gridcolor="#E9EEEE", rangemode="tozero")
                _plotly_chart(figure)
                _chart_caption("GA-shaped message triggers distinguish quick replies from free-text input in the demo data.")
            else:
                _show_missing("Question input-type data could not be loaded.")
    with intensity_col:
        with _chart_card(
            "Conversation intensity",
            "Conversations grouped by their recorded number of user questions.",
            key="conversation_intensity",
        ):
            if not intensity.empty:
                figure = px.bar(
                    intensity, x="intensity", y="conversations", text="conversations",
                    labels={"intensity": "Questions in conversation", "conversations": "Conversations"},
                    color_discrete_sequence=["#5A9A9D"],
                )
                figure.update_layout(height=300, margin=dict(l=5, r=10, t=10, b=10))
                figure.update_yaxes(gridcolor="#E9EEEE", rangemode="tozero")
                _plotly_chart(figure)
                _chart_caption("Conversation counts remain separate from GA website sessions and message events.")
            else:
                _show_missing("Conversation-intensity data could not be calculated.")

    languages = language_distribution(selected)
    if not languages.empty:
        _section_intro("Language Distribution", "Aggregated language labels supplied by the synthetic conversation source.")
        figure = px.bar(
            languages, x="question_count", y="language", orientation="h", text="question_count",
            labels={"question_count": "Questions", "language": "Language"},
            color_discrete_sequence=["#167C80"],
        )
        _plotly_chart(figure)
    with st.expander("Proposed metrics and source validation", expanded=False):
        st.dataframe(pd.DataFrame([
            {"Metric": "Quick-reply usage by pillar", "Status": "Partial", "Reason": "Quick-reply input is recorded, but the message events do not carry a verified pillar."},
            {"Metric": "Language distribution", "Status": "Requires Validation", "Reason": "No validated language field is present in the bundled conversation table."},
            {"Metric": "Translation request volume", "Status": "Requires Use-case Validation", "Reason": "No translation feature or request type is assumed."},
            {"Metric": "Chat activity by hour and weekday", "Status": "Requires Validation", "Reason": "The existing heatmap is Parent Portal page views, not Sprout conversation starts."},
        ]), hide_index=True, width="stretch")
    return

    trend = conversation_quality_trend(selected, granularity)
    _section_intro("Response Quality & Feedback", "Which recorded signals may deserve closer review and how users rated responses.")
    quality_col, feedback_col = st.container(key="conversation_quality").columns(2, gap="medium")
    with quality_col:
        with _chart_card(
            "Recorded evaluation flags",
            "Observed synthetic fallback, unresolved, and guardrail indicators.",
            key="evaluation_flags",
        ):
            if not trend.empty:
                figure = go.Figure()
                for label, column, color in [
                    ("Fallback / unable to answer", "fallback_count", "#B66832"),
                    ("Potentially unresolved", "unsuccessful_count", "#8A5A44"),
                    ("Guardrail triggered", "guardrail_count", "#6653A1"),
                ]:
                    figure.add_scatter(
                        x=trend["period"], y=trend[column], name=label,
                        mode="lines+markers", line=dict(width=2, color=color), marker=dict(size=5),
                    )
                figure.update_layout(
                    height=285, hovermode="x unified",
                    xaxis_title=granularity, yaxis_title="Questions",
                    legend=dict(orientation="h", y=1.18, x=0), margin=dict(l=5, r=10, t=55, b=5),
                )
                figure.update_yaxes(gridcolor="#E9EEEE", rangemode="tozero")
                _plotly_chart(figure)
            else:
                _show_missing("Evaluation trends could not be calculated.")
    with feedback_col:
        with _chart_card(
            "Submitted response ratings",
            "Positive and negative ratings recorded in the conversation layer.",
            key="response_ratings",
        ):
            if not trend.empty:
                figure = go.Figure()
                for label, column, color in [
                    ("Positive", "positive_feedback", "#167C80"),
                    ("Negative", "negative_feedback", "#B66832"),
                ]:
                    figure.add_bar(x=trend["period"], y=trend[column], name=label, marker_color=color)
                figure.update_layout(
                    height=285, barmode="group",
                    xaxis_title=granularity, yaxis_title="Ratings",
                    legend=dict(orientation="h", y=1.14, x=0), margin=dict(l=5, r=10, t=50, b=5),
                )
                figure.update_yaxes(gridcolor="#E9EEEE", rangemode="tozero")
                _plotly_chart(figure)
            else:
                _show_missing("Feedback trends could not be calculated.")
    st.caption("Fallback, guardrail, success, and feedback flags come from the separate synthetic conversation layer—not GA4.")

    languages = language_distribution(selected)
    if not languages.empty:
        st.markdown("#### Language distribution")
        figure = px.bar(
            languages, x="question_count", y="language", orientation="h", text="question_count",
            labels={"question_count": "Questions", "language": "Language"},
            color_discrete_sequence=["#167C80"],
        )
        _plotly_chart(figure)
    with st.expander("Metrics awaiting source validation", expanded=False):
        st.dataframe(pd.DataFrame([
            {"Metric": "Language distribution", "Required source": "Validated language field or approved classifier output", "Status": "Pending data validation"},
            {"Metric": "PII detection categories", "Required source": "Aggregated safety or DLP events; no sensitive text", "Status": "Pending data validation"},
            {"Metric": "LLM-as-a-Judge outcomes", "Required source": "Separate evaluation pipeline and calibration results", "Status": "Pending technical validation"},
            {"Metric": "PP application feedback", "Required source": "PP application-feedback service", "Status": "Source not currently ingested"},
        ]), hide_index=True, width="stretch")


def render_quality_page(
    tables: dict[str, pd.DataFrame], filters: dict[str, object],
    subpage: str = "Feedback Overview",
) -> None:
    cases = tables["feedback_cases"].rename(columns={
        "question_category": "primary_topic", "journey_stage": "user_type",
        "anonymised_session_id": "session_id"}).copy()
    cases = filter_extension_rows(
        cases, environment=str(filters["environment"]),
        date_range=filters["date_range"], date_column="timestamp",
    )
    links = tables["feedback_case_reasons"]
    if not cases.empty and "feedback_id" in cases:
        reason_map = links.drop_duplicates(["feedback_id", "feedback_reason"]).groupby("feedback_id")["feedback_reason"].agg(list).to_dict() if {"feedback_id", "feedback_reason"}.issubset(links) else {}
        cases["_feedback_reasons"] = cases["feedback_id"].map(lambda value: reason_map.get(value, []))
        cases["_submission_key"] = cases["feedback_id"].astype(str)
        if "primary_topic" not in cases:
            cases["primary_topic"] = "Unclassified"
        cases["primary_topic"] = cases["primary_topic"].fillna("Unclassified")
        cases["_has_written_feedback"] = cases.get("feedback_comment", pd.Series("", index=cases.index)).fillna("").astype(str).str.strip().ne("")
    submissions = cases

    if subpage == "Feedback Overview":
        events = filter_extension_rows(
            tables["cleaned_ga_events"], environment=str(filters["environment"]),
            date_range=filters["date_range"], date_column="event_timestamp",
        )
        labels = events.get("feedback_type", pd.Series(dtype="string")).fillna("unknown").astype(str).str.lower()
        counts = labels[labels.isin(["positive", "negative", "general"])].value_counts()
        feedback_overview = pd.DataFrame([{
            "total_feedback_submissions": int(counts.sum()),
            "positive_count": int(counts.get("positive", 0)),
            "negative_count": int(counts.get("negative", 0)),
            "general_count": int(counts.get("general", 0)),
            "negative_feedback_share": float(counts.get("negative", 0) / counts.sum()) if counts.sum() else 0,
        }]) if counts.sum() else tables["feedback_overview"]
        _render_feedback_overview(feedback_overview)
        if not submissions.empty:
            st.caption(
                f"Detailed reasons below use {len(submissions):,} negative feedback submissions "
                "from the separate conversation/backend layer."
            )
        filtered_reason_summary = feedback_reason_summary(submissions)
        if not filtered_reason_summary.empty:
            filtered_reason_summary = filtered_reason_summary.rename(columns={
                "Reason": "feedback_reason",
                "Count": "negative_case_count",
                "Percentage of negative feedback": "percentage_of_negative_cases",
            })
            filtered_reason_summary["negative_feedback_submissions"] = len(submissions)
        _render_feedback_reasons(
            filtered_reason_summary if not filtered_reason_summary.empty
            else tables["feedback_reason_summary"]
        )
    elif subpage == "Conversation Signals":
        render_conversation_signals(tables, filters)
    elif subpage == "Explore Feedback Cases":
        _render_feedback_explorer(tables, submissions)


def render_operations_page(tables: dict[str, pd.DataFrame], filters: dict[str, object]) -> None:
    """Render decision-useful proposed operations metrics for business stakeholders."""
    conversations = tables["cleaned_conversations"]
    environment = str(filters["environment"])
    dates = filters["date_range"]
    granularity = str(filters["granularity"])
    selected = filter_extension_rows(
        conversations, environment=environment, date_range=dates, date_column="timestamp",
    )
    operations = operational_demo(selected)
    _metric_status("Proposed Operational Metrics")
    st.caption("Illustrative data. Actual monitoring sources are pending validation.")
    if operations.empty:
        _show_missing("Illustrative operations data could not be generated from the selected demo period.")
        return

    _section_intro(
        "Service Reliability",
        "Are the proposed service indicators stable enough to support a dependable experience?",
    )
    availability = float(pd.to_numeric(operations["availability"], errors="coerce").mean())
    error_rate = float(pd.to_numeric(operations["api_error_rate"], errors="coerce").mean())
    latency_samples = operational_latency_samples(selected)
    latency_values = pd.to_numeric(latency_samples["latency_ms"], errors="coerce").dropna()
    average_ms = float(latency_values.mean()) if not latency_values.empty else 0
    p50_ms = float(latency_values.quantile(.50)) if not latency_values.empty else 0
    p95_ms = float(latency_values.quantile(.95)) if not latency_values.empty else 0
    cards = st.container(key="operations_metrics").columns(3)
    for card, label, value, help_text in [
        (cards[0], "System Availability", _format_rate(availability),
         "Share of monitored time the service was available. Helps indicate continuity. Requires uptime or synthetic-monitoring records."),
        (cards[1], "API Error Rate", _format_rate(error_rate),
         "Share of monitored API requests returning an error. Helps identify technical instability. Requires API gateway or application logs."),
        (cards[2], "P95 Response Time", f"{p95_ms / 1000:.2f}s",
         "95% of measured technical requests completed within this time. Helps identify slower requests. Requires application performance traces."),
    ]:
        card.metric(label, value, help=help_text)

    latency_col, alerts_col = st.container(key="operations_reliability").columns([7, 3], gap="medium")
    with latency_col:
        with _chart_card(
            f"Technical response time by {granularity.lower()}",
            "P95 is the primary slow-request indicator; P50 and average provide supporting context.",
            key="technical_response_time",
        ):
            latency_samples["period"] = time_bucket(latency_samples["timestamp"], granularity)
            latency_trend = latency_samples.groupby("period", as_index=False).agg(
                average_response_ms=("latency_ms", "mean"),
                p50_response_ms=("latency_ms", "median"),
                p95_response_ms=("latency_ms", lambda values: values.quantile(.95)),
            )
            figure = go.Figure()
            for label, column, color, width in [
                ("P95", "p95_response_ms", "#B66832", 3),
                ("Average", "average_response_ms", "#4F7FA3", 1.8),
                ("P50", "p50_response_ms", "#167C80", 1.8),
            ]:
                figure.add_scatter(
                    x=latency_trend["period"], y=latency_trend[column] / 1000, name=label,
                    mode="lines+markers", line=dict(width=width, color=color), marker=dict(size=4),
                )
            figure.update_layout(
                height=270, hovermode="x unified", xaxis_title=granularity, yaxis_title="Seconds",
                legend=dict(orientation="h", y=1.12, x=0), margin=dict(l=5, r=10, t=40, b=5),
            )
            figure.update_yaxes(gridcolor="#E9EEEE", rangemode="tozero")
            _plotly_chart(figure)
            _chart_caption("These synthetic technical request timings do not represent complete user-perceived AI response time.")
    with alerts_col:
        with _chart_card(
            "Alert summary", "Proposed alerts by severity in the selected period.",
            key="alert_summary",
        ):
            alert_totals = pd.DataFrame({
                "severity": ["Critical", "Warning"],
                "alerts": [int(operations["critical_alerts"].sum()), int(operations["warning_alerts"].sum())],
            })
            figure = px.bar(
                alert_totals, x="alerts", y="severity", orientation="h", text="alerts",
                labels={"severity": "", "alerts": "Alerts"}, color="severity",
                color_discrete_map={"Critical": "#984B4B", "Warning": "#B97835"},
            )
            maximum_alerts = max(int(alert_totals["alerts"].max()), 1)
            figure.update_traces(textposition="outside", cliponaxis=False)
            figure.update_xaxes(range=[0, maximum_alerts * 1.25], gridcolor="#E9EEEE", dtick=max(1, maximum_alerts // 3))
            figure.update_layout(height=235, showlegend=False, margin=dict(l=5, r=35, t=20, b=5))
            _plotly_chart(figure)
            _chart_caption("Illustrative counts; monitoring thresholds, severity rules, and ownership are not yet defined.")

    with st.expander("Supporting reliability metrics", expanded=False):
        support = st.columns(2)
        support[0].metric(
            "Average Technical Response Time", f"{average_ms / 1000:.2f}s",
            help="Average measured technical request duration. Useful for overall monitoring, but it can hide slower requests. Requires application performance traces.",
        )
        support[1].metric(
            "P50 Response Time", f"{p50_ms / 1000:.2f}s",
            help="50% of measured technical requests completed within this time. Shows typical technical performance. Requires application performance traces.",
        )

    _section_intro(
        "AI Evaluation & Human Review",
        "How much proposed automated screening and human review activity would require oversight?",
    )
    review = operations.copy()
    review["period"] = time_bucket(review["date"], granularity)
    review = review.groupby("period", as_index=False)[
        ["question_count", "llm_judge_flagged", "human_reviewed"]
    ].sum()
    total_questions = int(review["question_count"].sum())
    total_reviewed = int(review["human_reviewed"].sum())
    total_flagged = int(review["llm_judge_flagged"].sum())
    review_cards = st.container(key="review_metrics").columns(3)
    review_cards[0].metric(
        "LLM Judge Flagged Conversations", _format_count(float(total_flagged)),
        help="Conversations identified by an automated evaluator for possible further review. A flag is not a confirmed failure. Requires a calibrated evaluation pipeline.",
    )
    review_cards[1].metric(
        "Human Review Coverage", _format_rate(_safe_rate(total_reviewed, total_questions)),
        help="Percentage of the provisional demo population reviewed by a human. The eligible denominator must be defined before production use. Requires a review log and sampling policy.",
    )
    review_cards[2].metric(
        "Human-reviewed Conversations", _format_count(float(total_reviewed)),
        help="Number of synthetic conversations marked as human reviewed. Helps show review workload. Requires a governed review log.",
    )
    with _chart_card(
        f"Evaluation and review activity by {granularity.lower()}",
        "Automated flags indicate review candidates, while human-reviewed counts show illustrative review activity.",
        key="evaluation_review_activity",
    ):
        figure = go.Figure()
        for label, column, color in [
            ("LLM Judge flagged", "llm_judge_flagged", "#7162A6"),
            ("Human reviewed", "human_reviewed", "#167C80"),
        ]:
            figure.add_bar(x=review["period"], y=review[column], name=label, marker_color=color)
        figure.update_layout(
            height=265, barmode="group", xaxis_title=granularity, yaxis_title="Conversations",
            legend=dict(orientation="h", y=1.12, x=0), margin=dict(l=5, r=10, t=40, b=5),
        )
        figure.update_yaxes(gridcolor="#E9EEEE", rangemode="tozero")
        _plotly_chart(figure)
        _chart_caption("The dashboard does not run an evaluator; flags are synthetic review candidates, not confirmed response failures.")

    _section_intro(
        "Token Usage & Cost",
        "What level of illustrative AI processing activity and estimated cost would need monitoring?",
    )
    token_trend = operations.copy()
    token_trend["period"] = time_bucket(token_trend["date"], granularity)
    token_trend = token_trend.groupby("period", as_index=False)[
        ["input_tokens", "output_tokens", "estimated_cost_usd"]
    ].sum()
    total_tokens = float(token_trend["input_tokens"].sum() + token_trend["output_tokens"].sum())
    total_cost = float(token_trend["estimated_cost_usd"].sum())
    token_cards = st.container(key="token_metrics").columns(2)
    token_cards[0].metric(
        "Total Token Usage", f"{total_tokens:,.0f}",
        help="Combined illustrative input and output tokens processed. Helps estimate model workload. Requires model usage records.",
    )
    token_cards[1].metric(
        "Estimated Processing Cost", f"US${total_cost:.2f}",
        help="Illustrative estimated AI processing cost, not actual HPB billing expenditure. Requires validated usage records and approved billing rates.",
    )
    token_col, cost_col = st.container(key="token_breakdown").columns([7, 3], gap="medium")
    with token_col:
        with _chart_card(
            f"Token usage by {granularity.lower()}",
            "Input and output tokens remain visible as supporting workload detail.",
            key="token_usage",
        ):
            figure = go.Figure()
            for label, column, color in [
                ("Input tokens", "input_tokens", "#4F7FA3"),
                ("Output tokens", "output_tokens", "#167C80"),
            ]:
                figure.add_bar(x=token_trend["period"], y=token_trend[column], name=label, marker_color=color)
            figure.update_layout(
                height=255, barmode="group", xaxis_title=granularity, yaxis_title="Tokens",
                legend=dict(orientation="h", y=1.12, x=0), margin=dict(l=5, r=10, t=40, b=5),
            )
            figure.update_yaxes(gridcolor="#E9EEEE", rangemode="tozero")
            _plotly_chart(figure)
    with cost_col:
        with _summary_card("Processing breakdown", key="processing_breakdown"):
            st.caption("Supporting assumptions for the selected period.")
            _summary_item("Input tokens", f"{token_trend['input_tokens'].sum():,.0f}")
            _summary_item("Output tokens", f"{token_trend['output_tokens'].sum():,.0f}")
            _summary_item("Illustrative estimate", f"US${total_cost:.2f}")
            _chart_caption("Uses fixed synthetic assumptions—not vendor pricing, validated rates, or HPB billing records.")

    with st.expander("Metric Definitions & Data Requirements", expanded=False):
        st.dataframe(pd.DataFrame([
            {"Metric": "System Availability", "Definition": "Share of monitored time the service is available", "Why it matters": "Signals continuity", "Unit": "%", "Required source": "Uptime or synthetic-monitoring records", "Status": "Pending validation"},
            {"Metric": "API Error Rate", "Definition": "Share of monitored API requests returning an error", "Why it matters": "Signals technical instability", "Unit": "%", "Required source": "API gateway or application logs", "Status": "Pending validation"},
            {"Metric": "Average / P50 / P95 Response Time", "Definition": "Technical request-duration summaries", "Why it matters": "Shows typical and slower requests", "Unit": "seconds", "Required source": "Application performance traces", "Status": "Not end-to-end user response time"},
            {"Metric": "Critical / Warning Alerts", "Definition": "Alerts grouped by agreed severity", "Why it matters": "Supports incident prioritisation", "Unit": "alerts", "Required source": "Monitoring and incident platform", "Status": "Pending rules and ownership"},
            {"Metric": "LLM Judge Flagged Conversations", "Definition": "Automated-evaluator review candidates", "Why it matters": "Focuses further review", "Unit": "conversations", "Required source": "Calibrated evaluation pipeline", "Status": "Not confirmed failures"},
            {"Metric": "Human Review Coverage", "Definition": "Reviewed conversations divided by an eligible population", "Why it matters": "Shows review reach", "Unit": "%", "Required source": "Sampling policy and review log", "Status": "Denominator not validated"},
            {"Metric": "Total Token Usage", "Definition": "Combined input and output tokens", "Why it matters": "Indicates model workload", "Unit": "tokens", "Required source": "Model usage records", "Status": "Proposed metric"},
            {"Metric": "Estimated Processing Cost", "Definition": "Usage multiplied by transparent assumed rates", "Why it matters": "Supports planning", "Unit": "USD", "Required source": "Validated usage and approved rates", "Status": "Not actual expenditure"},
        ]), hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="Sprout AI Analytics Prototype", page_icon="🌱",
        layout="wide", initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --space-related: .25rem;
            --space-control: .5rem;
            --space-content: .75rem;
            --space-heading: 1rem;
            --space-section: 1.5rem;
            --space-page: 2rem;
            --color-teal: #167C80;
            --color-blue: #4F7FA3;
            --color-violet: #7162A6;
            --color-amber: #B97835;
            --color-critical: #984B4B;
        }
        .block-container {padding-top: 3rem; padding-bottom: var(--space-page); max-width: 1600px;}
        section[data-testid="stSidebar"] {min-width: 232px; max-width: 240px;}
        section[data-testid="stSidebar"] > div {width: 232px;}
        section[data-testid="stSidebar"] [data-testid="stButton"] button {
            justify-content: flex-start; min-height: 2.35rem; border-radius: 6px;
        }
        .workspace-title {font-size: .92rem; font-weight: 750; letter-spacing: .04em; line-height: 1.45; color: #183E42; margin: .35rem 0 1.25rem;}
        .nav-group {font-size: .68rem; font-weight: 750; letter-spacing: .08em; color: #707A80; margin: 1rem 0 .35rem;}
        [data-testid="stVerticalBlock"] {gap: var(--space-content);}
        [data-testid="stHorizontalBlock"] {gap: var(--space-heading);}
        :is(.st-key-activity_section, .st-key-entry_section, .st-key-needs_top,
            .st-key-needs_trend, .st-key-portal_trends,
            .st-key-portal_patterns, .st-key-operations_reliability) [data-testid="stHorizontalBlock"] {
            gap: var(--space-heading);
        }
        .st-key-feedback_explorer [data-testid="stHorizontalBlock"] {gap: var(--space-page);}
        .st-key-needs_trend [data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"] {
            gap: var(--space-heading);
        }
        h1 {font-size: 1.85rem !important; line-height: 1.2 !important; font-weight: 720 !important; padding: 0 0 .35rem !important; margin: 0;}
        h3 {
            font-size: 1.25rem !important; line-height: 1.3 !important; font-weight: 650 !important;
            padding: var(--space-content) 0 .45rem !important; margin: 0;
        }
        h4 {
            font-size: 1rem !important; line-height: 1.35 !important; font-weight: 650 !important;
            padding: var(--space-control) 0 var(--space-control) !important; margin: 0;
        }
        .section-description {
            color: #69747A; font-size: .84rem; line-height: 1.45;
            margin: -.15rem 0 var(--space-content);
        }
        h4.chart-title {
            color: #29343D; font-size: 1rem !important; font-weight: 650;
            line-height: 1.35 !important; padding: 0 !important; margin: 0 !important;
        }
        .chart-subtitle {color: #717B81; font-size: .8rem; line-height: 1.4; margin-top: .2rem; margin-bottom: .3rem;}
        [data-testid="stCaptionContainer"] p {
            font-size: .78rem; line-height: 1.45; color: #717B81;
            overflow-wrap: anywhere;
        }
        [data-testid="stCaptionContainer"] {margin-bottom: 0 !important;}
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #DFE7E8 !important; border-radius: 8px !important;
            background: #FFFFFF;
        }
        [data-testid="stVerticalBlockBorderWrapper"] > div {padding: 1rem 1rem .95rem;}
        [class*="st-key-chart_card_"] {
            width: 100%; height: 100%; min-width: 0;
            gap: var(--space-control);
        }
        [data-testid="stLayoutWrapper"]:has(> [class*="st-key-chart_card_"]) {
            flex: 1 1 auto;
        }
        [class*="st-key-chart_card_"] [data-testid="stPlotlyChart"] {
            min-width: 0;
        }
        [class*="st-key-chart_card_"] > [data-testid="stElementContainer"]:last-child
        [data-testid="stCaptionContainer"] {
            padding-top: var(--space-related);
            padding-bottom: var(--space-related);
        }
        [class*="st-key-chart_card_"] > [data-testid="stElementContainer"]:last-child:has(
            [data-testid="stCaptionContainer"]
        ) {
            margin-top: auto;
        }
        [class*="st-key-chart_card_"] [data-testid="stCaptionContainer"] {
            margin-bottom: 0 !important;
        }
        :is(.st-key-portal_trends, .st-key-portal_patterns,
            .st-key-activity_section, .st-key-entry_section, .st-key-needs_top,
            .st-key-needs_trend, .st-key-conversation_characteristics,
            .st-key-conversation_quality, .st-key-operations_reliability,
            .st-key-token_breakdown) [data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }
        :is(.st-key-portal_trends, .st-key-portal_patterns,
            .st-key-activity_section, .st-key-entry_section, .st-key-needs_top,
            .st-key-needs_trend, .st-key-conversation_characteristics,
            .st-key-conversation_quality, .st-key-operations_reliability,
            .st-key-token_breakdown) [data-testid="stColumn"] {
            display: flex; min-width: 0;
        }
        :is(.st-key-portal_trends, .st-key-portal_patterns,
            .st-key-activity_section, .st-key-entry_section, .st-key-needs_top,
            .st-key-needs_trend, .st-key-conversation_characteristics,
            .st-key-conversation_quality, .st-key-operations_reliability,
            .st-key-token_breakdown) [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
            width: 100%; height: 100%;
        }
        [role="tabpanel"] {padding-top: var(--space-heading);}
        [role="tabpanel"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:first-child:has([data-testid="stCaptionContainer"]) {
            margin-bottom: var(--space-content);
        }
        [data-testid="stExpander"] summary {padding: var(--space-control) var(--space-content);}
        [data-testid="stExpanderDetails"] {padding: var(--space-heading);}
        section[data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 0; background: transparent;
        }
        section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
            min-height: 2.35rem; padding: .35rem .15rem;
        }
        section[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
            padding: .15rem 0 .1rem .7rem;
        }
        section[data-testid="stSidebar"] [data-testid="stExpanderDetails"] button {
            min-height: 2.05rem; font-size: .84rem;
        }
        [data-testid="stCaptionContainer"] p {margin-bottom: 0;}
        .grouped-item {
            padding: var(--space-related) 0 var(--space-content); margin-bottom: var(--space-related);
            border-bottom: 1px solid #E9EEEE;
            overflow-wrap: anywhere;
        }
        .grouped-item-title {color: #29343D; font-size: 1rem; font-weight: 600; line-height: 1.4;}
        .grouped-item-meta {
            color: #707680; font-size: .85rem; line-height: 1.4;
            margin-top: var(--space-related); white-space: pre-line;
        }
        .grouped-item-reasons {margin-top: var(--space-control);}
        h4.summary-heading {
            color: #35434D; font-size: 1rem !important; font-weight: 600;
            padding: var(--space-control) 0 var(--space-heading) !important; margin: 0;
        }
        .summary-insight {
            padding-bottom: var(--space-content); margin-bottom: var(--space-related);
            border-bottom: 1px solid #E9EEEE; overflow-wrap: anywhere;
        }
        .summary-value {color: #202D36; font-size: 1.25rem; font-weight: 700; line-height: 1.3;}
        .summary-descriptor, .summary-note {
            color: #707680; font-size: .85rem; font-weight: 400; line-height: 1.45;
            margin-top: var(--space-related);
        }
        .summary-note {margin-top: var(--space-control);}
        .st-key-repeated_details [data-testid="stExpander"] {
            border: none; border-radius: 0;
        }
        .st-key-repeated_details [data-testid="stExpander"] summary {
            padding: var(--space-related) 0; min-height: 0;
        }
        .st-key-repeated_details [data-testid="stExpander"] summary p {font-size: .85rem;}
        [data-testid="stTabs"] [role="tablist"] {gap: var(--space-section);}
        [data-testid="stTabs"] [role="tab"] {height: 2.5rem;}
        .demo-indicator {
            display: inline-block; color: #4C5C61; background: #EEF3F3;
            border: 1px solid #D8E2E2; border-radius: 999px;
            padding: .18rem .58rem; margin: 0 0 var(--space-heading);
            font-size: .78rem; font-weight: 650;
        }
        .st-key-global_filter_toolbar {
            background: #F7F9F9; border: 1px solid #E3E9E9; border-radius: 8px;
            padding: .6rem .75rem; margin-bottom: var(--space-related);
        }
        .st-key-global_filter_toolbar [data-testid="stHorizontalBlock"] {gap: .75rem; align-items: end;}
        .st-key-global_filter_toolbar [data-testid="stSelectbox"] > div > div,
        .st-key-global_filter_toolbar [data-testid="stDateInput"] input,
        .st-key-global_filter_toolbar [data-testid="stButton"] button {min-height: 2.4rem;}
        .st-key-global_filter_toolbar [data-testid="stButtonGroup"] {min-height: 2.4rem;}
        .filter-label {
            color: #667176; font-size: .69rem; font-weight: 650; line-height: 1.2;
            letter-spacing: .02em; margin: 0 0 .3rem .1rem;
        }
        .filter-label-spacer {visibility: hidden;}
        .filter-date {
            display: flex; align-items: center; min-height: 2.4rem; padding: 0 .7rem;
            color: #35434D; background: #FFFFFF; border: 1px solid #D8E1E2;
            border-radius: 6px; font-size: .83rem; margin-bottom: 1rem;
        }
        .metric-status {
            display: inline-block; color: #5D666D; background: #F1F4F5;
            border-radius: 999px; padding: .2rem .6rem; margin-bottom: var(--space-content);
            font-size: .82rem; font-weight: 600;
        }
        [data-testid="stMetric"] {
            background: #F7FAFA; border: 1px solid #DDE8E8;
            border-radius: 8px; padding: var(--space-content);
            min-height: 6.6rem; height: 100%;
        }
        [data-testid="stMetricValue"] {font-size: 1.8rem; font-weight: 680;}
        [data-testid="stMetricLabel"] {
            white-space: normal; height: auto; min-height: 2.4rem;
            font-size: .8rem; line-height: 1.35;
        }
        [data-testid="stMetricLabel"] p {
            white-space: normal; overflow: visible; text-overflow: clip;
        }
        :is(.st-key-portal_metrics, .st-key-feedback_metrics,
            .st-key-conversation_signal_metrics, .st-key-operations_metrics,
            .st-key-review_metrics, .st-key-token_metrics,
            .st-key-journey_conversion, .st-key-journey_activity)
        [data-testid="stHorizontalBlock"] {align-items: stretch;}
        :is(.st-key-portal_metrics, .st-key-feedback_metrics,
            .st-key-conversation_signal_metrics, .st-key-operations_metrics,
            .st-key-review_metrics, .st-key-token_metrics,
            .st-key-journey_conversion, .st-key-journey_activity)
        [data-testid="stColumn"] {display: flex; min-width: 0;}
        :is(.st-key-portal_metrics, .st-key-feedback_metrics,
            .st-key-conversation_signal_metrics, .st-key-operations_metrics,
            .st-key-review_metrics, .st-key-token_metrics,
            .st-key-journey_conversion, .st-key-journey_activity)
        [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {width: 100%; height: 100%;}
        .st-key-journey_metrics [data-testid="stHorizontalBlock"] {align-items: stretch;}
        .st-key-journey_metrics > [data-testid="stVerticalBlock"] >
        [data-testid="stElementContainer"] > [data-testid="stHorizontalBlock"] >
        [data-testid="stColumn"] {display: flex; min-width: 0;}
        .st-key-journey_conversion, .st-key-journey_activity {
            width: 100%; height: 100%; padding: .65rem;
            border: 1px solid #E3E9E9; border-radius: 8px; background: #FCFDFD;
        }
        .metric-group-label {
            color: #5F6B70; font-size: .72rem; line-height: 1.25;
            font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
            margin: 0 0 var(--space-control) .1rem;
        }
        .st-key-journey_conversion [data-testid="stColumn"]:nth-child(-n+2) {position: relative;}
        .st-key-journey_conversion [data-testid="stColumn"]:nth-child(-n+2)::after {
            content: '→'; position: absolute; right: -.85rem; top: 50%; z-index: 2;
            color: #5D7D81; transform: translateY(-50%);
        }
        .audience-mix {
            display: flex; flex-direction: column; justify-content: center;
            min-height: 13.5rem; padding: .75rem 0 .35rem;
        }
        .audience-row + .audience-row {margin-top: 1.1rem;}
        .audience-row-header {
            display: grid; grid-template-columns: minmax(0, 1fr) auto auto;
            gap: .55rem; align-items: baseline; margin-bottom: .4rem;
        }
        .audience-label {color: #33434A; font-size: .86rem; font-weight: 650;}
        .audience-count {color: #27363D; font-size: 1rem; font-weight: 720;}
        .audience-share {color: #69757A; font-size: .78rem; font-variant-numeric: tabular-nums;}
        .audience-track {height: .72rem; background: #E8EEEE; border-radius: 999px; overflow: hidden;}
        .audience-fill {display: block; height: 100%; border-radius: inherit;}
        .st-key-feedback_explorer button {min-height: 2.15rem; padding: var(--space-control) var(--space-content);}
        .st-key-feedback_explorer [data-testid="stVerticalBlock"] {gap: var(--space-control);}
        @media (min-width: 761px) and (max-width: 1100px) {
            .st-key-global_filter_toolbar [data-testid="stHorizontalBlock"] {
                display: grid !important; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
                gap: .65rem !important; align-items: end;
            }
            .st-key-global_filter_toolbar [data-testid="stColumn"] {
                width: 100% !important; min-width: 0 !important; flex: none !important;
            }
            :is(.st-key-portal_trends, .st-key-portal_patterns,
                .st-key-activity_section, .st-key-entry_section,
                .st-key-needs_top, .st-key-needs_trend,
                .st-key-conversation_characteristics, .st-key-conversation_quality,
                .st-key-operations_reliability, .st-key-token_breakdown)
                [data-testid="stHorizontalBlock"] {
                flex-direction: column;
            }
            :is(.st-key-portal_trends, .st-key-portal_patterns,
                .st-key-activity_section, .st-key-entry_section,
                .st-key-needs_top, .st-key-needs_trend,
                .st-key-conversation_characteristics, .st-key-conversation_quality,
                .st-key-operations_reliability, .st-key-token_breakdown)
                [data-testid="stColumn"] {
                width: 100% !important; flex: 1 1 auto !important;
            }
            .st-key-journey_metrics [data-testid="stHorizontalBlock"]:has(.st-key-journey_conversion) {
                flex-direction: column;
            }
            .st-key-journey_metrics [data-testid="stHorizontalBlock"]:has(.st-key-journey_conversion) >
            [data-testid="stColumn"] {
                width: 100% !important; flex: 1 1 auto !important;
            }
            .st-key-feedback_metrics [data-testid="stHorizontalBlock"],
            .st-key-portal_metrics [data-testid="stHorizontalBlock"],
            .st-key-conversation_signal_metrics [data-testid="stHorizontalBlock"],
            .st-key-operations_metrics [data-testid="stHorizontalBlock"],
            .st-key-review_metrics [data-testid="stHorizontalBlock"],
            .st-key-token_metrics [data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
            .st-key-feedback_metrics [data-testid="stColumn"],
            .st-key-portal_metrics [data-testid="stColumn"],
            .st-key-conversation_signal_metrics [data-testid="stColumn"],
            .st-key-operations_metrics [data-testid="stColumn"],
            .st-key-review_metrics [data-testid="stColumn"],
            .st-key-token_metrics [data-testid="stColumn"] {flex: 1 1 30% !important; width: 30% !important;}
        }
        @media (min-width: 761px) and (max-width: 850px) {
            .st-key-feedback_explorer [data-testid="stHorizontalBlock"] {
                flex-direction: column;
            }
            .st-key-feedback_explorer [data-testid="stColumn"] {
                width: 100% !important; flex: 1 1 auto !important;
            }
        }
        .reason-chip {
            display: inline-block; background: #E7F0F0; color: #214F55;
            border: 1px solid #BFD3D5; border-radius: 999px;
            padding: 0.25rem 0.65rem; margin: 0 0.35rem 0.35rem 0;
            font-size: 0.9rem; font-weight: 600;
        }
        .category-chip {
            display: inline-block; background: #F1F4F5; color: #354B52;
            border: 1px solid #D7E0E2; border-radius: 999px;
            padding: 0.2rem 0.6rem; margin-bottom: 0.4rem;
            font-size: 0.82rem; font-weight: 700;
        }
        @media (max-width: 760px) {
            [data-testid="stHorizontalBlock"] {
                flex-direction: column; gap: var(--space-section) !important;
            }
            [data-testid="stColumn"] {
                width: 100% !important; flex: 1 1 auto !important;
            }
            :is(.st-key-journey_metrics, .st-key-feedback_metrics, .st-key-portal_metrics,
                .st-key-conversation_signal_metrics, .st-key-operations_metrics,
                .st-key-review_metrics, .st-key-token_metrics) [data-testid="stHorizontalBlock"] {
                gap: var(--space-content) !important;
            }
            .st-key-journey_metrics [data-testid="stColumn"]::after {display: none;}
            .block-container {padding-left: var(--space-heading); padding-right: var(--space-heading);}
            .st-key-global_filter_toolbar [data-testid="stHorizontalBlock"] {
                display: grid !important;
                grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
                gap: .65rem !important; align-items: end;
            }
            .st-key-global_filter_toolbar [data-testid="stColumn"] {
                width: 100% !important; min-width: 0 !important; flex: none !important;
            }
            .st-key-global_filter_toolbar [data-testid="stButtonGroup"] {min-width: 0;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    tables = load_dashboard_tables(dashboard_data_root())
    page, _ = render_sidebar_navigation()
    filters = render_report_header(page, tables)
    if page == "Parent Portal Reach":
        render_engagement_page(tables, filters, view="portal")
    elif page == "Sprout Usage":
        render_conversation_signals(tables, filters)
    elif page == "Entry & Journey":
        render_engagement_page(tables, filters, view="journey")
    elif page == "Topics & Trends":
        render_parent_needs_page(tables, filters)
    elif page == "Repeated Questions":
        render_repeated_questions_page(tables, filters)
    elif page == "Feedback Overview":
        render_quality_page(tables, filters, "Feedback Overview")
    elif page == "Explore Feedback Cases":
        render_quality_page(tables, filters, "Explore Feedback Cases")
    elif page == "Operations & AI Quality":
        render_operations_page(tables, filters)
    render_metric_traceability(page)


if __name__ == "__main__":
    main()
