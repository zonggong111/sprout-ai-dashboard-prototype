"""Business-facing Streamlit prototype over Phase 2 / 2.5 CSV outputs."""

from __future__ import annotations

import html
import hashlib
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.feedback_analysis import (parse_feedback_reasons, normalise_feedback_submissions,
                                   feedback_reason_summary, feedback_reason_matrix)


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
    "cleaned_conversations": "cleaned/cleaned_conversations.csv",
    **{name: f"feedback/{name}.csv" for name in [
        'feedback_overview', 'feedback_reason_summary', 'feedback_reason_by_category',
        'feedback_reason_matrix', 'feedback_cases', 'feedback_case_reasons']},
}


def dashboard_data_root() -> Path:
    """Use only the synthetic exports bundled with this deployment."""
    return DEFAULT_DATA_ROOT


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


def render_engagement_page(tables: dict[str, pd.DataFrame]) -> None:
    st.caption("Are parents using Sprout AI—and how do they enter?")
    daily = tables["engagement_daily"]
    with st.expander("Filters", expanded=False):
        environment = _environment_control([daily], key="engagement_environment")
        dates = _date_control(daily, "event_date", key="engagement_dates")
        st.caption("Dates affect the journey and activity chart. Entry-point totals cover the full exported period.")
    filtered = _filter(daily, environment=environment, date_range=dates, date_column="event_date")
    values = _engagement_values(filtered, daily, tables["engagement_kpis"])
    st.markdown("### Conversation journey")
    cards = st.container(key="journey_metrics").columns(5)
    for card, label, value, help_text in zip(cards, [
        "Entered Sprout AI", "Started a conversation", "Entry-to-conversation conversion",
        "User messages sent", "Average messages per active conversation",
    ], [_format_count(values["entries"]), _format_count(values["active"]), _format_rate(values["activation"]),
        _format_count(values["messages"]), _format_number(values["depth"])], [
        "Distinct sessions reaching the chat experience.",
        "Distinct entered sessions sending at least one user message.",
        "Started conversation sessions divided by entered sessions.",
        "Tracked user-message events.", "User messages divided by active conversation sessions.",
    ]):
        card.metric(label, value, help=help_text)
    st.markdown("### Activity over time")
    chart, notes = st.container(key="activity_section").columns([7, 3], gap="large")
    with chart:
        mode = st.segmented_control("Trend measure", ["Both", "Started conversations", "User messages sent"],
                                    default="Both", key="engagement_trend_mode", label_visibility="collapsed")
        if not filtered.empty and "event_date" in filtered:
            columns = [c for c in ["active_chat_sessions", "chat_messages_sent"] if c in filtered]
            trend = filtered.groupby("event_date", as_index=False)[columns].sum().sort_values("event_date")
            st.plotly_chart(engagement_trend_figure(trend, mode or "Both"), width="stretch")
        else:
            trend = pd.DataFrame()
            _show_missing("Activity data could not be loaded.")
    with notes:
        _summary_heading("Activity summary")
        if not trend.empty:
            _summary_item("Recorded period", f"{len(trend)} recorded days")
            for field, label in [("active_chat_sessions", "Peak started conversations"), ("chat_messages_sent", "Peak user messages")]:
                if field in trend:
                    _summary_item(label, _format_count(trend[field].max()))
            if "active_chat_sessions" in trend:
                peak = trend.sort_values(["active_chat_sessions", "event_date"], ascending=[False, True]).iloc[0]
                _summary_item("Most active date · conversation starts", str(peak["event_date"])[:10])
            st.caption("Peaks describe recorded usage; release timing and entry routes may help explain changes.")

    st.markdown("### How parents enter Sprout AI")
    entry = tables["entry_point_effectiveness"]
    chart, notes = st.container(key="entry_section").columns([7, 3], gap="large")
    with chart:
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
            st.plotly_chart(fig, width="stretch")
        else:
            _show_missing("Entry-point data could not be loaded.")
    with notes:
        _summary_heading("Entry-point summary")
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
            st.caption("Different routes can reflect different visitor intentions. Ties use alphabetical order.")


def render_parent_needs_page(tables: dict[str, pd.DataFrame]) -> None:
    st.caption("What are parents asking about—and which needs repeat?")
    environment = _environment_control([tables["topic_summary"]], key="needs_environment")
    summary = _filter(tables["topic_summary"], environment=environment)
    repeated = _filter(tables["repeated_question_clusters"], environment=environment)
    valid_repeated = not repeated.empty and {"representative_question","number_of_similar_questions"}.issubset(repeated)
    if valid_repeated:
        repeated = repeated.sort_values("number_of_similar_questions", ascending=False)
    chart, preview = st.container(key="needs_top").columns([6.5, 3.5], gap="large")
    with chart:
        st.markdown("### Top Question Categories")
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
            st.plotly_chart(figure, width="stretch")
            st.caption("Counts and shares use all questions in the current dataset.")
        else:
            _show_missing("Question Category data could not be loaded.")
    with preview:
        st.markdown("### Frequently repeated needs")
        if valid_repeated:
            for _, row in repeated.head(3).iterrows():
                _grouped_item(
                    row['representative_question'],
                    f"{row.get('primary_topic', 'Unclassified')} · {int(row['number_of_similar_questions'])} similar variants",
                )
            with st.container(key="repeated_details"), st.expander("View all repeated questions", expanded=False):
                _display_table(repeated, ["representative_question","primary_topic","number_of_similar_questions"], {
                    "representative_question":"Representative question", "primary_topic":"Question Category",
                    "number_of_similar_questions":"Similar variants"})
                example_columns = [c for c in repeated if c.startswith("example_question_")]
                if example_columns:
                    with st.expander("Example phrasings", expanded=False):
                        for _, row in repeated.head(5).iterrows():
                            phrases = dict.fromkeys(str(row[c]) for c in example_columns if pd.notna(row[c]))
                            _grouped_item(row['representative_question'], "\n".join(phrases))
        else:
            _show_missing("Repeated-question data could not be loaded.")

    st.markdown("### Question trends over time")
    chart, notes = st.container(key="needs_trend").columns([7, 3], gap="large")
    with chart:
        controls = st.columns([1.4, 1])
        with controls[0]:
            choice = st.selectbox("Topic trend", ["Top 3 topics", *_values([summary], "primary_topic")], key="needs_trend_topic", label_visibility="collapsed")
        with controls[1]:
            with st.expander("Trend dates", expanded=False):
                dates = _date_control(tables["topic_weekly_trend"], "week_start", key="needs_dates")
        weekly = _filter(tables["topic_weekly_trend"], environment=environment, date_range=dates, date_column="week_start")
        selected = summary.nlargest(3, "query_count")["primary_topic"].tolist() if choice == "Top 3 topics" and "query_count" in summary else [choice]
        if not weekly.empty and {"week_start","primary_topic","query_count"}.issubset(weekly):
            weekly = weekly.loc[weekly["primary_topic"].isin(selected)].groupby(["week_start","primary_topic"], as_index=False)["query_count"].sum().sort_values("week_start")
            figure = px.line(weekly, x="week_start", y="query_count", color="primary_topic", markers=True,
                labels={"week_start":"Week starting", "query_count":"Questions", "primary_topic":"Question Category"},
                color_discrete_sequence=["#167C80","#B66832","#6653A1"])
            figure.update_traces(line_width=2, marker_size=5)
            figure.update_yaxes(gridcolor="#E9EEEE")
            figure.update_layout(height=320, legend=dict(orientation="h",y=1.22), legend_title_text="",
                                 margin=dict(l=5,r=10,t=65,b=10))
            st.plotly_chart(figure, width="stretch")
        else:
            weekly = pd.DataFrame()
            _show_missing("Weekly question data could not be loaded.")
    with notes:
        _summary_heading("Trend summary")
        _summary_item("Current selection", choice)
        if not weekly.empty:
            periods = sorted(weekly["week_start"].unique())
            latest = weekly.loc[weekly["week_start"].eq(periods[-1])]
            leader = latest.sort_values(["query_count","primary_topic"], ascending=[False,True]).iloc[0]
            _summary_item("Highest recent volume", f"{leader['primary_topic']} · {int(leader['query_count'])}",
                          note=f"Latest recorded week: {str(periods[-1])[:10]}")
            if len(periods)>1:
                pivot = weekly.pivot(index="week_start", columns="primary_topic", values="query_count").fillna(0)
                delta = (pivot.loc[periods[-1]] - pivot.loc[periods[-2]]).sort_values(ascending=False)
                if delta.iloc[0]>0:
                    _summary_item("Largest recent increase", f"{delta.index[0]} · +{int(delta.iloc[0])} questions")
                else:
                    _summary_item("Recent change", "No selected category increased")
                st.caption("Comparison uses the last two recorded weeks; these are descriptive counts, not a significance test.")
    features = _filter(tables["feature_interest_summary"], environment=environment)
    if not features.empty and {"related_feature","query_count"}.issubset(features):
        with st.expander("Additional analysis", expanded=False):
            st.markdown("#### Related Parent Portal feature interest")
            st.caption("Small feature counts are exploratory and do not establish importance.")
            _display_table(features, ["related_feature","query_count"], {"related_feature":"Related feature","query_count":"Questions"})


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
    st.markdown(
        f"**{_format_count(counts['negative'])} of {_format_count(counts['total'])} "
        f"feedback submissions were negative ({counts['negative_share']:.1%}).**"
    )


def _render_feedback_reasons(exported: pd.DataFrame) -> None:
    st.markdown("### Most common reasons for negative feedback")
    summary = exported.rename(columns={"feedback_reason": "Reason", "negative_case_count": "Count",
                                        "percentage_of_negative_cases": "Percentage of negative feedback"}).copy()
    if summary.empty:
        st.info("Detailed dissatisfaction reason data is not available in the current dataset.")
        return
    summary = summary.sort_values(["Count", "Reason"], ascending=[False, True])
    denominator = int(summary["negative_feedback_submissions"].iloc[0])
    summary["Count / denominator"] = summary["Count"].map(lambda n: f"{n} of {denominator} negative feedback submissions")
    summary["Display label"] = summary.apply(
        lambda row: (
            f"{int(row['Count'])} cases · "
            f"{float(row['Percentage of negative feedback']):.1%}"
        ),
        axis=1,
    )
    figure = px.bar(
        summary.sort_values("Count"), x="Count", y="Reason", orientation="h",
        text="Display label",
        custom_data=["Percentage of negative feedback", "Count / denominator"],
        labels={"Count": "Negative feedback submissions", "Reason": "Feedback reason"},
        color_discrete_sequence=["#2F6B73"],
    )
    figure.update_traces(
        textposition="outside",
        hovertemplate=(
            "%{y}<br>%{customdata[1]}<br>Share of negative submissions: "
            "%{customdata[0]:.1%}<extra></extra>"
        ),
        cliponaxis=False,
    )
    figure.update_layout(
        height=max(260, min(340, 36 * len(summary) + 90)), showlegend=False,
        margin=dict(l=10, r=100, t=10, b=10),
    )
    figure.update_xaxes(rangemode="tozero", dtick=1)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Parents can select more than one reason, so percentages may not sum to 100%."
    )
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
    st.plotly_chart(figure, width="stretch")
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


def render_quality_page(tables: dict[str, pd.DataFrame]) -> None:
    cases = tables["feedback_cases"].rename(columns={
        "question_category": "primary_topic", "journey_stage": "user_type",
        "anonymised_session_id": "session_id"}).copy()
    links = tables["feedback_case_reasons"]
    if not cases.empty and "feedback_id" in cases:
        reason_map = links.drop_duplicates(["feedback_id", "feedback_reason"]).groupby("feedback_id")["feedback_reason"].agg(list).to_dict() if {"feedback_id", "feedback_reason"}.issubset(links) else {}
        cases["_feedback_reasons"] = cases["feedback_id"].map(lambda value: reason_map.get(value, []))
        if "primary_topic" not in cases:
            cases["primary_topic"] = "Unclassified"
        cases["primary_topic"] = cases["primary_topic"].fillna("Unclassified")
        cases["_has_written_feedback"] = cases.get("feedback_comment", pd.Series("", index=cases.index)).fillna("").astype(str).str.strip().ne("")
    submissions = cases

    reasons_tab, explorer_tab = st.tabs(["Feedback Overview", "Explore Feedback Cases"])
    with reasons_tab:
        _render_feedback_overview(tables["feedback_overview"])
        if not submissions.empty:
            st.caption(
                f"Detailed reasons below use {len(submissions):,} negative feedback submissions "
                "from the separate conversation/backend layer."
            )
        _render_feedback_reasons(tables["feedback_reason_summary"])
    with explorer_tab:
        _render_feedback_explorer(tables, submissions)


def main() -> None:
    st.set_page_config(
        page_title="Sprout AI Analytics Prototype", page_icon="🌱",
        layout="wide", initial_sidebar_state="collapsed",
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
        }
        .block-container {padding-top: 2.6rem; padding-bottom: var(--space-page); max-width: 1600px;}
        [data-testid="stVerticalBlock"] {gap: var(--space-content);}
        [data-testid="stHorizontalBlock"] {gap: var(--space-heading);}
        :is(.st-key-activity_section, .st-key-entry_section, .st-key-needs_top,
            .st-key-needs_trend, .st-key-feedback_explorer) [data-testid="stHorizontalBlock"] {
            gap: var(--space-page);
        }
        .st-key-needs_trend [data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"] {
            gap: var(--space-heading);
        }
        h1 {font-size: 1.9rem !important; padding: 0 0 var(--space-content) !important; margin: 0;}
        h3 {
            font-size: 1.2rem !important;
            padding: var(--space-heading) 0 var(--space-content) !important; margin: 0;
        }
        h4 {
            font-size: 1.05rem !important;
            padding: var(--space-control) 0 var(--space-control) !important; margin: 0;
        }
        [role="tabpanel"] {padding-top: var(--space-heading);}
        [role="tabpanel"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:first-child:has([data-testid="stCaptionContainer"]) {
            margin-bottom: var(--space-content);
        }
        [data-testid="stExpander"] summary {padding: var(--space-control) var(--space-content);}
        [data-testid="stExpanderDetails"] {padding: var(--space-heading);}
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
        .synthetic-banner {
            background: #FFF3CD; color: #664D03; border: 1px solid #FFECB5;
            border-radius: 8px; padding: var(--space-control) var(--space-content); font-weight: 700;
            letter-spacing: 0.01em; margin: 0;
        }
        [data-testid="stMetric"] {
            background: #F7FAFA; border: 1px solid #DDE8E8;
            border-radius: 8px; padding: var(--space-content);
        }
        [data-testid="stMetricValue"] {font-size: 2rem;}
        [data-testid="stMetricLabel"] {white-space: normal; height: auto; min-height: 2.5rem;}
        [data-testid="stMetricLabel"] p {
            white-space: normal; overflow: visible; text-overflow: clip;
        }
        .st-key-journey_metrics [data-testid="stMetric"] {height: 100%;}
        .st-key-journey_metrics [data-testid="stColumn"]:nth-child(n+4) [data-testid="stMetric"] {background: #FCFDFD;}
        .st-key-journey_metrics [data-testid="stColumn"]:nth-child(-n+2) {position: relative;}
        .st-key-journey_metrics [data-testid="stColumn"]:nth-child(-n+2)::after {
            content: '→'; position: absolute; right: -1rem; top: 45%; color: #5D7D81;
        }
        .st-key-feedback_explorer button {min-height: 2.15rem; padding: var(--space-control) var(--space-content);}
        .st-key-feedback_explorer [data-testid="stVerticalBlock"] {gap: var(--space-control);}
        @media (min-width: 761px) and (max-width: 1100px) {
            .st-key-journey_metrics [data-testid="stHorizontalBlock"],
            .st-key-feedback_metrics [data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
            .st-key-journey_metrics [data-testid="stColumn"],
            .st-key-feedback_metrics [data-testid="stColumn"] {flex: 1 1 30% !important; width: 30% !important;}
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
            :is(.st-key-journey_metrics, .st-key-feedback_metrics) [data-testid="stHorizontalBlock"] {
                gap: var(--space-content) !important;
            }
            .st-key-journey_metrics [data-testid="stColumn"]::after {display: none;}
            .block-container {padding-left: var(--space-heading); padding-right: var(--space-heading);}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Sprout AI Usage & Parent Needs Analytics")
    st.markdown(
        '<div class="synthetic-banner">SYNTHETIC / PROTOTYPE DATA — DEMONSTRATION ONLY</div>',
        unsafe_allow_html=True,
    )
    tables = load_dashboard_tables(dashboard_data_root())

    engagement_tab, needs_tab, quality_tab = st.tabs([
        "1 · Usage & Entry", "2 · Parent Needs", "3 · Feedback & Response Quality",
    ])
    with engagement_tab:
        render_engagement_page(tables)
    with needs_tab:
        render_parent_needs_page(tables)
    with quality_tab:
        render_quality_page(tables)


if __name__ == "__main__":
    main()
