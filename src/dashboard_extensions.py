"""Deterministic demo aggregations for the stakeholder-facing dashboard.

These helpers use only bundled synthetic records. Operational measures are
illustrative formulas, not observations from production monitoring or billing.
"""

from __future__ import annotations

from datetime import date

import pandas as pd


__all__ = [
    "aggregate_engagement",
    "aggregate_portal_usage",
    "aggregate_topic_trend",
    "conversation_intensity",
    "conversation_quality_trend",
    "engagement_funnel",
    "entry_point_performance",
    "filter_rows",
    "language_distribution",
    "operational_demo",
    "operational_latency_samples",
    "portal_activity_heatmap",
    "portal_monthly",
    "portal_summary",
    "portal_user_mix",
    "question_input_types",
    "time_bucket",
]


DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
GRANULARITIES = ("Day", "Week", "Month")
DEMO_COST_PER_MILLION_INPUT_TOKENS = 1.50
DEMO_COST_PER_MILLION_OUTPUT_TOKENS = 4.50


def _truthy(series: pd.Series) -> pd.Series:
    """Return a nullable-boolean-like source as an ordinary boolean mask."""
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes", "y"})


def filter_rows(
    table: pd.DataFrame,
    *,
    environment: str = "All",
    date_range: tuple[date, date] | None = None,
    date_column: str,
) -> pd.DataFrame:
    """Apply dashboard environment and inclusive date filters safely."""
    output = table.copy()
    if environment != "All" and "beta_or_production" in output:
        output = output.loc[output["beta_or_production"].astype(str).eq(environment)]
    if date_range and date_column in output:
        dates = pd.to_datetime(output[date_column], errors="coerce", utc=True).dt.date
        output = output.loc[dates.between(*date_range)]
    return output.reset_index(drop=True)


def time_bucket(values: pd.Series, granularity: str) -> pd.Series:
    """Return calendar-aligned, timezone-naive buckets for chart aggregation."""
    if granularity not in GRANULARITIES:
        raise ValueError(f"Unsupported granularity: {granularity}")
    timestamps = pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None)
    if granularity == "Day":
        return timestamps.dt.floor("D")
    if granularity == "Week":
        return timestamps.dt.to_period("W-SUN").dt.start_time
    return timestamps.dt.to_period("M").dt.start_time


def aggregate_engagement(daily: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Aggregate additive conversation activity to the selected calendar grain."""
    columns = ["entry_sessions", "active_chat_sessions", "chat_messages_sent"]
    if daily.empty or "event_date" not in daily:
        return pd.DataFrame(columns=["period", *columns])
    work = daily.copy()
    work["period"] = time_bucket(work["event_date"], granularity)
    available = [column for column in columns if column in work]
    if not available:
        return pd.DataFrame(columns=["period", *columns])
    for column in available:
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0)
    return work.dropna(subset=["period"]).groupby("period", as_index=False)[available].sum()


def aggregate_topic_trend(
    conversations: pd.DataFrame,
    topics: list[str],
    granularity: str,
) -> pd.DataFrame:
    """Align topic counts to one calendar index without treating missing periods as zero.

    A topic receives a recorded zero when other conversations prove that the period is
    represented in the selected export. Calendar periods with no conversation records
    remain missing so charts can display a genuine gap instead of an invented value.
    """
    columns = ["period", "primary_topic", "query_count", "period_status"]
    required = {"timestamp", "primary_topic"}
    selected_topics = list(dict.fromkeys(str(topic) for topic in topics if str(topic).strip()))
    if conversations.empty or not selected_topics or not required.issubset(conversations.columns):
        return pd.DataFrame(columns=columns)

    work = conversations.copy()
    work["period"] = time_bucket(work["timestamp"], granularity)
    work = work.dropna(subset=["period"])
    if work.empty:
        return pd.DataFrame(columns=columns)

    recorded_periods = pd.DatetimeIndex(sorted(work["period"].unique()))
    if granularity == "Day":
        calendar = pd.date_range(recorded_periods.min(), recorded_periods.max(), freq="D")
    elif granularity == "Week":
        calendar = pd.date_range(recorded_periods.min(), recorded_periods.max(), freq="7D")
    else:
        calendar = pd.date_range(recorded_periods.min(), recorded_periods.max(), freq="MS")

    observed = (
        work.loc[work["primary_topic"].astype(str).isin(selected_topics)]
        .groupby(["period", "primary_topic"], as_index=False)
        .size()
        .rename(columns={"size": "query_count"})
    )
    grid = pd.MultiIndex.from_product(
        [calendar, selected_topics], names=["period", "primary_topic"]
    ).to_frame(index=False)
    grid = grid.merge(observed, on=["period", "primary_topic"], how="left")
    recorded_mask = grid["period"].isin(recorded_periods)
    grid["period_status"] = recorded_mask.map({True: "Recorded", False: "No records"})
    grid.loc[recorded_mask, "query_count"] = grid.loc[recorded_mask, "query_count"].fillna(0)
    return grid[columns]


def portal_summary(events: pd.DataFrame) -> dict[str, float]:
    """Summarise selected synthetic Parent Portal events without double counting users."""
    if events.empty:
        return {
            "unique_users": 0, "first_time_users": 0, "returning_users": 0,
            "page_views": 0, "average_engagement_seconds": 0,
        }
    users = events.get("user_id_anonymised", pd.Series(dtype="string")).dropna().astype(str)
    unique_users = set(users)
    first_time_users: set[str] = set()
    if {"user_id_anonymised", "is_first_login"}.issubset(events.columns):
        first_time_users = set(
            events.loc[_truthy(events["is_first_login"]), "user_id_anonymised"]
            .dropna().astype(str)
        ) & unique_users
    returning_users = unique_users - first_time_users
    sessions = events.get("session_id", pd.Series(dtype="string")).dropna().nunique()
    engagement = pd.to_numeric(
        events.get("engagement_time", pd.Series(0, index=events.index)), errors="coerce"
    ).fillna(0).clip(lower=0).sum()
    return {
        "unique_users": float(len(unique_users)),
        "first_time_users": float(len(first_time_users)),
        "returning_users": float(len(returning_users)),
        "page_views": float(events.get("event_name", pd.Series(dtype="string")).eq("page_view").sum()),
        "average_engagement_seconds": float(engagement / sessions) if sessions else 0,
    }


def aggregate_portal_usage(events: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Aggregate portal activity; unique users are recomputed inside each period."""
    required = {"event_timestamp", "event_name", "user_id_anonymised", "session_id"}
    if events.empty or not required.issubset(events.columns):
        return pd.DataFrame(columns=["period", "unique_users", "page_views", "sessions"])
    work = events.copy()
    timestamps = pd.to_datetime(work["event_timestamp"], errors="coerce", utc=True)
    work = work.loc[timestamps.notna()].copy()
    work["period"] = time_bucket(work["event_timestamp"], granularity)
    return (
        work.groupby("period", as_index=False)
        .agg(
            unique_users=("user_id_anonymised", "nunique"),
            sessions=("session_id", "nunique"),
            page_views=("event_name", lambda values: values.eq("page_view").sum()),
        )
        .sort_values("period")
    )


def portal_monthly(events: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible monthly portal aggregation."""
    return aggregate_portal_usage(events, "Month").rename(columns={"period": "month"})


def portal_user_mix(events: pd.DataFrame) -> pd.DataFrame:
    summary = portal_summary(events)
    total = summary["unique_users"]
    output = pd.DataFrame({
        "user_type": ["First-time users", "Returning users"],
        "user_count": [summary["first_time_users"], summary["returning_users"]],
    })
    output["share"] = output["user_count"] / total if total else 0
    return output


def portal_activity_heatmap(events: pd.DataFrame) -> pd.DataFrame:
    """Count synthetic PP page views by weekday and hour in Singapore Time."""
    if events.empty or not {"event_timestamp", "event_name"}.issubset(events.columns):
        matrix = pd.DataFrame(float("nan"), index=DAY_ORDER, columns=range(24))
        matrix.attrs.update(invalid_timestamp_count=0, coverage_available=False)
        return matrix
    work = events.loc[events["event_name"].eq("page_view")].copy()
    timestamps = pd.to_datetime(work["event_timestamp"], errors="coerce", utc=True)
    invalid_count = int(timestamps.isna().sum())
    valid_mask = timestamps.notna()
    if not valid_mask.any():
        matrix = pd.DataFrame(float("nan"), index=DAY_ORDER, columns=range(24))
        matrix.attrs.update(
            invalid_timestamp_count=invalid_count,
            coverage_available=not work.empty and invalid_count == 0,
        )
        return matrix
    work = work.loc[valid_mask].copy()
    timestamps = timestamps.loc[valid_mask].dt.tz_convert("Asia/Singapore")
    work["weekday"] = timestamps.dt.day_name()
    work["hour"] = timestamps.dt.hour
    matrix = work.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    matrix = matrix.reindex(index=DAY_ORDER, columns=range(24), fill_value=0)
    matrix.attrs.update(invalid_timestamp_count=invalid_count, coverage_available=True)
    return matrix


def engagement_funnel(events: pd.DataFrame) -> pd.DataFrame:
    """Create a proposed user-level funnel from available synthetic identifiers."""
    if events.empty or "user_id_anonymised" not in events:
        return pd.DataFrame({"stage": [], "users": []})
    users = set(events["user_id_anonymised"].dropna().astype(str))
    entry_mask = (
        _truthy(events["chat_entry_event_flag"])
        if "chat_entry_event_flag" in events else pd.Series(False, index=events.index)
    )
    message_mask = (
        _truthy(events["chat_message_event_flag"])
        if "chat_message_event_flag" in events else events.get("event_name", pd.Series(index=events.index)).eq("chat_message_sent")
    )
    opened = set(events.loc[entry_mask, "user_id_anonymised"].dropna().astype(str)) & users
    messaged = set(events.loc[message_mask, "user_id_anonymised"].dropna().astype(str)) & opened
    return pd.DataFrame({
        "stage": ["PP users", "Opened Sprout", "Sent a user message"],
        "users": [len(users), len(opened), len(messaged)],
    })


def entry_point_performance(events: pd.DataFrame) -> pd.DataFrame:
    """Return date-filterable session-level entry and activation counts by route."""
    required = {"session_id", "entry_point"}
    if events.empty or not required.issubset(events.columns):
        return pd.DataFrame(columns=["entry_point", "entry_sessions", "active_chat_sessions", "activation_rate"])
    work = events.dropna(subset=["session_id"]).copy()
    work["entry_point"] = work["entry_point"].fillna("unknown").astype(str)
    entry = _truthy(work.get("chat_entry_event_flag", pd.Series(False, index=work.index)))
    active = _truthy(work.get("chat_message_event_flag", pd.Series(False, index=work.index)))
    entry_sessions = work.loc[entry].groupby("entry_point")["session_id"].nunique()
    active_sessions = work.loc[active].groupby("entry_point")["session_id"].nunique()
    output = pd.concat(
        [entry_sessions.rename("entry_sessions"), active_sessions.rename("active_chat_sessions")],
        axis=1,
    ).fillna(0).reset_index()
    for column in ["entry_sessions", "active_chat_sessions"]:
        output[column] = output[column].astype(int)
    output["activation_rate"] = (
        output["active_chat_sessions"] / output["entry_sessions"].replace(0, pd.NA)
    )
    return output.sort_values("entry_sessions", ascending=False).reset_index(drop=True)


def question_input_types(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "event_name" not in events:
        return pd.DataFrame(columns=["input_type", "message_count"])
    messages = events.loc[events["event_name"].eq("chat_message_sent")].copy()
    if messages.empty:
        return pd.DataFrame(columns=["input_type", "message_count"])
    triggers = messages.get("trigger", pd.Series("unknown", index=messages.index)).fillna("unknown").astype(str)
    labels = triggers.map({"quick_reply": "Quick reply", "free_text": "Free text"}).fillna("Unknown")
    return labels.value_counts().rename_axis("input_type").reset_index(name="message_count")


def conversation_intensity(conversations: pd.DataFrame) -> pd.DataFrame:
    key = "conversation_id" if "conversation_id" in conversations else "session_id"
    if conversations.empty or key not in conversations:
        return pd.DataFrame(columns=["intensity", "conversations"])
    counts = conversations.dropna(subset=[key]).groupby(key).size()
    labels = pd.cut(
        counts, bins=[0, 1, 3, float("inf")],
        labels=["1 question", "2–3 questions", "4+ questions"], right=True,
    )
    return (
        labels.value_counts(sort=False)
        .rename_axis("intensity").reset_index(name="conversations")
    )


def conversation_quality_trend(
    conversations: pd.DataFrame, granularity: str = "Week"
) -> pd.DataFrame:
    """Aggregate available synthetic evaluation and feedback flags by calendar period."""
    if conversations.empty or "timestamp" not in conversations:
        return pd.DataFrame()
    work = conversations.copy()
    timestamps = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
    work = work.loc[timestamps.notna()].copy()
    if work.empty:
        return pd.DataFrame()
    timestamps = timestamps.loc[timestamps.notna()]
    work["period"] = time_bucket(work["timestamp"], granularity)
    work["fallback_count"] = _truthy(work.get("fallback_flag", pd.Series(False, index=work.index))).astype(int)
    work["guardrail_count"] = _truthy(work.get("guardrail_flag", pd.Series(False, index=work.index))).astype(int)
    success = work.get("success_flag", pd.Series(pd.NA, index=work.index)).astype("string").str.lower()
    work["unsuccessful_count"] = success.isin({"false", "0", "no", "n"}).astype(int)
    feedback = work.get("user_feedback", pd.Series("", index=work.index)).fillna("").astype(str).str.lower()
    work["positive_feedback"] = feedback.eq("positive").astype(int)
    work["negative_feedback"] = feedback.eq("negative").astype(int)
    return (
        work.groupby("period", as_index=False)
        .agg(
            questions=("query_id", "count") if "query_id" in work else ("period", "size"),
            fallback_count=("fallback_count", "sum"),
            guardrail_count=("guardrail_count", "sum"),
            unsuccessful_count=("unsuccessful_count", "sum"),
            positive_feedback=("positive_feedback", "sum"),
            negative_feedback=("negative_feedback", "sum"),
        )
        .sort_values("period")
    )


def language_distribution(conversations: pd.DataFrame) -> pd.DataFrame:
    """Use a supplied language field only; never infer language from sensitive text."""
    if conversations.empty or "language" not in conversations:
        return pd.DataFrame(columns=["language", "question_count"])
    labels = conversations["language"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    return labels.value_counts().rename_axis("language").reset_index(name="question_count")


def operational_latency_samples(conversations: pd.DataFrame) -> pd.DataFrame:
    """Create one deterministic illustrative latency observation per demo question."""
    if conversations.empty or "timestamp" not in conversations:
        return pd.DataFrame(columns=["timestamp", "latency_ms"])
    timestamps = pd.to_datetime(conversations["timestamp"], errors="coerce", utc=True)
    output = pd.DataFrame({"timestamp": timestamps}).dropna().sort_values("timestamp").reset_index(drop=True)
    if output.empty:
        return pd.DataFrame(columns=["timestamp", "latency_ms"])
    daily_sequence = output.groupby(output["timestamp"].dt.date).cumcount()
    day_number = (output["timestamp"].dt.floor("D") - output["timestamp"].dt.floor("D").min()).dt.days
    output["latency_ms"] = 650 + ((day_number * 41 + daily_sequence * 137) % 1500)
    return output


def operational_demo(conversations: pd.DataFrame) -> pd.DataFrame:
    """Build stable illustrative operations data from the demo activity calendar.

    The formulas intentionally avoid random values. They demonstrate a future
    monitoring contract and must not be interpreted as measured service levels.
    """
    if conversations.empty or "timestamp" not in conversations:
        return pd.DataFrame()
    timestamps = pd.to_datetime(conversations["timestamp"], errors="coerce", utc=True)
    valid = conversations.loc[timestamps.notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["date"] = timestamps.loc[timestamps.notna()].dt.date
    volume = valid.groupby("date").size()
    latency_samples = operational_latency_samples(valid)
    latency_samples["date"] = latency_samples["timestamp"].dt.date
    calendar = pd.date_range(min(volume.index), max(volume.index), freq="D")
    rows = []
    for index, timestamp in enumerate(calendar):
        day = timestamp.date()
        questions = int(volume.get(day, 0))
        request_count = questions
        availability = min(0.9999, max(0.99, 0.9972 + ((index * 7) % 6) * 0.00035 - (0.0012 if index and index % 23 == 0 else 0)))
        error_rate = (1 + ((index * 5) % 4) + questions * 0.15) / 1000
        latency_population = latency_samples.loc[latency_samples["date"].eq(day), "latency_ms"]
        p50 = float(latency_population.quantile(.50)) if request_count else 0
        p95 = float(latency_population.quantile(.95)) if request_count else 0
        average = float(latency_population.mean()) if request_count else 0
        input_tokens = questions * (850 + (index % 5) * 45)
        output_tokens = questions * (260 + (index % 4) * 30)
        cost = (
            input_tokens / 1_000_000 * DEMO_COST_PER_MILLION_INPUT_TOKENS
            + output_tokens / 1_000_000 * DEMO_COST_PER_MILLION_OUTPUT_TOKENS
        )
        rows.append({
            "date": day,
            "question_count": questions,
            "request_count": request_count,
            "availability": availability,
            "api_error_rate": error_rate,
            "critical_alerts": 1 if index and index % 31 == 0 else 0,
            "warning_alerts": int((index + questions) % 3 == 0) + int(index > 0 and index % 17 == 0),
            "average_response_ms": round(average),
            "p50_response_ms": round(p50),
            "p95_response_ms": round(p95),
            "llm_judge_flagged": min(questions, round(questions * 0.08)),
            "human_reviewed": min(questions, round(questions * 0.30)),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": cost,
            "data_status": "Illustrative data — pending technical validation",
        })
    return pd.DataFrame(rows)
