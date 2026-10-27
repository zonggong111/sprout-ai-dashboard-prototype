"""Reusable feedback submission normalization and analytics."""
import json
import re
import pandas as pd


def feedback_outputs(ga_feedback: pd.DataFrame, conversations: pd.DataFrame) -> dict:
    """Export GA feedback overview separately from response-linked negative cases."""
    counts = (ga_feedback.set_index('feedback_type')['feedback_count']
              if not ga_feedback.empty else pd.Series(dtype=float))
    total = int(counts.sum())
    negative = int(counts.get('negative', 0))
    overview = pd.DataFrame([dict(
        total_feedback_submissions=total, positive_count=int(counts.get('positive', 0)),
        negative_count=negative, general_count=int(counts.get('general', 0)),
        negative_feedback_share=negative / total if total else None,
        source_dataset='GA engagement',
    )])
    field = 'user_feedback' if 'user_feedback' in conversations else 'feedback_type'
    raw = conversations.loc[conversations[field].fillna('').astype(str).str.lower().eq('negative')] if field in conversations else pd.DataFrame()
    normalized = normalise_feedback_submissions(raw)
    aliases = {'primary_topic': 'question_category', 'user_feedback': 'feedback_type',
               'user_type': 'journey_stage', 'session_id': 'anonymised_session_id'}
    columns = ['feedback_id', 'query_id', 'conversation_id', 'response_id', 'message_id',
               'question_category', 'user_query', 'ai_response', 'feedback_type',
               'feedback_comment', 'timestamp', 'journey_stage', 'entry_point',
               'anonymised_session_id', 'beta_or_production', 'data_notice']
    cases = normalized.copy()
    for source, target in aliases.items():
        if source in cases and target not in cases:
            cases[target] = cases[source]
    for column in columns:
        if column not in cases:
            cases[column] = pd.Series(index=cases.index, dtype=object)
    if not cases.empty:
        missing = cases['feedback_id'].fillna('').astype(str).str.strip().eq('')
        cases.loc[missing, 'feedback_id'] = normalized.loc[missing, '_submission_key'].map(lambda value: 'generated:' + value)
    links = pd.DataFrame([
        {'feedback_id': cases.loc[i, 'feedback_id'], 'feedback_reason': reason}
        for i, row in normalized.iterrows() for reason in row['_feedback_reasons']
    ], columns=['feedback_id', 'feedback_reason']).drop_duplicates()
    summary = links.groupby('feedback_reason').size().reset_index(name='negative_case_count')
    summary['negative_feedback_submissions'] = len(cases)
    summary['percentage_of_negative_cases'] = summary['negative_case_count'] / len(cases) if len(cases) else pd.Series(dtype=float)
    joined = links.merge(cases[['feedback_id', 'question_category']], on='feedback_id', validate='many_to_one')
    joined['question_category'] = joined['question_category'].fillna('Unclassified')
    category = joined.groupby(['feedback_reason', 'question_category']).size().reset_index(name='case_count')
    totals = cases.assign(question_category=cases['question_category'].fillna('Unclassified')).groupby('question_category').size()
    category['category_negative_cases'] = category['question_category'].map(totals)
    return {'feedback_overview': overview, 'feedback_cases': cases[columns],
            'feedback_case_reasons': links, 'feedback_reason_summary': summary,
            'feedback_reason_by_category': category,
            'feedback_reason_matrix': category.copy()}

OFFICIAL_FEEDBACK_REASONS = [
    "Wrong information", "Missing details", "Not relevant", "Hard to understand",
    "Slow or not working", "Something else",
]

def parse_feedback_reasons(value: object) -> list[str]:
    """Preserve recorded reason labels while supporting common multi-select storage."""
    if value is None or (not isinstance(value, (list, tuple, set)) and pd.isna(value)):
        return []
    values: list[object]
    if isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        text_value = str(value).strip()
        if not text_value:
            return []
        values = []
        if text_value.startswith("["):
            try:
                decoded = json.loads(text_value)
                if isinstance(decoded, list):
                    values = decoded
            except (json.JSONDecodeError, TypeError):
                values = []
        if not values:
            values = re.split(r"\s*[|;]\s*", text_value)
    reasons: list[str] = []
    for item in values:
        reason = str(item).strip()
        if reason and reason not in reasons:
            reasons.append(reason)
    return reasons


def normalise_feedback_submissions(cases: pd.DataFrame) -> pd.DataFrame:
    """Collapse optional one-row-per-reason data to one row per feedback submission."""
    if cases.empty:
        return cases.copy()
    working = cases.copy().reset_index(drop=True)
    display_aliases = {
        "question_category": "primary_topic",
        "journey_stage": "user_type",
        "anonymised_session_id": "session_id",
    }
    for source, target in display_aliases.items():
        if target not in working and source in working:
            working[target] = working[source]
    id_columns = [
        column for column in ["feedback_id", "response_id", "message_id", "query_id"]
        if column in working
    ]

    def submission_key(row: pd.Series) -> str:
        for column in id_columns:
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                return f"{column}:{str(value).strip()}"
        return f"row:{row.name}"

    working["_submission_key"] = working.apply(submission_key, axis=1)
    records: list[dict] = []
    for key, group in working.groupby("_submission_key", sort=False, dropna=False):
        record = group.iloc[0].to_dict()
        reasons: list[str] = []
        if "feedback_reason" in group:
            for value in group["feedback_reason"]:
                for reason in parse_feedback_reasons(value):
                    if reason not in reasons:
                        reasons.append(reason)
        comments = (
            group.get("feedback_comment", pd.Series(dtype=object))
            .dropna().astype(str).str.strip()
        )
        comments = comments.loc[comments.ne("")]
        record["feedback_comment"] = comments.iloc[0] if not comments.empty else ""
        record["_feedback_reasons"] = tuple(reasons)
        record["feedback_reason"] = " | ".join(reasons)
        record["_has_written_feedback"] = bool(record["feedback_comment"])
        record["_submission_key"] = key
        records.append(record)
    return pd.DataFrame(records).reset_index(drop=True)


def feedback_reason_summary(submissions: pd.DataFrame) -> pd.DataFrame:
    """Count each selected reason once per negative feedback submission."""
    columns = ["Reason", "Count", "Count / denominator", "Percentage of negative feedback"]
    if submissions.empty or "_feedback_reasons" not in submissions:
        return pd.DataFrame(columns=columns)
    denominator = len(submissions)
    rows = [
        {"_submission_key": row["_submission_key"], "Reason": reason}
        for _, row in submissions.iterrows()
        for reason in row["_feedback_reasons"]
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    exploded = pd.DataFrame(rows).drop_duplicates()
    summary = exploded.groupby("Reason")["_submission_key"].nunique().reset_index(name="Count")
    order = {reason: position for position, reason in enumerate(OFFICIAL_FEEDBACK_REASONS)}
    summary["_order"] = summary["Reason"].map(order).fillna(len(order))
    summary = summary.sort_values(["_order", "Count", "Reason"], ascending=[True, False, True])
    summary["Count / denominator"] = summary["Count"].map(
        lambda count: f"{count} of {denominator} negative feedback submissions"
    )
    summary["Percentage of negative feedback"] = summary["Count"] / denominator
    return summary[columns].reset_index(drop=True)


def feedback_reason_matrix(submissions: pd.DataFrame) -> pd.DataFrame:
    """Return submission counts by parent need and selected feedback reason."""
    if submissions.empty or "primary_topic" not in submissions or "_feedback_reasons" not in submissions:
        return pd.DataFrame()
    rows = [
        {
            "Parent Need": row.get("primary_topic", "Unclassified"),
            "Feedback Reason": reason,
            "_submission_key": row["_submission_key"],
        }
        for _, row in submissions.iterrows()
        for reason in row["_feedback_reasons"]
    ]
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows).drop_duplicates()
    matrix = long.pivot_table(
        index="Parent Need", columns="Feedback Reason", values="_submission_key",
        aggfunc="nunique", fill_value=0,
    )
    ordered = [reason for reason in OFFICIAL_FEEDBACK_REASONS if reason in matrix.columns]
    extras = sorted(column for column in matrix.columns if column not in ordered)
    return matrix.reindex(columns=[*ordered, *extras]).astype(int)

