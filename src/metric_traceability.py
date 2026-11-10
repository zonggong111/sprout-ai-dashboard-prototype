"""Stable business-metric traceability for the synthetic dashboard.

The catalogue is code-only runtime metadata.  It mirrors the reviewed metric
names in ``PP_Sprout_Analytics_Metrics_Mapping_v0.2 @CH.xlsx`` without shipping
that working workbook in the public deployment repository.
"""

from __future__ import annotations

import pandas as pd


METRIC_TRACEABILITY = [
    # Overview & Usage
    ("PP-01", "Unique PP users", "Parent Portal Reach", "Reach KPIs", "pp-reach-kpis", "Shown"),
    ("PP-02", "First-time PP users", "Parent Portal Reach", "Reach KPIs", "pp-reach-kpis", "Shown"),
    ("PP-03", "Returning PP users", "Parent Portal Reach", "Reach KPIs", "pp-reach-kpis", "Shown"),
    ("PP-04", "Page views by page", "Parent Portal Reach", "Reach and activity", "pp-usage-trend", "Partial"),
    ("PP-05", "Average engagement time", "Parent Portal Reach", "Reach KPIs", "pp-reach-kpis", "Shown"),
    ("PP-06", "Usage by access channel", "Parent Portal Reach", "Awaiting source validation", "pp-validation", "Requires Validation"),
    ("PP-07", "CTA clicks by action", "Parent Portal Reach", "Tracked interactions", "pp-actions", "Shown"),
    ("PP-08", "View Health Plan clicks", "Parent Portal Reach", "Tracked interactions", "pp-actions", "Shown"),
    ("PP-09", "Resource card clicks by pillar", "Parent Portal Reach", "Tracked interactions", "pp-resource-pillars", "Shown"),
    ("PP-10", "Scroll depth", "Parent Portal Reach", "Awaiting source validation", "pp-validation", "Requires Validation"),
    ("SU-01", "Chat initiation rate", "Entry & Journey", "Conversion journey", "journey-kpis", "Shown"),
    ("SU-02", "PP-to-Sprout engagement funnel", "Entry & Journey", "Engagement funnel", "journey-funnel", "Shown"),
    ("SU-03", "Sprout entries by entry point", "Entry & Journey", "Entry-point performance", "entry-performance", "Shown"),
    ("SU-04", "Quick-reply usage by pillar", "Sprout Usage", "Input method", "sprout-input-method", "Partial"),
    ("SU-05", "User question input type", "Sprout Usage", "Input method", "sprout-input-method", "Shown"),
    ("SU-06", "Unique Sprout chat users", "Sprout Usage", "Usage KPIs", "sprout-usage-kpis", "Shown"),
    ("SU-07", "Chat message volume", "Sprout Usage", "Usage KPIs", "sprout-usage-kpis", "Shown"),
    ("SU-08", "Average user messages per chat user", "Sprout Usage", "Usage KPIs", "sprout-usage-kpis", "Shown"),
    ("SU-09", "Chat sessions (conversations)", "Sprout Usage", "Usage KPIs", "sprout-usage-kpis", "Shown"),
    ("SU-10", "Average user messages per session", "Sprout Usage", "Usage KPIs", "sprout-usage-kpis", "Shown"),
    ("SU-11", "Chat activity by hour and weekday", "Sprout Usage", "Awaiting source validation", "sprout-validation", "Requires Validation"),
    ("SU-12", "Bot response latency", "Operations & AI Quality", "Service Reliability", "ops-latency", "Proposed"),
    # Parent Needs / Feedback & Quality
    ("FQ-01", "Sprout response ratings", "Feedback Overview", "Feedback balance", "feedback-balance", "Shown"),
    ("FQ-02", "Negative response feedback share", "Feedback Overview", "Feedback KPIs", "feedback-kpis", "Shown"),
    ("FQ-03", "Chat feedback reasons", "Feedback Overview", "Reason breakdown", "feedback-reasons", "Shown"),
    ("FQ-04", "PP application feedback", "Feedback Overview", "Excluded pending clarification", "feedback-out-of-scope", "Out of Scope"),
    ("FQ-05", "Top topics discussed", "Topics & Trends", "Topic distribution", "topic-distribution", "Shown"),
    ("FQ-06", "User message language distribution", "Sprout Usage", "Awaiting source validation", "sprout-validation", "Requires Validation"),
    ("FQ-07", "Translation request volume", "Sprout Usage", "Awaiting use-case validation", "sprout-validation", "Requires Validation"),
    ("FQ-08", "Repeated or rephrased questions", "Repeated Questions", "Repeated-question clusters", "repeated-clusters", "Shown"),
    ("FQ-09", "Follow-up questions within a session", "Sprout Usage", "Conversation intensity", "sprout-intensity", "Shown"),
    ("FQ-10", "Multi-turn conversation distribution", "Sprout Usage", "Conversation intensity", "sprout-intensity", "Shown"),
    ("FQ-11", "Potentially unresolved questions", "Feedback Overview", "Response-quality signals", "quality-signals", "Partial"),
    ("FQ-12", "LLM Judge flagged conversations", "Operations & AI Quality", "AI Evaluation & Human Review", "ops-review", "Proposed"),
    ("FQ-13", "Conversation human-review coverage", "Operations & AI Quality", "AI Evaluation & Human Review", "ops-review", "Proposed"),
    ("FQ-14", "Guardrail-triggering topics", "Feedback Overview", "Awaiting source validation", "quality-validation", "Requires Validation"),
    ("FQ-15", "PII detection events by category", "Operations & AI Quality", "Awaiting source validation", "ops-validation", "Requires Validation"),
    ("FQ-16", "Fallback / no-response incidents by reason", "Feedback Overview", "Response-quality signals", "quality-signals", "Partial"),
    ("FQ-17", "Unable-to-answer topics", "Feedback Overview", "Response-quality signals", "quality-signals", "Partial"),
    ("FQ-18", "RAG retrieval source distribution", "Operations & AI Quality", "Awaiting source validation", "ops-validation", "Requires Validation"),
    # Operations
    ("OP-01", "Sprout service availability", "Operations & AI Quality", "Service Reliability", "ops-reliability", "Proposed"),
    ("OP-02", "Sprout API error rate", "Operations & AI Quality", "Service Reliability", "ops-reliability", "Proposed"),
    ("OP-03", "Critical and warning alerts", "Operations & AI Quality", "Service Reliability", "ops-alerts", "Proposed"),
    ("OP-04", "API requests, token usage and cost", "Operations & AI Quality", "Token Usage & Cost", "ops-cost", "Proposed"),
]


def metric_traceability(page: str | None = None) -> pd.DataFrame:
    """Return the stable catalogue, optionally scoped to one dashboard page."""
    columns = ["Metric ID", "Metric", "Page", "Section", "Visual ID", "Status"]
    output = pd.DataFrame(METRIC_TRACEABILITY, columns=columns)
    if page is not None:
        output = output.loc[output["Page"].eq(page)]
    return output.reset_index(drop=True)

