"""
AI Utilities for Insightera Dashboard
======================================
Builds data context summaries and wraps Claude API calls.

Design notes:
- Data context goes into the `system` prompt so it's sent once per turn rather
  than re-injected into every user message (token savings on long chats).
- Conversation history is trimmed to the last `MAX_HISTORY_TURNS` turns to keep
  request size bounded.
- Calls are wrapped with exponential-backoff retry on transient errors.
"""

from __future__ import annotations

import os
import random
import time
from typing import Optional

import anthropic
import pandas as pd

_client: Optional[anthropic.Anthropic] = None

MODEL_ID = "claude-opus-4-6"
MAX_TOKENS_CHAT = 1024
MAX_TOKENS_INSIGHT = 512
MAX_HISTORY_TURNS = 12  # keep last 12 messages (≈6 round-trips)
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0  # seconds, exponential backoff


# ══════════════════════════════════════════════════════════════════════════════
#  CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it with: export ANTHROPIC_API_KEY=your_key"
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def reset_client() -> None:
    """Force the client to be re-created on next call (e.g. after API key change)."""
    global _client
    _client = None


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CONTEXT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_kpi_context(profiles: pd.DataFrame, events: pd.DataFrame) -> str:
    total_users = len(profiles)
    total_events = len(events)
    if total_users == 0:
        return "KEY METRICS:\n- No users in current filter."
    conversion_rate = (profiles["converted"].mean() * 100).round(1) if "converted" in profiles else 0
    churn_rate = (profiles["churned"].mean() * 100).round(1) if "churned" in profiles else 0
    onboarding_rate = (profiles["completed_onboarding"].mean() * 100).round(1) if "completed_onboarding" in profiles else 0
    avg_active_days = profiles["active_days"].mean().round(1) if "active_days" in profiles else 0
    avg_events_per_user = total_events / total_users

    top_events = events["event_name"].value_counts().head(8)
    top_events_str = ", ".join(f"{evt} ({cnt:,})" for evt, cnt in top_events.items())

    return f"""
KEY METRICS:
- Total users: {total_users:,}
- Total events: {total_events:,}
- Avg events/user: {avg_events_per_user:.1f}
- Conversion rate (paid at least once): {conversion_rate}%
- Churn rate: {churn_rate}%
- Onboarding completion rate: {onboarding_rate}%
- Avg active days per user: {avg_active_days}

TOP EVENTS (by volume):
{top_events_str}
""".strip()


def build_cluster_context(profiles: pd.DataFrame) -> str:
    if "cluster_label" not in profiles.columns or profiles.empty:
        return "USER SEGMENTS: none available."
    summary = (
        profiles.groupby("cluster_label")
        .agg(
            users=("total_events", "size"),
            avg_events=("total_events", "mean"),
            avg_revenue=("revenue_events", "mean"),
            avg_active_days=("active_days", "mean"),
            avg_features=("feature_breadth", "mean"),
            churn_pct=("churned", "mean"),
        )
        .round(2)
    )
    summary["churn_pct"] = (summary["churn_pct"] * 100).round(1)
    lines = ["USER SEGMENTS:"]
    for label, row in summary.iterrows():
        lines.append(
            f"  {label}: {int(row['users'])} users, "
            f"avg {row['avg_events']:.0f} events, "
            f"avg {row['avg_revenue']:.1f} revenue events, "
            f"{row['churn_pct']}% churn, "
            f"{row['avg_features']:.1f} features used"
        )
    return "\n".join(lines)


def build_funnel_context(events: pd.DataFrame, profiles: pd.DataFrame) -> str:
    funnel_steps = [
        ("signup", "Signup"),
        ("onboarding_step1", "Onboarding Step 1"),
        ("onboarding_step2", "Onboarding Step 2"),
        ("view_pricing", "View Pricing"),
        ("start_checkout", "Start Checkout"),
        ("payment_success", "Payment Success"),
    ]
    total = len(profiles)
    if total == 0:
        return "CONVERSION FUNNEL: no users in current filter."
    lines = ["CONVERSION FUNNEL:"]
    for evt, label in funnel_steps:
        n = events[events["event_name"] == evt]["user_id"].nunique()
        pct = (n / total * 100) if total else 0
        lines.append(f"  {label}: {n:,} users ({pct:.1f}%)")
    return "\n".join(lines)


def build_churn_context(profiles: pd.DataFrame) -> str:
    if "churned" not in profiles.columns or profiles.empty:
        return "CHURN ANALYSIS: unavailable."
    churned = profiles[profiles["churned"] == 1]
    retained = profiles[profiles["churned"] == 0]
    if churned.empty or retained.empty:
        return f"CHURN ANALYSIS: {len(churned)} churned vs {len(retained)} retained — skipping breakdown."

    metrics = ["total_events", "active_days", "feature_breadth",
               "revenue_events", "n_rage_click", "n_error", "session_count"]
    metrics = [m for m in metrics if m in profiles.columns]

    lines = ["CHURN ANALYSIS (churned vs retained avg):"]
    for m in metrics:
        c_val = churned[m].mean()
        r_val = retained[m].mean()
        lines.append(f"  {m}: churned={c_val:.2f}, retained={r_val:.2f}")
    return "\n".join(lines)


def build_full_context(
    profiles: pd.DataFrame,
    events: pd.DataFrame,
    filter_label: Optional[str] = None,
) -> str:
    """Full analytics context for the AI.

    filter_label: optional human-readable filter description, e.g.
    "segment=Rage Clickers, date 2025-01-01 → 2025-03-31". When set, it's
    prepended so Claude knows the numbers below reflect a filtered slice.
    """
    parts = []
    if filter_label:
        parts.append(f"CURRENT FILTER: {filter_label}")
        parts.append(
            "(All numbers below reflect this filter. If the user asks about "
            "unfiltered totals, note that the current view is restricted.)")
        parts.append("")
    parts.extend([
        build_kpi_context(profiles, events),
        "",
        build_cluster_context(profiles),
        "",
        build_funnel_context(events, profiles),
        "",
        build_churn_context(profiles),
    ])
    return "\n".join(parts)


SYSTEM_PROMPT_BASE = """You are an expert product analytics AI assistant for Insightera, a SaaS analytics platform.
You have access to real user behavioural data and financial metrics.
Your job is to provide clear, actionable insights to help product teams understand their users, improve retention, and grow revenue.

When answering:
- Be concise and direct
- Lead with the most important finding
- Provide specific numbers from the data when relevant
- Suggest concrete next actions when asked
- Use plain language, avoid jargon
- Format with short paragraphs or bullet points for readability"""


def _build_system_prompt(data_context: str) -> str:
    """Compose the system prompt with the current data context embedded."""
    return (
        f"{SYSTEM_PROMPT_BASE}\n\n"
        f"=== CURRENT PRODUCT ANALYTICS SNAPSHOT ===\n{data_context}\n"
        f"=== END SNAPSHOT ===\n\n"
        f"Always ground your answers in the snapshot above. If the user asks "
        f"about something not present in the snapshot, say so explicitly."
    )


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep only the most recent MAX_HISTORY_TURNS messages."""
    if len(history) <= MAX_HISTORY_TURNS:
        return list(history)
    # Keep the most recent messages, ensure we start with a user turn
    trimmed = history[-MAX_HISTORY_TURNS:]
    while trimmed and trimmed[0].get("role") != "user":
        trimmed = trimmed[1:]
    return trimmed


# ══════════════════════════════════════════════════════════════════════════════
#  CLAUDE API CALLS WITH RETRY
# ══════════════════════════════════════════════════════════════════════════════

_TRANSIENT_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


def _call_with_retry(create_fn, **kwargs):
    """Call Anthropic API with exponential-backoff retry on transient errors."""
    last_err = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return create_fn(**kwargs)
        except _TRANSIENT_ERRORS as e:
            last_err = e
            if attempt + 1 == RETRY_ATTEMPTS:
                break
            # Exponential backoff with jitter
            delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(delay)
    raise last_err  # all retries exhausted


def ask_claude(
    question: str,
    data_context: str,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """Send a question to Claude with the data context in the system prompt.

    conversation_history is pure turns (user + assistant), no context prefix.
    Returns the assistant's reply as a string.
    """
    client = get_client()
    history = _trim_history(conversation_history or [])
    messages = history + [{"role": "user", "content": question}]

    response = _call_with_retry(
        client.messages.create,
        model=MODEL_ID,
        max_tokens=MAX_TOKENS_CHAT,
        system=_build_system_prompt(data_context),
        messages=messages,
    )
    return response.content[0].text


def generate_tab_insight(tab_name: str, data_context: str) -> str:
    """Generate a concise insight summary for a specific dashboard tab."""
    prompts = {
        # Current dashboard_flows.py tab names
        "overview": (
            "Based on the data, give me the 3 most important KPI-level insights: "
            "what's healthy, what's concerning, and what is the single highest-leverage "
            "fix. Be specific with numbers. Max 150 words."
        ),
        "conversion": (
            "Based on the data, give me 3 key insights about the conversion funnel — "
            "where users drop off, which actions most strongly predict payment, and "
            "what should be changed in onboarding. Be specific with numbers. Max 150 words."
        ),
        "engagement": (
            "Based on the data, give me 3 key insights about user engagement — "
            "activity patterns, friction points / self-loops, and which user segments "
            "are the most engaged. Be specific with numbers. Max 150 words."
        ),
        "retention": (
            "Based on the data, give me 3 key insights about retention — "
            "cohort behaviour, stickiness (DAU/MAU), and the biggest retention "
            "leaks by revenue impact. Be specific with numbers. Max 150 words."
        ),
        "churn": (
            "Based on the data, give me 3 key insights about churn — "
            "what distinguishes churned from retained users, which clusters are "
            "most at risk, and the top win-back actions to prioritise. "
            "Be specific with numbers. Max 150 words."
        ),
        # Legacy names kept for backwards compatibility
        "flows": "Give me 3 key insights about user journey flows and conversion drop-offs. Max 150 words.",
        "features": "Give me 3 key insights about feature adoption. Max 150 words.",
        "clusters": "Give me 3 key insights about user segments and how to move them up-tier. Max 150 words.",
        "revenue": "Give me 3 key insights about revenue patterns and churn risk. Max 150 words.",
    }
    question = prompts.get(tab_name, "Give me 3 key insights from this data. Max 150 words.")

    client = get_client()
    response = _call_with_retry(
        client.messages.create,
        model=MODEL_ID,
        max_tokens=MAX_TOKENS_INSIGHT,
        system=_build_system_prompt(data_context),
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


# ══════════════════════════════════════════════════════════════════════════════
#  AI-ASSISTED ETL: infer a mapping from a sample of raw data
# ══════════════════════════════════════════════════════════════════════════════

MAPPING_SYSTEM_PROMPT = """You are a senior data engineer specialising in product
analytics ETL. Your job is to map a customer's raw event data to a canonical
event taxonomy used by the Insightera analytics platform.

You will be given:
  1. The list of canonical events Insightera expects.
  2. The column names in the customer's events (and users) table.
  3. A histogram of raw event names with their frequencies.
  4. A small sample of rows.

You MUST respond with STRICT YAML (no commentary before or after) matching this
exact schema:

  source_name: <short slug for this data source>
  description: <one-line human-readable summary>
  events_columns:
    user_id:      <raw column name>
    event_name:   <raw column name>
    timestamp:    <raw column name>
    # Optional: session_id, properties, platform, country
  event_map:
    # canonical_name: [list of raw event names to fold into it]
    signup:         [<raw>, <raw>, ...]
    payment_success:[<raw>]
    # ...
  users_columns:
    user_id:   <raw column>
    signup_at: <raw column or omit>
    email:     <raw column or omit>
  unmapped_events:
    - <raw event name that doesn't match any canonical event>
  notes: |
    Brief rationale (2-4 lines) about decisions you made, especially
    any raw events you left unmapped or merged aggressively.

RULES:
  - Only use CANONICAL event names from the list you're given.
  - One raw event can map to AT MOST one canonical event.
  - If a raw event has no good canonical home, put it in `unmapped_events`
    rather than forcing a bad mapping.
  - Prefer precision over coverage: when in doubt, leave unmapped.
  - If a canonical event has no raw equivalent, omit it (don't fabricate).
  - Respond with YAML only. No prose before or after.
"""


def infer_data_mapping(
    probe_summary: str,
    canonical_events: list[str],
    extra_hints: str = "",
) -> str:
    """Ask Claude to propose a mapping YAML for a new data source.

    Args:
        probe_summary: text block from `SourceProbe.to_summary()`.
        canonical_events: list of canonical event names the taxonomy defines.
        extra_hints: optional free-form hints from the user (e.g. "this is
            from Segment, our signup event is called user_registered").

    Returns:
        Raw YAML text. Caller should parse with yaml.safe_load and construct
        a DataMapping from it. The caller is expected to validate + show the
        proposed mapping to the user for review.
    """
    canon_text = ", ".join(sorted(canonical_events))
    hints = f"\n\nADDITIONAL HINTS FROM USER:\n{extra_hints}\n" if extra_hints.strip() else ""
    user_msg = (
        f"CANONICAL EVENTS (use these exact names in event_map keys):\n"
        f"{canon_text}\n\n"
        f"SOURCE PROBE:\n{probe_summary}"
        f"{hints}\n\n"
        f"Produce the YAML mapping now."
    )

    client = get_client()
    response = _call_with_retry(
        client.messages.create,
        model=MODEL_ID,
        max_tokens=2048,
        system=MAPPING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    # Claude sometimes wraps YAML in ``` fences — strip them defensively.
    text = response.content[0].text.strip()
    for fence in ("```yaml", "```YAML", "```yml", "```"):
        if text.startswith(fence):
            text = text[len(fence):].lstrip("\n")
    if text.endswith("```"):
        text = text[: -3].rstrip()
    return text
