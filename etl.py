"""
ETL Pipeline for User Event Data + Stripe Financial Data
=========================================================
- Cleans messy event names, normalizes to canonical form
- Detects rage_click from rapid click bursts (not from source data)
- Standardises Stripe charges/subscriptions/invoices
- Engineers user profiles joining behaviour + financial data
"""

import os
import pandas as pd
import numpy as np
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#  EVENT NAME NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

EVENT_ALIASES = {
    "signup": "signup", "sigup": "signup", "user_signup": "signup",
    "onboarding_1": "onboarding_step1", "onboarding1": "onboarding_step1",
    "complete_onboarding_step1": "onboarding_step1",
    "onboarding_step_2": "onboarding_step2", "onboading_step2": "onboarding_step2",
    "complete_onboarding_step2": "onboarding_step2",
    "view_dashboard": "view_dashboard", "view_dash": "view_dashboard",
    "dashboard_view": "view_dashboard",
    "view_feature_A": "view_feature_A", "feat_a": "view_feature_A",
    "feature_a_click": "view_feature_A",
    "view_feature_B": "view_feature_B", "feautre_b": "view_feature_B",
    "featureB": "view_feature_B",
    "view_feature_C": "view_feature_C", "view_featc": "view_feature_C",
    "feature_c_view": "view_feature_C",
    "view_feature_D": "view_feature_D", "featureD": "view_feature_D",
    "f_d": "view_feature_D",
    "start_checkout": "start_checkout", "checkout_start": "start_checkout",
    "upgrade_plan": "plan_upgrade", "plan_upgrade": "plan_upgrade",
    "payment_success": "payment_success", "payment_ok": "payment_success",
    "payment_failed": "payment_failed", "payment_error": "payment_failed",
    "error_occurred": "error", "404_error": "error",
    "view_pricing": "view_pricing",
    "session_start": "session_start", "session_end": "session_end",
    "page_view": "page_view",
    "click": "click", "page_click": "click", "btn_click": "click",
    "invite_team": "invite_team",
    "contact_support": "contact_support",
}

EVENT_CATEGORIES = {
    "signup": "onboarding",
    "onboarding_step1": "onboarding",
    "onboarding_step2": "onboarding",
    "view_dashboard": "engagement",
    "view_feature_A": "feature_usage",
    "view_feature_B": "feature_usage",
    "view_feature_C": "feature_usage",
    "view_feature_D": "feature_usage",
    "view_pricing": "monetization",
    "start_checkout": "monetization",
    "plan_upgrade": "monetization",
    "payment_success": "monetization",
    "payment_failed": "monetization",
    "page_view": "engagement",
    "click": "engagement",
    "session_start": "session",
    "session_end": "session",
    "rage_click": "friction",
    "error": "friction",
    "invite_team": "engagement",
    "contact_support": "friction",
}


def normalize_event_name(raw_name: str) -> str:
    clean = raw_name.strip()
    if "_dup_" in clean:
        clean = clean[: clean.index("_dup_")]
    return EVENT_ALIASES.get(clean, clean)


# ══════════════════════════════════════════════════════════════════════════════
#  RAGE CLICK DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_rage_clicks(df: pd.DataFrame, threshold_sec: float = 2.0,
                       min_clicks: int = 3) -> pd.DataFrame:
    """
    Detect rage clicks: ≥ min_clicks 'click' events within threshold_sec
    seconds for the same user.  Replaces those bursts with a single
    'rage_click' event at the timestamp of the first click in the burst.
    """
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    click_mask = df["event_name"] == "click"
    clicks = df[click_mask].copy()
    if clicks.empty:
        return df

    clicks["prev_ts"] = clicks.groupby("user_id")["timestamp"].shift(1)
    clicks["gap_sec"] = (clicks["timestamp"] - clicks["prev_ts"]).dt.total_seconds()

    # Mark burst boundaries: a new burst starts when gap > threshold
    clicks["new_burst"] = (clicks["gap_sec"].isna()) | (clicks["gap_sec"] > threshold_sec)
    clicks["burst_id"] = clicks.groupby("user_id")["new_burst"].cumsum()

    # Find bursts with ≥ min_clicks
    burst_sizes = clicks.groupby(["user_id", "burst_id"]).size().reset_index(name="burst_size")
    rage_bursts = burst_sizes[burst_sizes["burst_size"] >= min_clicks]

    if rage_bursts.empty:
        return df

    # Get indices of click events that are part of rage bursts.
    # Preserve the original df index before merge — merge() resets it to 0,1,2,...
    # which would cause df.drop() to delete the wrong rows.
    clicks["_orig_idx"] = clicks.index
    clicks_with_burst = clicks.merge(rage_bursts, on=["user_id", "burst_id"], how="inner")
    rage_indices = clicks_with_burst["_orig_idx"].tolist()

    # For each rage burst, keep first click's timestamp as rage_click
    burst_firsts = (
        clicks_with_burst.groupby(["user_id", "burst_id"])
        .agg(timestamp=("timestamp", "first"), burst_size=("burst_size", "first"))
        .reset_index()
    )

    # Remove all clicks in rage bursts from main df
    df_clean = df.drop(index=rage_indices)

    # Insert rage_click events
    rage_events = pd.DataFrame({
        "user_id": burst_firsts["user_id"],
        "timestamp": burst_firsts["timestamp"],
        "event_name": "rage_click",
        "event_raw": "detected_rage_click",
        "event_category": "friction",
    })

    df_out = pd.concat([df_clean, rage_events], ignore_index=True)
    df_out = df_out.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
#  STRIPE DATA STANDARDISATION
# ══════════════════════════════════════════════════════════════════════════════

def load_stripe_data(data_dir: str) -> dict:
    """Load and standardise Stripe CSVs.  Returns dict of clean DataFrames."""
    charges_path = os.path.join(data_dir, "stripe_charges.csv")
    subs_path = os.path.join(data_dir, "stripe_subscriptions.csv")
    invoices_path = os.path.join(data_dir, "stripe_invoices.csv")

    result = {}

    if os.path.exists(charges_path):
        ch = pd.read_csv(charges_path)
        ch["created"] = pd.to_datetime(ch["created"])
        # Normalise currency to lowercase
        ch["currency"] = ch["currency"].str.lower().str.strip()
        # Amount from cents to dollars
        ch["amount_usd"] = ch["amount"] / 100.0
        ch["amount_refunded_usd"] = ch["amount_refunded"] / 100.0
        ch["net_amount_usd"] = ch["amount_usd"] - ch["amount_refunded_usd"]
        # Remove duplicate idempotency charges
        ch["is_duplicate"] = ch["failure_code"] == "idempotency_key_in_use"
        result["charges"] = ch

    if os.path.exists(subs_path):
        subs = pd.read_csv(subs_path)
        for col in ["created", "current_period_start", "current_period_end",
                     "canceled_at", "trial_start", "trial_end"]:
            if col in subs.columns:
                subs[col] = pd.to_datetime(subs[col], errors="coerce")
        subs["plan_currency"] = subs["plan_currency"].str.lower().str.strip()
        subs["plan_amount_usd"] = subs["plan_amount"] / 100.0
        result["subscriptions"] = subs

    if os.path.exists(invoices_path):
        inv = pd.read_csv(invoices_path)
        for col in ["created", "due_date", "paid_at", "period_start", "period_end",
                     "next_payment_attempt"]:
            if col in inv.columns:
                inv[col] = pd.to_datetime(inv[col], errors="coerce")
        inv["currency"] = inv["currency"].str.lower().str.strip()
        inv["amount_due_usd"] = inv["amount_due"] / 100.0
        inv["amount_paid_usd"] = inv["amount_paid"] / 100.0
        result["invoices"] = inv

    return result


def build_user_financials(stripe: dict) -> pd.DataFrame:
    """
    Build one row per user_id with financial metrics from Stripe data.
    """
    charges = stripe.get("charges")
    subs = stripe.get("subscriptions")
    if charges is None:
        return pd.DataFrame()

    # Exclude duplicate charges
    real_charges = charges[~charges["is_duplicate"]].copy()

    fin = pd.DataFrame()
    fin.index.name = "user_id"

    # Revenue metrics — single agg pass per filtered DataFrame
    successful = real_charges[real_charges["paid"] == True]
    succ_agg = successful.groupby("user_id").agg(
        total_revenue=("net_amount_usd", "sum"),
        n_successful_charges=("net_amount_usd", "size"),
        total_refunded=("amount_refunded_usd", "sum"),
        avg_charge_amount=("amount_usd", "mean"),
        first_charge_date=("created", "min"),
        last_charge_date=("created", "max"),
        n_payment_methods=("payment_method", "nunique"),
    )
    for col in succ_agg.columns:
        fin[col] = succ_agg[col]
    fin["n_failed_charges"] = (
        real_charges[real_charges["paid"] == False].groupby("user_id").size()
    )

    # Risk
    fin["avg_risk_score"] = real_charges.groupby("user_id")["risk_score"].mean()

    # Subscription info
    if subs is not None and not subs.empty:
        fin["plan_amount_usd"] = subs.groupby("user_id")["plan_amount_usd"].max()
        fin["sub_status"] = subs.groupby("user_id")["status"].first()
        _trial_users = subs.loc[subs["trial_start"].notna(), "user_id"].unique()
        fin["has_trial"] = fin.index.isin(_trial_users).astype(int)
    fin = fin.fillna(0)
    return fin


# ══════════════════════════════════════════════════════════════════════════════
#  EVENT LOADING + CLEANING
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_from_db(db_url: str, events_table: str = "events") -> pd.DataFrame:
    """
    Load raw events from a SQLite or PostgreSQL database.

    The table must have columns: user_id, timestamp, event_name
    (or: user_id, ts, event  — auto-remapped)

    db_url examples:
      SQLite:     sqlite:///path/to/mydb.sqlite
      PostgreSQL: postgresql://user:pass@host:5432/dbname
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)
    with engine.connect() as conn:
        # Inspect columns to handle alternate naming
        insp = engine.dialect.get_columns(conn, events_table)
        col_names = [c["name"] for c in insp]

        df = pd.read_sql_table(events_table, con=conn)

    # Remap common alternative column names
    rename_map = {}
    if "ts" in col_names and "timestamp" not in col_names:
        rename_map["ts"] = "timestamp"
    if "event" in col_names and "event_name" not in col_names:
        rename_map["event"] = "event_name"
    if "uid" in col_names and "user_id" not in col_names:
        rename_map["uid"] = "user_id"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"user_id", "timestamp", "event_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Events table '{events_table}' is missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    return df[["user_id", "timestamp", "event_name"]]


def load_stripe_from_db(db_url: str) -> dict:
    """
    Load Stripe tables (charges, subscriptions, invoices) from a database.
    Returns empty dict if tables are not found.
    """
    from sqlalchemy import create_engine, inspect as sa_inspect

    engine = create_engine(db_url)
    insp = sa_inspect(engine)
    existing_tables = set(insp.get_table_names())

    result = {}
    table_map = {
        "stripe_charges": "charges",
        "stripe_subscriptions": "subscriptions",
        "stripe_invoices": "invoices",
    }
    with engine.connect() as conn:
        for table, key in table_map.items():
            if table in existing_tables:
                result[key] = pd.read_sql_table(table, con=conn)
    return result


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = ["user_id", "timestamp", "event_name"]
    df["timestamp"] = to_utc_naive(df["timestamp"])
    df = df[df["timestamp"].notna()]
    df["event_raw"] = df["event_name"]
    # Vectorized normalization: strip → remove _dup_ suffix → dict map
    _names = df["event_name"].str.strip()
    _has_dup = _names.str.contains("_dup_", regex=False, na=False)
    _names = _names.where(~_has_dup, _names.str.split("_dup_").str[0])
    df["event_name"] = _names.map(EVENT_ALIASES).fillna(_names)
    df["event_category"] = df["event_name"].map(EVENT_CATEGORIES).fillna("other")
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Detect rage clicks from rapid click bursts
    df = detect_rage_clicks(df)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  USER PROFILES  (behaviour + financial)
# ══════════════════════════════════════════════════════════════════════════════

def to_utc_naive(series: pd.Series) -> pd.Series:
    """Normalise a timestamp series to a single UTC wall clock.

    Source data routinely mixes timezones (or mixes aware and naive values),
    which silently corrupts any "time between events" calculation. This parses
    everything to UTC, then drops tzinfo so that all downstream comparisons —
    including against date-picker values, which are naive — stay consistent.
    """
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)


def reference_now(df: pd.DataFrame, reference_time=None) -> pd.Timestamp:
    """The "now" used for recency and churn calculations.

    Uses real wall-clock time for live data. Falls back to the dataset's last
    timestamp when the data is stale (>7 days behind), so that historical or
    demo datasets don't mark every single user as churned.
    """
    if reference_time is not None:
        return pd.Timestamp(reference_time).tz_localize(None) \
            if pd.Timestamp(reference_time).tzinfo else pd.Timestamp(reference_time)
    dataset_end = pd.Timestamp(df["timestamp"].max())
    if dataset_end.tzinfo is not None:
        dataset_end = dataset_end.tz_localize(None)
    now = pd.Timestamp.utcnow().tz_localize(None)
    return now if (now - dataset_end).days <= 7 else dataset_end


def infer_churn_threshold(df: pd.DataFrame,
                          floor_days: float = 14.0,
                          cap_days: float = 90.0) -> float:
    """Derive a churn threshold from the dataset's own activity rhythm.

    A fixed 14-day rule suits a product used daily, but wrongly marks almost
    everyone churned in a product used weekly or monthly. Since every customer's
    product has a different cadence, the threshold is inferred: take the 75th
    percentile of users' inter-event gaps and allow roughly three missed usage
    cycles, clamped to a defensible range.
    """
    s = df.sort_values(["user_id", "timestamp"])
    gaps = s.groupby("user_id")["timestamp"].diff().dt.total_seconds() / 86400.0
    typical = gaps[gaps > 0].quantile(0.75)
    if pd.isna(typical):
        return floor_days
    return float(min(max(typical * 3.0, floor_days), cap_days))


def build_user_profiles(df: pd.DataFrame, financials: pd.DataFrame = None,
                        reference_time=None,
                        churn_threshold_days: float = None) -> pd.DataFrame:
    user_groups = df.groupby("user_id")

    p = pd.DataFrame()
    p["total_events"] = user_groups.size()
    p["first_event"] = user_groups["timestamp"].min()
    p["last_event"] = user_groups["timestamp"].max()

    # Distinct calendar dates on which the user did something. NOT the span
    # between first and last event — a user seen on day 1 and day 30 has two
    # active days, not thirty.
    p["active_days"] = (
        df.assign(_date=df["timestamp"].dt.normalize())
        .groupby("user_id")["_date"].nunique()
    )
    # Span from first to last event. Kept separately because cohort and
    # lifecycle analysis genuinely needs elapsed time.
    p["tenure_days"] = (
        (p["last_event"] - p["first_event"]).dt.total_seconds() / 86400
    ).round(1)
    p["events_per_day"] = (p["total_events"] / p["active_days"].clip(lower=1)).round(2)

    # Sessions
    p["session_count"] = df[df["event_name"] == "session_start"].groupby("user_id").size()
    p["session_count"] = p["session_count"].fillna(0).astype(int)
    p["unique_events"] = user_groups["event_name"].nunique()

    # Category counts
    cat_counts = df.groupby(["user_id", "event_category"]).size().unstack(fill_value=0)
    for col in cat_counts.columns:
        p[f"cat_{col}"] = cat_counts[col]
    p = p.fillna(0)

    # Specific event counts — single cross-groupby instead of 12 separate masks
    _evt_list = [
        "signup", "onboarding_step1", "onboarding_step2",
        "payment_success", "payment_failed", "plan_upgrade",
        "start_checkout", "rage_click", "error", "contact_support",
        "invite_team", "view_pricing",
    ]
    _evt_counts = (
        df[df["event_name"].isin(_evt_list)]
        .groupby(["user_id", "event_name"]).size()
        .unstack(fill_value=0)
        .reindex(columns=_evt_list, fill_value=0)
    )
    for evt in _evt_list:
        p[f"n_{evt}"] = _evt_counts[evt].reindex(p.index, fill_value=0)
    p = p.fillna(0)

    # Behavioural flags
    p["revenue_events"] = p["n_payment_success"]
    p["converted"] = (p["revenue_events"] > 0).astype(int)

    p["feature_breadth"] = (
        df[df["event_name"].str.startswith("view_feature_")]
        .groupby("user_id")["event_name"].nunique()
        .reindex(p.index).fillna(0).astype(int)
    )

    # ── Churn: recency-based, not a composite heuristic ─────────────────
    # Previously churn was inferred from "low events + short span + few
    # features", which labelled brand-new active users as churned and
    # long-tenured dormant users as retained. Churn is a recency question.
    ref_now = reference_now(df, reference_time)
    threshold = (churn_threshold_days if churn_threshold_days is not None
                 else infer_churn_threshold(df))
    p["days_since_last"] = (
        (ref_now - p["last_event"]).dt.total_seconds() / 86400
    ).round(1)
    p["churned"] = (p["days_since_last"] > threshold).astype(int)
    # Expose what was actually used, so the UI can state the definition rather
    # than presenting an unexplained churn number.
    p.attrs["churn_threshold_days"] = round(threshold, 1)
    p.attrs["reference_now"] = str(ref_now)

    # Engagement depth is still useful — just not as a churn proxy.
    p["low_engagement"] = (
        (p["total_events"] < p["total_events"].median())
        & (p["feature_breadth"] < 2)
    ).astype(int)

    p["completed_onboarding"] = (
        (p["n_onboarding_step1"] > 0) & (p["n_onboarding_step2"] > 0)
    ).astype(int)
    p["checkout_conversion"] = np.where(
        p["n_start_checkout"] > 0,
        p["n_payment_success"] / p["n_start_checkout"],
        0,
    )

    # ── Merge financial data ────────────────────────────────────────────
    if financials is not None and not financials.empty:
        for col in financials.columns:
            p[col] = financials[col].reindex(p.index).fillna(0)
        p["total_revenue"] = p.get("total_revenue", 0)
    else:
        p["total_revenue"] = 0

    # Revenue actually collected to date. This is NOT lifetime value — LTV is a
    # forward projection (expected revenue over the remaining relationship).
    # `ltv` is retained as an alias so existing chart code keeps working, but
    # `revenue_to_date` is the honest name and should be preferred in new code.
    p["revenue_to_date"] = p["total_revenue"]
    p["ltv"] = p["revenue_to_date"]

    return p


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSITION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def build_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.sort_values(["user_id", "timestamp"])
    sorted_df["next_event"] = sorted_df.groupby("user_id")["event_name"].shift(-1)
    sorted_df["next_time"] = sorted_df.groupby("user_id")["timestamp"].shift(-1)
    sorted_df["time_to_next"] = (
        sorted_df["next_time"] - sorted_df["timestamp"]
    ).dt.total_seconds()

    transitions = sorted_df.dropna(subset=["next_event"])

    def _ws_avg(x):
        ws = x[x < 1800]
        return ws.mean() if len(ws) > 0 else float("nan")

    trans_agg = (
        transitions.groupby(["event_name", "next_event"])
        .agg(
            count=("time_to_next", "size"),
            avg_time_seconds=("time_to_next", "mean"),
            avg_time_seconds_ws=("time_to_next", _ws_avg),
        )
        .reset_index()
        .rename(columns={"event_name": "source", "next_event": "target"})
    )
    trans_agg["avg_time_seconds"]    = trans_agg["avg_time_seconds"].round(0)
    trans_agg["avg_time_seconds_ws"] = trans_agg["avg_time_seconds_ws"].round(0)
    return trans_agg.sort_values("count", ascending=False)


def build_step_sequences(df: pd.DataFrame, max_steps: int = 10) -> pd.DataFrame:
    sorted_df = df.sort_values(["user_id", "timestamp"])
    sorted_df["step"] = sorted_df.groupby("user_id").cumcount() + 1
    return sorted_df[sorted_df["step"] <= max_steps][["user_id", "step", "event_name"]]


def build_self_loops(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.sort_values(["user_id", "timestamp"])
    sorted_df["prev_event"] = sorted_df.groupby("user_id")["event_name"].shift(1)
    loops = sorted_df[sorted_df["event_name"] == sorted_df["prev_event"]]
    return (
        loops.groupby(["user_id", "event_name"])
        .size()
        .reset_index(name="loop_count")
        .sort_values("loop_count", ascending=False)
    )


def prepare_retentioneering_df(df: pd.DataFrame) -> pd.DataFrame:
    rete_df = df[["user_id", "event_name", "timestamp"]].copy()
    rete_df.columns = ["user_id", "event", "timestamp"]
    return rete_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ETL ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _finalise_events(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Shared post-mapping cleanup: event_category, sort, rage-click detection.

    Expects `raw_df` to already have canonical `user_id`, `event_name`,
    `timestamp` columns (either because the source was already canonical, or
    because a taxonomy mapping has been applied upstream).
    """
    df = raw_df.copy()
    df["timestamp"] = to_utc_naive(df["timestamp"])
    df = df[df["timestamp"].notna()]
    if "event_raw" not in df.columns:
        df["event_raw"] = df["event_name"]
    # Typo / duplicate cleanup still happens after the semantic mapping —
    # downstream data may have minor noise even post-mapping.
    _names = df["event_name"].astype(str).str.strip()
    _has_dup = _names.str.contains("_dup_", regex=False, na=False)
    _names = _names.where(~_has_dup, _names.str.split("_dup_").str[0])
    df["event_name"] = _names.map(EVENT_ALIASES).fillna(_names)
    df["event_category"] = df["event_name"].map(EVENT_CATEGORIES).fillna("other")
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return detect_rage_clicks(df)


def run_etl_from_connector(connector, mapping=None, users_override=None,
                           financials_override=None) -> dict:
    """
    New connector-based entry point.

    Parameters:
        connector           — an instance of connectors.Connector
        mapping             — taxonomy.DataMapping (if None, an identity
                              mapping is constructed, which works when the
                              source already uses canonical names)
        users_override      — optional pre-loaded users DataFrame (e.g. from
                              the UI upload flow); overrides connector.read_users()
        financials_override — optional pre-loaded Stripe dict

    Returns the same dict shape as run_etl() — drop-in compatible with
    the rest of the dashboard.
    """
    # Deferred imports so etl.py stays usable even if those modules are
    # absent during unit-testing a subset of the pipeline.
    from taxonomy import apply_mapping, identity_mapping, apply_user_mapping

    raw_events = connector.read_events()
    raw_users = users_override if users_override is not None else connector.read_users()
    financials_raw = (financials_override
                      if financials_override is not None
                      else connector.read_financials())

    _mapping = mapping or identity_mapping()
    normalised = apply_mapping(raw_events, _mapping)

    df = _finalise_events(normalised)

    # Stripe-style financials: accept either the connector's dict OR synthesise
    # from CSVs if connector didn't provide any.
    stripe = financials_raw or {}
    financials = (build_user_financials(stripe)
                  if stripe else pd.DataFrame())

    # Users override: reserved for the future "user dimension table" flow.
    # Today the dashboard derives user_profiles from events, so we don't need
    # raw_users, but we apply the user mapping for completeness.
    if raw_users is not None:
        _ = apply_user_mapping(raw_users, _mapping)

    return {
        "events": df,
        "user_profiles": build_user_profiles(df, financials),
        "transitions": build_transition_matrix(df),
        "step_sequences": build_step_sequences(df),
        "self_loops": build_self_loops(df),
        "rete_df": prepare_retentioneering_df(df),
        "stripe": stripe,
        "user_financials": financials,
        "source": getattr(connector, "source_name", "connector"),
        "mapping": _mapping,
    }


def run_etl(csv_path: str, db_url: Optional[str] = None,
            mapping=None) -> dict:
    """
    Run the full ETL pipeline.

    Priority:
      1. db_url  — load events (and Stripe tables) from a database
      2. csv_path — fall back to CSV file + Stripe CSVs in the same directory

    db_url examples:
      sqlite:///mydb.sqlite
      postgresql://user:pass@host:5432/dbname

    `mapping`: optional taxonomy.DataMapping — when omitted, source events
    are assumed to already use canonical names (identity mapping).
    """
    # New connector-based path — preferred for any non-legacy caller.
    try:
        from connectors import CSVConnector, SQLConnector
        if db_url:
            print(f"  Loading events from database: {db_url[:40]}…")
            connector = SQLConnector(db_url)
            # Stripe tables come from the same DB if present
            try:
                stripe = load_stripe_from_db(db_url)
            except Exception:
                stripe = {}
            result = run_etl_from_connector(connector, mapping=mapping,
                                            financials_override=stripe or None)
            result["source"] = "db"
            return result
        else:
            data_dir = os.path.dirname(os.path.abspath(csv_path))
            connector = CSVConnector(csv_path)
            stripe = load_stripe_data(data_dir)
            result = run_etl_from_connector(connector, mapping=mapping,
                                            financials_override=stripe or None)
            result["source"] = "csv"
            return result
    except Exception:
        # Fall back to the legacy monolithic path so existing deployments keep
        # working even if the new modules are absent or broken.
        pass

    if db_url:
        print(f"  Loading events from database: {db_url[:40]}…")
        raw_df = load_from_db(db_url)
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
        raw_df["event_raw"] = raw_df["event_name"]
        _names = raw_df["event_name"].str.strip()
        _has_dup = _names.str.contains("_dup_", regex=False, na=False)
        _names = _names.where(~_has_dup, _names.str.split("_dup_").str[0])
        raw_df["event_name"] = _names.map(EVENT_ALIASES).fillna(_names)
        raw_df["event_category"] = raw_df["event_name"].map(EVENT_CATEGORIES).fillna("other")
        raw_df = raw_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        df = detect_rage_clicks(raw_df)
        stripe = load_stripe_from_db(db_url)
    else:
        data_dir = os.path.dirname(os.path.abspath(csv_path))
        df = load_and_clean(csv_path)
        stripe = load_stripe_data(data_dir)

    financials = build_user_financials(stripe) if stripe else pd.DataFrame()

    return {
        "events": df,
        "user_profiles": build_user_profiles(df, financials),
        "transitions": build_transition_matrix(df),
        "step_sequences": build_step_sequences(df),
        "self_loops": build_self_loops(df),
        "rete_df": prepare_retentioneering_df(df),
        "stripe": stripe,
        "user_financials": financials,
        "source": "db" if db_url else "csv",
    }
