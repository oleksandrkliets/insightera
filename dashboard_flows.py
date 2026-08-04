"""
Insightera Analytics — User Intelligence Platform  v4
======================================================
Journey analysis · Conversion intelligence · Revenue patterns
Churn intelligence · User clustering · Friction hotspots
PM opportunity matrix · DAU/MAU trends · Custom user segments

Run:  python dashboard_flows.py
Open: http://127.0.0.1:8050
"""

import os, sys, warnings, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import dash
from dash import dcc, html, Input, Output, State, callback, ctx, no_update
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc

from retentioneering.eventstream import Eventstream

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from etl import run_etl, EVENT_CATEGORIES, run_etl_from_connector
try:
    from ai_utils import (ask_claude, generate_tab_insight, build_full_context,
                          infer_data_mapping)
    _AI_AVAILABLE = True
except ImportError:
    _AI_AVAILABLE = False

# Taxonomy + connectors — core of the AI-ETL flow
from taxonomy import (CANONICAL_EVENTS, DataMapping, apply_mapping,
                      identity_mapping, load_mapping, save_mapping,
                      coverage_report)
from connectors import (CSVConnector, SQLConnector, DataFrameConnector,
                        Connector)

# ══════════════════════════════════════════════════════════════
#  ETL
# ══════════════════════════════════════════════════════════════
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "messy_complete_events_simple.csv")
print("Running ETL …")
data = run_etl(CSV_PATH)
events_raw    = data["events"]
user_profiles = data["user_profiles"]
user_financials = data["user_financials"]
stripe        = data["stripe"]
transitions   = data["transitions"]
step_seq      = data["step_sequences"]
self_loops    = data["self_loops"]
total_users   = events_raw["user_id"].nunique()

# ── Filter generic/noise events for all visualisations
GENERIC_EVENTS = {"page_view", "click", "session_start", "session_end"}
events = events_raw[~events_raw["event_name"].isin(GENERIC_EVENTS)].copy()

rete_df_filtered = (
    events[["user_id", "event_name", "timestamp"]]
    .rename(columns={"event_name": "event"})
    .sort_values(["user_id", "timestamp"])
    .reset_index(drop=True)
)

print(f"  {len(events_raw):,} raw → {len(events):,} meaningful · {total_users:,} users")
print("Building Eventstream …")
es = Eventstream(rete_df_filtered)

# ── AI data context ─────────────────────────────────────────────
# Cached per filter key so filtered tabs / chat see the right numbers.
_DATA_CONTEXT_CACHE: dict = {}

def _filter_label_from_store(store) -> str | None:
    """Human-readable description of the active filter for AI context."""
    if not store:
        return None
    segment = store.get("segment", "all")
    mode = store.get("mode", "all")
    d_s = store.get("date_start")
    d_e = store.get("date_end")
    parts = []
    if mode == "custom":
        n = len(store.get("user_ids") or [])
        parts.append(f"custom user list ({n} users)")
    elif segment and segment != "all":
        parts.append(f"segment = {segment}")
    if d_s and d_e:
        parts.append(f"date {d_s} \u2192 {d_e}")
    return ", ".join(parts) if parts else None


def _get_data_context(store=None) -> str:
    """Build AI data context reflecting the current filter.
    Cached per filter key; falls back to unfiltered context when no filter."""
    if not _AI_AVAILABLE:
        return ""
    key = _store_key(store) if "_store_key" in globals() else None
    cached = _DATA_CONTEXT_CACHE.get(key)
    if cached is not None:
        return cached
    # Resolve filtered data (if filter active) or use globals
    if key is None:
        up_used, ev_used = user_profiles, events
        label = None
    else:
        ev_used, up_used, _ = _resolve_filter(store)
        label = _filter_label_from_store(store)
    ctx = build_full_context(up_used, ev_used, filter_label=label)
    _DATA_CONTEXT_CACHE[key] = ctx
    return ctx


def _invalidate_ai_caches() -> None:
    """Call when underlying data is reloaded (e.g. new CSV uploaded)."""
    _DATA_CONTEXT_CACHE.clear()

# ══════════════════════════════════════════════════════════════
#  DESIGN SYSTEM  (Insightera brand)
# ══════════════════════════════════════════════════════════════
BG_PAGE    = "#E8F0FE"
BG_CARD    = "#F4F8FF"
BG_SURFACE = "#EBF2FF"
BG_INPUT   = "#FFFFFF"

BORDER     = "rgba(59, 82, 217, 0.22)"
BORDER_LIT = "rgba(59, 82, 217, 0.50)"

BRAND_BLUE   = "#3B52D9"
BRAND_PURPLE = "#7C22C4"
BRAND_TEAL   = "#059669"
BRAND_AMBER  = "#D97706"

CATEGORY_COLORS = {
    "onboarding":    "#3B52D9",
    "engagement":    "#059669",
    "feature_usage": "#7C22C4",
    "monetization":  "#D97706",
    "friction":      "#DC2626",
    "session":       "#0891B2",
    "other":         "#475569",
}

TEXT_PRI   = "#1A2845"
TEXT_SEC   = "#2D4A7A"
TEXT_MUTED = "#6B82A8"

COL_HI   = "#059669"
COL_MID  = "#D97706"
COL_LO   = "#DC2626"
COL_FREE = "#475569"

SEG_COLORS = {
    "Free": COL_FREE, "Low Revenue": COL_LO,
    "Mid Revenue": COL_MID, "High Revenue": COL_HI,
}

CLUSTER_PALETTE = ["#3B52D9", "#059669", "#7C22C4", "#D97706",
                   "#DC2626", "#0891B2", "#EA6C00", "#C4376A"]

# Individual KPI-area card IDs — used for per-card visibility toggle.
# Each tuple: (element_id, display_label)
KPI_CARD_IDS = [
    ("kpi-total-users", "Total Users"),
    ("kpi-dau",         "DAU"),
    ("kpi-ratio",       "DAU/MAU Ratio"),
    ("kpi-revenue",     "Total Revenue"),
    ("kpi-ltv",         "Avg LTV"),
    ("kpi-churn",       "Churn Risk"),
]
HEALTH_CARD_IDS = [
    ("health-activation",  "Activation Rate"),
    ("health-conversion",  "Conversion Rate"),
    ("health-retention",   "Retention Rate"),
    ("health-new-users",   "New Users / Month"),
    ("health-ttv",         "Median TTV"),
]
PERIOD_CARD_IDS = [
    ("period-new-signups",   "New Signups"),
    ("period-activation",    "Activation %"),
    ("period-conversion",    "Conversion %"),
    ("period-active-users",  "Active Users"),
    ("period-events-day",    "Avg Events / Day"),
    ("period-payments",      "Payments"),
]
# Flat list used by callbacks  (PERIOD_CARD_IDS removed — deltas are now inline in KPI cards)
ALL_KPI_CARDS = KPI_CARD_IDS + HEALTH_CARD_IDS
# Map metric label → period card id (kept for build_period_comparison_card legacy support)
_PERIOD_ID_MAP = {label: cid for cid, label in PERIOD_CARD_IDS}


def _hex_rgb(h):
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)

def _rgba(hex_c, a=1.0):
    r, g, b = _hex_rgb(hex_c)
    return f"rgba({r},{g},{b},{a})"

def _cdefaults(**extra):
    d = dict(
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_PAGE,
        font=dict(family="Inter, system-ui, -apple-system, sans-serif",
                  color=TEXT_SEC, size=13),
        xaxis=dict(gridcolor=_rgba(BRAND_BLUE, 0.15),
                   zerolinecolor=_rgba(BRAND_BLUE, 0.25),
                   tickfont=dict(color=TEXT_SEC, size=11)),
        yaxis=dict(gridcolor=_rgba(BRAND_BLUE, 0.15),
                   zerolinecolor=_rgba(BRAND_BLUE, 0.25),
                   tickfont=dict(color=TEXT_SEC, size=11)),
        legend=dict(bgcolor="rgba(244,248,255,0.85)", font=dict(color=TEXT_SEC, size=11),
                    orientation="h", y=-0.18),
        margin=dict(l=14, r=14, t=32, b=14),
    )
    d.update(extra)
    return d


# ══════════════════════════════════════════════════════════════
#  PRE-COMPUTATIONS
# ══════════════════════════════════════════════════════════════

# ── DAU / MAU (133-day window, set-union method)
def _compute_dau_mau(ev=None):
    """Compute DAU/MAU from events. Pass ev to use filtered events."""
    ud = (ev if ev is not None else events_raw)[["user_id", "timestamp"]].copy()
    ud["date"] = ud["timestamp"].dt.normalize()
    ud = ud.drop_duplicates(["user_id", "date"])

    dau = ud.groupby("date")["user_id"].nunique()
    daily_sets = ud.groupby("date")["user_id"].agg(set)
    sorted_dates = daily_sets.index.sort_values()

    # Sliding 30-day window: add today's users, evict users from 30+ days ago.
    # O(total_user_day_pairs) vs the previous O(n_dates × 30 × users/day).
    from collections import Counter, deque
    user_counter: Counter = Counter()
    window: deque = deque()          # deque of (date, user_set)
    mau_map = {}
    for d in sorted_dates:
        cutoff = d - pd.Timedelta(days=29)
        # Evict dates outside the 30-day window
        while window and window[0][0] < cutoff:
            old_d, old_set = window.popleft()
            for uid in old_set:
                user_counter[uid] -= 1
                if user_counter[uid] == 0:
                    del user_counter[uid]
        # Add today
        today_set = daily_sets[d]
        window.append((d, today_set))
        for uid in today_set:
            user_counter[uid] += 1
        mau_map[d] = len(user_counter)

    mau = pd.Series(mau_map, name="mau")
    df = pd.concat([dau.rename("dau"), mau], axis=1).reset_index()
    df.columns = ["date", "dau", "mau"]
    df["ratio"] = (df["dau"] / df["mau"].clip(lower=1)).round(3)
    return df

print("Computing DAU/MAU …")
dau_mau_df = _compute_dau_mau()

# ── Monthly cohort join counts
up = user_profiles
_mc = up["first_event"].dt.to_period("M").value_counts().sort_index().reset_index()
_mc.columns = ["month", "new_users"]
_mc["month_str"] = _mc["month"].astype(str)
monthly_cohorts = _mc

# ── Monthly revenue trend
if stripe and "charges" in stripe:
    _ch = stripe["charges"]
    _real = _ch[~_ch["is_duplicate"] & (_ch["paid"] == True)].copy()
    _real["month"] = _real["created"].dt.to_period("M").astype(str)
    ltv_trend = _real.groupby("month")["net_amount_usd"].sum().reset_index()
    ltv_trend.columns = ["month", "revenue"]
else:
    ltv_trend = pd.DataFrame(columns=["month", "revenue"])

# ── MRR from invoices (more complete than charges)
if stripe and "invoices" in stripe:
    _inv = stripe["invoices"]
    _inv_paid = _inv[_inv["amount_paid"] > 0].copy()
    _inv_paid["month"] = _inv_paid["period_start"].dt.to_period("M").astype(str)
    mrr_df = _inv_paid.groupby("month")["amount_paid_usd"].sum().reset_index()
    mrr_df.columns = ["month", "mrr"]
    mrr_df["mrr_prev"] = mrr_df["mrr"].shift(1)
    mrr_df["mom_pct"] = (
        (mrr_df["mrr"] - mrr_df["mrr_prev"]) / mrr_df["mrr_prev"] * 100
    ).round(1)
else:
    mrr_df = pd.DataFrame(columns=["month", "mrr", "mom_pct"])

# ── Journey insights (static metrics)
def _journey_insights():
    up_ = user_profiles
    ob1 = (up_["n_onboarding_step1"] > 0).mean() * 100
    ob2 = (up_["n_onboarding_step2"] > 0).mean() * 100
    conv = up_["converted"].mean() * 100
    started_co = (up_["n_start_checkout"] > 0).sum()
    paid_n = (up_["n_payment_success"] > 0).sum()
    co_aband = (1 - paid_n / max(started_co, 1)) * 100
    ltv_ob = up_[up_["completed_onboarding"] == 1]["ltv"].mean()
    ltv_no = up_[up_["completed_onboarding"] == 0]["ltv"].mean()
    ltv_lift = ltv_ob / max(ltv_no, 0.01)
    # Avg events to first payment
    converters_idx = up_[up_["converted"] == 1].index
    conv_evts = events[events["user_id"].isin(converters_idx)].sort_values(
        ["user_id", "timestamp"])
    first_pay = (conv_evts[conv_evts["event_name"] == "payment_success"]
                 .groupby("user_id")["timestamp"].min().rename("pay_ts"))
    cb = conv_evts.merge(first_pay, on="user_id")
    cb = cb[cb["timestamp"] <= cb["pay_ts"]]
    avg_to_pay = cb.groupby("user_id").size().mean()
    return dict(ob1=ob1, ob2=ob2, ob_drop=ob1 - ob2,
                conv=conv, co_aband=co_aband,
                ltv_lift=ltv_lift, avg_to_pay=avg_to_pay)

print("Computing journey insights …")
JI = _journey_insights()

# ── Engagement-matched impact of friction events on conversion ───────────
#
# For each candidate event, stratify users by total-event quartile, then within
# each stratum compare conversion rates between users who touched vs didn't.
# Returned DataFrame has one row per event: conv_delta_pp (percentage-point
# change; negative = event hurts conversion), retention_delta_pp, reach.
#
# This is the same propensity-stratification idea used for
# build_engagement_matched_lift, applied specifically to the friction events
# so we can feed the result into the Priority Matrix.
def _compute_negative_event_impact(_ev=None, _up=None, events_subset=None):
    ev_ = _ev if _ev is not None else events
    up_ = _up if _up is not None else user_profiles
    if up_.empty or "converted" not in up_.columns or "churned" not in up_.columns:
        return pd.DataFrame(columns=["event", "conv_delta_pp",
                                      "retention_delta_pp", "reach"])
    target_events = events_subset if events_subset is not None else [
        "error", "payment_failed", "rage_click", "contact_support",
    ]
    te = up_["total_events"].fillna(0)
    try:
        eng_tier = pd.qcut(te, q=3, labels=["Low", "Mid", "High"],
                           duplicates="drop")
    except Exception:
        return pd.DataFrame(columns=["event", "conv_delta_pp",
                                      "retention_delta_pp", "reach"])
    up_t = up_.assign(eng_tier=eng_tier)

    rows = []
    for evt in target_events:
        ev_sub = ev_[ev_["event_name"] == evt]
        if ev_sub.empty:
            continue
        touchers = set(ev_sub["user_id"].unique())
        if len(touchers) < 30:
            continue
        conv_parts, ret_parts, total_w = [], [], 0
        for _tier, tg in up_t.groupby("eng_tier", observed=True):
            tu = set(tg.index)
            t_in = tg.loc[list(tu & touchers)]
            t_out = tg.loc[list(tu - touchers)]
            if len(t_in) < 15 or len(t_out) < 15:
                continue
            w = len(t_in) + len(t_out)
            conv_parts.append(((t_in["converted"].mean() -
                                t_out["converted"].mean()) * 100, w))
            ret_parts.append((((1 - t_in["churned"].mean()) -
                               (1 - t_out["churned"].mean())) * 100, w))
            total_w += w
        if not conv_parts or total_w == 0:
            continue
        rows.append({
            "event": evt,
            "conv_delta_pp": round(
                sum(d * w for d, w in conv_parts) / total_w, 2),
            "retention_delta_pp": round(
                sum(d * w for d, w in ret_parts) / total_w, 2),
            "reach": len(touchers),
        })
    return pd.DataFrame(rows)


# ── Opportunity matrix (pre-computed)
def _opp_matrix_data(_ev=None, _up=None):
    ev_ = _ev if _ev is not None else events
    up_ = _up if _up is not None else user_profiles
    sd = ev_.sort_values(["user_id", "timestamp"]).copy()
    sd["next_event"] = sd.groupby("user_id")["event_name"].shift(-1)

    # Vectorized boolean columns — computed once across the full DataFrame
    sd["_is_loop"] = sd["event_name"] == sd["next_event"]
    sd["_is_rage"] = sd["next_event"] == "rage_click"
    sd["_is_err"]  = sd["next_event"] == "error"
    sd["_is_drop"] = sd["next_event"].isna() | sd["next_event"].eq("session_end")

    # Single groupby aggregation (no per-event Python loop)
    agg = sd.groupby("event_name").agg(
        n=("user_id", "size"),
        nu=("user_id", "nunique"),
        n_loop=("_is_loop", "sum"),
        n_rage=("_is_rage", "sum"),
        n_err=("_is_err", "sum"),
        n_drop=("_is_drop", "sum"),
    )
    agg = agg[agg["n"] >= 50].copy()

    agg["dp"] = agg["n_drop"] / agg["n"] * 100
    agg["Friction"] = (
        (agg["n_loop"]/agg["n"])*30 + (agg["n_rage"]/agg["n"])*30 +
        (agg["n_err"]/agg["n"])*20  + (agg["n_drop"]/agg["n"])*20
    ) * 100

    # LTV impact: join avg_ltv per event user-set in one merge pass
    event_avg_ltv = (
        sd[["event_name", "user_id"]].drop_duplicates()
        .merge(up_[["ltv"]], left_on="user_id", right_index=True, how="left")
        .groupby("event_name")["ltv"].mean()
    )
    agg = agg.join(event_avg_ltv.rename("avg_ltv"))
    agg["n_drop_u"] = (agg["dp"] / 100 * agg["nu"]).clip(lower=0).astype(int)
    agg["LTV_Impact"] = (agg["avg_ltv"] * agg["n_drop_u"]).where(
        agg["avg_ltv"].notna(), 0.0).round(0)

    # ── Merge engagement-matched conversion-impact estimate ─────────
    # Populated for friction events (rage_click, error, payment_failed,
    # contact_support). NaN for other events (structurally unavailable).
    impact_df = _compute_negative_event_impact(_ev=ev_, _up=up_)
    if not impact_df.empty:
        impact_df = impact_df.set_index("event")
        agg = agg.join(impact_df[["conv_delta_pp"]]
                       .rename(columns={"conv_delta_pp": "ConvImpactPP"}))
    else:
        agg["ConvImpactPP"] = np.nan

    # ConvImpactPP is a signed pp delta (negative = hurts conversion).
    # Boost priority proportionally to magnitude of the hurt: -5pp → ×2.
    hurt = (-agg["ConvImpactPP"]).clip(lower=0).fillna(0)
    agg["impact_boost"] = 1 + hurt / 5.0

    agg["Priority"] = (
        agg["Friction"] * np.log1p(agg["nu"]) * (1 + agg["dp"] / 200)
        * agg["impact_boost"]
    ).round(1)

    df = agg.reset_index().rename(columns={
        "event_name": "Event", "nu": "Users", "dp": "Dropoff"})
    df["Category"] = df["Event"].map(EVENT_CATEGORIES).fillna("other")
    df["Friction"]  = df["Friction"].round(1)
    df["Dropoff"]   = df["Dropoff"].round(1)
    df = df[["Event","Category","Friction","Users","Dropoff","LTV_Impact",
             "ConvImpactPP","Priority"]]\
           .sort_values("Priority", ascending=False)

    # Vectorised action classification (no apply)
    med_f = df["Friction"].median()
    med_u = df["Users"].median()
    df["Action"] = np.select(
        [(df["Friction"] >= med_f) & (df["Users"] >= med_u),
         df["Friction"] >= med_f,
         df["Users"] >= med_u],
        ["🔴 Fix First", "🟡 Monitor", "🟢 Healthy"],
        default="⚪ Low Priority",
    )
    return df

print("Computing opportunity matrix …")
opp_df = _opp_matrix_data()

# ── Comeback counts (vectorised)
def _comeback_counts():
    sd = events.sort_values(["user_id", "timestamp"]).copy()
    sd["prev_event"] = sd.groupby("user_id")["event_name"].shift(1)
    sd["seen_before"] = sd.duplicated(subset=["user_id", "event_name"], keep="first")
    sd["is_comeback"] = sd["seen_before"] & (sd["prev_event"] != sd["event_name"])
    return sd[sd["is_comeback"]].groupby("event_name").size().to_dict()

_comeback_dict = _comeback_counts()


# ══════════════════════════════════════════════════════════════
#  SEGMENT HELPERS
# ══════════════════════════════════════════════════════════════
SEGMENTS = [
    {"label": "All Users",           "value": "all"},
    {"label": "Paying Users",        "value": "paying"},
    {"label": "Free Users",          "value": "free"},
    {"label": "Churned",             "value": "churned"},
    {"label": "Active (not churned)","value": "active"},
    {"label": "High LTV (top 25%)",  "value": "high_ltv"},
    {"label": "Rage Clickers",       "value": "rage"},
    {"label": "No Onboarding",       "value": "no_ob"},
]


def get_user_ids(segment="all", custom_ids=None):
    if custom_ids:
        return pd.Index([int(x) for x in custom_ids if str(x).strip().isdigit()])
    up_ = user_profiles
    m = {"paying":   up_["total_revenue"] > 0,
         "free":     up_["total_revenue"] == 0,
         "churned":  up_["churned"] == 1,
         "active":   up_["churned"] == 0,
         "high_ltv": up_["ltv"] >= up_[up_["ltv"] > 0]["ltv"].quantile(0.75) if (up_["ltv"] > 0).any() else (up_["ltv"] >= 0),
         "rage":     up_["n_rage_click"] > 0,
         "no_ob":    up_["completed_onboarding"] == 0}
    return up_[m[segment]].index if segment in m else up_.index


def _segment_users():
    up_ = user_profiles.copy()
    pay = up_[up_["total_revenue"] > 0]
    if len(pay) < 4:
        up_["revenue_segment"] = np.where(up_["converted"] > 0, "Paying", "Free")
        return up_
    q75 = pay["total_revenue"].quantile(0.75)
    q25 = pay["total_revenue"].quantile(0.25)
    rev = up_["total_revenue"]
    up_["revenue_segment"] = np.select(
        [rev <= 0, rev >= q75, rev >= q25],
        ["Free", "High Revenue", "Mid Revenue"],
        default="Low Revenue",
    )
    return up_


# ══════════════════════════════════════════════════════════════
#  SPARKLINE HELPER
# ══════════════════════════════════════════════════════════════
def _sparkline(values, color=BRAND_BLUE, height=46):
    r, g, b = _hex_rgb(color)
    fig = go.Figure(go.Scatter(
        y=list(values), mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy",
        fillcolor=f"rgba({r},{g},{b},0.12)",
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hovermode=False,
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  SANKEY
# ══════════════════════════════════════════════════════════════
def build_rete_sankey(max_steps=8, threshold=0.03, user_ids=None, eventstream=None):
    """Build retentioneering sankey. Pass `eventstream` to use a pre-built Eventstream."""
    if eventstream is not None:
        _es = eventstream
    elif user_ids is not None and len(user_ids) < total_users:
        rf = rete_df_filtered[rete_df_filtered["user_id"].isin(user_ids)]
        try:
            _es = Eventstream(rf)
        except Exception:
            return _empty_fig("Not enough data for selected filter")
    else:
        _es = es
    try:
        sk = _es.step_sankey(max_steps=max_steps, threshold=threshold, show_plot=False)
    except Exception as e:
        return _empty_fig(f"Sankey error: {e}")

    ndf = sk.data_grp_nodes.copy()
    ldf = sk.data_grp_links.copy()
    ndf["nk"] = ndf["step"].astype(str) + "|" + ndf["event"]
    ldf["sk"] = ldf["step"].astype(str) + "|" + ldf["event"]
    ldf["tk"] = ldf["next_step"].astype(str) + "|" + ldf["next_event"]

    all_keys = list(dict.fromkeys(ndf["nk"].tolist()))
    for k in ldf["tk"]:
        if k not in all_keys:
            all_keys.append(k)
    idx = {k: i for i, k in enumerate(all_keys)}

    labels, colors, cd = [], [], []
    for key in all_keys:
        parts = key.split("|", 1)
        ev = parts[1] if len(parts) > 1 else key
        labels.append(ev)
        cat = EVENT_CATEGORIES.get(ev, "other")
        colors.append(CATEGORY_COLORS.get(cat, "#64748B"))
        row = ndf[ndf["nk"] == key]
        u = int(row["usr_cnt"].values[0]) if len(row) else 0
        p = float(row["perc"].values[0]) if len(row) else 0
        cd.append(f"Step {parts[0]} · {u:,} users · {p:.1f}%")

    srcs, tgts, vals, lcols = [], [], [], []
    for _, row in ldf.iterrows():
        si = idx.get(row["sk"]); ti = idx.get(row["tk"])
        if si is None or ti is None: continue
        srcs.append(si); tgts.append(ti); vals.append(int(row["usr_cnt"]))
        r2, g2, b2 = _hex_rgb(colors[si])
        lcols.append(f"rgba({r2},{g2},{b2},0.20)")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=18, thickness=22, label=labels, color=colors,
                  customdata=cd, hovertemplate="%{label}<br>%{customdata}<extra></extra>"),
        link=dict(source=srcs, target=tgts, value=vals, color=lcols,
                  hovertemplate="%{source.label} → %{target.label}: %{value:,}<extra></extra>"),
    ))
    fig.update_layout(
        font=dict(size=11, color=TEXT_SEC,
                  family="Inter, system-ui, sans-serif"),
        height=620, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
    )
    return fig


def _empty_fig(msg="No data"):
    return go.Figure().update_layout(
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PAGE,
        font=dict(color=TEXT_PRI),
        annotations=[dict(text=msg, showarrow=False,
                          font=dict(size=14, color=TEXT_SEC))],
        height=300, margin=dict(l=0, r=0, t=0, b=0),
    )


# ── Journey insights callouts (static HTML elements)
def build_journey_insight_cards():
    def _card(value, label, color=BRAND_BLUE, note=None):
        return html.Div([
            html.Div(value, style={"fontSize": "1.8rem", "fontWeight": "700",
                                   "color": color, "lineHeight": "1"}),
            html.Div(label, style={"fontSize": "0.72rem", "color": TEXT_SEC,
                                   "marginTop": "3px", "textTransform": "uppercase",
                                   "letterSpacing": "0.5px"}),
            html.Div(note, style={"fontSize": "0.68rem", "color": TEXT_MUTED,
                                  "marginTop": "2px"}) if note else None,
        ], style={"backgroundColor": BG_SURFACE, "border": f"1px solid {BORDER}",
                  "borderRadius": "10px", "padding": "14px 16px",
                  "borderLeft": f"3px solid {color}"})

    return dbc.Row([
        dbc.Col(_card(f"{JI['ob1']:.0f}%", "Complete Onboarding S.1",
                      BRAND_BLUE), md=3),
        dbc.Col(_card(f"{JI['ob_drop']:.0f}%", "Drop after Step 1",
                      COL_LO, "never reach step 2"), md=3),
        dbc.Col(_card(f"{JI['conv']:.1f}%", "Convert to Paid",
                      COL_HI, f"Onboarding 2x LTV lift"), md=3),
        dbc.Col(_card(f"{JI['co_aband']:.0f}%", "Checkout Abandonment",
                      COL_MID, f"avg {JI['avg_to_pay']:.0f} events to pay"), md=3),
    ], className="mb-3 g-2")


# ══════════════════════════════════════════════════════════════
#  CONVERSION LIFT  (replaces bridge events — much more insightful)
# ══════════════════════════════════════════════════════════════
# ── Event classification for conversion analyses ──────────────────
#
# Tautological — these ARE the conversion target or a consequence of paying.
# Including them in a "conversion lift" chart is circular ("users who paid are
# 100% likely to have paid").
CONVERSION_TAUTOLOGICAL_EVENTS = {"payment_success", "plan_upgrade"}
#
# Negative-signal — correlate with conversion only because they mark high
# engagement (you have to use the product a lot to hit an error or rage-click).
# We isolate these into a separate "Conversion Decliners" chart that asks the
# correct question: do users with MORE exposure to friction retain worse?
CONVERSION_NEGATIVE_EVENTS = {"error", "payment_failed", "rage_click", "contact_support"}


def build_conversion_lift(_ev=None, _up=None):
    """
    Positive-events-only conversion lift:
      lift = P(converted | touched event) / P(converted overall)

    What's excluded and why:
      • payment_success, plan_upgrade — tautological (they ARE conversion).
      • error, payment_failed, rage_click, contact_support — negative-signal
        events whose apparent "lift" is reverse-causal. High engagement drives
        both conversion AND friction exposure. See build_conversion_decliners
        (long-term retention impact) and build_engagement_matched_lift
        (confounder-controlled analysis) for correct treatment.
    """
    ev_ = _ev if _ev is not None else events
    up_ = _up if _up is not None else user_profiles
    overall_conv = up_["converted"].mean()
    if overall_conv == 0:
        return _empty_fig("No conversion data")

    excluded = CONVERSION_TAUTOLOGICAL_EVENTS | CONVERSION_NEGATIVE_EVENTS
    ev_filtered = ev_[~ev_["event_name"].isin(excluded)]
    ev_dedup = ev_filtered[["event_name", "user_id"]].drop_duplicates()
    ev_dedup = ev_dedup.join(up_[["converted"]], on="user_id")
    evt_stats = ev_dedup.groupby("event_name").agg(
        users=("user_id", "size"),
        conv_sum=("converted", "sum"),
    )
    evt_stats = evt_stats[evt_stats["users"] >= 30].copy()
    if evt_stats.empty:
        return _empty_fig("Not enough data after excluding tautological/negative events")
    evt_stats["conv_among"] = evt_stats["conv_sum"] / evt_stats["users"]
    evt_stats["lift"]       = (evt_stats["conv_among"] / overall_conv).round(2)
    evt_stats["conv_rate"]  = (evt_stats["conv_among"] * 100).round(1)
    evt_stats["reach"]      = (evt_stats["users"] / len(up_) * 100).round(1)
    df = evt_stats.reset_index().rename(columns={"event_name": "event"})
    df = df.sort_values("lift", ascending=True)

    def _col(l):
        if l >= 2.0: return COL_HI
        if l >= 1.3: return BRAND_BLUE
        if l >= 0.8: return COL_MID
        return COL_LO

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["event"], x=df["lift"], orientation="h",
        marker_color=[_col(l) for l in df["lift"]],
        customdata=np.stack([df["conv_rate"], df["reach"], df["users"]], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>Lift: %{x:.2f}x<br>"
            "Conv rate: %{customdata[0]:.1f}%<br>"
            "Reach: %{customdata[1]:.1f}% users (%{customdata[2]:,})"
            "<extra></extra>"
        ),
    ))
    fig.add_vline(x=1.0, line_dash="dash", line_color=TEXT_MUTED,
                  annotation_text=f"Baseline {overall_conv*100:.1f}%",
                  annotation_font_color=TEXT_MUTED, annotation_font_size=10)
    fig.add_vline(x=1.5, line_dash="dot", line_color=_rgba(COL_HI, 0.3))

    fig.update_layout(
        **_cdefaults(
            xaxis_title="Conversion Lift (1.0 = average)",
            height=400,
        )
    )
    return fig


def build_conversion_decliners(_ev=None, _up=None):
    """
    Negative-event exposure vs long-term retention.

    For each friction / failure event, split users into exposure tiers:
      • None  — never experienced the event
      • Low   — experienced it 1 or 2 times
      • High  — experienced it 3 or more times
    and show retention rate (1 − churn rate) per tier.

    A widening gap from None → High is evidence that the event meaningfully
    hurts retention; a flat line means the event is incidental to engagement.
    """
    ev_ = _ev if _ev is not None else events
    up_ = _up if _up is not None else user_profiles
    if up_.empty or "churned" not in up_.columns:
        return _empty_fig("No churn data available")

    tiers = ["None", "Low (1–2)", "High (3+)"]
    tier_colors = [COL_HI, BRAND_AMBER, COL_LO]

    events_to_show = [e for e in CONVERSION_NEGATIVE_EVENTS
                      if (ev_["event_name"] == e).any()]
    if not events_to_show:
        return _empty_fig("No negative events in current slice")

    fig = go.Figure()
    overall_retention = (1 - up_["churned"].mean()) * 100

    for t_idx, (tier, color) in enumerate(zip(tiers, tier_colors)):
        ys, xs, hover_n = [], [], []
        for evt in events_to_show:
            cnt = (ev_[ev_["event_name"] == evt]
                   .groupby("user_id").size()
                   .reindex(up_.index, fill_value=0))
            if tier == "None":
                mask = cnt == 0
            elif tier == "Low (1–2)":
                mask = (cnt >= 1) & (cnt <= 2)
            else:
                mask = cnt >= 3
            cohort = up_[mask]
            if len(cohort) < 20:
                ys.append(evt); xs.append(None); hover_n.append(len(cohort))
                continue
            retention = (1 - cohort["churned"].mean()) * 100
            ys.append(evt)
            xs.append(retention)
            hover_n.append(len(cohort))
        fig.add_trace(go.Bar(
            x=[e.replace("_", " ").title() for e in ys],
            y=xs, name=tier, marker_color=color,
            customdata=np.array(hover_n).reshape(-1, 1),
            hovertemplate=(
                f"<b>%{{x}}</b><br>Exposure: {tier}<br>"
                "Retention: %{y:.1f}%<br>Cohort: %{customdata[0]:,} users"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(y=overall_retention, line_dash="dash",
                  line_color=TEXT_MUTED,
                  annotation_text=f"Overall retention {overall_retention:.1f}%",
                  annotation_font_color=TEXT_MUTED, annotation_font_size=10)
    fig.update_layout(**_cdefaults(
        yaxis_title="Retention Rate (%)",
        barmode="group",
        height=400,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    ))
    return fig


def build_negative_signal_impact(_ev=None, _up=None):
    """
    Estimated conversion & retention impact per negative (friction) event.

    Two horizontal bars per event: engagement-matched conversion-rate delta
    (pp) and retention-rate delta (pp). Negative bars = the event meaningfully
    hurts the metric even after controlling for engagement level.

    The conversion-impact value feeds directly into the Priority Matrix as an
    `impact_boost` multiplier, so events that demonstrably hurt conversion
    rise to the top of the fix-first ranking.
    """
    df = _compute_negative_event_impact(_ev=_ev, _up=_up)
    if df.empty:
        return _empty_fig("Not enough data for negative-signal impact")
    df = df.sort_values("conv_delta_pp", ascending=True)
    labels = df["event"].str.replace("_", " ").str.title()

    def _col(delta):
        return COL_LO if delta < 0 else (COL_HI if delta > 0 else TEXT_MUTED)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=df["conv_delta_pp"], orientation="h",
        name="Conversion \u0394 (pp)",
        marker_color=[_col(d) for d in df["conv_delta_pp"]],
        customdata=df["reach"].values.reshape(-1, 1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Conversion \u0394: %{x:+.2f} pp<br>"
            "Reach: %{customdata[0]:,} users<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        y=labels, x=df["retention_delta_pp"],
        mode="markers", name="Retention \u0394 (pp)",
        marker=dict(symbol="diamond-tall", size=11,
                    color=BRAND_AMBER,
                    line=dict(width=1, color=_rgba(BRAND_AMBER, 0.6))),
        hovertemplate="<b>%{y}</b><br>Retention \u0394: %{x:+.2f} pp<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=TEXT_MUTED, line_width=1)
    fig.update_layout(**_cdefaults(
        xaxis_title="Percentage-Point Change (engagement-matched)",
        height=360,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    ))
    return fig


def build_engagement_matched_lift(_ev=None, _up=None):
    """
    Confounder-controlled conversion lift (a.k.a. engagement-matched lift).

    Industry pattern commonly used for feature-impact analysis at companies
    like Airbnb (propensity stratification), Meta (DID uplift), Intercom
    (habit-matched cohorts): before claiming that event X causes conversion,
    split users into engagement tiers by total activity, then compute the
    within-tier conversion-rate difference between users who did/didn't touch
    the event. The reported "matched lift" is the average across tiers.

    Interpretation:
      • Matched lift ≫ 1  → event retains predictive power even among equally
        engaged users → likely a real causal signal.
      • Matched lift ≈ 1  → the naive lift was driven by engagement confound
        (engaged users do everything, not this specifically).
    """
    ev_ = _ev if _ev is not None else events
    up_ = _up if _up is not None else user_profiles
    overall_conv = up_["converted"].mean() if "converted" in up_.columns else 0
    if overall_conv == 0 or up_.empty:
        return _empty_fig("No conversion data")

    excluded = CONVERSION_TAUTOLOGICAL_EVENTS | CONVERSION_NEGATIVE_EVENTS
    ev_pos = ev_[~ev_["event_name"].isin(excluded)]
    if ev_pos.empty:
        return _empty_fig("No positive events in current slice")

    # Engagement tier by total_events quartile
    up_with_tier = up_.copy()
    te = up_with_tier["total_events"].fillna(0)
    try:
        up_with_tier["eng_tier"] = pd.qcut(
            te, q=3, labels=["Low", "Mid", "High"], duplicates="drop"
        )
    except Exception:
        return _empty_fig("Not enough variance in engagement to stratify")

    # Touched-event mapping
    touched = ev_pos[["event_name", "user_id"]].drop_duplicates()
    rows = []
    for evt, grp in touched.groupby("event_name"):
        touchers = set(grp["user_id"].unique())
        if len(touchers) < 60:
            continue
        matched_lifts = []
        total_weight = 0
        for tier, tier_users in up_with_tier.groupby("eng_tier", observed=True):
            tu_set = set(tier_users.index)
            t_in = tier_users.loc[list(tu_set & touchers)]
            t_out = tier_users.loc[list(tu_set - touchers)]
            if len(t_in) < 15 or len(t_out) < 15:
                continue
            p_in = t_in["converted"].mean()
            p_out = t_out["converted"].mean()
            if p_out <= 0:
                continue
            # Relative lift within this tier
            matched_lifts.append((p_in / p_out, len(t_in) + len(t_out)))
            total_weight += len(t_in) + len(t_out)
        if not matched_lifts or total_weight == 0:
            continue
        weighted = sum(l * w for l, w in matched_lifts) / total_weight
        naive = ((touched[touched["event_name"] == evt]
                    .merge(up_[["converted"]], left_on="user_id", right_index=True)
                    ["converted"].mean()) / overall_conv)
        rows.append({
            "event": evt,
            "matched_lift": round(weighted, 2),
            "naive_lift": round(naive, 2),
            "reach": len(touchers),
        })

    if not rows:
        return _empty_fig("Not enough data for engagement-matched lift")

    df = pd.DataFrame(rows).sort_values("matched_lift", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["event"], x=df["naive_lift"], orientation="h",
        name="Naïve lift",
        marker_color=_rgba(TEXT_MUTED, 0.45),
        hovertemplate="<b>%{y}</b><br>Naïve lift: %{x:.2f}x<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=df["event"], x=df["matched_lift"], orientation="h",
        name="Engagement-matched lift",
        marker_color=BRAND_BLUE,
        customdata=df["reach"].values.reshape(-1, 1),
        hovertemplate=(
            "<b>%{y}</b><br>Matched lift: %{x:.2f}x<br>"
            "Reach: %{customdata[0]:,} users<extra></extra>"
        ),
    ))
    fig.add_vline(x=1.0, line_dash="dash", line_color=TEXT_MUTED,
                  annotation_text="No effect", annotation_font_size=10,
                  annotation_font_color=TEXT_MUTED)
    fig.update_layout(**_cdefaults(
        xaxis_title="Lift (1.0 = no effect)",
        height=400,
        barmode="group",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    ))
    return fig


# ══════════════════════════════════════════════════════════════
#  CONVERSION FUNNEL
# ══════════════════════════════════════════════════════════════
def build_conversion_funnel(user_ids=None):
    evts_f = events if user_ids is None else events[events["user_id"].isin(user_ids)]
    stages = [("signup", "Signup"), ("onboarding_step1", "Onboarding 1"),
              ("onboarding_step2", "Onboarding 2"), ("view_pricing", "Pricing"),
              ("start_checkout", "Checkout"), ("payment_success", "Payment")]
    rows = [{"stage": lb, "users": evts_f[evts_f["event_name"] == ev]["user_id"].nunique()}
            for ev, lb in stages]
    fdf = pd.DataFrame(rows)
    fig = go.Figure(go.Funnel(
        y=fdf["stage"], x=fdf["users"],
        textinfo="value+percent initial+percent previous",
        marker=dict(color=["#4B63F5","#06D6A0","#9333EA","#F59E0B","#EF4444","#22D3EE"],
                    line=dict(color=BG_CARD, width=1)),
        connector=dict(line=dict(color=BORDER)),
    ))
    fig.update_layout(**_cdefaults(height=400))
    return fig


# ══════════════════════════════════════════════════════════════
#  REVENUE BEHAVIOUR PATTERNS
# ══════════════════════════════════════════════════════════════
def build_segment_radar():
    """
    Heatmap: % of each revenue segment who ever performed each feature event.
    Breadth (did they use it?) is more actionable than volume (how much?).
    """
    up_ = _segment_users()
    features = [
        ("view_feature_A",  "Feature A"),
        ("view_feature_B",  "Feature B"),
        ("view_feature_C",  "Feature C"),
        ("view_feature_D",  "Feature D"),
        ("view_dashboard",  "Dashboard"),
        ("view_pricing",    "Pricing"),
        ("start_checkout",  "Checkout"),
        ("invite_team",     "Team Invite"),
        ("contact_support", "Support"),
        ("payment_success", "Payment"),
    ]
    feat_events = [e for e, _ in features]
    feat_labels = [l for _, l in features]
    seg_order   = ["Free", "Low Revenue", "Mid Revenue", "High Revenue"]

    rel_users  = up_[up_["revenue_segment"].isin(seg_order)][["revenue_segment"]]
    seg_totals = rel_users["revenue_segment"].value_counts()

    ev_rel = (
        events[events["user_id"].isin(rel_users.index) &
               events["event_name"].isin(feat_events)]
        .drop_duplicates(["user_id", "event_name"])
        .join(rel_users, on="user_id")
    )
    counts = (ev_rel.groupby(["revenue_segment", "event_name"])["user_id"]
              .nunique().reset_index(name="n"))
    counts["tot"] = counts["revenue_segment"].map(seg_totals).clip(lower=1)
    counts["pct"] = (counts["n"] / counts["tot"] * 100).round(1)

    segs_present = [s for s in seg_order if s in seg_totals.index]
    if not segs_present:
        return _empty_fig("No segment data")

    matrix = []
    text   = []
    for seg in segs_present:
        row_vals = []
        row_text = []
        for evt in feat_events:
            mask = (counts["revenue_segment"] == seg) & (counts["event_name"] == evt)
            val  = counts.loc[mask, "pct"].values
            v    = float(val[0]) if len(val) else 0.0
            row_vals.append(v)
            row_text.append(f"{v:.0f}%")
        matrix.append(row_vals)
        text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=feat_labels,
        y=segs_present,
        colorscale=[[0, BG_SURFACE], [0.35, _rgba(BRAND_BLUE, 0.55)], [1, BRAND_TEAL]],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11, color=TEXT_PRI,
                      family="Inter, system-ui, sans-serif"),
        hovertemplate="%{y} — %{x}: %{z:.1f}% of segment<extra></extra>",
        showscale=True,
        colorbar=dict(
            ticksuffix="%",
            tickfont=dict(color=TEXT_SEC, size=9),
            outlinewidth=0,
        ),
        zmin=0, zmax=100,
    ))
    fig.update_layout(**_cdefaults(
        height=320,
        margin=dict(l=14, r=44, t=24, b=32),
        xaxis=dict(tickfont=dict(color=TEXT_SEC, size=10), side="bottom"),
        yaxis=dict(tickfont=dict(color=TEXT_SEC, size=10), autorange="reversed"),
    ))
    return fig


def build_revenue_segment_kpis():
    up_ = _segment_users()
    metrics = ["Avg Sessions", "Onboarding %", "Features", "Conv %", "Events/Day"]
    seg_order = ["Free", "Low Revenue", "Mid Revenue", "High Revenue"]
    # Single groupby aggregation instead of filtering in a loop
    kdf = up_.groupby("revenue_segment").agg(
        _sessions=("session_count", "mean"),
        _ob=("completed_onboarding", "mean"),
        _feats=("feature_breadth", "mean"),
        _conv=("checkout_conversion", "mean"),
        _epd=("events_per_day", "mean"),
    ).rename(columns={"_sessions": "Avg Sessions", "_feats": "Features",
                      "_epd": "Events/Day"})
    kdf["Onboarding %"] = kdf["_ob"] * 100
    kdf["Conv %"]       = kdf["_conv"] * 100
    kdf = kdf[metrics].reindex([s for s in seg_order if s in kdf.index])
    if kdf.empty: return _empty_fig("No segment data")
    kdf.index.name = "Segment"
    kdf = kdf.reset_index()
    fig = go.Figure()
    for _, row in kdf.iterrows():
        seg = row["Segment"]
        fig.add_trace(go.Bar(
            x=metrics, y=row[metrics].tolist(), name=seg,
            marker_color=_rgba(SEG_COLORS.get(seg, "#64748B"), 0.72),
            hovertemplate="%{x}: %{y:.1f}<extra>" + seg + "</extra>",
        ))
    fig.update_layout(**_cdefaults(barmode="group", height=320,
                                   xaxis_title="", yaxis_title="Value"))
    return fig


def build_revenue_journey_comparison():
    up_ = _segment_users()
    stages = [("signup","Signup"),("onboarding_step1","Onboarding 1"),
              ("onboarding_step2","Onboarding 2"),("view_pricing","Pricing"),
              ("start_checkout","Checkout"),("payment_success","Payment")]
    segs = {"High Revenue": COL_HI, "Free": COL_FREE}
    stage_events = [ev for ev, _ in stages]
    stage_label  = dict(stages)

    # Filter once to relevant users + events, then groupby — no nested loop
    rel_users = up_[up_["revenue_segment"].isin(segs)][["revenue_segment"]]
    seg_totals = rel_users["revenue_segment"].value_counts()
    ev_rel = (
        events[events["user_id"].isin(rel_users.index) &
               events["event_name"].isin(stage_events)]
        .drop_duplicates(["user_id", "event_name"])
        .join(rel_users, on="user_id")
    )
    counts = (ev_rel.groupby(["revenue_segment", "event_name"])["user_id"]
              .nunique().reset_index(name="Users"))
    counts["Stage"] = counts["event_name"].map(stage_label)
    counts["tot"]   = counts["revenue_segment"].map(seg_totals)
    counts["Pct"]   = (counts["Users"] / counts["tot"] * 100).round(1)
    counts.rename(columns={"revenue_segment": "Segment"}, inplace=True)

    # Ensure all stage rows exist (fill zeros for missing combinations)
    all_combos = pd.MultiIndex.from_product(
        [list(segs.keys()), stage_events], names=["Segment", "event_name"])
    counts = (counts.set_index(["Segment", "event_name"])
              .reindex(all_combos, fill_value=0).reset_index())
    counts["Stage"] = counts["event_name"].map(stage_label)
    counts["tot"]   = counts["Segment"].map(seg_totals).clip(lower=1)
    counts["Pct"]   = (counts["Users"] / counts["tot"] * 100).round(1)
    jdf = counts
    if jdf.empty: return _empty_fig()
    fig = go.Figure()
    for seg, col in segs.items():
        sd = jdf[jdf["Segment"] == seg]
        fig.add_trace(go.Bar(
            x=sd["Stage"], y=sd["Pct"], name=seg,
            marker_color=_rgba(col, 0.72),
            text=(sd["Pct"].round(0).astype(int).astype(str) + "%"),
            textposition="outside", textfont=dict(size=10),
            customdata=sd["Users"],
            hovertemplate="%{x}: %{y:.1f}% (%{customdata:,})<extra>" + seg + "</extra>",
        ))
    fig.update_layout(**_cdefaults(barmode="group", height=320,
                                   yaxis_title="% of Segment"))
    return fig


def build_ltv_distribution():
    up_ = _segment_users()
    pay = up_[up_["total_revenue"] > 0]
    if pay.empty: return _empty_fig("No paying users")
    fig = go.Figure()
    for seg in ["Low Revenue", "Mid Revenue", "High Revenue"]:
        sd = pay[pay["revenue_segment"] == seg]
        if sd.empty: continue
        fig.add_trace(go.Histogram(
            x=sd["total_revenue"], name=seg,
            marker_color=_rgba(SEG_COLORS.get(seg, "#64748B"), 0.72),
            bingroup=1,
            hovertemplate="$%{x:.0f}: %{y} users<extra>" + seg + "</extra>",
        ))
    fig.update_layout(**_cdefaults(barmode="overlay", height=320,
                                   xaxis_title="Total Revenue ($)", yaxis_title="Users"))
    return fig


# ══════════════════════════════════════════════════════════════
#  CHURN INTELLIGENCE
# ══════════════════════════════════════════════════════════════
def build_churn_donut(user_ids=None):
    up_ = user_profiles if user_ids is None else user_profiles[user_profiles.index.isin(user_ids)]
    churned = up_[up_["churned"] == 1]
    active = up_[up_["churned"] == 0]
    never_conv = (churned["converted"] == 0).sum()
    post_conv = (churned["converted"] == 1).sum()

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=["Active", "Churned — Never Paid", "Churned — Post-Paid"],
        values=[len(active), never_conv, post_conv],
        hole=0.62,
        marker=dict(colors=[COL_HI, COL_LO, COL_MID],
                    line=dict(color=BG_CARD, width=2)),
        textinfo="label+percent",
        textfont=dict(size=10, color=TEXT_PRI),
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
    ))
    churn_pct = len(churned) / max(len(up_), 1) * 100
    fig.add_annotation(text=f"{churn_pct:.1f}%<br><span style='font-size:11px'>churn</span>",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=22, color=TEXT_PRI,
                                 family="Inter, system-ui, sans-serif"))
    fig.update_layout(**_cdefaults(height=320,
                                   margin=dict(l=10, r=10, t=20, b=20)))
    return fig


def build_cohort_timeline(user_ids=None):
    up_ = user_profiles.copy() if user_ids is None else user_profiles[user_profiles.index.isin(user_ids)].copy()
    up_["join_month"] = up_["first_event"].dt.to_period("M").astype(str)
    coh = up_.groupby("join_month").agg(
        total=("churned", "count"), churned=("churned", "sum")).reset_index()
    coh["active"] = coh["total"] - coh["churned"]
    coh["churn_rate"] = (coh["churned"] / coh["total"] * 100).round(1)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=coh["join_month"], y=coh["active"],
                         name="Active", marker_color=_rgba(COL_HI, 0.72)),
                  secondary_y=False)
    fig.add_trace(go.Bar(x=coh["join_month"], y=coh["churned"],
                         name="Churned", marker_color=_rgba(COL_LO, 0.72)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=coh["join_month"], y=coh["churn_rate"],
                              name="Churn Rate %", mode="lines+markers",
                              line=dict(color=COL_MID, width=2),
                              marker=dict(size=7),
                              hovertemplate="%{x}: %{y:.1f}%<extra>Churn Rate</extra>"),
                  secondary_y=True)
    fig.update_layout(barmode="stack", height=320,
                      paper_bgcolor=BG_CARD, plot_bgcolor=BG_PAGE,
                      font=dict(family="Inter, system-ui, sans-serif",
                                color=TEXT_SEC, size=10),
                      legend=dict(bgcolor="rgba(0,0,0,0)",
                                  font=dict(color=TEXT_SEC, size=10),
                                  orientation="h", y=-0.22),
                      margin=dict(l=14, r=14, t=24, b=14),
                      xaxis=dict(gridcolor=_rgba(BRAND_BLUE, 0.08),
                                 tickfont=dict(color=TEXT_SEC, size=10)))
    fig.update_yaxes(title_text="Users", gridcolor=_rgba(BRAND_BLUE, 0.08),
                     tickfont=dict(color=TEXT_SEC, size=10), secondary_y=False)
    fig.update_yaxes(title_text="Churn %", ticksuffix="%",
                     gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(color=TEXT_SEC, size=10), secondary_y=True)
    return fig


def build_onboarding_funnel(_ev=None, _up=None):
    """
    Days from first event to completing each onboarding step.
    Distinct from TTV (time-to-payment): measures how quickly users progress
    through the onboarding sequence, regardless of whether they ever paid.
    """
    ev_ = _ev if _ev is not None else events
    up_ = (_up if _up is not None else user_profiles)[["first_event"]].copy()
    ob_events = ev_[ev_["event_name"].isin(["onboarding_step1", "onboarding_step2"])]

    if ob_events.empty:
        return _empty_fig("No onboarding events found")

    first_ob = (
        ob_events.groupby(["user_id", "event_name"])["timestamp"].min()
        .reset_index()
        .join(up_["first_event"], on="user_id")
    )
    first_ob["days_to_step"] = (
        (first_ob["timestamp"].dt.normalize() - first_ob["first_event"].dt.normalize())
        .dt.days
    ).clip(lower=0, upper=60)

    step1 = first_ob[first_ob["event_name"] == "onboarding_step1"]["days_to_step"]
    step2 = first_ob[first_ob["event_name"] == "onboarding_step2"]["days_to_step"]

    fig = go.Figure()
    for data, name, color in [
        (step1, "Step 1 completed", BRAND_TEAL),
        (step2, "Step 2 completed", BRAND_BLUE),
    ]:
        if len(data) == 0:
            continue
        fig.add_trace(go.Histogram(
            x=data, name=name,
            marker_color=_rgba(color, 0.72),
            nbinsx=30,
            hovertemplate=f"{name}<br>Day %{{x}}: %{{y}} users<extra></extra>",
        ))
        med = data.median()
        fig.add_vline(x=med, line_dash="dash", line_color=color,
                      annotation_text=f"Med {med:.0f}d",
                      annotation_font_color=color, annotation_font_size=10)

    fig.update_layout(**_cdefaults(
        barmode="overlay", height=280,
        xaxis=dict(
            range=[-0.5, 30],
            title="Days from First Event  (0 = completed same day as signup; capped at 60)",
            gridcolor=_rgba(BRAND_BLUE, 0.08),
            zerolinecolor=_rgba(BRAND_BLUE, 0.12),
            tickfont=dict(color=TEXT_SEC, size=11),
        ),
        yaxis_title="Users",
        margin=dict(l=14, r=14, t=24, b=14),
    ))
    return fig


# ══════════════════════════════════════════════════════════════
#  USER CLUSTERING  (callback-driven: features + n_clusters)
# ══════════════════════════════════════════════════════════════
CLUSTER_FEATURES = [
    {"label": "Lifetime Value ($)",     "value": "ltv"},
    {"label": "Avg Charge ($)",         "value": "avg_charge_amount"},
    {"label": "Successful Charges",     "value": "n_successful_charges"},
    {"label": "Session Count",          "value": "session_count"},
    {"label": "Active Days",            "value": "active_days"},
    {"label": "Events / Day",           "value": "events_per_day"},
    {"label": "Feature Breadth",        "value": "feature_breadth"},
    {"label": "Rage Clicks",            "value": "n_rage_click"},
    {"label": "Onboarding Done",        "value": "completed_onboarding"},
    {"label": "Checkout Conversion",    "value": "checkout_conversion"},
    {"label": "Pricing Views",          "value": "n_view_pricing"},
    {"label": "Support Contacts",       "value": "n_contact_support"},
    {"label": "Team Invites",           "value": "n_invite_team"},
    {"label": "Failed Charges",         "value": "n_failed_charges"},
    {"label": "Total Events",           "value": "total_events"},
    {"label": "Churn Flag",            "value": "churned"},
    {"label": "Paid Conversion",       "value": "converted"},
    {"label": "Time to Value (days)",  "value": "median_ttv"},
]
DEFAULT_CLUSTER_FEATS = ["ltv", "session_count", "active_days",
                         "feature_breadth", "checkout_conversion",
                         "n_rage_click", "median_ttv"]


def build_clustering(features=None, user_ids=None, highlight_ids=None, n_clusters_override=None):
    """
    n_clusters_override: None / "auto" → scan k=2..8 and pick best silhouette.
                         int (2-8)      → use that k directly (still compute score).
    Returns (fig, cluster_summary, cluster_user_map, k_scores_dict)
    k_scores_dict: {k: silhouette_score} for all k tested.
    """
    if not features or len(features) < 2:
        return _empty_fig("Select at least 2 features"), pd.DataFrame(), {}, {}

    up_ = user_profiles.copy()
    if user_ids is not None and len(user_ids) < len(up_):
        up_ = up_[up_.index.isin(user_ids)]
    if len(up_) < 20:
        return _empty_fig("Not enough users for clustering"), pd.DataFrame(), {}, {}

    feats = [f for f in features if f in up_.columns]
    X = up_[feats].fillna(0)
    Xs = StandardScaler().fit_transform(X)

    _sample_size = min(3000, len(Xs))
    _k_max = min(8, len(Xs) // 15)
    k_range = range(2, max(3, _k_max + 1))
    k_scores: dict = {}

    if n_clusters_override is None or n_clusters_override == "auto":
        # Scan all k and pick best
        best_k, best_score = 2, -1.0
        for k in k_range:
            _km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=150)
            _lbl = _km.fit_predict(Xs)
            _sc = silhouette_score(Xs, _lbl, sample_size=_sample_size, random_state=42)
            k_scores[k] = round(float(_sc), 3)
            if _sc > best_score:
                best_score, best_k = _sc, k
        n_clusters = best_k
        best_score = k_scores[n_clusters]
        chosen_by = "auto"
    else:
        # Use the user-chosen k; still compute scores for all k (for the display)
        n_clusters = int(n_clusters_override)
        for k in k_range:
            _km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=150)
            _lbl = _km.fit_predict(Xs)
            _sc = silhouette_score(Xs, _lbl, sample_size=_sample_size, random_state=42)
            k_scores[k] = round(float(_sc), 3)
        best_score = k_scores.get(n_clusters, 0.0)
        chosen_by = "manual"

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(Xs)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)

    up_ = up_.copy()
    up_["cluster"] = labels
    up_["px"] = coords[:, 0]
    up_["py"] = coords[:, 1]

    ev_pct = pca.explained_variance_ratio_

    fig = go.Figure()
    for cl in sorted(up_["cluster"].unique()):
        sd = up_[up_["cluster"] == cl]
        is_hi = highlight_ids is not None and len(highlight_ids) > 0
        hi_mask = sd.index.isin(highlight_ids) if is_hi else pd.Series(False, index=sd.index)

        size_col = np.log1p(sd["ltv"].clip(lower=0)).values
        size_norm = 6 + (size_col / max(size_col.max(), 1)) * 14

        fig.add_trace(go.Scatter(
            x=sd["px"], y=sd["py"],
            mode="markers",
            name=f"Cluster {cl + 1}",
            marker=dict(
                color=CLUSTER_PALETTE[cl % len(CLUSTER_PALETTE)],
                size=size_norm,
                opacity=0.7,
                line=dict(width=0.5, color=BG_CARD),
            ),
            customdata=np.stack(
                [sd["ltv"].round(0), sd["session_count"],
                 sd["active_days"].round(0), sd["feature_breadth"],
                 sd["checkout_conversion"].round(3)], axis=-1),
            hovertemplate=(
                f"<b>Cluster {cl + 1}</b><br>"
                "LTV: $%{customdata[0]:.0f}<br>"
                "Sessions: %{customdata[1]:.0f}<br>"
                "Active Days: %{customdata[2]:.0f}<br>"
                "Features: %{customdata[3]:.0f}<br>"
                "Conv Rate: %{customdata[4]:.1%}"
                "<extra></extra>"
            ),
        ))

    var_total = ev_pct.sum() * 100
    k_label = f"k={n_clusters} ({'auto' if chosen_by == 'auto' else 'manual'})"
    fig.update_layout(
        **_cdefaults(
            xaxis_title=f"PC1 ({ev_pct[0]*100:.1f}% variance)",
            yaxis_title=f"PC2 ({ev_pct[1]*100:.1f}% variance)",
            height=500,
            title=dict(
                text=(f"User Clusters · {k_label} · "
                      f"silhouette={best_score:.3f} · "
                      f"{var_total:.0f}% PCA variance"),
                font=dict(size=12, color=TEXT_SEC), x=0, xanchor="left",
            ),
            margin=dict(l=14, r=14, t=44, b=14),
        )
    )

    # Cluster summary
    summary_rows = []
    for cl in sorted(up_["cluster"].unique()):
        sd = up_[up_["cluster"] == cl]
        row = {
            "Cluster": f"#{cl + 1}",
            "Users": len(sd),
            "Avg LTV ($)": round(sd["ltv"].mean(), 0),
            "Avg Sessions": round(sd["session_count"].mean(), 1),
            "Onboarding %": round(sd["completed_onboarding"].mean() * 100, 0),
            "Conv %": round(sd["checkout_conversion"].mean() * 100, 1),
            "Avg Rage": round(sd["n_rage_click"].mean(), 1),
            "Churn %": round(sd["churned"].mean() * 100, 1) if "churned" in sd else "—",
            "Converted %": round(sd["converted"].mean() * 100, 1) if "converted" in sd else "—",
            "Median TTV (d)": (round(sd["median_ttv"].dropna().median(), 1)
                               if "median_ttv" in sd and sd["median_ttv"].notna().any() else "—"),
        }
        # Add raw feature means for archetype generation
        for fc in features:
            if fc in sd.columns:
                row[fc] = round(float(sd[fc].mean()), 4)
        summary_rows.append(row)
    cluster_summary = pd.DataFrame(summary_rows)
    # Cluster user map: dict {cluster_label → list of user_ids}
    cluster_user_map = {
        f"Cluster {cl + 1}": up_[up_["cluster"] == cl].index.tolist()
        for cl in sorted(up_["cluster"].unique())
    }
    return fig, cluster_summary, cluster_user_map, k_scores


# ══════════════════════════════════════════════════════════════
#  FRICTION SCORE  (filterable by user group)
# ══════════════════════════════════════════════════════════════
def _friction_for_segment(segment, user_ids=None):
    """Compute friction scores per event for a single segment. Returns DataFrame."""
    uid = get_user_ids(segment) if user_ids is None else user_ids
    evts_f = events[events["user_id"].isin(uid)]
    sd = evts_f.sort_values(["user_id", "timestamp"]).copy()
    sd["next_event"] = sd.groupby("user_id")["event_name"].shift(-1)

    rows = []
    for evt in evts_f["event_name"].unique():
        ef = sd[sd["event_name"] == evt]
        n = len(ef)
        if n < 20:
            continue
        n_loop = (ef["next_event"] == evt).sum()
        n_rage = (ef["next_event"] == "rage_click").sum()
        n_err  = (ef["next_event"] == "error").sum()
        n_drop = (ef["next_event"].isna() | ef["next_event"].eq("session_end")).sum()
        fr = ((n_loop/n)*30 + (n_rage/n)*30 + (n_err/n)*20 + (n_drop/n)*20) * 100
        rows.append(dict(event=evt, friction=round(fr, 1),
                         loop=round(n_loop/n*100, 1), rage=round(n_rage/n*100, 1),
                         error=round(n_err/n*100, 1), dropoff=round(n_drop/n*100, 1),
                         n=n))
    fdf = pd.DataFrame(rows)
    if fdf.empty:
        return fdf
    return fdf[(fdf["friction"] > 0.5) & (fdf["n"] > 20)]


def build_friction_score(segments=None, user_ids=None):
    """Friction score chart. Accepts a list of segment values for comparison."""
    if segments is None:
        segments = ["all"]
    if isinstance(segments, str):
        segments = [segments]

    seg_labels = {s["value"]: s["label"] for s in SEGMENTS}
    # Fixed distinct color per segment for consistent visual identity
    _SEG_COLORS = {
        "all":      BRAND_BLUE,
        "paying":   COL_HI,       # green — revenue positive
        "free":     BRAND_AMBER,  # amber — not yet monetised
        "churned":  COL_LO,       # red — lost
        "active":   BRAND_TEAL,   # teal — healthy
        "high_ltv": BRAND_PURPLE, # purple — premium
        "rage":     "#F06292",    # pink — friction
        "no_ob":    COL_MID,      # orange — incomplete
    }

    # Single segment → original colour-coded view
    if len(segments) == 1:
        fdf = _friction_for_segment(segments[0], user_ids=user_ids)
        if fdf.empty:
            return _empty_fig("No friction data for this segment")
        fdf = fdf.sort_values("friction")

        def _fc(s):
            if s >= 15: return COL_LO
            if s >= 8:  return COL_MID
            return COL_HI

        fig = go.Figure(go.Bar(
            y=fdf["event"], x=fdf["friction"], orientation="h",
            marker_color=[_fc(s) for s in fdf["friction"]],
            customdata=np.stack(
                [fdf["loop"], fdf["rage"], fdf["error"], fdf["dropoff"], fdf["n"]],
                axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>Friction: %{x:.1f}<br>"
                "─────────────────<br>"
                "Self-loop: %{customdata[0]:.1f}%<br>"
                "Rage click after: %{customdata[1]:.1f}%<br>"
                "Error after: %{customdata[2]:.1f}%<br>"
                "Drop-off: %{customdata[3]:.1f}%<br>"
                "─────────────────<br>"
                "Sample: %{customdata[4]:,} events"
                "<extra></extra>"
            ),
        ))
        fig.update_layout(**_cdefaults(xaxis_title="Friction Score (0–100)", height=420))
        return fig

    # Multi-segment → grouped horizontal bars (like friction by revenue segment)
    all_rows = []
    for seg in segments:
        fdf = _friction_for_segment(seg)
        if fdf.empty:
            continue
        fdf["segment"] = seg_labels.get(seg, seg)
        all_rows.append(fdf)

    if not all_rows:
        return _empty_fig("No friction data")

    sdf = pd.concat(all_rows, ignore_index=True)
    top_evts = (sdf.groupby("event")["friction"].mean()
                .nlargest(12).index.tolist())
    sdf = sdf[sdf["event"].isin(top_evts)]
    pivot = sdf.pivot_table(index="event", columns="segment",
                            values="friction", aggfunc="mean").fillna(0)
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=True).index)

    fig = go.Figure()
    for i, seg_val in enumerate(segments):
        seg_name = seg_labels.get(seg_val, seg_val)
        if seg_name not in pivot.columns:
            continue
        color = _SEG_COLORS.get(seg_val, BRAND_BLUE)
        fig.add_trace(go.Bar(
            y=pivot.index, x=pivot[seg_name],
            name=seg_name, orientation="h",
            marker=dict(color=_rgba(color, 0.75),
                        line=dict(color=_rgba(color, 0.95), width=0.5)),
            hovertemplate=f"<b>%{{y}}</b><br>{seg_name} friction: %{{x:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        **_cdefaults(
            height=max(380, len(top_evts) * 45),
            xaxis=dict(title="Friction Score (0–100)",
                       gridcolor=_rgba(BRAND_BLUE, 0.08)),
            yaxis=dict(tickfont=dict(size=10, color=TEXT_SEC)),
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)),
            margin=dict(l=14, r=14, t=50, b=14),
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  COMEBACK HOTSPOTS
# ══════════════════════════════════════════════════════════════
def build_comeback_loops():
    if not _comeback_dict:
        return _empty_fig("No comeback data")
    cdf = pd.DataFrame(
        [{"event": k, "count": v} for k, v in _comeback_dict.items()]
    ).sort_values("count", ascending=True)
    evt_users = events.groupby("event_name")["user_id"].nunique()
    cdf["total_users"] = cdf["event"].map(evt_users).fillna(1)
    cdf["rate"] = (cdf["count"] / cdf["total_users"] * 100).round(1)
    cdf = cdf[cdf["count"] > 20]
    if cdf.empty: return _empty_fig("No comeback data")

    fig = go.Figure()
    fig.add_trace(go.Bar(y=cdf["event"], x=cdf["count"], orientation="h",
                         name="Comeback Count", marker_color=_rgba(BRAND_BLUE, 0.72),
                         hovertemplate="%{y}: %{x:,}<extra></extra>"))
    fig.add_trace(go.Scatter(
        y=cdf["event"], x=cdf["rate"], mode="markers+text", name="Rate %",
        marker=dict(color=COL_MID, size=9, symbol="diamond"),
        text=cdf["rate"].apply(lambda v: f"{v:.0f}%"),
        textposition="middle right", textfont=dict(color=COL_MID, size=9),
        xaxis="x2",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(**_cdefaults(
        height=400,
        xaxis=dict(title="Comeback Count", side="bottom",
                   gridcolor=_rgba(BRAND_BLUE, 0.08)),
        xaxis2=dict(title="Comeback Rate %", side="top", overlaying="x",
                    gridcolor="rgba(0,0,0,0)", ticksuffix="%"),
        barmode="overlay",
    ))
    return fig


# ══════════════════════════════════════════════════════════════
#  FRICTION SIGNAL BREAKDOWN (stacked bar per event)
# ══════════════════════════════════════════════════════════════
def build_friction_breakdown(_ev=None):
    """
    For the top 12 highest-friction events, show a 100% stacked bar breaking
    the friction signal into its four components: loop, rage, error, drop-off.
    Helps PMs understand WHY each event is high-friction.
    """
    ev_ = _ev if _ev is not None else events
    uid = get_user_ids("all")
    evts_f = ev_[ev_["user_id"].isin(uid)]
    sd = evts_f.sort_values(["user_id", "timestamp"]).copy()
    sd["next_event"] = sd.groupby("user_id")["event_name"].shift(-1)

    rows = []
    for evt in evts_f["event_name"].unique():
        ef = sd[sd["event_name"] == evt]
        n = len(ef)
        if n < 20: continue
        n_loop = (ef["next_event"] == evt).sum()
        n_rage = (ef["next_event"] == "rage_click").sum()
        n_err  = (ef["next_event"] == "error").sum()
        n_drop = (ef["next_event"].isna() | ef["next_event"].eq("session_end")).sum()
        fr = ((n_loop/n)*30 + (n_rage/n)*30 + (n_err/n)*20 + (n_drop/n)*20) * 100
        rows.append(dict(event=evt, friction=round(fr,1),
                         loop_pct=round(n_loop/n*100,1),
                         rage_pct=round(n_rage/n*100,1),
                         error_pct=round(n_err/n*100,1),
                         drop_pct=round(n_drop/n*100,1)))

    if not rows:
        return _empty_fig("No friction data")
    fdf = pd.DataFrame(rows).sort_values("friction", ascending=False).head(12)

    # Sort ascending so highest friction appears at top of horizontal bar
    fdf = fdf.sort_values("friction", ascending=True)

    signals = [
        ("drop_pct",  "Drop-off",    COL_LO),
        ("rage_pct",  "Rage Click",  BRAND_AMBER),
        ("error_pct", "Error After", BRAND_PURPLE),
        ("loop_pct",  "Self-Loop",   BRAND_BLUE),
    ]

    fig = go.Figure()
    for col, label, color in signals:
        fig.add_trace(go.Bar(
            y=fdf["event"],
            x=fdf[col],
            name=label,
            orientation="h",
            marker=dict(color=_rgba(color, 0.78), line=dict(color=_rgba(color, 0.95), width=0.5)),
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        **_cdefaults(
            height=360,
            xaxis=dict(title="Signal Composition (%)", ticksuffix="%",
                       gridcolor=_rgba(BRAND_BLUE, 0.08)),
            yaxis=dict(tickfont=dict(size=9, color=TEXT_SEC)),
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)),
            margin=dict(l=14, r=14, t=50, b=14),
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  FRICTION SEGMENT COMPARISON (which segments feel pain more?)
# ══════════════════════════════════════════════════════════════
def build_friction_segment_comparison():
    """
    Grouped horizontal bar: top 8 events by avg friction score,
    one bar per revenue segment. Shows if friction is universal or segment-specific.
    """
    seg_order = ["Free", "Low Revenue", "Mid Revenue", "High Revenue"]
    seg_colors = [COL_FREE, COL_LO, COL_MID, COL_HI]

    up_ = _segment_users()
    seg_ev = events.join(
        up_[["revenue_segment"]], on="user_id"
    ).dropna(subset=["revenue_segment"])

    all_rows = []
    for seg in seg_order:
        evts_f = seg_ev[seg_ev["revenue_segment"] == seg]
        if evts_f.empty:
            continue
        sd = evts_f.sort_values(["user_id", "timestamp"]).copy()
        sd["next_event"] = sd.groupby("user_id")["event_name"].shift(-1)
        for evt in evts_f["event_name"].unique():
            ef = sd[sd["event_name"] == evt]
            n = len(ef)
            if n < 10: continue
            n_loop = (ef["next_event"] == evt).sum()
            n_rage = (ef["next_event"] == "rage_click").sum()
            n_err  = (ef["next_event"] == "error").sum()
            n_drop = (ef["next_event"].isna() | ef["next_event"].eq("session_end")).sum()
            fr = ((n_loop/n)*30 + (n_rage/n)*30 + (n_err/n)*20 + (n_drop/n)*20) * 100
            all_rows.append({"segment": seg, "event": evt, "friction": round(fr, 1)})

    if not all_rows:
        return _empty_fig("No segment friction data")

    sdf = pd.DataFrame(all_rows)
    # Top 8 events by mean friction across all segments
    top_evts = (sdf.groupby("event")["friction"].mean()
                .nlargest(8).index.tolist())
    sdf = sdf[sdf["event"].isin(top_evts)]
    sdf_pivot = sdf.pivot_table(index="event", columns="segment",
                                values="friction", aggfunc="mean").fillna(0)
    sdf_pivot = sdf_pivot.reindex(
        sdf_pivot.mean(axis=1).sort_values(ascending=True).index
    )

    fig = go.Figure()
    for seg, color in zip(seg_order, seg_colors):
        if seg not in sdf_pivot.columns:
            continue
        fig.add_trace(go.Bar(
            y=sdf_pivot.index,
            x=sdf_pivot[seg],
            name=seg,
            orientation="h",
            marker=dict(color=_rgba(color, 0.75), line=dict(color=_rgba(color, 0.95), width=0.5)),
            hovertemplate=f"<b>%{{y}}</b><br>{seg} friction: %{{x:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        **_cdefaults(
            height=360,
            xaxis=dict(title="Friction Score", gridcolor=_rgba(BRAND_BLUE, 0.08)),
            yaxis=dict(tickfont=dict(size=9, color=TEXT_SEC)),
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)),
            margin=dict(l=14, r=14, t=50, b=14),
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  DROP-OFF RISK MAP
# ══════════════════════════════════════════════════════════════
def build_dropoff_risk_map():
    journey = ["signup", "onboarding_step1", "onboarding_step2",
               "view_dashboard", "view_feature_A", "view_feature_B",
               "view_feature_C", "view_feature_D", "view_pricing",
               "start_checkout", "payment_success"]
    journey_rank = {ev: i for i, ev in enumerate(journey)}  # O(1) lookup

    # Map every event to its journey rank (NaN for events not in journey)
    sd = events.copy()
    sd["rank"] = sd["event_name"].map(journey_rank)

    # Per user: max rank reached among journey events (vectorized groupby)
    user_max_rank = (
        sd.dropna(subset=["rank"])
        .groupby("user_id")["rank"].max()
    )
    rank_to_event = {v: k for k, v in journey_rank.items()}
    terminal = user_max_rank.map(rank_to_event).value_counts()

    # Reached counts: single groupby nunique on filtered events
    reached = (
        sd[sd["rank"].notna()]
        .groupby("event_name")["user_id"].nunique()
    )
    ddf = pd.DataFrame({
        "Stage":   journey,
        "Reached": [reached.get(ev, 0) for ev in journey],
        "Stopped": [int(terminal.get(ev, 0)) for ev in journey],
    })
    ddf["Dropoff"] = (ddf["Stopped"] / ddf["Reached"].clip(lower=1) * 100).round(1)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=ddf["Stage"], y=ddf["Reached"],
                         name="Users Reached",
                         marker_color=_rgba(BRAND_BLUE, 0.72)),
                  secondary_y=False)
    fig.add_trace(go.Bar(x=ddf["Stage"], y=ddf["Stopped"],
                         name="Stopped Here",
                         marker_color=_rgba(COL_LO, 0.72)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=ddf["Stage"], y=ddf["Dropoff"],
                              mode="lines+markers+text", name="Drop-off %",
                              line=dict(color=COL_MID, width=2.5),
                              marker=dict(size=9, color=COL_MID,
                                          line=dict(width=2, color="white")),
                              text=ddf["Dropoff"].apply(lambda v: f"{v:.0f}%"),
                              textposition="top center",
                              textfont=dict(color=TEXT_PRI, size=12,
                                            family="Inter, system-ui, sans-serif"),
                              cliponaxis=False),
                  secondary_y=True)
    fig.update_layout(barmode="overlay", height=400,
                      paper_bgcolor=BG_CARD, plot_bgcolor=BG_PAGE,
                      font=dict(family="Inter, system-ui, sans-serif",
                                color=TEXT_SEC, size=10),
                      legend=dict(bgcolor="rgba(0,0,0,0)",
                                  font=dict(color=TEXT_SEC, size=10),
                                  orientation="h", y=-0.25),
                      margin=dict(l=14, r=14, t=24, b=60),
                      xaxis=dict(tickangle=-30,
                                 gridcolor=_rgba(BRAND_BLUE, 0.08),
                                 tickfont=dict(color=TEXT_SEC, size=10)))
    fig.update_yaxes(title_text="Users", gridcolor=_rgba(BRAND_BLUE, 0.08),
                     tickfont=dict(color=TEXT_SEC, size=10), secondary_y=False)
    fig.update_yaxes(title_text="Drop-off %", ticksuffix="%",
                     gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(color=TEXT_SEC, size=10), secondary_y=True)
    return fig


# ══════════════════════════════════════════════════════════════
#  IMPROVEMENT OPPORTUNITY MATRIX  (upgraded)
# ══════════════════════════════════════════════════════════════
def build_opportunity_matrix(y_mode="users", _opp=None):
    """y_mode: 'users' = unique users, 'ltv' = LTV at risk ($),
    'impact' = estimated conversion impact in pp (engagement-matched)."""
    df = (_opp if _opp is not None else opp_df).copy()
    if df.empty: return _empty_fig("No opportunity data")

    if y_mode == "ltv":
        y_col, y_title = "LTV_Impact", "LTV at Risk ($)"
    elif y_mode == "impact":
        if "ConvImpactPP" not in df.columns or df["ConvImpactPP"].isna().all():
            return _empty_fig("No conversion-impact estimate available")
        # Show absolute hurt (positive = bigger loss). Events without an
        # estimate are dropped from this view.
        df = df[df["ConvImpactPP"].notna()].copy()
        df["ConvImpactAbs"] = (-df["ConvImpactPP"]).clip(lower=0)
        y_col, y_title = "ConvImpactAbs", "Est. Conversion Loss (pp, engagement-matched)"
    else:
        y_col, y_title = "Users", "Unique Users (Traffic)"
    med_f = df["Friction"].median()
    med_u = df[y_col].median()

    # Size = priority score (normalised 8–14) — kept small to prevent overlap
    p_min, p_max = df["Priority"].min(), df["Priority"].max()
    if p_max > p_min:
        sizes = 8 + (df["Priority"] - p_min) / (p_max - p_min) * 6
    else:
        sizes = pd.Series([11] * len(df))

    fig = go.Figure()
    for cat in sorted(df["Category"].unique()):
        cd = df[df["Category"] == cat]
        cat_col = CATEGORY_COLORS.get(cat, "#64748B")
        fig.add_trace(go.Scatter(
            x=cd["Friction"], y=cd[y_col],
            mode="markers",
            name=cat.replace("_", " ").title(),
            text=cd["Event"],
            marker=dict(
                size=sizes[cd.index].values,
                color=cat_col,
                opacity=0.82,
                line=dict(width=1, color=_rgba(cat_col, 0.4)),
            ),
            customdata=np.stack(
                [cd["Priority"], cd["Dropoff"], cd["Action"],
                 cd.get("LTV_Impact", pd.Series(0, index=cd.index)),
                 cd.get("ConvImpactPP", pd.Series(np.nan, index=cd.index))],
                axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"Friction: %{{x:.1f}} · {y_title}: %{{y:,.2f}}<br>"
                "Drop-off: %{customdata[1]:.1f}%<br>"
                "LTV at Risk: $%{customdata[3]:,.0f}<br>"
                "Est. Conv Impact: %{customdata[4]:+.2f} pp<br>"
                "Priority: %{customdata[0]:.0f}<br>"
                "%{customdata[2]}"
                "<extra>" + cat + "</extra>"
            ),
        ))

    x_min = df["Friction"].min()
    x_max = df["Friction"].max()
    y_min = df[y_col].min()
    y_max = df[y_col].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = max((y_max - y_min) * 0.05, 1)

    for x0, x1, y0, y1, col in [
        (med_f, x_max + x_pad, med_u, y_max + y_pad, _rgba(COL_LO,     0.07)),
        (x_min - x_pad, med_f, med_u, y_max + y_pad, _rgba(COL_HI,     0.06)),
        (med_f, x_max + x_pad, y_min - y_pad, med_u, _rgba(COL_MID,    0.05)),
        (x_min - x_pad, med_f, y_min - y_pad, med_u, _rgba(TEXT_MUTED, 0.04)),
    ]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=col, line_width=0, layer="below")

    fig.add_hline(y=med_u, line_dash="dot", line_color=_rgba(TEXT_MUTED, 0.30))
    fig.add_vline(x=med_f, line_dash="dot", line_color=_rgba(TEXT_MUTED, 0.30))

    # Annotate only the top-5 highest-priority events — keeps chart readable
    _ann_offsets = [
        (30, -25), (-30, -25), (30, 25), (-30, 25), (40, 0),
    ]
    top5 = df.nlargest(5, "Priority")
    for i, (_, row) in enumerate(top5.iterrows()):
        ax, ay = _ann_offsets[i % len(_ann_offsets)]
        fig.add_annotation(
            x=row["Friction"], y=row[y_col],
            text=row["Event"].replace("_", " ").title()[:20],
            showarrow=True,
            arrowhead=2, arrowsize=0.8, arrowwidth=1,
            arrowcolor=_rgba(TEXT_MUTED, 0.4),
            ax=ax, ay=ay,
            font=dict(size=8, color=TEXT_SEC,
                      family="Inter, system-ui, sans-serif"),
            bgcolor=_rgba(BG_CARD, 0.85),
            borderpad=2,
            bordercolor=BORDER,
            borderwidth=0.5,
        )

    fig.update_layout(**_cdefaults(
        xaxis_title="Friction Score (higher = more user pain)",
        yaxis_title=y_title,
        height=520,
        margin=dict(l=14, r=14, t=40, b=14),
    ))
    return fig


def build_opportunity_table(y_mode="users"):
    df = opp_df.copy()
    if y_mode == "ltv":
        df = df.sort_values("LTV_Impact", ascending=False).head(20).copy()
    else:
        df = df.head(20).copy()
    df["Priority"] = df["Priority"].astype(int)
    df["LTV_Impact"] = df["LTV_Impact"].astype(int)

    if y_mode == "ltv":
        cols = [
            {"name": "Event",           "id": "Event"},
            {"name": "Category",        "id": "Category"},
            {"name": "Friction",        "id": "Friction"},
            {"name": "LTV at Risk ($)", "id": "LTV_Impact", "type": "numeric",
             "format": {"specifier": "$,"}},
            {"name": "Users Affected",  "id": "Users", "type": "numeric",
             "format": {"specifier": ","}},
            {"name": "Drop-off %",      "id": "Dropoff"},
            {"name": "Priority Score",  "id": "Priority"},
            {"name": "Action",          "id": "Action"},
        ]
        sort_by = [{"column_id": "LTV_Impact", "direction": "desc"}]
    else:
        cols = [
            {"name": "Event",           "id": "Event"},
            {"name": "Category",        "id": "Category"},
            {"name": "Friction",        "id": "Friction"},
            {"name": "Users",           "id": "Users", "type": "numeric",
             "format": {"specifier": ","}},
            {"name": "Drop-off %",      "id": "Dropoff"},
            {"name": "Priority Score",  "id": "Priority"},
            {"name": "Action",          "id": "Action"},
        ]
        sort_by = [{"column_id": "Priority", "direction": "desc"}]

    return DataTable(
        data=df.to_dict("records"),
        columns=cols,
        sort_action="native",
        sort_by=sort_by,
        style_table={"overflowX": "auto", "borderRadius": "8px",
                     "border": f"1px solid {BORDER}"},
        style_header={"backgroundColor": BG_SURFACE,
                      "color": TEXT_PRI,
                      "fontWeight": "600",
                      "fontSize": "12px",
                      "border": f"1px solid {BORDER}",
                      "fontFamily": "Inter, system-ui, sans-serif"},
        style_data={"backgroundColor": BG_CARD,
                    "color": TEXT_PRI,
                    "fontSize": "11px",
                    "border": f"1px solid {_rgba(BRAND_BLUE, 0.08)}",
                    "fontFamily": "Inter, system-ui, sans-serif"},
        style_data_conditional=[
            {"if": {"filter_query": '{Action} contains "Fix First"'},
             "color": COL_LO, "fontWeight": "600"},
            {"if": {"filter_query": '{Action} contains "Monitor"'},
             "color": COL_MID},
            {"if": {"filter_query": '{Action} contains "Healthy"'},
             "color": COL_HI},
            {"if": {"row_index": "odd"},
             "backgroundColor": BG_SURFACE},
        ],
        page_size=10,
    )


def _opp_summary():
    top3 = opp_df[opp_df["Action"].str.contains("Fix First")].head(3)
    total_critical = len(opp_df[opp_df["Action"].str.contains("Fix First")])
    affected_users = opp_df[opp_df["Action"].str.contains("Fix First")]["Users"].sum()
    top_event = opp_df.iloc[0]["Event"] if len(opp_df) else "N/A"
    top_priority = opp_df.iloc[0]["Priority"] if len(opp_df) else 0

    return html.Div([
        html.Div([
            html.Span("⚡ ", style={"fontSize": "1.1rem"}),
            html.Strong(f"{total_critical} events ", style={"color": COL_LO}),
            html.Span("need immediate attention · Affects "),
            html.Strong(f"{affected_users:,} users ", style={"color": COL_MID}),
            html.Span("· Top issue: "),
            html.Strong(f"{top_event}", style={"color": TEXT_PRI}),
            html.Span(f" (Priority: {top_priority:.0f})", style={"color": TEXT_MUTED}),
        ], style={"padding": "12px 18px",
                  "backgroundColor": _rgba(BRAND_BLUE, 0.07),
                  "border": f"1px solid {_rgba(BRAND_BLUE, 0.25)}",
                  "borderRadius": "8px", "fontSize": "0.85rem",
                  "color": TEXT_SEC, "marginBottom": "16px"}),
        html.Div([
            html.P([
                html.Strong("How to read this matrix: ", style={"color": TEXT_PRI}),
                "Each bubble = one event in the product. ",
                html.Span("X-axis = friction score", style={"color": COL_LO}),
                " (loops + rage clicks + errors + drop-offs). ",
                html.Span("Y-axis = traffic", style={"color": BRAND_BLUE}),
                " (unique users). ",
                html.Span("Bubble size = overall priority score", style={"color": COL_MID}),
                " (friction × traffic × drop-off). ",
                "Top-right quadrant = your biggest wins.",
            ], style={"fontSize": "0.78rem", "color": TEXT_MUTED,
                      "marginBottom": "0"}),
        ]),
    ])


# ══════════════════════════════════════════════════════════════
#  TRANSITION HEATMAP  (from dashboard.py)
# ══════════════════════════════════════════════════════════════
def build_transition_table():
    """Sorted DataTable of top event-to-event transition probabilities."""
    if transitions is None or transitions.empty:
        return html.P("No transition data available", style={"color": TEXT_MUTED})

    t = transitions.copy()
    # Exclude session boundaries and self-loops
    _sess = {"session_start", "session_end"}
    t = t[
        ~t["source"].isin(_sess) & ~t["target"].isin(_sess) &
        (t["source"] != t["target"])
    ]
    if t.empty:
        return html.P("No meaningful transitions found", style={"color": TEXT_MUTED})

    src_totals = t.groupby("source")["count"].sum().rename("src_total")
    t = t.join(src_totals, on="source")
    t["probability"] = (t["count"] / t["src_total"] * 100).round(1)
    t = t.nlargest(60, "count")[["source", "target", "count", "probability"]].reset_index(drop=True)
    t.columns = ["From Event", "To Event", "Count", "Probability (%)"]

    return DataTable(
        data=t.to_dict("records"),
        columns=[{"name": c, "id": c} for c in t.columns],
        sort_action="native",
        sort_by=[{"column_id": "Count", "direction": "desc"}],
        filter_action="native",
        page_size=20,
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": BG_CARD, "color": TEXT_SEC,
                    "border": f"1px solid {BORDER}", "fontSize": "11px",
                    "padding": "6px 10px", "fontFamily": "Inter, monospace",
                    "textAlign": "left"},
        style_header={"backgroundColor": BG_SURFACE, "color": TEXT_PRI,
                      "fontWeight": "600", "border": f"1px solid {BORDER}",
                      "fontFamily": "Inter, system-ui, sans-serif", "fontSize": "11px"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": BG_SURFACE},
            {"if": {"column_id": "Probability (%)",
                    "filter_query": "{Probability (%)} > 40"},
             "color": COL_HI, "fontWeight": "600"},
        ],
    )


# Keep old name used in pre-build (returns empty fig — table is built inline in layout)
def build_transition_heatmap():
    return build_transition_table()


# ══════════════════════════════════════════════════════════════
#  AVG TIME BETWEEN EVENTS  (from dashboard.py)
# ══════════════════════════════════════════════════════════════
def build_avg_time_between(within_session=True):
    """
    Average hours between top event transitions.
    within_session=True  → only transitions that completed within 30 min (same session)
    within_session=False → all transitions, including cross-session gaps
    """
    if transitions is None or transitions.empty or "avg_time_seconds" not in transitions.columns:
        return _empty_fig("No transition timing data")
    top_t = transitions.copy()
    # Remove session-boundary and self-loop transitions
    _sess_evts = {"session_end", "session_start"}
    top_t = top_t[
        ~top_t["source"].isin(_sess_evts) &
        ~top_t["target"].isin(_sess_evts) &
        (top_t["source"] != top_t["target"])
    ]

    has_ws = "avg_time_seconds_ws" in top_t.columns
    if within_session and has_ws:
        top_t = top_t.dropna(subset=["avg_time_seconds_ws"])
        time_col   = "avg_time_seconds_ws"
        scope_text = "within session"
        x_title    = "Avg Minutes Between Events (within session)"
        divisor    = 60.0   # show minutes for within-session (more readable)
    else:
        time_col   = "avg_time_seconds"
        scope_text = "all transitions"
        x_title    = "Avg Hours Between Events (all transitions)"
        divisor    = 3600.0

    top_t["label"]     = top_t["source"] + " → " + top_t["target"]
    top_t["avg_units"] = (top_t[time_col] / divisor).round(2)
    top_t = top_t[top_t["avg_units"] > 0]
    top_t = top_t.sort_values("avg_units", ascending=True).tail(15)

    fig = go.Figure(go.Bar(
        y=top_t["label"], x=top_t["avg_units"], orientation="h",
        marker=dict(
            color=top_t["avg_units"],
            colorscale=[[0, BRAND_BLUE], [0.5, BRAND_TEAL], [1, BRAND_AMBER]],
            showscale=False,
        ),
        hovertemplate=f"%{{y}}<br>%{{x:.2f}} ({'min' if within_session and has_ws else 'hrs'}) "
                      f"({scope_text})<extra></extra>",
    ))
    fig.update_layout(**_cdefaults(
        height=400, xaxis_title=x_title,
        margin=dict(l=14, r=14, t=24, b=14),
    ))
    return fig


# ══════════════════════════════════════════════════════════════
#  STEP DISTRIBUTION  (from dashboard.py)
# ══════════════════════════════════════════════════════════════
def build_step_distribution():
    """Stacked % bar: which events occur at post-signup steps 1–10."""
    if step_seq is None or step_seq.empty:
        return _empty_fig("No step sequence data")

    # Keep only post-signup steps
    signup_steps = (
        step_seq[step_seq["event_name"] == "signup"]
        .groupby("user_id")["step"].min()
        .rename("signup_step")
    )
    sq = step_seq.join(signup_steps, on="user_id")
    sq["signup_step"] = sq["signup_step"].fillna(0)
    sq = sq[sq["step"] > sq["signup_step"]].copy()

    # Re-number steps from 1 within each user's post-signup sequence
    sq = sq.sort_values(["user_id", "step"])
    sq["post_step"] = sq.groupby("user_id").cumcount() + 1
    sq = sq[sq["post_step"] <= 10]

    top_events = (
        sq.groupby("event_name")["user_id"].nunique()
        .nlargest(10).index.tolist()
    )
    sq["event_display"] = sq["event_name"].where(
        sq["event_name"].isin(top_events), "other"
    )
    # Use EVENT_CATEGORIES to map events → category colours for visual coherence
    def _evt_color(evt_name, idx):
        for cat, evts in EVENT_CATEGORIES.items():
            if evt_name in evts:
                return CATEGORY_COLORS.get(cat, "#64748B")
        fallback = ["#4B63F5","#06D6A0","#9333EA","#F59E0B","#EF4444",
                    "#22D3EE","#F97316","#EC4899","#94A3B8","#CBD5E1","#64748B"]
        return fallback[idx % len(fallback)]

    unique_evts = top_events + (["other"] if "other" in sq["event_display"].values else [])
    color_map = {e: _evt_color(e, i) for i, e in enumerate(unique_evts)}

    pivot = (
        sq.groupby(["post_step", "event_display"]).size()
        .reset_index(name="count")
    )
    total = pivot.groupby("post_step")["count"].transform("sum")
    pivot["pct"] = (pivot["count"] / total * 100).round(1)

    fig = go.Figure()
    for evt in unique_evts:
        d = pivot[pivot["event_display"] == evt]
        if d.empty:
            continue
        base_color = color_map[evt]
        fig.add_trace(go.Bar(
            x=d["post_step"],
            y=d["pct"],
            name=evt,
            marker=dict(
                color=_rgba(base_color, 0.72),
                line=dict(color=_rgba(base_color, 0.95), width=0.6),
                opacity=1.0,
                pattern=dict(shape="", fillmode="overlay"),
            ),
            hovertemplate=(
                f"<b>{evt}</b><br>"
                "Step %{x}<br>"
                "Share: %{y:.1f}%"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="stack",
        bargap=0.10,
        height=380,
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_PAGE,
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT_SEC, size=11),
        legend=dict(
            bgcolor=_rgba(BG_SURFACE, 0.9),
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(color=TEXT_SEC, size=9),
            orientation="h",
            y=-0.28, x=0,
        ),
        margin=dict(l=14, r=14, t=32, b=14),
        xaxis=dict(
            title="Post-Signup Step",
            dtick=1,
            gridcolor=_rgba(BRAND_BLUE, 0.06),
            zeroline=False,
            tickfont=dict(color=TEXT_SEC, size=10),
        ),
        yaxis=dict(
            title="Share of Events (%)",
            ticksuffix="%",
            range=[0, 100],
            gridcolor=_rgba(BRAND_BLUE, 0.06),
            zeroline=False,
            tickfont=dict(color=TEXT_SEC, size=10),
        ),
        uniformtext=dict(minsize=7, mode="hide"),
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  SELF-LOOP CHART  (from dashboard.py)
# ══════════════════════════════════════════════════════════════
def build_self_loop_chart(_loops=None):
    """Top 12 events where users get stuck (consecutive repeats)."""
    loops_ = _loops if _loops is not None else self_loops
    if loops_ is None or loops_.empty:
        return _empty_fig("No self-loop data")
    loop_agg = (
        loops_.groupby("event_name")["loop_count"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "total_loops", "count": "users_affected"})
        .sort_values("total_loops", ascending=True)
        .tail(12)
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=loop_agg["event_name"], x=loop_agg["total_loops"],
        orientation="h", name="Total Loops",
        marker_color=_rgba(COL_LO, 0.72),
        hovertemplate="%{y}: %{x:,} repeats · %{customdata:,} users affected<extra></extra>",
        customdata=loop_agg["users_affected"],
    ))
    fig.update_layout(**_cdefaults(
        height=400, xaxis_title="Total Consecutive Repeats",
        margin=dict(l=14, r=14, t=24, b=14),
    ))
    return fig


# ══════════════════════════════════════════════════════════════
#  COHORT RETENTION CURVES  (proper per-cohort normalization)
# ══════════════════════════════════════════════════════════════
def build_cohort_retention_heatmap(roll_mode="7d", _up=None, _ev_raw=None):
    """Cohort retention heatmap — rows = signup cohort, cols = period N after signup.
    roll_mode: '7d' (weekly), '14d' (bi-weekly), 'monthly' (30-day periods)."""
    up__ = _up if _up is not None else user_profiles
    ev_raw_ = _ev_raw if _ev_raw is not None else events_raw
    up_ = up__.reset_index()[["user_id", "first_event"]].copy()
    up_["join_month"] = up_["first_event"].dt.to_period("M").astype(str)
    month_counts = up_["join_month"].value_counts()
    valid_months = sorted(month_counts[month_counts >= 30].index.tolist())[-10:]
    if not valid_months:
        return _empty_fig("Not enough cohort data")
    up_ = up_[up_["join_month"].isin(valid_months)]
    cohort_sizes = up_.groupby("join_month")["user_id"].count().to_dict()

    merged = ev_raw_.merge(up_[["user_id", "first_event", "join_month"]], on="user_id")
    merged["day_n"] = (
        (merged["timestamp"].dt.normalize() - merged["first_event"].dt.normalize())
        .dt.days
    )

    if roll_mode == "7d":
        period_days, max_p = 7, 9
        col_labels = [f"W{p}" for p in range(max_p)]
    elif roll_mode == "14d":
        period_days, max_p = 14, 7
        col_labels = [f"d{p*14}–{(p+1)*14}" for p in range(max_p)]
    else:  # monthly
        period_days, max_p = 30, 5
        col_labels = [f"M{p}" for p in range(max_p)]

    matrix, text_matrix = [], []
    for month in valid_months:
        cohort_size = cohort_sizes[month]
        m_evts = merged[merged["join_month"] == month]
        row, t_row = [], []
        for p in range(max_p):
            d0, d1 = p * period_days, (p + 1) * period_days
            active = m_evts[m_evts["day_n"].between(d0, d1 - 1)]["user_id"].nunique()
            pct = round(active / max(cohort_size, 1) * 100, 1)
            row.append(pct)
            t_row.append(f"{pct:.0f}%")
        matrix.append(row)
        text_matrix.append(t_row)

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=col_labels,
        y=[f"{m} (n={cohort_sizes[m]:,})" for m in valid_months],
        colorscale=[[0, BG_SURFACE], [0.3, _rgba(BRAND_BLUE, 0.25)],
                    [0.6, _rgba(BRAND_BLUE, 0.55)], [1, BRAND_BLUE]],
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=10, color=TEXT_PRI),
        zmin=0, zmax=100,
        hovertemplate="Cohort: %{y}<br>Period: %{x}<br>Retention: %{z:.1f}%<extra></extra>",
        showscale=True,
        colorbar=dict(ticksuffix="%", tickfont=dict(color=TEXT_SEC, size=9)),
    ))
    fig.update_layout(**_cdefaults(
        height=max(300, len(valid_months) * 42 + 60),
        xaxis=dict(tickfont=dict(color=TEXT_SEC, size=10)),
        yaxis=dict(tickfont=dict(color=TEXT_SEC, size=10)),
        margin=dict(l=14, r=44, t=28, b=14),
    ))
    return fig


# Keep old name as an alias for the pre-build (will be overridden in layout by callback)
def build_cohort_retention(_up=None, _ev_raw=None):
    return build_cohort_retention_heatmap("7d", _up=_up, _ev_raw=_ev_raw)


# ══════════════════════════════════════════════════════════════
#  MRR TREND
# ══════════════════════════════════════════════════════════════
def build_mrr_trend():
    """Monthly MRR bar + MoM % change line (dual-axis)."""
    if mrr_df is None or mrr_df.empty:
        return _empty_fig("No invoice data for MRR")
    df = mrr_df.copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df["month"], y=df["mrr"],
        name="MRR ($)",
        marker_color=_rgba(BRAND_BLUE, 0.72),
        hovertemplate="%{x}<br>MRR: $%{y:,.0f}<extra></extra>",
    ), secondary_y=False)
    if "mom_pct" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["month"], y=df["mom_pct"],
            name="MoM Growth %", mode="lines+markers",
            line=dict(color=BRAND_TEAL, width=2),
            marker=dict(size=6, color=BRAND_TEAL),
            hovertemplate="%{x}<br>MoM: %{y:+.1f}%<extra></extra>",
        ), secondary_y=True)
    fig.update_layout(
        barmode="group", height=320,
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PAGE,
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT_SEC, size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SEC, size=10),
                    orientation="h", y=-0.22),
        margin=dict(l=14, r=14, t=24, b=14),
        xaxis=dict(gridcolor=_rgba(BRAND_BLUE, 0.08),
                   tickfont=dict(color=TEXT_SEC, size=10)),
    )
    fig.update_yaxes(title_text="Monthly Revenue ($)", gridcolor=_rgba(BRAND_BLUE, 0.08),
                     tickfont=dict(color=TEXT_SEC, size=10), tickprefix="$",
                     secondary_y=False)
    fig.update_yaxes(title_text="MoM Growth %", ticksuffix="%",
                     gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(color=TEXT_SEC, size=10), secondary_y=True)
    return fig


# ══════════════════════════════════════════════════════════════
#  FEATURE ADOPTION CURVE
# ══════════════════════════════════════════════════════════════
def build_feature_adoption(user_ids=None):
    """Cumulative % of users who have used each feature by day N after signup."""
    features = ["view_feature_A", "view_feature_B", "view_feature_C", "view_feature_D"]
    up_ = user_profiles.reset_index()[["user_id", "first_event"]].copy()
    if user_ids is not None:
        up_ = up_[up_["user_id"].isin(user_ids)]
    if up_.empty:
        return _empty_fig("No users in selection")
    n_total = len(up_)
    evts_ = events if user_ids is None else events[events["user_id"].isin(user_ids)]
    merged = evts_.merge(up_[["user_id", "first_event"]], on="user_id")
    merged["day_n"] = (
        (merged["timestamp"].dt.normalize() - merged["first_event"].dt.normalize())
        .dt.days
    )
    palette = [BRAND_BLUE, BRAND_TEAL, BRAND_PURPLE, BRAND_AMBER]
    fig = go.Figure()
    for i, feat in enumerate(features):
        fd = merged[merged["event_name"] == feat]
        if fd.empty:
            continue
        first_use = fd.groupby("user_id")["day_n"].min()
        sorted_days = np.sort(first_use.values)
        days = np.arange(0, 31)
        # searchsorted gives count of users who adopted by each day threshold
        n_adopted = np.searchsorted(sorted_days, days, side="right")
        cum_df = pd.DataFrame({"day_n": days,
                               "pct": n_adopted / n_total * 100})
        fig.add_trace(go.Scatter(
            x=cum_df["day_n"], y=cum_df["pct"],
            mode="lines", name=feat.replace("view_", ""),
            line=dict(color=palette[i], width=2),
            fill="tozeroy" if i == 0 else None,
            fillcolor=_rgba(palette[i], 0.06) if i == 0 else None,
            hovertemplate=f"{feat}<br>Day %{{x}}: %{{y:.1f}}% adopted<extra></extra>",
        ))
    fig.update_layout(**_cdefaults(
        height=320, xaxis_title="Days Since Signup",
        yaxis=dict(title="% Users Adopted", ticksuffix="%",
                   tickfont=dict(color=TEXT_SEC, size=10)),
        margin=dict(l=14, r=14, t=24, b=14),
    ))
    return fig


# ══════════════════════════════════════════════════════════════
#  INDIVIDUAL USER JOURNEY VIEWER  (FullStory-like)
# ══════════════════════════════════════════════════════════════
def _detect_sessions(df, gap_minutes=30):
    """Assign session numbers based on inactivity gaps."""
    df = df.sort_values("timestamp").copy()
    prev = df["timestamp"].shift(1)
    gap = (df["timestamp"] - prev).dt.total_seconds() / 60
    df["session_id"] = ((gap > gap_minutes) | gap.isna()).cumsum()
    return df


def build_user_journey_viewer(user_id, include_categories=None):
    """FullStory-like session swimlane timeline for a single user."""
    ud = events_raw[events_raw["user_id"] == user_id].sort_values("timestamp")
    if ud.empty:
        return _empty_fig(f"User {user_id} not found in data")
    ud = ud.copy()

    # Detect sessions on the full event set (including session_start/end) so that
    # 30-min inactivity gaps are measured correctly, then drop the raw
    # session_start / session_end events from the plotted points — they often span
    # days in the source data (synthetic artefact) and the swimlane rows already
    # communicate session boundaries clearly.
    ud = _detect_sessions(ud)
    ud = ud[~ud["event_name"].isin({"session_start", "session_end"})].copy()

    ud["category"] = ud["event_name"].map(EVENT_CATEGORIES).fillna("other")

    if include_categories:
        ud = ud[ud["category"].isin(include_categories)]
    if ud.empty:
        return _empty_fig("No events match the selected categories")

    # Build session labels: "S1 · Jan 15" etc.
    session_info = (
        ud.groupby("session_id")["timestamp"]
        .agg(["min", "max"])
        .reset_index()
    )
    session_info["label"] = (
        "S" + session_info["session_id"].astype(int).astype(str) +
        " · " + session_info["min"].dt.strftime("%b %d, %H:%M")
    )
    session_info["duration_min"] = (
        (session_info["max"] - session_info["min"]).dt.total_seconds() / 60
    ).round(0)
    label_map = dict(zip(session_info["session_id"], session_info["label"]))
    ud["session_label"] = ud["session_id"].map(label_map)

    # Time-since-previous event within session
    ud["prev_ts"] = ud.groupby("session_id")["timestamp"].shift(1)
    ud["delta_s"] = (ud["timestamp"] - ud["prev_ts"]).dt.total_seconds()
    _s = ud["delta_s"].fillna(-1)
    ud["delta_str"] = np.where(
        ud["delta_s"].isna(), "—",
        np.where(
            _s < 60,
            _s.clip(lower=0).astype(int).astype(str) + "s",
            (_s // 60).astype(int).astype(str) + "m " +
            (_s % 60).astype(int).astype(str) + "s",
        ),
    )

    fig = go.Figure()
    for cat in sorted(ud["category"].unique()):
        grp = ud[ud["category"] == cat]
        color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])
        fig.add_trace(go.Scatter(
            x=grp["timestamp"],
            y=grp["session_label"],
            mode="markers",
            name=cat,
            marker=dict(
                color=color, size=11, opacity=0.88,
                symbol="circle",
                line=dict(color=BG_CARD, width=1.5),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{x|%H:%M:%S}  ·  Δ %{customdata[1]}<br>"
                f"<span style='color:{color}'>■</span> {cat}"
                "<extra></extra>"
            ),
            customdata=list(zip(grp["event_name"], grp["delta_str"])),
        ))

    n_sessions = ud["session_id"].nunique()
    n_events = len(ud)
    first_ts = ud["timestamp"].min()
    last_ts = ud["timestamp"].max()
    span_days = max((last_ts - first_ts).days, 1)

    fig.update_layout(**_cdefaults(
        height=max(340, min(700, n_sessions * 54 + 80)),
        title=dict(
            text=(f"User {user_id} · {n_events:,} events · "
                  f"{n_sessions} sessions · {span_days} day span"),
            font=dict(color=TEXT_PRI, size=12),
            x=0, pad=dict(l=0),
        ),
        xaxis=dict(
            title="Time",
            tickfont=dict(color=TEXT_SEC, size=9),
            gridcolor=_rgba(BRAND_BLUE, 0.08),
            showgrid=True,
        ),
        yaxis=dict(
            tickfont=dict(color=TEXT_SEC, size=10),
            gridcolor=_rgba(BRAND_BLUE, 0.06),
            categoryorder="array",
            categoryarray=session_info["label"].tolist(),  # oldest first
        ),
        margin=dict(l=14, r=14, t=44, b=14),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SEC, size=9),
            orientation="h", y=-0.12, x=0,
        ),
    ))
    return fig


def build_journey_event_table(user_id, include_categories=None):
    """Return DataTable data+columns for the user's event log."""
    ud = events_raw[events_raw["user_id"] == user_id].sort_values("timestamp")
    if ud.empty:
        return [], []
    ud = ud.copy()
    # Detect sessions first on full event set, then drop session boundary markers
    ud = _detect_sessions(ud)
    ud = ud[~ud["event_name"].isin({"session_start", "session_end"})].copy()
    ud["category"] = ud["event_name"].map(EVENT_CATEGORIES).fillna("other")
    if include_categories:
        ud = ud[ud["category"].isin(include_categories)]
    ud["prev_ts"] = ud.groupby("session_id")["timestamp"].shift(1)
    ud["delta_s"] = (ud["timestamp"] - ud["prev_ts"]).dt.total_seconds()
    ud["Δt"] = ud["delta_s"].apply(
        lambda x: f"{int(x)}s" if pd.notna(x) and x < 60 else
                  (f"{int(x/60)}m {int(x%60)}s" if pd.notna(x) else "—")
    )
    ud["Session"] = "S" + ud["session_id"].astype(int).astype(str)
    ud["Time"] = ud["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    table_df = ud[["Session", "Time", "event_name", "category", "Δt"]].copy()
    table_df.columns = ["Session", "Time", "Event", "Category", "Δt"]
    cols = [{"name": c, "id": c} for c in table_df.columns]
    return table_df.to_dict("records"), cols


def build_journey_profile(user_id):
    """Return an html.Div profile panel like FullStory's left sidebar."""
    ud = events_raw[events_raw["user_id"] == user_id]
    if ud.empty:
        return html.Div(f"User {user_id} not found", style={"color": COL_LO})

    up_row = user_profiles[user_profiles.index == user_id]
    ud_s = _detect_sessions(ud.copy())
    n_sessions = ud_s["session_id"].nunique()
    avg_evts = len(ud) / max(n_sessions, 1)

    def _row(label, value, color=TEXT_PRI):
        return html.Div([
            html.Span(label, className="journey-profile-label"),
            html.Span(str(value), className="journey-profile-value",
                      style={"color": color}),
        ])

    items = [
        html.H6(f"User #{user_id}", style={"color": TEXT_PRI, "fontWeight": "700",
                                            "fontSize": "1rem", "marginBottom": "14px"}),
    ]

    if not up_row.empty:
        r = up_row.iloc[0]
        ltv_val = f"${r['ltv']:.0f}" if r.get("ltv", 0) > 0 else "$0"
        status = "Churned" if r.get("churned") else "Active"
        status_color = COL_LO if r.get("churned") else COL_HI
        items += [
            _row("Status", status, status_color),
            _row("Lifetime Value", ltv_val, BRAND_TEAL),
            _row("Onboarding",
                 "Completed" if r.get("completed_onboarding") else "Incomplete",
                 COL_HI if r.get("completed_onboarding") else COL_MID),
            _row("Converted", "Yes" if r.get("converted") else "No",
                 COL_HI if r.get("converted") else TEXT_MUTED),
            _row("Active Days", f"{int(r.get('active_days', 0))}"),
            _row("First Seen",
                 pd.Timestamp(r["first_event"]).strftime("%b %d, %Y")
                 if pd.notna(r.get("first_event")) else "—"),
            _row("Last Seen",
                 pd.Timestamp(r["last_event"]).strftime("%b %d, %Y")
                 if pd.notna(r.get("last_event")) else "—"),
            html.Hr(style={"borderColor": BORDER, "margin": "10px 0"}),
        ]

    items += [
        _row("Total Events", f"{len(ud):,}"),
        _row("Sessions", f"{n_sessions:,}"),
        _row("Avg Events/Session", f"{avg_evts:.1f}"),
    ]

    return html.Div(items, className="journey-profile-panel")


# ══════════════════════════════════════════════════════════════
#  NEW PRODUCT ANALYTICS CHARTS
# ══════════════════════════════════════════════════════════════

TTV_EVENT_OPTIONS = [
    {"label": "Payment Success",         "value": "payment_success"},
    {"label": "Onboarding Step 2",       "value": "onboarding_step2"},
    {"label": "First Feature Use (A)",   "value": "view_feature_A"},
    {"label": "View Dashboard",          "value": "view_dashboard"},
    {"label": "Start Checkout",          "value": "start_checkout"},
    {"label": "Plan Upgrade",            "value": "plan_upgrade"},
    {"label": "Invite Team Member",      "value": "invite_team"},
]


def build_time_to_value(value_event="payment_success", user_ids=None):
    """CDF of days from signup to first occurrence of value_event."""
    evts_ = events_raw if user_ids is None else events_raw[events_raw["user_id"].isin(user_ids)]
    up_   = user_profiles if user_ids is None else user_profiles[user_profiles.index.isin(user_ids)]

    val_evts = evts_[evts_["event_name"] == value_event]
    first_val = val_evts.groupby("user_id")["timestamp"].min().rename("val_ts")
    up_m = up_.reset_index()[["user_id", "first_event"]].join(
        first_val, on="user_id", how="inner"
    )
    if up_m.empty:
        label = next((o["label"] for o in TTV_EVENT_OPTIONS if o["value"] == value_event),
                     value_event)
        return _empty_fig(f"No users reached '{label}'")

    up_m["ttv_days"] = (
        (up_m["val_ts"] - up_m["first_event"]).dt.total_seconds() / 86400
    ).clip(lower=0, upper=90)
    ttv = np.sort(up_m["ttv_days"].dropna().values)
    cdf = np.arange(1, len(ttv) + 1) / len(ttv) * 100
    p50 = float(np.percentile(ttv, 50))
    p75 = float(np.percentile(ttv, 75))
    label = next((o["label"] for o in TTV_EVENT_OPTIONS if o["value"] == value_event),
                 value_event)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ttv, y=cdf, mode="lines",
        line=dict(color=BRAND_TEAL, width=2.5),
        fill="tozeroy", fillcolor=_rgba(BRAND_TEAL, 0.10),
        hovertemplate="Day %{x:.0f}: %{y:.1f}% reached<extra></extra>",
        name=f"CDF – {label}",
    ))
    for pct_val, days, col in [(50, p50, BRAND_AMBER), (75, p75, COL_MID)]:
        fig.add_vline(x=days, line_dash="dot", line_color=_rgba(col, 0.65))
        fig.add_annotation(x=days + 0.5, y=pct_val + 6,
                           text=f"p{pct_val}: {days:.0f}d",
                           showarrow=False, font=dict(size=9, color=col))
    n_reached = len(ttv)
    n_total   = len(up_)
    fig.update_layout(**_cdefaults(
        height=320,
        title=dict(
            text=f"{label} — {n_reached:,}/{n_total:,} users reached "
                 f"({n_reached/max(n_total,1)*100:.0f}%)",
            font=dict(size=11, color=TEXT_SEC), x=0,
        ),
        xaxis_title="Days since first event",
        yaxis=dict(title="% of users reached", ticksuffix="%",
                   gridcolor=_rgba(BRAND_BLUE, 0.08)),
        margin=dict(l=14, r=14, t=44, b=14),
    ))
    return fig


def build_activity_heatmap(_ev_raw=None):
    """Hour-of-day × day-of-week heatmap of event volume — reveals when users are active."""
    ev_raw_ = _ev_raw if _ev_raw is not None else events_raw
    ud = ev_raw_[["timestamp"]].copy()
    ud["hour"] = ud["timestamp"].dt.hour
    ud["dow"]  = ud["timestamp"].dt.dayofweek  # 0=Mon
    heat = (ud.groupby(["dow", "hour"])
              .size()
              .unstack(fill_value=0)
              .reindex(range(7), fill_value=0))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    z    = heat.values.tolist()
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"{h:02d}:00" for h in range(24)],
        y=days,
        colorscale=[
            [0.00, BG_SURFACE],
            [0.01, _rgba(BRAND_AMBER, 0.15)],
            [0.25, _rgba(BRAND_AMBER, 0.40)],
            [0.55, BRAND_AMBER],
            [0.85, "#EA6C00"],
            [1.00, "#C4376A"],
        ],
        hovertemplate="%{y}  %{x}<br>Events: %{z:,}<extra></extra>",
        showscale=False,
        xgap=2, ygap=2,
    ))
    fig.update_layout(**_cdefaults(
        height=260,
        xaxis=dict(tickfont=dict(size=9, color=TEXT_MUTED), tickangle=-45,
                   gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(tickfont=dict(size=10, color=TEXT_MUTED),
                   gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=42, r=14, t=24, b=44),
    ))
    return fig


def build_revenue_concentration():
    """Pareto / Lorenz curve — top X% of users account for Y% of revenue."""
    up_ = user_profiles[user_profiles["ltv"] > 0].sort_values("ltv", ascending=False)
    if up_.empty:
        return _empty_fig("No revenue data")
    cum_rev   = up_["ltv"].cumsum() / up_["ltv"].sum() * 100
    cum_users = np.arange(1, len(up_) + 1) / len(up_) * 100
    idx_20 = max(int(len(up_) * 0.20) - 1, 0)
    rev_20 = float(cum_rev.iloc[idx_20])

    fig = go.Figure()
    # Perfect equality diagonal
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], mode="lines", name="Perfect equality",
        line=dict(color=_rgba(TEXT_MUTED, 0.3), dash="dash", width=1),
    ))
    # Actual Lorenz curve
    fig.add_trace(go.Scatter(
        x=cum_users, y=cum_rev.values, mode="lines",
        name="Revenue concentration",
        line=dict(color=BRAND_BLUE, width=2.5),
        fill="tonexty", fillcolor=_rgba(BRAND_BLUE, 0.09),
        hovertemplate="Top %{x:.1f}% users → %{y:.1f}% revenue<extra></extra>",
    ))
    fig.add_annotation(
        x=28, y=rev_20 - 9,
        text=f"Top 20% = {rev_20:.0f}% revenue",
        showarrow=False, font=dict(size=10, color=BRAND_BLUE),
    )
    fig.update_layout(**_cdefaults(
        height=320,
        xaxis=dict(title="Cumulative % of users (high → low LTV)",
                   ticksuffix="%", gridcolor=_rgba(BRAND_BLUE, 0.08)),
        yaxis=dict(title="Cumulative % of revenue",
                   ticksuffix="%", gridcolor=_rgba(BRAND_BLUE, 0.08)),
        margin=dict(l=14, r=14, t=28, b=14),
    ))
    return fig


def build_session_duration(user_ids=None):
    """Histogram of session durations (minutes) — reveals engagement depth & patterns."""
    src = events_raw if user_ids is None else events_raw[events_raw["user_id"].isin(user_ids)]
    ud = src[["user_id", "timestamp"]].sort_values(["user_id", "timestamp"]).copy()
    ud["prev_ts"] = ud.groupby("user_id")["timestamp"].shift(1)
    ud["gap_min"] = (ud["timestamp"] - ud["prev_ts"]).dt.total_seconds() / 60
    ud["new_sess"] = ud["gap_min"].isna() | (ud["gap_min"] > 30)
    # Cumulative session counter per user
    ud["sess_id"] = ud.groupby("user_id")["new_sess"].cumsum()
    sess = ud.groupby(["user_id", "sess_id"])["timestamp"].agg(["min", "max"])
    sess["dur_min"] = (sess["max"] - sess["min"]).dt.total_seconds() / 60
    dur = sess.loc[sess["dur_min"] > 0, "dur_min"].clip(upper=120)
    if dur.empty:
        return _empty_fig("No session data")

    median_dur = float(dur.median())
    fig = go.Figure(go.Histogram(
        x=dur, nbinsx=40,
        marker_color=_rgba(BRAND_PURPLE, 0.72),
        marker_line_color=_rgba(BRAND_PURPLE, 0.25),
        marker_line_width=0.5,
        hovertemplate="Duration: %{x:.0f}m<br>Sessions: %{y:,}<extra></extra>",
        name="Sessions",
    ))
    fig.add_vline(x=median_dur, line_dash="dot", line_color=_rgba(BRAND_AMBER, 0.8))
    fig.add_annotation(x=median_dur + 2, y=0, yref="paper", ay=0, yanchor="bottom",
                       text=f"Median: {median_dur:.0f}m",
                       showarrow=False, font=dict(size=9, color=BRAND_AMBER))
    fig.update_layout(**_cdefaults(
        height=300,
        xaxis=dict(title="Session Duration (minutes)",
                   gridcolor=_rgba(BRAND_BLUE, 0.08)),
        yaxis=dict(title="Sessions", gridcolor=_rgba(BRAND_BLUE, 0.08)),
        margin=dict(l=14, r=14, t=28, b=14),
    ))
    return fig


def build_activation_health():
    """Activation rate by signup cohort — shows if product improves over time.
    Answers: Are users getting enough value? (Did they complete onboarding?)"""
    up_ = user_profiles.copy()
    up_["join_month"] = up_["first_event"].dt.to_period("M").astype(str)
    month_counts = up_["join_month"].value_counts()
    valid = sorted(month_counts[month_counts >= 20].index.tolist())[-10:]
    if not valid:
        return _empty_fig("Not enough cohort data")

    rows = []
    for m in valid:
        grp = up_[up_["join_month"] == m]
        n = len(grp)
        rows.append({
            "month": m,
            "ob_rate": round(grp["completed_onboarding"].mean() * 100, 1),
            "conv_rate": round(grp["converted"].mean() * 100, 1),
            "n": n,
        })
    df = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["month"], y=df["ob_rate"], name="Onboarding Completion %",
        marker_color=_rgba(BRAND_BLUE, 0.72),
        hovertemplate="%{x}<br>Onboarding: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["month"], y=df["conv_rate"], name="Conversion %",
        mode="lines+markers", line=dict(color=BRAND_TEAL, width=2),
        marker=dict(size=7, color=BRAND_TEAL),
        hovertemplate="%{x}<br>Conversion: %{y:.1f}%<extra></extra>",
        yaxis="y2",
    ))
    fig.update_layout(
        height=320, barmode="group",
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PAGE,
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT_SEC, size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SEC, size=9),
                    orientation="h", y=-0.22),
        margin=dict(l=14, r=60, t=28, b=14),
        xaxis=dict(gridcolor=_rgba(BRAND_BLUE, 0.08), tickfont=dict(color=TEXT_SEC, size=9),
                   tickangle=-30),
        yaxis=dict(title="Onboarding %", ticksuffix="%",
                   gridcolor=_rgba(BRAND_BLUE, 0.08), tickfont=dict(color=TEXT_SEC, size=10)),
        yaxis2=dict(title="Conversion %", ticksuffix="%", overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)", tickfont=dict(color=BRAND_TEAL, size=10)),
    )
    return fig


def build_stickiness_trend(_dau=None):
    """DAU/MAU (stickiness) ratio over time — answers: Are users staying?"""
    dau_ = _dau if _dau is not None else dau_mau_df
    df = dau_.copy()
    if df.empty:
        return _empty_fig("No DAU/MAU data")
    # 7-day rolling average for smoothing
    df = df.sort_values("date")
    df["ratio_7ma"] = df["ratio"].rolling(7, min_periods=1).mean()
    df["mau_7ma"]   = df["mau"].rolling(7, min_periods=1).mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df["date"], y=(df["ratio_7ma"] * 100).round(1),
        name="Stickiness (DAU/MAU %) — 7d MA",
        mode="lines", line=dict(color=BRAND_TEAL, width=2.5),
        fill="tozeroy", fillcolor=_rgba(BRAND_TEAL, 0.09),
        hovertemplate="%{x|%b %d}<br>Stickiness: %{y:.1f}%<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["mau_7ma"].round(0),
        name="MAU — 7d MA",
        mode="lines", line=dict(color=_rgba(BRAND_BLUE, 0.5), width=1.5, dash="dot"),
        hovertemplate="%{x|%b %d}<br>MAU: %{y:,.0f}<extra></extra>",
    ), secondary_y=True)
    fig.update_layout(
        height=300,
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PAGE,
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT_SEC, size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SEC, size=9),
                    orientation="h", y=-0.22),
        margin=dict(l=14, r=60, t=28, b=14),
        xaxis=dict(gridcolor=_rgba(BRAND_BLUE, 0.08), tickfont=dict(color=TEXT_SEC, size=9)),
    )
    fig.update_yaxes(title_text="Stickiness %", ticksuffix="%",
                     gridcolor=_rgba(BRAND_BLUE, 0.08),
                     tickfont=dict(color=TEXT_SEC, size=10), secondary_y=False)
    fig.update_yaxes(title_text="MAU", gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(color=TEXT_SEC, size=10), secondary_y=True)
    return fig


# ══════════════════════════════════════════════════════════════
#  CHURN RISK DEEP-DIVE
# ══════════════════════════════════════════════════════════════

def _churn_risk_precompute():
    """Return at-risk stats used across churn deep-dive charts."""
    up = user_profiles.copy()
    churned  = up[up["churned"] == 1]
    active   = up[up["churned"] == 0]

    # Churn score for active users — use active-only medians so the comparison
    # is within the healthy population, not dragged down by churned users.
    med_ev  = active["total_events"].median()
    med_act = active["active_days"].median()
    med_fb  = active["feature_breadth"].median()
    a = active.copy()
    a["_score"] = (
        (a["total_events"]  < med_ev).astype(int) +
        (a["active_days"]   < med_act).astype(int) +
        (a["feature_breadth"] < med_fb).astype(int)
    )
    # High risk = 2+ factors (score >= 2 is more actionable than >= 1)
    at_risk = a[a["_score"] >= 2]

    # Churn signal stats
    never_onboarded_pct = (
        float((churned["completed_onboarding"] == 0).mean() * 100)
        if len(churned) else 0.0
    )
    post_paid_count = int((churned["converted"] == 1).sum()) if len(churned) else 0
    post_paid_pct   = (
        float((churned["converted"] == 1).mean() * 100)
        if len(churned) else 0.0
    )

    return {
        "at_risk_count":       len(at_risk),
        "active_count":        len(active),
        "churned_count":       len(churned),
        "never_onboarded_pct": round(never_onboarded_pct, 1),
        "post_paid_count":     post_paid_count,
        "post_paid_pct":       round(post_paid_pct, 1),
        "at_risk_df":          at_risk,
        "active_df":           active,
        "churned_df":          churned,
    }

_CHURN_RISK = _churn_risk_precompute()


def build_churn_inactivity_dist():
    """Histogram: days_since_last for churned vs active — shows inactivity window."""
    active  = _CHURN_RISK["active_df"]
    churned = _CHURN_RISK["churned_df"]
    thresh  = _CHURN_RISK["threshold_days"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=active["days_since_last"].clip(upper=90),
        name="Active users",
        nbinsx=30, opacity=0.70,
        marker_color=_rgba(COL_HI, 0.72),
        hovertemplate="~%{x:.0f} days inactive: %{y} active users<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=churned["days_since_last"].clip(upper=90),
        name="Churned users",
        nbinsx=30, opacity=0.70,
        marker_color=_rgba(COL_LO, 0.72),
        hovertemplate="~%{x:.0f} days inactive: %{y} churned users<extra></extra>",
    ))
    fig.add_vline(
        x=thresh,
        line=dict(color=BRAND_AMBER, width=2, dash="dash"),
        annotation_text=f"Risk zone >{thresh:.0f}d",
        annotation=dict(font_color=BRAND_AMBER, font_size=10,
                        bgcolor=_rgba(BRAND_AMBER, 0.12)),
    )
    fig.update_layout(**_cdefaults(
        height=300, barmode="overlay",
        title=dict(text="Inactivity Distribution: Active vs Churned",
                   font=dict(color=TEXT_PRI, size=11), x=0, pad=dict(l=0)),
        xaxis=dict(title="Days since last event (capped at 90)",
                   tickfont=dict(color=TEXT_SEC, size=9),
                   gridcolor=_rgba(BRAND_BLUE, 0.08)),
        yaxis=dict(title="Users",
                   tickfont=dict(color=TEXT_SEC, size=9),
                   gridcolor=_rgba(BRAND_BLUE, 0.08)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SEC, size=9),
                    orientation="h", y=-0.22),
        margin=dict(l=14, r=14, t=40, b=14),
    ))
    return fig


def build_churn_behavior_comparison(_up=None):
    """Grouped bar: % of each group who completed key behaviours."""
    if _up is not None:
        active  = _up[_up["churned"] == 0]
        churned = _up[_up["churned"] == 1]
    else:
        active  = _CHURN_RISK["active_df"]
        churned = _CHURN_RISK["churned_df"]
    churn_np = churned[churned["converted"] == 0]
    churn_pp = churned[churned["converted"] == 1]

    metrics = [
        ("Completed Onboarding",  "completed_onboarding"),
        ("≥2 Features Used",       None),          # derived
        ("Viewed Pricing",         "n_view_pricing"),
        ("Any Payment",            "converted"),
        ("Invited Team",           "n_invite_team"),
        ("Contacted Support",      "n_contact_support"),
        ("Had Rage Clicks",        "n_rage_click"),
    ]

    def _pct(grp, col, derived=False):
        if len(grp) == 0:
            return 0.0
        if col == "n_rage_click":
            return (grp[col] > 0).mean() * 100
        if col is None:                     # feature_breadth >= 2
            return (grp["feature_breadth"] >= 2).mean() * 100
        return grp[col].clip(upper=1).mean() * 100

    labels  = [m[0] for m in metrics]
    cols    = [m[1] for m in metrics]
    g_act   = [_pct(active,   c) for c in cols]
    g_cnp   = [_pct(churn_np, c) for c in cols]
    g_cpp   = [_pct(churn_pp, c) for c in cols]

    fig = go.Figure()
    for vals, name, color in [
        (g_act,  "Active",              COL_HI),
        (g_cnp,  "Churned — Never Paid", COL_LO),
        (g_cpp,  "Churned — Post-Paid",  COL_MID),
    ]:
        fig.add_trace(go.Bar(
            y=labels, x=vals, name=name,
            orientation="h",
            marker_color=_rgba(color, 0.72),
            hovertemplate="%{y}: %{x:.1f}%<extra>" + name + "</extra>",
        ))
    fig.update_layout(**_cdefaults(
        height=340, barmode="group",
        title=dict(text="Behaviour Profile: Active vs Churned Segments",
                   font=dict(color=TEXT_PRI, size=11), x=0, pad=dict(l=0)),
        xaxis=dict(title="% of segment", ticksuffix="%",
                   tickfont=dict(color=TEXT_SEC, size=9),
                   gridcolor=_rgba(BRAND_BLUE, 0.08), range=[0, 105]),
        yaxis=dict(tickfont=dict(color=TEXT_SEC, size=10),
                   autorange="reversed"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SEC, size=9),
                    orientation="h", y=-0.18, x=0),
        margin=dict(l=14, r=14, t=40, b=14),
    ))
    return fig


def build_churn_vs_clusters(cluster_user_map=None, show_cluster="all"):
    """Bar: churn rate % per cluster. Uses the last-run clustering."""
    cum = cluster_user_map or _cluster_user_map
    if not cum:
        return _empty_fig("Run clustering first (Section 5)")

    rows = []
    for cname, uids in cum.items():
        if show_cluster != "all" and cname != show_cluster:
            continue
        grp = user_profiles[user_profiles.index.isin(uids)]
        if grp.empty:
            continue
        cr = grp["churned"].mean() * 100
        n_ch = grp["churned"].sum()
        n_ac = len(grp) - n_ch
        rows.append({"cluster": cname, "churn_pct": round(cr, 1),
                     "n_churned": int(n_ch), "n_active": int(n_ac),
                     "total": len(grp),
                     "avg_ltv": round(grp["ltv"].mean(), 0)})
    if not rows:
        return _empty_fig("No cluster data")
    df = pd.DataFrame(rows).sort_values("churn_pct", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["cluster"], y=df["n_active"],
        name="Active", marker_color=_rgba(COL_HI, 0.72),
        hovertemplate="%{x}<br>Active: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df["cluster"], y=df["n_churned"],
        name="Churned", marker_color=_rgba(COL_LO, 0.72),
        hovertemplate="%{x}<br>Churned: %{y:,}<br>Rate: %{customdata:.1f}%<extra></extra>",
        customdata=df["churn_pct"],
    ))
    # Churn % annotation above each bar
    for _, r in df.iterrows():
        fig.add_annotation(
            x=r["cluster"], y=r["total"] + max(df["total"].max() * 0.03, 2),
            text=f"{r['churn_pct']:.1f}%",
            showarrow=False,
            font=dict(color=COL_MID, size=10, family="Inter, system-ui, sans-serif"),
        )
    fig.update_layout(**_cdefaults(
        height=320, barmode="stack",
        title=dict(text="Churn Rate by User Cluster",
                   font=dict(color=TEXT_PRI, size=11), x=0, pad=dict(l=0)),
        xaxis=dict(tickfont=dict(color=TEXT_SEC, size=10)),
        yaxis=dict(title="Users", tickfont=dict(color=TEXT_SEC, size=9),
                   gridcolor=_rgba(BRAND_BLUE, 0.08)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SEC, size=9),
                    orientation="h", y=-0.18, x=0),
        margin=dict(l=14, r=14, t=40, b=14),
    ))
    return fig


def _intervention_label(row):
    """Rule-based recommended action for an at-risk active user."""
    if row.get("completed_onboarding", 0) == 0:
        return "📚 Send Onboarding Reminder"
    if row.get("n_view_pricing", 0) == 0:
        return "💰 Trigger Pricing Nudge"
    if row.get("n_view_pricing", 0) > 0 and row.get("converted", 0) == 0:
        return "🎯 Offer Demo / Conversion CTA"
    if row.get("n_rage_click", 0) > 3:
        return "🐛 Flag UX Friction"
    if row.get("days_since_last", 0) > 14:
        return "📧 Send Re-engagement Email"
    return "👀 Monitor Closely"


# Populated by build_at_risk_table() — used by dropdown-filter callback.
_AT_RISK_RECORDS: list = []

# Dropdown options (hardcoded from _intervention_label possibilities)
_AT_RISK_RISK_OPTIONS = [
    {"label": "All risk levels", "value": "all"},
    {"label": "🔴 High",         "value": "🔴 High"},
    {"label": "🟡 Medium",       "value": "🟡 Medium"},
]
_AT_RISK_ACTION_OPTIONS = [
    {"label": "All actions",                    "value": "all"},
    {"label": "📚 Send Onboarding Reminder",    "value": "📚 Send Onboarding Reminder"},
    {"label": "💰 Trigger Pricing Nudge",       "value": "💰 Trigger Pricing Nudge"},
    {"label": "🎯 Offer Demo / Conversion CTA", "value": "🎯 Offer Demo / Conversion CTA"},
    {"label": "🐛 Flag UX Friction",            "value": "🐛 Flag UX Friction"},
    {"label": "📧 Send Re-engagement Email",    "value": "📧 Send Re-engagement Email"},
    {"label": "👀 Monitor Closely",             "value": "👀 Monitor Closely"},
]


def build_at_risk_table():
    """DataTable of active users sorted by churn risk score then inactivity.
    Uses the already-scored _CHURN_RISK['at_risk_df']. Records stored in
    _AT_RISK_RECORDS for use by the dropdown-filter callback."""
    global _AT_RISK_RECORDS
    at_risk = (_CHURN_RISK["at_risk_df"]
               .sort_values(["_score", "days_since_last"], ascending=[False, False])
               .head(200)
               .copy())

    at_risk["Risk"]               = at_risk["_score"].map({3: "🔴 High", 2: "🟡 Medium"})
    at_risk["User ID"]            = at_risk.index
    at_risk["Last Seen (days)"]   = at_risk["days_since_last"].round(0).astype(int)
    at_risk["LTV ($)"]            = at_risk["ltv"].round(0).astype(int)
    at_risk["Sessions"]           = at_risk["session_count"].astype(int)
    at_risk["Features Used"]      = at_risk["feature_breadth"].astype(int)
    at_risk["Onboarding"]         = at_risk["completed_onboarding"].map({1: "✅", 0: "✗"})
    at_risk["Rage Clicks"]        = at_risk["n_rage_click"].astype(int)
    at_risk["Recommended Action"] = at_risk.apply(_intervention_label, axis=1)

    show_cols = ["Risk", "User ID", "Last Seen (days)", "LTV ($)",
                 "Sessions", "Features Used", "Onboarding", "Rage Clicks",
                 "Recommended Action"]
    tbl_df = at_risk[show_cols].reset_index(drop=True)
    _AT_RISK_RECORDS = tbl_df.to_dict("records")

    return DataTable(
        id="at-risk-table",
        data=_AT_RISK_RECORDS,
        columns=[{"name": c, "id": c} for c in show_cols],
        page_size=12,
        sort_action="native",
        row_selectable="single",
        style_table={"overflowX": "auto", "borderRadius": "8px",
                     "border": f"1px solid {BORDER}"},
        style_header={"backgroundColor": BG_SURFACE, "color": TEXT_PRI,
                      "fontWeight": "600", "fontSize": "11px",
                      "border": f"1px solid {BORDER}",
                      "fontFamily": "Inter, system-ui, sans-serif"},
        style_data={"backgroundColor": BG_CARD, "color": TEXT_SEC,
                    "fontSize": "11px",
                    "border": f"1px solid {_rgba(BRAND_BLUE, 0.08)}",
                    "fontFamily": "Inter, system-ui, sans-serif"},
        style_cell_conditional=[
            {"if": {"column_id": "Recommended Action"},
             "color": TEXT_PRI, "fontStyle": "normal", "minWidth": "200px"},
            {"if": {"column_id": "User ID"},
             "color": BRAND_BLUE, "fontWeight": "600", "cursor": "pointer"},
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": BG_SURFACE},
            {"if": {"filter_query": '{Risk} = "🔴 High"'},
             "color": COL_LO, "fontWeight": "600"},
            {"if": {"filter_query": '{Risk} = "🟡 Medium"'},
             "color": BRAND_AMBER},
        ],
        tooltip_header={
            "User ID": "Click any row to load that user's journey in the Individual Journey section",
            "Recommended Action": "Rule-based suggested next action for each at-risk user",
        },
        tooltip_delay=400,
        tooltip_duration=None,
    )


# ══════════════════════════════════════════════════════════════
#  CHURN PREDICTOR TABLE — first-N-event signals
# ══════════════════════════════════════════════════════════════

def _churn_predictor_data(within_first_n=5):
    """Compute per-event churn predictor stats for the first N actions per user.
    Returns list-of-dicts (DataTable records), sorted by Gap descending."""
    ev = events_raw.sort_values(["user_id", "timestamp"]).copy()
    ev["_rank"] = ev.groupby("user_id").cumcount() + 1
    first_n = ev[ev["_rank"] <= within_first_n].copy()

    cs = user_profiles[["churned"]]
    first_n = first_n.merge(cs, left_on="user_id", right_index=True)

    churned_up = user_profiles[user_profiles["churned"] == 1]
    active_up  = user_profiles[user_profiles["churned"] == 0]
    n_ch = max(len(churned_up), 1)
    n_ac = max(len(active_up),  1)

    rows = []
    for evt in sorted(first_n["event_name"].unique()):
        did_ch = first_n[(first_n["churned"] == 1) & (first_n["event_name"] == evt)]["user_id"].nunique()
        did_ac = first_n[(first_n["churned"] == 0) & (first_n["event_name"] == evt)]["user_id"].nunique()
        pct_ch = round(did_ch / n_ch * 100, 1)
        pct_ac = round(did_ac / n_ac * 100, 1)
        gap    = round(pct_ac - pct_ch, 1)
        if abs(gap) < 1:
            signal = "Neutral"
        elif gap > 0:
            signal = "✅ Retention"
        else:
            signal = "🔴 Churn risk"
        rows.append({
            "Event":                     evt,
            "Signal":                    signal,
            "Active %":                  pct_ac,
            "Churned %":                 pct_ch,
            "Gap (Active − Churned) pp": gap,
        })

    df = pd.DataFrame(rows)
    # Drop events with trivially tiny gap (< 2 pp) — not actionable
    df = df[df["Gap (Active − Churned) pp"].abs() >= 2].copy()
    return df.sort_values("Gap (Active − Churned) pp", ascending=False).to_dict("records")


def build_churn_predictor_table():
    """DataTable: early-action churn predictors (data updated by callback on N change)."""
    cols = [
        {"name": "Event",                     "id": "Event"},
        {"name": "Signal",                    "id": "Signal"},
        {"name": "Active %",                  "id": "Active %"},
        {"name": "Churned %",                 "id": "Churned %"},
        {"name": "Gap (Active − Churned) pp", "id": "Gap (Active − Churned) pp"},
    ]
    return DataTable(
        id="churn-predictor-table",
        data=_churn_predictor_data(5),
        columns=cols,
        sort_action="native",
        style_table={"overflowX": "auto", "borderRadius": "8px",
                     "border": f"1px solid {BORDER}"},
        style_header={"backgroundColor": BG_SURFACE, "color": TEXT_PRI,
                      "fontWeight": "600", "fontSize": "11px",
                      "border": f"1px solid {BORDER}",
                      "fontFamily": "Inter, system-ui, sans-serif"},
        style_data={"backgroundColor": BG_CARD, "color": TEXT_SEC,
                    "fontSize": "12px",
                    "border": f"1px solid {_rgba(BRAND_BLUE, 0.08)}",
                    "fontFamily": "Inter, system-ui, sans-serif"},
        style_cell_conditional=[
            {"if": {"column_id": "Event"},   "fontWeight": "600", "color": TEXT_PRI},
            {"if": {"column_id": "Active %"},  "textAlign": "right"},
            {"if": {"column_id": "Churned %"}, "textAlign": "right"},
            {"if": {"column_id": "Gap (Active − Churned) pp"},
             "fontWeight": "700", "textAlign": "right"},
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": BG_SURFACE},
            {"if": {"filter_query": '{Signal} contains "Retention"'},
             "color": COL_HI, "fontWeight": "600"},
            {"if": {"filter_query": '{Signal} contains "Churn"'},
             "color": COL_LO, "fontWeight": "600"},
            {"if": {"column_id": "Gap (Active − Churned) pp",
                    "filter_query": "{Gap (Active − Churned) pp} > 0"},
             "color": COL_HI, "fontWeight": "700"},
            {"if": {"column_id": "Gap (Active − Churned) pp",
                    "filter_query": "{Gap (Active − Churned) pp} < 0"},
             "color": COL_LO, "fontWeight": "700"},
        ],
        tooltip_header={
            "Gap (Active − Churned) pp": (
                "Percentage-point difference. Positive = active users did this more (retention signal). "
                "Negative = churned users did this more (churn signal)."
            ),
            "Signal": "Retention = retained users did this early. Churn risk = churned users did this early.",
        },
        tooltip_delay=400,
        tooltip_duration=None,
    )


# ══════════════════════════════════════════════════════════════
#  CHURN PREDICTION MODEL — sklearn + SHAP feature importance
# ══════════════════════════════════════════════════════════════

def _build_churn_model():
    """Train a RandomForest classifier on user_profiles to predict churn,
    then compute SHAP values.  Returns (shap_fig, model_accuracy, feature_table_records)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import shap

    up = user_profiles.copy()

    # ── Engineer ratio-based features (more actionable than raw counts)
    te = up["total_events"].clip(lower=1)
    sc = up["session_count"].clip(lower=1)
    ad = up["active_days"].clip(lower=1)
    up["rage_per_100_events"]    = up["n_rage_click"]      / te * 100
    up["errors_per_100_events"]  = up["n_error"]           / te * 100
    up["support_per_session"]    = up["n_contact_support"]  / sc
    up["pricing_views_per_sess"] = up["n_view_pricing"]     / sc
    up["events_per_session"]     = up["total_events"]       / sc
    up["events_per_active_day"]  = up["total_events"]       / ad
    up["failed_charge_rate"]     = up["n_failed_charges"]   / (up["n_successful_charges"] + up["n_failed_charges"]).clip(lower=1)

    # Feature columns — exclude target, dates, non-numeric, and raw activity-volume
    # features that trivially separate churned (who have near-zero activity) from
    # active users. Excluding these forces the model to learn behavioural quality
    # signals that are more actionable for PM decisions.
    exclude = {"churned", "first_event", "last_event", "first_charge_date",
               "last_charge_date", "sub_status",
               # Raw volume features — trivially correlated with churn definition
               "total_events", "active_days", "session_count", "unique_events",
               "days_since_last",
               # Raw event category counts — also pure volume proxies
               "cat_session", "cat_feature_usage", "cat_engagement",
               "cat_onboarding", "cat_conversion", "cat_navigation",
               "cat_support", "cat_social", "cat_friction", "cat_monetization",
               "revenue_events",
               # Raw action counts — confusing because active users have
               # more of everything; ratio features replace them
               "n_rage_click", "n_error", "n_contact_support",
               "n_view_pricing", "n_invite_team",
               "n_onboarding_step1", "n_onboarding_step2",
               "n_start_checkout", "n_plan_upgrade", "n_payment_failed",
               "n_successful_charges", "n_failed_charges"}
    feat_cols = [c for c in up.columns
                 if c not in exclude and up[c].dtype in ("float64", "int64", "float32", "int32")]
    X = up[feat_cols].fillna(0)
    y = up["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    # SHAP — use a subsample for speed
    sample_idx = np.random.RandomState(42).choice(len(X_test), size=min(500, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx]
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)   # [class0, class1]
    # shap_values for class 1 (churned) — handle both old (list) and new (3-D) formats
    if isinstance(shap_values, list):
        sv = shap_values[1]                       # old format: list of 2D arrays
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]                  # new format: (n, features, classes)
    else:
        sv = shap_values                           # binary single-output

    # Mean |SHAP| per feature — global importance (ensure 1-D)
    mean_abs_shap = np.asarray(np.abs(sv).mean(axis=0)).ravel()

    # Direction: correlation between feature value and its SHAP value
    # Positive corr → higher feature value pushes toward churn (class 1)
    # Negative corr → higher feature value pushes toward retention
    sv_arr = np.asarray(sv)
    x_arr  = np.asarray(X_sample)
    dir_corr = np.zeros(len(feat_cols))
    for fi in range(len(feat_cols)):
        fv = x_arr[:, fi]
        if np.std(fv) > 1e-9 and np.std(sv_arr[:, fi]) > 1e-9:
            dir_corr[fi] = float(np.corrcoef(fv, sv_arr[:, fi])[0, 1])

    order = np.argsort(mean_abs_shap)[::-1].ravel()
    top_n = min(15, len(feat_cols))
    top_idx = order[:top_n].tolist()     # plain Python ints

    feat_names  = [feat_cols[i] for i in top_idx]
    importances = [float(mean_abs_shap[i]) for i in top_idx]
    directions  = [float(dir_corr[i]) for i in top_idx]

    # ── Bar chart: SHAP importance with direction coloring
    colors = [COL_LO if d > 0 else COL_HI for d in directions]
    fig = go.Figure(go.Bar(
        y=feat_names[::-1],
        x=importances[::-1],
        orientation="h",
        marker_color=colors[::-1],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "SHAP importance: %{x:.4f}<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(**_cdefaults(
        height=max(320, top_n * 28),
        xaxis=dict(title=dict(text="Mean |SHAP value|", font=dict(size=11))),
        yaxis=dict(tickfont=dict(size=11)),
        margin=dict(l=160, r=14, t=28, b=36),
    ))
    # Add annotation legend
    fig.add_annotation(
        text=f"<span style='color:{COL_LO}'>Red = higher value → churn</span>  ·  "
             f"<span style='color:{COL_HI}'>Green = higher value → retention</span>  ·  "
             f"Model AUC = {auc:.3f}",
        xref="paper", yref="paper", x=0.5, y=1.02,
        showarrow=False, font=dict(size=10, color=TEXT_MUTED),
    )

    # ── Table data
    records = []
    for i, idx in enumerate(top_idx):
        fname = feat_cols[idx]
        imp   = float(mean_abs_shap[idx])
        d     = float(dir_corr[idx])
        if d > 0.05:
            signal = "🔴 Higher → churn"
        elif d < -0.05:
            signal = "✅ Higher → retention"
        else:
            signal = "— Mixed"
        # Interpretation: mean value for churned vs active
        cv = float(up.loc[up["churned"] == 1, fname].mean())
        av = float(up.loc[up["churned"] == 0, fname].mean())
        records.append({
            "Rank": i + 1,
            "Feature": fname,
            "SHAP Importance": round(imp, 4),
            "Direction": signal,
            "Avg (Active)": round(av, 2),
            "Avg (Churned)": round(cv, 2),
        })

    return fig, auc, records, clf, feat_cols, sv, X_sample


def build_shap_feature_table(records):
    """DataTable for SHAP feature importance breakdown."""
    cols = [
        {"name": "Rank",            "id": "Rank"},
        {"name": "Feature",         "id": "Feature"},
        {"name": "SHAP Importance", "id": "SHAP Importance"},
        {"name": "Direction",       "id": "Direction"},
        {"name": "Avg (Active)",    "id": "Avg (Active)"},
        {"name": "Avg (Churned)",   "id": "Avg (Churned)"},
    ]
    return DataTable(
        id="shap-feature-table",
        data=records,
        columns=cols,
        sort_action="native",
        page_size=15,
        style_table={"overflowX": "auto", "borderRadius": "8px",
                     "border": f"1px solid {BORDER}"},
        style_header={"backgroundColor": BG_SURFACE, "color": TEXT_PRI,
                      "fontWeight": "600", "fontSize": "11px",
                      "border": f"1px solid {BORDER}",
                      "fontFamily": "Inter, system-ui, sans-serif"},
        style_data={"backgroundColor": BG_CARD, "color": TEXT_SEC,
                    "fontSize": "11px",
                    "border": f"1px solid {_rgba(BRAND_BLUE, 0.08)}",
                    "fontFamily": "Inter, system-ui, sans-serif"},
        style_cell_conditional=[
            {"if": {"column_id": "Feature"}, "fontWeight": "600", "color": TEXT_PRI},
            {"if": {"column_id": "SHAP Importance"}, "fontWeight": "700", "textAlign": "right"},
            {"if": {"column_id": "Avg (Active)"},  "textAlign": "right"},
            {"if": {"column_id": "Avg (Churned)"}, "textAlign": "right"},
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": BG_SURFACE},
            {"if": {"filter_query": '{Direction} contains "churn"'},
             "color": COL_LO, "fontWeight": "600"},
            {"if": {"filter_query": '{Direction} contains "retention"'},
             "color": COL_HI, "fontWeight": "600"},
        ],
    )


# ══════════════════════════════════════════════════════════════
#  CHURN FEATURE 1 — PER-USER CHURN PROBABILITY
# ══════════════════════════════════════════════════════════════

def _compute_churn_probabilities(clf, feat_cols):
    """Score every user with churn probability using the trained RF model."""
    up = user_profiles.copy()
    # Engineer same ratio features as in training
    te = up["total_events"].clip(lower=1)
    sc = up["session_count"].clip(lower=1)
    ad = up["active_days"].clip(lower=1)
    up["rage_per_100_events"]    = up["n_rage_click"]       / te * 100
    up["errors_per_100_events"]  = up["n_error"]            / te * 100
    up["support_per_session"]    = up["n_contact_support"]   / sc
    up["pricing_views_per_sess"] = up["n_view_pricing"]      / sc
    up["events_per_session"]     = up["total_events"]        / sc
    up["events_per_active_day"]  = up["total_events"]        / ad
    up["failed_charge_rate"]     = up["n_failed_charges"] / (up["n_successful_charges"] + up["n_failed_charges"]).clip(lower=1)

    X_all = up[feat_cols].fillna(0)
    probs = clf.predict_proba(X_all)[:, 1]
    return pd.Series(probs, index=up.index, name="churn_prob")


def build_churn_prob_histogram(probs):
    """Histogram of churn probabilities for active users."""
    active_probs = probs[user_profiles["churned"] == 0]
    fig = go.Figure(go.Histogram(
        x=active_probs * 100,
        nbinsx=20,
        marker_color=_rgba(BRAND_BLUE, 0.72),
        hovertemplate="Churn Prob: %{x:.0f}%<br>Users: %{y:,}<extra></extra>",
    ))
    # Add risk zone annotations
    fig.add_vrect(x0=70, x1=100, fillcolor=_rgba(COL_LO, 0.08),
                  line_width=0, annotation_text="High Risk", annotation_position="top")
    fig.add_vrect(x0=40, x1=70, fillcolor=_rgba(COL_MID, 0.06),
                  line_width=0, annotation_text="Medium", annotation_position="top")
    fig.update_layout(**_cdefaults(
        height=260,
        xaxis=dict(title="Churn Probability (%)", range=[0, 100]),
        yaxis=dict(title="Active Users"),
        margin=dict(l=14, r=14, t=30, b=36),
    ))
    return fig


# ══════════════════════════════════════════════════════════════
#  CHURN FEATURE 2 — PERSONALISED NEXT-BEST-ACTION
# ══════════════════════════════════════════════════════════════

# Human-readable action map for SHAP features
_FEATURE_ACTIONS = {
    "feature_breadth":        "Guide user to explore more features (in-app tour, feature spotlight)",
    "completed_onboarding":   "Send onboarding reminder / simplify onboarding flow",
    "events_per_session":     "Improve session depth (reduce exit points, add nudges)",
    "events_per_active_day":  "Encourage daily micro-engagement (streak rewards, daily tips)",
    "pricing_views_per_sess": "Show pricing page / trigger upgrade prompt",
    "events_per_day":         "Investigate binge-then-leave pattern (pace engagement)",
    "rage_per_100_events":    "Fix UX friction points causing rage clicks",
    "errors_per_100_events":  "Prioritise bug fixes for this user's workflow",
    "support_per_session":    "Proactive support outreach / knowledge base prompt",
    "checkout_conversion":    "Simplify checkout / offer incentive to complete purchase",
    "n_payment_methods":      "Prompt to add payment method",
    "plan_amount_usd":        "Consider downgrade option to retain user",
    "avg_charge_amount":      "Review pricing tier fit",
    "converted":              "Trigger conversion CTA / trial extension",
    "failed_charge_rate":     "Alert user about failed payment / update billing",
}


def _compute_next_best_actions(clf, feat_cols, shap_sv, shap_X_sample):
    """For each at-risk user, find which feature improvement would most reduce churn prob.
    Uses per-feature SHAP contribution direction: the feature with highest positive SHAP
    (pushing toward churn) is the best candidate for intervention."""
    at_risk = _CHURN_RISK.get("at_risk_df")
    if at_risk is None or at_risk.empty:
        return {}

    up = user_profiles.copy()
    te = up["total_events"].clip(lower=1)
    sc = up["session_count"].clip(lower=1)
    ad = up["active_days"].clip(lower=1)
    up["rage_per_100_events"]    = up["n_rage_click"]       / te * 100
    up["errors_per_100_events"]  = up["n_error"]            / te * 100
    up["support_per_session"]    = up["n_contact_support"]   / sc
    up["pricing_views_per_sess"] = up["n_view_pricing"]      / sc
    up["events_per_session"]     = up["total_events"]        / sc
    up["events_per_active_day"]  = up["total_events"]        / ad
    up["failed_charge_rate"]     = up["n_failed_charges"] / (up["n_successful_charges"] + up["n_failed_charges"]).clip(lower=1)

    # Mean SHAP direction per feature (from training sample)
    sv_arr = np.asarray(shap_sv)
    x_arr = np.asarray(shap_X_sample)
    # For each feature, compute: does higher value → churn or retention?
    feat_direction = {}  # feature → +1 (higher=churn) or -1 (higher=retention)
    for fi, fname in enumerate(feat_cols):
        fv = x_arr[:, fi]
        if np.std(fv) > 1e-9 and np.std(sv_arr[:, fi]) > 1e-9:
            corr = float(np.corrcoef(fv, sv_arr[:, fi])[0, 1])
            feat_direction[fname] = 1 if corr > 0 else -1
        else:
            feat_direction[fname] = 0

    # For at-risk users: find features where user is below/above active median
    active = up[up["churned"] == 0]
    active_medians = {fc: float(active[fc].median()) for fc in feat_cols if fc in active.columns}

    actions = {}  # user_id → (action_text, feature, gap_description)
    at_risk_ids = at_risk.index.tolist()[:200]

    for uid in at_risk_ids:
        if uid not in up.index:
            continue
        user_row = up.loc[uid]
        best_feat = None
        best_impact = 0

        for fname in feat_cols:
            if fname not in active_medians or feat_direction.get(fname, 0) == 0:
                continue
            user_val = float(user_row.get(fname, 0))
            med_val = active_medians[fname]
            direction = feat_direction[fname]

            # Feature pushes toward churn when high AND user is above median → reduce it
            # Feature pushes toward retention when high AND user is below median → increase it
            if direction == -1 and user_val < med_val:
                # User is below the "retention" threshold — biggest gap = most impactful
                gap = (med_val - user_val) / max(abs(med_val), 0.001)
                if gap > best_impact:
                    best_impact = gap
                    best_feat = fname
            elif direction == 1 and user_val > med_val:
                gap = (user_val - med_val) / max(abs(med_val), 0.001)
                if gap > best_impact:
                    best_impact = gap
                    best_feat = fname

        if best_feat:
            action = _FEATURE_ACTIONS.get(best_feat, f"Improve {best_feat}")
            actions[uid] = action
        else:
            actions[uid] = "Monitor closely — no single dominant risk factor"

    return actions


# ══════════════════════════════════════════════════════════════
#  CHURN FEATURE 3 — TIME-TO-CHURN SURVIVAL ANALYSIS
# ══════════════════════════════════════════════════════════════

def build_survival_curve():
    """Kaplan-Meier survival curve: P(user still active after N days from signup)."""
    up = user_profiles.copy()
    up["tenure_days"] = (up["last_event"] - up["first_event"]).dt.days.clip(lower=0)
    # Event = 1 means churned (event occurred), 0 = still active (censored)
    up["event"] = up["churned"].astype(int)

    max_days = int(up["tenure_days"].quantile(0.95))
    if max_days < 1:
        max_days = 60
    timeline = np.arange(0, max_days + 1)

    # Kaplan-Meier estimator
    n_total = len(up)
    survival = np.ones(len(timeline))
    at_risk = n_total

    for i, t in enumerate(timeline):
        if t == 0:
            continue
        # Events at time t: users who churned with tenure_days == t
        events_at_t = int(((up["tenure_days"] == t) & (up["event"] == 1)).sum())
        # Censored at time t: active users with tenure_days == t (still alive, observation ends)
        censored_at_t = int(((up["tenure_days"] == t) & (up["event"] == 0)).sum())

        if at_risk > 0 and events_at_t > 0:
            survival[i] = survival[i - 1] * (1 - events_at_t / at_risk)
        else:
            survival[i] = survival[i - 1]

        at_risk -= (events_at_t + censored_at_t)
        if at_risk <= 0:
            survival[i:] = survival[i]
            break

    # Find median survival time (where survival drops below 50%)
    median_idx = np.where(survival <= 0.5)[0]
    median_days = int(timeline[median_idx[0]]) if len(median_idx) > 0 else None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline, y=survival * 100,
        mode="lines", fill="tozeroy",
        line=dict(color=BRAND_BLUE, width=2),
        fillcolor=_rgba(BRAND_BLUE, 0.08),
        hovertemplate="Day %{x}<br>Survival: %{y:.1f}%<extra></extra>",
    ))
    if median_days is not None:
        fig.add_hline(y=50, line_dash="dot", line_color=COL_MID,
                      annotation_text=f"Median: {median_days}d", annotation_position="right")
        fig.add_vline(x=median_days, line_dash="dot", line_color=COL_MID)

    # Add danger zone
    fig.add_vrect(x0=0, x1=7, fillcolor=_rgba(COL_LO, 0.06), line_width=0,
                  annotation_text="Critical first week", annotation_position="top left")

    fig.update_layout(**_cdefaults(
        height=300,
        xaxis=dict(title="Days Since First Event", range=[0, max_days]),
        yaxis=dict(title="% Users Still Active", range=[0, 105]),
        margin=dict(l=14, r=14, t=30, b=36),
    ))
    return fig, median_days


# ══════════════════════════════════════════════════════════════
#  CHURN FEATURE 4 — CHURN COHORT COMPARISON
# ══════════════════════════════════════════════════════════════

def build_churn_cohort_comparison():
    """Compare recent churners (last 14 days of data) vs older churners."""
    up = user_profiles.copy()
    churned = up[up["churned"] == 1].copy()
    active  = up[up["churned"] == 0].copy()
    if churned.empty:
        return _empty_fig("No churned users"), []

    data_end = up["last_event"].max()
    cutoff   = data_end - pd.Timedelta(days=14)
    recent   = churned[churned["last_event"] >= cutoff]
    older    = churned[churned["last_event"] <  cutoff]

    metrics = ["total_events", "session_count", "active_days", "feature_breadth",
               "n_rage_click", "ltv", "completed_onboarding", "checkout_conversion"]
    labels  = ["Total Events", "Sessions", "Active Days", "Feature Breadth",
               "Rage Clicks", "LTV ($)", "Onboarded %", "Checkout Conv %"]

    rows = []
    for m, label in zip(metrics, labels):
        mult = 100 if "%" in label else 1
        rv = float(recent[m].mean() * mult) if not recent.empty else 0
        ov = float(older[m].mean() * mult)  if not older.empty else 0
        av = float(active[m].mean() * mult) if not active.empty else 0
        rows.append({"Metric": label, "Recent Churners": round(rv, 1),
                     "Older Churners": round(ov, 1), "Active Users": round(av, 1)})

    fig = go.Figure()
    for col, color, name in [
        ("Active Users", COL_HI, "Active"),
        ("Recent Churners", BRAND_AMBER, f"Recent Churners ({len(recent):,})"),
        ("Older Churners", COL_LO, f"Older Churners ({len(older):,})"),
    ]:
        vals = [r[col] for r in rows]
        fig.add_trace(go.Bar(
            y=[r["Metric"] for r in rows], x=vals, name=name,
            orientation="h", marker_color=_rgba(color, 0.72),
            hovertemplate=f"<b>%{{y}}</b><br>{name}: %{{x:.1f}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        **_cdefaults(
            height=max(300, len(metrics) * 38),
            xaxis=dict(title="Average Value"),
            yaxis=dict(tickfont=dict(size=10)),
            legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
            margin=dict(l=14, r=14, t=50, b=36),
        ),
    )
    return fig, rows


# ══════════════════════════════════════════════════════════════
#  CHURN FEATURE 5 — WIN-BACK CANDIDATES
# ══════════════════════════════════════════════════════════════

def _compute_winback_scores():
    """Score churned users by likelihood of win-back.
    Higher score = more recoverable.
    Factors: pre-churn engagement, recency, payment history, feature adoption."""
    up = user_profiles.copy()
    churned = up[up["churned"] == 1].copy()
    if churned.empty:
        return []

    # Normalise each factor to 0-25 range, total = 0-100
    def _norm(s, invert=False):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        n = (s - mn) / (mx - mn)
        return (1 - n) if invert else n

    # 1. Engagement (0-25): higher events+sessions = more engaged before leaving
    eng = (_norm(churned["total_events"]) * 0.5 + _norm(churned["session_count"]) * 0.5) * 25

    # 2. Recency (0-25): more recent churn = easier to win back
    rec = _norm(churned["days_since_last"], invert=True) * 25

    # 3. Monetisation (0-25): paid users are higher value win-backs
    mon = (_norm(churned["ltv"]) * 0.6 +
           churned["converted"].astype(float) * 0.4) * 25

    # 4. Feature adoption (0-25): users who explored features had found value
    feat = (_norm(churned["feature_breadth"]) * 0.5 +
            churned["completed_onboarding"].astype(float) * 0.5) * 25

    churned["winback_score"] = (eng + rec + mon + feat).round(1)
    churned["engagement_score"] = eng.round(1)
    churned["recency_score"]    = rec.round(1)
    churned["monetisation_score"] = mon.round(1)
    churned["adoption_score"]   = feat.round(1)

    # Classify
    def _tier(score):
        if score >= 60:
            return "🟢 High"
        if score >= 35:
            return "🟡 Medium"
        return "🔴 Low"

    top = churned.nlargest(100, "winback_score")
    records = []
    for uid, r in top.iterrows():
        records.append({
            "User ID": int(uid),
            "Win-back Score": float(r["winback_score"]),
            "Tier": _tier(r["winback_score"]),
            "Engagement": float(r["engagement_score"]),
            "Recency": float(r["recency_score"]),
            "Monetisation": float(r["monetisation_score"]),
            "Adoption": float(r["adoption_score"]),
            "LTV ($)": round(float(r["ltv"]), 0),
            "Days Since Last": int(r["days_since_last"]),
            "Sessions": int(r["session_count"]),
        })
    return records


def build_winback_table(records):
    cols = [
        {"name": c, "id": c} for c in
        ["Tier", "User ID", "Win-back Score", "LTV ($)", "Days Since Last",
         "Sessions", "Engagement", "Recency", "Monetisation", "Adoption"]
    ]
    return DataTable(
        id="winback-table", data=records, columns=cols,
        sort_action="native", page_size=15,
        style_table={"overflowX": "auto", "borderRadius": "8px",
                     "border": f"1px solid {BORDER}"},
        style_header={"backgroundColor": BG_SURFACE, "color": TEXT_PRI,
                      "fontWeight": "600", "fontSize": "11px",
                      "border": f"1px solid {BORDER}",
                      "fontFamily": "Inter, system-ui, sans-serif"},
        style_data={"backgroundColor": BG_CARD, "color": TEXT_SEC,
                    "fontSize": "11px",
                    "border": f"1px solid {_rgba(BRAND_BLUE, 0.08)}",
                    "fontFamily": "Inter, system-ui, sans-serif"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": BG_SURFACE},
            {"if": {"filter_query": '{Tier} contains "High"'}, "color": COL_HI, "fontWeight": "600"},
            {"if": {"filter_query": '{Tier} contains "Medium"'}, "color": COL_MID},
        ],
    )


# ══════════════════════════════════════════════════════════════
#  DASHBOARD FEATURE 6 — USER HEALTH SCORE (0-100)
# ══════════════════════════════════════════════════════════════

def _compute_health_scores(_up=None):
    """Composite health score (0-100) for each user.
    25 pts each: engagement, feature adoption, monetisation, recency."""
    up = (_up if _up is not None else user_profiles).copy()

    def _norm_clip(s):
        mn, mx = s.quantile(0.02), s.quantile(0.98)
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        return ((s - mn) / (mx - mn)).clip(0, 1)

    # Engagement (25 pts): events_per_day, session_count
    eng = (_norm_clip(up["events_per_day"]) * 0.5 +
           _norm_clip(up["session_count"]) * 0.5) * 25

    # Feature adoption (25 pts): feature_breadth, completed_onboarding
    adopt = (_norm_clip(up["feature_breadth"]) * 0.5 +
             up["completed_onboarding"].astype(float) * 0.5) * 25

    # Monetisation (25 pts): ltv, converted
    mon = (_norm_clip(up["ltv"]) * 0.6 +
           up["converted"].astype(float) * 0.4) * 25

    # Recency (25 pts): inverse of days_since_last (lower = healthier)
    rec_raw = up["days_since_last"].clip(lower=0)
    rec = (1 - _norm_clip(rec_raw)) * 25

    scores = (eng + adopt + mon + rec).round(1)
    return scores


def build_health_score_distribution(scores, _up=None):
    """Histogram of user health scores."""
    up_ = _up if _up is not None else user_profiles
    active_scores = scores[up_["churned"] == 0]
    churned_scores = scores[up_["churned"] == 1]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=active_scores, nbinsx=20, name="Active",
        marker_color=_rgba(COL_HI, 0.72),
        hovertemplate="Score: %{x:.0f}<br>Active Users: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=churned_scores, nbinsx=20, name="Churned",
        marker_color=_rgba(COL_LO, 0.72),
        hovertemplate="Score: %{x:.0f}<br>Churned Users: %{y:,}<extra></extra>",
    ))
    fig.update_layout(**_cdefaults(
        height=260, barmode="overlay",
        xaxis=dict(title="Health Score (0–100)", range=[0, 100]),
        yaxis=dict(title="Users"),
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)),
        margin=dict(l=14, r=14, t=40, b=36),
    ))
    return fig


# ══════════════════════════════════════════════════════════════
#  DASHBOARD FEATURE 1 — GOAL TRACKING / NORTH STAR
# ══════════════════════════════════════════════════════════════

GOAL_DEFAULTS = [
    ("activation", "Activation %", 70),
    ("conversion", "Conversion %", 25),
    ("dau-mau",    "DAU/MAU %",    20),
    ("churn",      "Churn %",      10),
    ("ltv",        "Avg LTV $",    800),
    ("health",     "Health /100",  65),
]


def build_goal_tracking(targets=None):
    """Progress bars for key PM metrics vs targets.
    targets: optional dict {"activation": 70, "conversion": 25, ...} to override defaults."""
    up = user_profiles
    active = up[up["churned"] == 0]

    activation_pct = float(up["completed_onboarding"].mean() * 100)
    conversion_pct = float(up["converted"].mean() * 100)
    dau_mau = float(_DAU_MAU_RATIO * 100) if _DAU_MAU_RATIO else 0
    churn_pct = float(up["churned"].mean() * 100)
    avg_ltv = float(up.loc[up["ltv"] > 0, "ltv"].mean()) if (up["ltv"] > 0).any() else 0
    avg_health = float(_health_scores[up["churned"] == 0].mean()) if hasattr(_health_scores, "mean") else 50

    t = targets or {}
    goals = [
        {"metric": "Activation Rate",  "current": activation_pct,
         "target": t.get("activation", 70), "unit": "%",
         "color": BRAND_BLUE, "direction": "higher"},
        {"metric": "Conversion Rate",  "current": conversion_pct,
         "target": t.get("conversion", 25), "unit": "%",
         "color": BRAND_TEAL, "direction": "higher"},
        {"metric": "DAU/MAU Ratio",    "current": dau_mau,
         "target": t.get("dau-mau", 20), "unit": "%",
         "color": BRAND_PURPLE, "direction": "higher"},
        {"metric": "Churn Rate",       "current": churn_pct,
         "target": t.get("churn", 10), "unit": "%",
         "color": COL_LO, "direction": "lower"},
        {"metric": "Avg LTV",          "current": avg_ltv,
         "target": t.get("ltv", 800), "unit": "$",
         "color": BRAND_AMBER, "direction": "higher"},
        {"metric": "Avg Health Score", "current": avg_health,
         "target": t.get("health", 65), "unit": "/100",
         "color": COL_HI, "direction": "higher"},
    ]

    cards = []
    for g in goals:
        cur, tgt = g["current"], g["target"]
        if g["direction"] == "lower":
            progress = max(0, min(100, (1 - cur / max(tgt * 2, 1)) * 100)) if tgt > 0 else 50
            on_track = cur <= tgt
        else:
            progress = max(0, min(100, cur / max(tgt, 0.01) * 100))
            on_track = cur >= tgt

        status_color = COL_HI if on_track else (COL_MID if progress >= 60 else COL_LO)
        status_text  = "On Track" if on_track else ("Close" if progress >= 60 else "Needs Work")

        val_text = f"${cur:,.0f}" if g["unit"] == "$" else f"{cur:.1f}{g['unit']}"
        tgt_text = f"${tgt:,.0f}" if g["unit"] == "$" else f"{tgt}{g['unit']}"

        cards.append(dbc.Col(html.Div([
            html.Div([
                html.Span(g["metric"], style={"color": TEXT_MUTED, "fontSize": "0.63rem",
                    "textTransform": "uppercase", "letterSpacing": "0.5px"}),
                html.Span(status_text, style={"color": status_color, "fontSize": "0.6rem",
                    "fontWeight": "700", "float": "right"}),
            ]),
            html.Div(val_text, style={"color": TEXT_PRI, "fontSize": "1.2rem",
                "fontWeight": "700", "lineHeight": "1.3"}),
            html.Div(style={
                "height": "6px", "borderRadius": "3px",
                "backgroundColor": _rgba(g["color"], 0.15),
                "marginTop": "4px", "marginBottom": "3px",
                "overflow": "hidden",
            }, children=[
                html.Div(style={
                    "height": "100%", "borderRadius": "3px",
                    "width": f"{min(progress, 100):.0f}%",
                    "backgroundColor": g["color"],
                    "transition": "width 0.5s",
                }),
            ]),
            html.Div(f"Target: {tgt_text}", style={"color": TEXT_MUTED, "fontSize": "0.58rem"}),
        ], style={"backgroundColor": BG_CARD, "border": f"1px solid {BORDER}",
                  "borderRadius": "10px", "padding": "10px 14px"}),
        md=2, className="mb-2"))

    return dbc.Row(cards, className="g-2 mb-3")


# ══════════════════════════════════════════════════════════════
#  DASHBOARD FEATURE 3 — FEATURE ADOPTION FUNNEL
# ══════════════════════════════════════════════════════════════

def build_feature_adoption_funnel():
    """For each product feature: discovered → tried → adopted → power-used."""
    feature_events = [e for e in events["event_name"].unique() if e.startswith("view_feature_")]
    if not feature_events:
        return _empty_fig("No feature events found")

    rows = []
    total_users = len(user_profiles)
    for feat in sorted(feature_events):
        user_counts = events[events["event_name"] == feat].groupby("user_id").size()
        discovered = int((user_counts >= 1).sum())  # used at least once
        tried      = int((user_counts >= 3).sum())   # 3+ times
        adopted    = int((user_counts >= 7).sum())   # 7+ times (weekly habit)
        power      = int((user_counts >= 15).sum())  # 15+ times (power user)
        fname = feat.replace("view_feature_", "Feature ").title()
        rows.append({"feature": fname, "Discovered": discovered, "Tried (3+)": tried,
                     "Adopted (7+)": adopted, "Power (15+)": power})

    df = pd.DataFrame(rows)
    stages = ["Discovered", "Tried (3+)", "Adopted (7+)", "Power (15+)"]
    stage_colors = [BRAND_BLUE, BRAND_TEAL, BRAND_PURPLE, BRAND_AMBER]

    fig = go.Figure()
    for stage, color in zip(stages, stage_colors):
        fig.add_trace(go.Bar(
            y=df["feature"], x=df[stage],
            name=stage, orientation="h",
            marker_color=_rgba(color, 0.72),
            hovertemplate=f"<b>%{{y}}</b><br>{stage}: %{{x:,}} users<br>"
                          f"(%{{customdata:.1f}}% of total)<extra></extra>",
            customdata=df[stage] / total_users * 100,
        ))

    fig.update_layout(
        barmode="group",
        **_cdefaults(
            height=max(280, len(feature_events) * 55),
            xaxis=dict(title="Users"),
            yaxis=dict(tickfont=dict(size=10)),
            legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
            margin=dict(l=14, r=14, t=50, b=36),
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  DASHBOARD FEATURE 4 — REVENUE IMPACT ATTRIBUTION
# ══════════════════════════════════════════════════════════════

def build_revenue_impact_attribution():
    """For each high-friction event, estimate revenue at risk.
    Revenue at risk = potential LTV of at-risk + churned users who experienced that event."""
    # Get friction scores per event
    fdf = _friction_for_segment("all")
    if fdf.empty:
        return _empty_fig("No friction data"), []

    at_risk_ids = set(_CHURN_RISK["at_risk_df"].index.tolist()) if not _CHURN_RISK["at_risk_df"].empty else set()
    churned_ids = set(user_profiles[user_profiles["churned"] == 1].index.tolist())
    risk_ids = at_risk_ids | churned_ids

    # For each friction event, find affected users and their LTV
    impact_rows = []
    for _, row in fdf.nlargest(15, "friction").iterrows():
        evt = row["event"]
        affected_users = events[events["event_name"] == evt]["user_id"].unique()
        affected_at_risk = set(affected_users) & risk_ids
        n_affected = len(affected_at_risk)
        if n_affected == 0:
            continue

        # Revenue at risk: avg LTV of active users × at-risk users (potential loss)
        # + actual LTV lost from churned users who hit this event
        at_risk_in_evt = set(affected_users) & at_risk_ids
        churned_in_evt = set(affected_users) & churned_ids

        avg_active_ltv = float(user_profiles.loc[user_profiles["churned"] == 0, "ltv"].mean())
        revenue_at_risk = len(at_risk_in_evt) * avg_active_ltv
        revenue_lost    = float(user_profiles.loc[user_profiles.index.isin(churned_in_evt), "ltv"].sum())

        impact_rows.append({
            "Event": evt,
            "Friction Score": float(row["friction"]),
            "At-Risk Users": len(at_risk_in_evt),
            "Churned Users": len(churned_in_evt),
            "Revenue at Risk ($)": round(revenue_at_risk, 0),
            "Revenue Lost ($)": round(revenue_lost, 0),
            "Total Impact ($)": round(revenue_at_risk + revenue_lost, 0),
        })

    if not impact_rows:
        return _empty_fig("No revenue impact data"), []

    idf = pd.DataFrame(impact_rows).sort_values("Total Impact ($)", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=idf["Event"], x=idf["Revenue at Risk ($)"],
        name="Revenue at Risk", orientation="h",
        marker_color=_rgba(COL_MID, 0.72),
        hovertemplate="<b>%{y}</b><br>At Risk: $%{x:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=idf["Event"], x=idf["Revenue Lost ($)"],
        name="Revenue Already Lost", orientation="h",
        marker_color=_rgba(COL_LO, 0.72),
        hovertemplate="<b>%{y}</b><br>Lost: $%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        **_cdefaults(
            height=max(300, len(impact_rows) * 32),
            xaxis=dict(title="Revenue Impact ($)"),
            yaxis=dict(tickfont=dict(size=10)),
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)),
            margin=dict(l=14, r=14, t=44, b=36),
        ),
    )
    return fig, impact_rows


# ══════════════════════════════════════════════════════════════
#  PM INTELLIGENCE — PRODUCT HEALTH TRENDS
# ══════════════════════════════════════════════════════════════
def build_product_health_trends(_up=None, _ev=None, window_days=30):
    """
    Five compact sparkline-metric cards showing monthly cohort trends.
    _up: optional filtered user_profiles, _ev: optional filtered events.
    window_days: controls comparison period (30=MoM, 60=2M, 90=quarter).
    """
    n_periods = max(1, round(window_days / 30))
    up_ = (_up if _up is not None else user_profiles).copy()
    up_["cohort_m"] = up_["first_event"].dt.to_period("M").astype(str)

    cohort_stats = (
        up_.groupby("cohort_m")
        .agg(
            new_users=("churned", "count"),
            activation_rate=("completed_onboarding", "mean"),
            conversion_rate=("converted", "mean"),
            retention_rate=("churned", lambda x: (1 - x).mean()),
        )
        .reset_index()
        .sort_values("cohort_m")
    )

    # Median TTV per cohort (days from first event to first payment)
    _ev_src = _ev if _ev is not None else events
    pay_ev = _ev_src[_ev_src["event_name"] == "payment_success"].groupby("user_id")["timestamp"].min()
    ttv = (
        pay_ev.reset_index()
        .join(up_[["first_event", "cohort_m"]], on="user_id")
        .assign(ttv_days=lambda d: (d["timestamp"] - d["first_event"]).dt.days.clip(lower=0))
    )
    med_ttv = ttv.groupby("cohort_m")["ttv_days"].median().reset_index()
    med_ttv.columns = ["cohort_m", "median_ttv"]
    cohort_stats = cohort_stats.merge(med_ttv, on="cohort_m", how="left")
    cohort_stats["median_ttv"] = cohort_stats["median_ttv"].fillna(0)
    cohort_stats = cohort_stats[cohort_stats["new_users"] >= 5]
    if len(cohort_stats) < 2:
        return html.Div()

    def _period_delta(series, n):
        """Delta between latest value and n periods ago."""
        v = series.dropna().values
        if len(v) > n:
            return float(v[-1]) - float(v[-(1 + n)])
        elif len(v) >= 2:
            return float(v[-1]) - float(v[0])
        return 0.0

    vs_label = "MoM" if n_periods == 1 else f"vs {n_periods}M ago"

    def _health_card(title, subtitle, series, fmt, color, is_better_up=True, is_pct=False, card_id=None):
        vals = series.dropna().values
        if not len(vals):
            return dbc.Col()
        curr  = vals[-1]
        delta = _period_delta(series, n_periods)
        good  = (delta >= 0) if is_better_up else (delta <= 0)
        d_col = COL_HI if good else COL_LO
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        d_str = f"{arrow} {abs(delta):.1f}{'pp' if is_pct else ''} {vs_label}"
        sp    = _sparkline(vals, color)
        extra = {"id": card_id} if card_id else {}
        return dbc.Col(_card(title, subtitle, [
            html.Div([
                html.Span(fmt.format(curr),
                          style={"fontSize": "1.85rem", "fontWeight": "700",
                                 "color": color, "lineHeight": "1.1"}),
                html.Span(d_str,
                          style={"color": d_col, "fontSize": "0.74rem",
                                 "marginLeft": "8px", "fontWeight": "600"}),
            ], style={"marginBottom": "4px"}),
            dcc.Graph(figure=sp, config={"displayModeBar": False},
                      style={"height": "40px", "marginTop": "2px"}),
        ]), md=True, **extra)

    return dbc.Row([
        _health_card("Activation Rate",  "% new users completing onboarding per cohort",
                     cohort_stats["activation_rate"] * 100, "{:.1f}%", BRAND_TEAL,  is_pct=True,
                     card_id="health-activation"),
        _health_card("Conversion Rate",  "% new users who paid per cohort",
                     cohort_stats["conversion_rate"]  * 100, "{:.1f}%", BRAND_BLUE,  is_pct=True,
                     card_id="health-conversion"),
        _health_card("Retention Rate",    "% of signup cohort retained (composite: event volume, activity, feature breadth)",
                     cohort_stats["retention_rate"]   * 100, "{:.1f}%", BRAND_PURPLE, is_pct=True,
                     card_id="health-retention"),
        _health_card("New Users / Month", "Monthly signups trend",
                     cohort_stats["new_users"], "{:,.0f}", COL_HI,
                     card_id="health-new-users"),
        _health_card("Median TTV",       "Days from signup to first payment (lower = better)",
                     cohort_stats["median_ttv"], "{:.0f}d", BRAND_AMBER, is_better_up=False,
                     card_id="health-ttv"),
    ], className="mb-3 g-2")


def _compute_period_deltas(window_days=30):
    """
    Compute MoM (or multi-month) period deltas for each of the 6 key metrics.
    Returns {metric_name: (arrow, dpct, d_col, vs_label)}.
    Used to embed trend indicators directly inside KPI cards.
    """
    n_months  = max(1, round(window_days / 30))
    ref       = user_profiles["first_event"].max().to_period("M")
    pa_months = pd.period_range(end=ref,            periods=n_months, freq="M")
    pb_months = pd.period_range(end=ref - n_months, periods=n_months, freq="M")
    vs_label  = "vs prior month" if n_months == 1 else f"vs prior {n_months}M"

    def _metrics(months):
        up_mask  = user_profiles["first_event"].dt.to_period("M").isin(months)
        new_     = user_profiles[up_mask]
        n_new    = len(new_)
        ev_start = months.min().start_time
        ev_end   = months.max().end_time + pd.Timedelta(days=1)
        ev       = events[(events["timestamp"] >= ev_start) & (events["timestamp"] < ev_end)]
        days     = max((ev_end - ev_start).days, 1)
        return {
            "New Signups":      n_new,
            "Activation %":     round(new_["completed_onboarding"].mean() * 100, 1) if n_new else 0,
            "Conversion %":     round(new_["converted"].mean() * 100, 1) if n_new else 0,
            "Active Users":     ev["user_id"].nunique(),
            "Avg Events / Day": round(len(ev) / days, 1),
            "Payments":         int(ev[ev["event_name"] == "payment_success"]["user_id"].nunique()),
        }

    a = _metrics(pa_months)
    b = _metrics(pb_months)
    better_up = {"New Signups", "Activation %", "Conversion %",
                 "Active Users", "Avg Events / Day", "Payments"}
    result = {}
    for metric, a_v in a.items():
        b_v   = b.get(metric, 0)
        delta = a_v - b_v
        dpct  = (delta / max(abs(b_v), 0.01)) * 100
        good  = (delta >= 0) if (metric in better_up) else (delta <= 0)
        result[metric] = (
            "↑" if delta > 0 else ("↓" if delta < 0 else "→"),
            dpct,
            COL_HI if good else COL_LO,
            vs_label,
        )
    return result


def build_period_comparison_card(window_days=30, visible_ids=None):
    """
    Compare key product metrics between two consecutive calendar-month windows.
    window_days=30  → current month vs previous month
    window_days=60  → last 2 months vs 2 months before that
    window_days=90  → last quarter vs prior quarter

    Uses the same calendar-month grouping as build_product_health_trends() so that
    "New Signups" here matches "New Users / Month" in the health trends row above.

    visible_ids: set of card IDs to show (None = show all).
    """
    n_months = max(1, round(window_days / 30))
    ref      = user_profiles["first_event"].max().to_period("M")

    # Period A = last n_months calendar months (ending with ref)
    pa_months = pd.period_range(end=ref,                  periods=n_months, freq="M")
    # Period B = n_months calendar months immediately before Period A
    pb_months = pd.period_range(end=ref - n_months,       periods=n_months, freq="M")

    vs_label = "vs prior month" if n_months == 1 else f"vs prior {n_months}M"

    def _metrics(months):
        up_mask = user_profiles["first_event"].dt.to_period("M").isin(months)
        new_    = user_profiles[up_mask]
        n_new   = len(new_)
        # Event window aligned to those exact calendar months
        ev_start = months.min().start_time
        ev_end   = months.max().end_time + pd.Timedelta(days=1)
        ev   = events[(events["timestamp"] >= ev_start) & (events["timestamp"] < ev_end)]
        days = max((ev_end - ev_start).days, 1)
        return {
            "New Signups":         n_new,
            "Activation %":        round(new_["completed_onboarding"].mean() * 100, 1) if n_new else 0,
            "Conversion %":        round(new_["converted"].mean() * 100, 1) if n_new else 0,
            "Active Users":        ev["user_id"].nunique(),
            "Avg Events / Day":    round(len(ev) / days, 1),
            "Payments":            int(ev[ev["event_name"] == "payment_success"]["user_id"].nunique()),
        }

    a = _metrics(pa_months)
    b = _metrics(pb_months)
    better_up = {"New Signups", "Activation %", "Conversion %",
                 "Active Users", "Avg Events / Day", "Payments"}

    cols = []
    for metric, a_v in a.items():
        card_id   = _PERIOD_ID_MAP.get(metric)
        hidden    = (visible_ids is not None) and (card_id not in visible_ids)
        b_v   = b.get(metric, 0)
        delta = a_v - b_v
        dpct  = (delta / max(abs(b_v), 0.01)) * 100
        good  = (delta >= 0) if (metric in better_up) else (delta <= 0)
        d_col = COL_HI if good else COL_LO
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        is_pct = "%" in metric
        fmt    = "{:.1f}%" if is_pct else ("{:,.0f}" if a_v >= 10 else "{:.1f}")
        col_kwargs = {"md": True}
        if card_id:
            col_kwargs["id"] = card_id
        if hidden:
            col_kwargs["style"] = {"display": "none"}
        cols.append(dbc.Col(html.Div([
            html.Div(metric, style={"color": TEXT_MUTED, "fontSize": "0.64rem",
                                    "textTransform": "uppercase", "letterSpacing": "0.5px",
                                    "marginBottom": "3px"}),
            html.Div(fmt.format(a_v), style={"color": TEXT_PRI, "fontSize": "1.3rem",
                                              "fontWeight": "700", "lineHeight": "1.1",
                                              "marginBottom": "3px"}),
            html.Div([
                html.Span(f"{arrow} {abs(dpct):.1f}% ", style={"color": d_col, "fontWeight": "600"}),
                html.Span(vs_label, style={"color": TEXT_MUTED}),
            ], style={"fontSize": "0.72rem"}),
        ], style={
            "padding": "12px 14px", "background": BG_SURFACE,
            "borderRadius": "10px", "border": f"1px solid {BORDER}",
        }), **col_kwargs))

    return dbc.Row(cols, className="g-2")


# ══════════════════════════════════════════════════════════════
#  PM INTELLIGENCE — ACTIVATION & AHA MOMENT
# ══════════════════════════════════════════════════════════════
def build_aha_moment_chart():
    """
    For each product feature, compare the D30 retention rate of users
    who adopted the feature within Day 3 / Day 7 / Day 14 vs those who
    never adopted it. The larger the gap, the more that feature acts as
    an 'aha moment' driver. Correlation-based, not causal.
    """
    features = [
        ("view_feature_A", "Feature A"),
        ("view_feature_B", "Feature B"),
        ("view_feature_C", "Feature C"),
        ("view_feature_D", "Feature D"),
    ]
    up_   = user_profiles[["first_event", "churned"]].copy()
    up_["retained"] = (1 - up_["churned"])
    baseline = up_["retained"].mean() * 100

    all_rows = []
    for feat_evt, feat_label in features:
        feat_ev = events[events["event_name"] == feat_evt]
        if feat_ev.empty:
            continue
        first_use = (
            feat_ev.groupby("user_id")["timestamp"].min()
            .reset_index()
            .join(up_[["first_event"]], on="user_id")
        )
        first_use["days"] = (
            (first_use["timestamp"].dt.normalize() -
             first_use["first_event"].dt.normalize()).dt.days
        ).clip(lower=0)

        never_ids  = set(up_.index) - set(first_use["user_id"])
        ret_never  = up_[up_.index.isin(never_ids)]["retained"].mean() * 100 if never_ids else 0

        for window, label in [(3, "≤ Day 3"), (7, "≤ Day 7"), (14, "≤ Day 14")]:
            ids = set(first_use[first_use["days"] <= window]["user_id"])
            ret = up_[up_.index.isin(ids)]["retained"].mean() * 100 if ids else 0
            all_rows.append({"feature": feat_label, "window": label,
                              "retention": ret, "uplift": round(ret - ret_never, 1),
                              "n": len(ids)})
        all_rows.append({"feature": feat_label, "window": "Never Adopted",
                         "retention": ret_never, "uplift": 0.0, "n": len(never_ids)})

    if not all_rows:
        return _empty_fig("No feature event data")

    df = pd.DataFrame(all_rows)
    windows = ["≤ Day 3", "≤ Day 7", "≤ Day 14", "Never Adopted"]
    colors  = [BRAND_TEAL, BRAND_BLUE, BRAND_PURPLE, COL_LO]

    fig = go.Figure()
    for win, col in zip(windows, colors):
        d = df[df["window"] == win]
        if d.empty:
            continue
        fig.add_trace(go.Bar(
            x=d["feature"], y=d["retention"], name=win,
            marker_color=_rgba(col, 0.72),
            text=(d["retention"].round(0).astype(int).astype(str) + "%"),
            textposition="outside",
            textfont=dict(size=9, color=col),
            customdata=np.stack([d["n"], d["uplift"]], axis=1),
            hovertemplate=(
                "<b>%{x}</b> — " + win +
                "<br>Retention: %{y:.1f}%"
                "<br>vs Never-Adopted: %{customdata[1]:+.1f}pp"
                "<br>N users: %{customdata[0]:,}"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(y=baseline,
                  line=dict(color=_rgba(TEXT_MUTED, 0.55), width=1, dash="dot"),
                  annotation_text=f"Overall baseline {baseline:.1f}%",
                  annotation=dict(font=dict(color=TEXT_MUTED, size=9)))

    fig.update_layout(**_cdefaults(
        height=340, barmode="group",
        xaxis=dict(tickfont=dict(color=TEXT_PRI, size=12)),
        yaxis=dict(title="% Users Retained", ticksuffix="%",
                   tickfont=dict(color=TEXT_SEC, size=10),
                   gridcolor=_rgba(BRAND_BLUE, 0.08), range=[0, 110]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SEC, size=10),
                    orientation="h", y=-0.18, x=0),
        margin=dict(l=14, r=14, t=40, b=14),
    ))
    return fig


def build_adoption_timing_chart():
    """
    For retained users only: median day they first used each feature.
    Features adopted earliest = most likely aha moments.
    """
    features = [
        ("view_feature_A", "Feature A"),
        ("view_feature_B", "Feature B"),
        ("view_feature_C", "Feature C"),
        ("view_feature_D", "Feature D"),
        ("view_dashboard",  "Dashboard"),
        ("view_pricing",    "Pricing"),
        ("invite_team",     "Team Invite"),
    ]
    retained = user_profiles[user_profiles["churned"] == 0]
    up_fe    = user_profiles[["first_event"]].copy()

    rows = []
    for feat_evt, feat_label in features:
        feat_ev = events[
            events["event_name"].eq(feat_evt) &
            events["user_id"].isin(retained.index)
        ]
        if feat_ev.empty:
            continue
        fu = (
            feat_ev.groupby("user_id")["timestamp"].min()
            .reset_index()
            .join(up_fe["first_event"], on="user_id")
        )
        fu["days"] = (
            (fu["timestamp"].dt.normalize() -
             fu["first_event"].dt.normalize()).dt.days
        ).clip(lower=0)
        rows.append({
            "feature":    feat_label,
            "median_day": fu["days"].median(),
            "pct_adopted": round(len(fu) / max(len(retained), 1) * 100, 1),
        })

    if not rows:
        return _empty_fig("No feature event data")

    df = pd.DataFrame(rows).sort_values("median_day", ascending=True)

    fig = go.Figure(go.Bar(
        y=df["feature"], x=df["median_day"], orientation="h",
        marker=dict(
            color=df["median_day"],
            colorscale=[[0, BRAND_TEAL], [0.5, BRAND_BLUE], [1, BRAND_AMBER]],
            showscale=False,
        ),
        text=("Day " + df["median_day"].round(1).astype(str)),
        textposition="outside",
        textfont=dict(size=10),
        customdata=df["pct_adopted"],
        hovertemplate=(
            "<b>%{y}</b><br>Median adoption: Day %{x:.1f}"
            "<br>%{customdata:.1f}% of retained users adopted it"
            "<extra></extra>"
        ),
    ))
    fig.add_vline(x=3,  line=dict(color=_rgba(BRAND_TEAL,  0.6), width=1, dash="dash"),
                  annotation_text="Day 3",  annotation=dict(font=dict(color=BRAND_TEAL,  size=9)))
    fig.add_vline(x=7,  line=dict(color=_rgba(BRAND_AMBER, 0.6), width=1, dash="dash"),
                  annotation_text="Day 7",  annotation=dict(font=dict(color=BRAND_AMBER, size=9)))
    fig.add_vline(x=14, line=dict(color=_rgba(COL_LO,      0.5), width=1, dash="dash"),
                  annotation_text="Day 14", annotation=dict(font=dict(color=COL_LO,      size=9)))

    fig.update_layout(**_cdefaults(
        height=340,
        xaxis=dict(title="Median Day of First Use (retained users only)",
                   tickfont=dict(color=TEXT_SEC, size=10),
                   gridcolor=_rgba(BRAND_BLUE, 0.08)),
        yaxis=dict(tickfont=dict(color=TEXT_SEC, size=11), autorange="reversed"),
        margin=dict(l=14, r=14, t=24, b=14),
    ))
    return fig


def build_feature_ltv_impact():
    """
    For each feature/event, compares avg LTV between adopters and non-adopters.
    The delta = monetisation value attributable to that feature's adoption.
    Hover shows the total LTV upside if all non-adopters could be converted.
    """
    features = [
        ("view_feature_A",  "Feature A"),
        ("view_feature_B",  "Feature B"),
        ("view_feature_C",  "Feature C"),
        ("view_feature_D",  "Feature D"),
        ("view_pricing",    "Pricing Page"),
        ("invite_team",     "Team Invite"),
        ("view_dashboard",  "Dashboard"),
    ]
    up_ = user_profiles[["ltv"]].copy()

    rows = []
    for feat_evt, feat_label in features:
        adopt_ids  = set(events[events["event_name"] == feat_evt]["user_id"].unique())
        non_ids    = set(up_.index) - adopt_ids
        if not adopt_ids:
            continue
        avg_a = up_[up_.index.isin(adopt_ids)]["ltv"].mean()
        avg_n = up_[up_.index.isin(non_ids)]["ltv"].mean() if non_ids else 0.0
        delta = avg_a - avg_n
        rows.append({
            "feature":   feat_label,
            "ltv_adopt": round(avg_a, 1),
            "ltv_non":   round(avg_n, 1),
            "delta":     round(delta, 1),
            "n_adopt":   len(adopt_ids),
            "n_non":     len(non_ids),
            "upside":    round(delta * len(non_ids), 0),
        })

    if not rows:
        return _empty_fig("No feature LTV data")

    df = pd.DataFrame(rows).sort_values("delta", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["feature"], x=df["ltv_non"], name="Non-Adopters avg LTV",
        orientation="h", marker_color=_rgba(TEXT_MUTED, 0.55),
        customdata=df["n_non"],
        hovertemplate="<b>%{y}</b> — Non-adopters<br>Avg LTV: $%{x:.0f}<br>N: %{customdata:,}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=df["feature"], x=df["ltv_adopt"], name="Adopters avg LTV",
        orientation="h", marker_color=_rgba(BRAND_TEAL, 0.72),
        customdata=np.stack([df["n_adopt"], df["upside"]], axis=1),
        hovertemplate=(
            "<b>%{y}</b> — Adopters<br>Avg LTV: $%{x:.0f}"
            "<br>N: %{customdata[0]:,}"
            "<br>LTV upside (all non-adopters): $%{customdata[1]:,.0f}"
            "<extra></extra>"
        ),
    ))

    x_max = df[["ltv_adopt", "ltv_non"]].values.max() * 1.25
    for _, r in df.iterrows():
        fig.add_annotation(
            y=r["feature"],
            x=max(r["ltv_adopt"], r["ltv_non"]) + x_max * 0.02,
            text=f"+${r['delta']:.0f}" if r["delta"] >= 0 else f"${r['delta']:.0f}",
            showarrow=False, xanchor="left",
            font=dict(color=COL_HI if r["delta"] >= 0 else COL_LO, size=10),
        )

    fig.update_layout(**_cdefaults(
        height=320, barmode="group",
        xaxis=dict(title="Avg LTV ($)", tickprefix="$",
                   tickfont=dict(color=TEXT_SEC, size=10),
                   gridcolor=_rgba(BRAND_BLUE, 0.08), range=[0, x_max]),
        yaxis=dict(tickfont=dict(color=TEXT_SEC, size=11), autorange="reversed"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SEC, size=10),
                    orientation="h", y=-0.18, x=0),
        margin=dict(l=14, r=14, t=24, b=14),
    ))
    return fig


def build_segmented_funnel_chart(mode="all"):
    """
    mode='all'     → standard single funnel (all users)
    mode='segment' → grouped bars per stage, one bar per revenue segment
    mode='cluster' → grouped bars per stage, one bar per cluster
    Shows relative drop-off rates so segments can be directly compared.
    """
    stages = [
        ("signup",           "Signup"),
        ("onboarding_step1", "Onboarding 1"),
        ("onboarding_step2", "Onboarding 2"),
        ("view_pricing",     "Pricing"),
        ("start_checkout",   "Checkout"),
        ("payment_success",  "Payment"),
    ]
    if mode == "all":
        return build_conversion_funnel()

    if mode == "segment":
        up_  = _segment_users()
        seg_order = ["Free", "Low Revenue", "Mid Revenue", "High Revenue"]
        groups = {s: up_[up_["revenue_segment"] == s].index
                  for s in seg_order if s in up_["revenue_segment"].values}
        colors = SEG_COLORS
    else:  # cluster
        if not _cluster_user_map:
            return _empty_fig("Run clustering first (Section 5), then return here.")
        groups = {k: pd.Index(v) for k, v in _cluster_user_map.items()}
        pal    = [BRAND_BLUE, BRAND_TEAL, BRAND_PURPLE, BRAND_AMBER, COL_MID, COL_HI]
        colors = {k: pal[i % len(pal)] for i, k in enumerate(groups)}

    fig = go.Figure()
    for grp, uid_idx in groups.items():
        if not len(uid_idx):
            continue
        evts_g = events[events["user_id"].isin(uid_idx)]
        n_grp  = len(uid_idx)
        x_lb, y_pct, y_cnt = [], [], []
        for ev, lb in stages:
            n = evts_g[evts_g["event_name"] == ev]["user_id"].nunique()
            x_lb.append(lb)
            y_pct.append(round(n / n_grp * 100, 1))
            y_cnt.append(n)

        fig.add_trace(go.Bar(
            x=x_lb, y=y_pct, name=grp,
            marker_color=_rgba(colors.get(grp, BRAND_BLUE), 0.72),
            text=[f"{p:.0f}%" for p in y_pct],
            textposition="outside", textfont=dict(size=8),
            customdata=y_cnt,
            hovertemplate=(
                f"<b>{grp}</b><br>%{{x}}: %{{y:.1f}}% of segment"
                "<br>(%{customdata:,} users)<extra></extra>"
            ),
        ))

    fig.update_layout(**_cdefaults(
        height=360, barmode="group",
        yaxis=dict(title="% of Segment Reaching Stage", ticksuffix="%",
                   tickfont=dict(color=TEXT_SEC, size=10),
                   gridcolor=_rgba(BRAND_BLUE, 0.08), range=[0, 118]),
        xaxis=dict(tickfont=dict(color=TEXT_PRI, size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SEC, size=10),
                    orientation="h", y=-0.22, x=0),
        margin=dict(l=14, r=14, t=24, b=14),
    ))
    return fig


def _generate_cluster_archetypes(cluster_summary, features):
    """
    Generate plain-English archetype name + description for each cluster row.
    Returns a list of (archetype_name, description) tuples, one per cluster row.
    """
    if cluster_summary is None or cluster_summary.empty:
        return []

    feat_cols = [f for f in features if f in cluster_summary.columns]
    if not feat_cols:
        return []

    overall_means = cluster_summary[feat_cols].mean()
    results = []

    # Ordered signal rules — first match wins for the primary name
    ltv_cols     = [c for c in feat_cols if "ltv" in c or "revenue" in c or "charge" in c]
    engage_cols  = [c for c in feat_cols if "session" in c or "active" in c or "events" in c]
    feature_cols = [c for c in feat_cols if "breadth" in c or "feature" in c]
    rage_cols    = [c for c in feat_cols if "rage" in c]
    team_cols    = [c for c in feat_cols if "invite" in c or "team" in c]
    pricing_cols = [c for c in feat_cols if "pricing" in c]
    ob_cols      = [c for c in feat_cols if "onboarding" in c]

    for _, row in cluster_summary.iterrows():
        name_parts = []
        traits     = []

        def _ratio(cols):
            if not cols:
                return 1.0
            vals   = [row.get(c, 0) for c in cols]
            means_ = [overall_means.get(c, 1) for c in cols]
            ratios = [v / max(m, 0.0001) for v, m in zip(vals, means_)]
            return sum(ratios) / len(ratios)

        # Revenue
        if ltv_cols:
            r = _ratio(ltv_cols)
            if r > 1.35:
                name_parts.append("High Value")
            elif r < 0.65:
                name_parts.append("Low Spend")

        # Engagement
        if engage_cols:
            r = _ratio(engage_cols)
            if r > 1.35:
                name_parts.append("Power User")
            elif r < 0.65:
                name_parts.append("Disengaged")

        # Features
        if feature_cols:
            r = _ratio(feature_cols)
            if r > 1.3:
                traits.append("broad feature adoption")
            elif r < 0.7:
                traits.append("narrow feature use")

        # Rage clicks
        if rage_cols and _ratio(rage_cols) > 1.5:
            traits.append("high UX friction")

        # Team invite
        if team_cols and _ratio(team_cols) > 1.3:
            name_parts.append("Collaborator")

        # Pricing interest
        if pricing_cols and _ratio(pricing_cols) > 1.3:
            traits.append("pricing-aware")

        # Onboarding
        if ob_cols:
            r = _ratio(ob_cols)
            if r < 0.6:
                traits.append("low onboarding completion")

        archetype = " & ".join(dict.fromkeys(name_parts[:2])) if name_parts else "Mixed Profile"
        desc = "Users with " + ", ".join(traits[:3]) + "." if traits else "Balanced across all metrics."
        results.append((archetype, desc))

    return results


# ══════════════════════════════════════════════════════════════
#  KPI CARDS WITH SPARKLINES + DAU/MAU
# ══════════════════════════════════════════════════════════════
def _kpi_card(title, value, subtext, sparkfig=None, color=BRAND_BLUE, card_id=None,
              alert_level=None, col_md=2, period_delta=None):
    """
    period_delta: tuple (arrow, dpct, d_col, vs_label) from _compute_period_deltas.
    When provided, a compact MoM delta strip is rendered between subtext and sparkline.
    """
    r2, g2, b2 = _hex_rgb(color)
    alert_cfg = {
        "danger": (COL_LO,      "⚠ Below threshold"),
        "warn":   (BRAND_AMBER, "▲ Watch"),
    }
    children = []
    if alert_level and alert_level in alert_cfg:
        a_col, a_text = alert_cfg[alert_level]
        children.append(html.Div(a_text, className="kpi-alert", style={
            "backgroundColor": _rgba(a_col, 0.12),
            "border": f"1px solid {_rgba(a_col, 0.35)}",
            "color": a_col,
        }))
    children += [
        html.Div(title, style={"color": TEXT_MUTED, "fontSize": "0.63rem",
                                "textTransform": "uppercase", "letterSpacing": "0.6px",
                                "marginBottom": "1px"}),
        html.Div(value, style={"color": color, "fontSize": "1.3rem",
                                "fontWeight": "700", "lineHeight": "1.1",
                                "fontFamily": "Inter, system-ui, sans-serif"}),
        html.Div(subtext, style={"color": TEXT_MUTED, "fontSize": "0.63rem",
                                  "marginTop": "2px", "marginBottom": "3px"}),
    ]
    if period_delta:
        p_arrow, p_dpct, p_col, p_label = period_delta
        # Inline delta — no separator, no extra height
        children.append(html.Div([
            html.Span(f"{p_arrow} {abs(p_dpct):.1f}%",
                      style={"color": p_col, "fontWeight": "700", "fontSize": "0.65rem",
                             "marginRight": "4px"}),
            html.Span(p_label,
                      style={"color": TEXT_MUTED, "fontSize": "0.60rem"}),
        ], style={"marginTop": "3px"}))
    # Sparklines removed from individual cards — see Product Health Trends row below
    extra = {"id": card_id} if card_id else {}
    return dbc.Col(html.Div(children, style={
        "backgroundColor": BG_CARD,
        "border": f"1px solid {BORDER}",
        "borderLeft": f"3px solid {color}",
        "borderRadius": "10px",
        "padding": "10px 14px 8px",
        "boxShadow": f"0 0 18px rgba({r2},{g2},{b2},0.07)",
        "height": "100%",
        "display": "flex",
        "flexDirection": "column",
        "justifyContent": "flex-start",
        "minHeight": "90px",
    }), md=col_md, sm=4, xs=6, className="mb-2 d-flex", **extra)


def build_kpi_row(window_days=30, _up=None):
    """
    KPI cards with embedded period-over-period delta indicators.
    window_days controls the comparison window (30=1M, 60=2M, 90=quarter).
    _up: optional filtered user_profiles DataFrame (pass when date filter is active).
    """
    up_ = _up if _up is not None else user_profiles
    pay = up_[up_["total_revenue"] > 0]
    n_pay = len(pay)
    total_rev = pay["total_revenue"].sum()
    avg_ltv = pay["ltv"].mean() if n_pay else 0
    churn_pct = up_["churned"].mean() * 100
    rage_total = int(up_["n_rage_click"].sum()) if "n_rage_click" in up_.columns else 0

    # Latest DAU/MAU — derived from filtered user set if available
    _dau_src = dau_mau_df
    if _up is not None:
        # Recompute DAU/MAU for the filtered user set
        _uid_set = set(up_.index.tolist())
        _ev_f = events_raw[events_raw["user_id"].isin(_uid_set)]
        if not _ev_f.empty:
            _dau_src = _compute_dau_mau(_ev_f)
    last_dau   = _dau_src["dau"].iloc[-1]   if len(_dau_src) else 0
    last_mau   = _dau_src["mau"].iloc[-1]   if len(_dau_src) else 1
    last_ratio = _dau_src["ratio"].iloc[-1] if len(_dau_src) else 0

    # Alert thresholds
    ratio_alert = "danger" if last_ratio < 0.10 else ("warn" if last_ratio < 0.20 else None)
    churn_alert = "danger" if churn_pct > 40 else ("warn" if churn_pct > 25 else None)

    # Period deltas — embedded inline in each KPI card
    _pd = _compute_period_deltas(window_days)
    def _d(metric):
        return _pd.get(metric)      # returns (arrow, dpct, d_col, vs_label) or None

    def _group_label(text, color):
        return html.Div(text, style={
            "color": color, "fontSize": "9px", "fontWeight": "700",
            "textTransform": "uppercase", "letterSpacing": "0.8px",
            "borderLeft": f"2px solid {color}", "paddingLeft": "7px",
            "marginBottom": "6px", "marginTop": "2px",
        })

    n_users = len(up_)
    return dbc.Row([
        _kpi_card("Total Users",
                  f"{n_users:,}",
                  f"{n_pay:,} paying ({n_pay/max(n_users,1)*100:.1f}%)",
                  None, BRAND_BLUE,   card_id="kpi-total-users", col_md=2,
                  period_delta=_d("New Signups")),
        _kpi_card("DAU",
                  f"{last_dau:,}",
                  "last active day",
                  None, BRAND_PURPLE, card_id="kpi-dau", col_md=2,
                  period_delta=_d("Active Users")),
        _kpi_card("DAU/MAU Ratio",
                  f"{last_ratio:.2f}",
                  f"MAU: {last_mau:,}  · depth",
                  None, BRAND_TEAL, card_id="kpi-ratio",
                  alert_level=ratio_alert, col_md=2,
                  period_delta=_d("Avg Events / Day")),
        _kpi_card("Total Revenue",
                  f"${total_rev:,.0f}",
                  f"from {n_pay:,} paying users",
                  None, BRAND_AMBER,  card_id="kpi-revenue", col_md=2,
                  period_delta=_d("Conversion %")),
        _kpi_card("Avg LTV",
                  f"${avg_ltv:.0f}",
                  "per paying user",
                  None, COL_HI,       card_id="kpi-ltv", col_md=2,
                  period_delta=_d("Payments")),
        _kpi_card("Churn Risk",
                  f"{churn_pct:.1f}%",
                  f"Rage clicks: {rage_total:,}",
                  None, COL_LO,         card_id="kpi-churn",
                  alert_level=churn_alert, col_md=2,
                  period_delta=_d("Activation %")),
    ], className="g-2 mb-3 align-items-stretch")


def _collapsible_header(title, subtitle, section_id):
    """Section header with a collapse toggle button."""
    return html.Div([
        html.Div([
            html.H4(title, style=DIVIDER_H),
            html.P(subtitle, style=DIVIDER_SUB),
        ], style={"flex": "1"}),
        html.Button(
            "⊟",
            id={"type": "sec-toggle", "index": section_id},
            className="section-collapse-btn",
            n_clicks=0,
            title="Collapse / expand section",
        ),
    ], className="section-header-row")

# ══════════════════════════════════════════════════════════════
#  LAYOUT HELPERS
# ══════════════════════════════════════════════════════════════
CARD_STYLE = {
    "backgroundColor": BG_CARD,
    "border": f"1px solid {BORDER}",
    "borderRadius": "14px",
    "padding": "20px 22px",
    "marginBottom": "16px",
}
SECTION_H = {"color": TEXT_PRI, "fontWeight": "600", "fontSize": "1rem",
             "marginBottom": "3px", "fontFamily": "Inter, system-ui, sans-serif"}
SECTION_SUB = {"color": TEXT_MUTED, "fontSize": "0.75rem", "marginBottom": "14px"}
DIVIDER_H = {"color": TEXT_PRI, "fontWeight": "700", "fontSize": "1.15rem",
             "marginTop": "8px", "marginBottom": "5px",
             "fontFamily": "Inter, system-ui, sans-serif"}
DIVIDER_SUB = {"color": TEXT_MUTED, "fontSize": "0.8rem", "marginBottom": "18px"}


def _card(title, subtitle, children):
    return html.Div([
        html.H6(title, style=SECTION_H),
        html.P(subtitle, style=SECTION_SUB),
        *children,
    ], style=CARD_STYLE, className="analytics-card")


def _section_header(title, subtitle):
    return html.Div([
        html.H4(title, style=DIVIDER_H),
        html.P(subtitle, style=DIVIDER_SUB),
    ])


def _chat_bubble(role: str, text: str):
    is_user = role == "user"
    return html.Div(
        dcc.Markdown(text, style={"margin": "0", "fontSize": "0.87rem"}),
        style={
            "backgroundColor": _rgba(BRAND_BLUE, 0.06) if is_user else _rgba(BRAND_PURPLE, 0.06),
            "border": f"1px solid {_rgba(BRAND_BLUE if is_user else BRAND_PURPLE, 0.2)}",
            "borderRadius": "10px",
            "padding": "10px 14px",
            "marginBottom": "8px",
            "color": TEXT_PRI,
            "marginLeft": "0" if is_user else "24px",
            "marginRight": "24px" if is_user else "0",
        },
    )


_TAB_PROMPTS = {
    "overview": "KPI overview, health trends, opportunity matrix, and goal tracking",
    "conversion": "conversion funnel, feature adoption, time-to-value, and onboarding",
    "engagement": "user flows, session duration, friction points, and clustering",
    "retention": "cohort retention, stickiness, and value/retention health",
    "churn": "churn risk, win-back priorities, and survival analysis",
}

def _ai_tab_panel(tab_key: str):
    """Collapsible AI Insights panel placed at the bottom of each tab."""
    return html.Div([
        html.Button([
            html.Span("✦ ", style={"color": BRAND_PURPLE}),
            html.Span("AI Insights"),
        ], id=f"ai-panel-toggle-{tab_key}", n_clicks=0,
           style={"background": BG_SURFACE, "border": f"1px solid {BORDER}",
                  "borderRadius": "8px", "color": TEXT_SEC,
                  "fontSize": "0.78rem", "padding": "8px 18px",
                  "cursor": "pointer", "fontWeight": "600",
                  "letterSpacing": "0.3px", "width": "100%",
                  "textAlign": "left", "marginTop": "20px"}),
        dbc.Collapse(id=f"ai-panel-collapse-{tab_key}", is_open=False, children=[
            html.Div([
                # Auto-summary area
                dbc.Spinner(
                    html.Div(id=f"ai-panel-summary-{tab_key}",
                             style={"minHeight": "40px", "marginBottom": "12px"}),
                    size="sm", color="primary",
                ),
                # Chat history
                html.Div(id=f"ai-panel-history-{tab_key}",
                    style={"maxHeight": "320px", "overflowY": "auto",
                           "marginBottom": "10px"}),
                # Chat input
                dbc.InputGroup([
                    dbc.Input(id=f"ai-panel-input-{tab_key}",
                        placeholder="Ask a follow-up question…",
                        style={"backgroundColor": BG_INPUT, "color": TEXT_PRI,
                               "border": f"1px solid {BORDER}",
                               "borderRadius": "8px 0 0 8px", "fontSize": "0.85rem"}),
                    dbc.Button("Ask", id=f"ai-panel-send-{tab_key}", color="primary",
                        size="sm", style={"borderRadius": "0 8px 8px 0", "fontWeight": "600"}),
                ], size="sm"),
                dcc.Store(id=f"ai-panel-store-{tab_key}", data=[]),
            ], style={"padding": "14px 0 8px"}),
        ]),
    ], style={"borderTop": f"1px solid {BORDER}", "paddingTop": "8px",
              "marginTop": "24px"})


# ══════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════
CUSTOM_CSS = f"""
* {{ box-sizing: border-box; }}
body, html {{
    background-color: {BG_PAGE} !important;
    font-family: Inter, system-ui, -apple-system, sans-serif;
    color: {TEXT_PRI};
    /* Override Dash 4's default light-theme CSS custom properties */
    --Dash-Fill-Inverse-Strong: {BG_INPUT};
    --Dash-Stroke-Strong: {BORDER};
    --Dash-Stroke-Weak: {_rgba(BRAND_BLUE, 0.08)};
    --Dash-Fill-Interactive-Strong: {BRAND_BLUE};
    --Dash-Fill-Interactive-Weak: {_rgba(BRAND_BLUE, 0.12)};
    --Dash-Text-Primary: {TEXT_PRI};
    --Dash-Text-Strong: {TEXT_PRI};
    --Dash-Text-Weak: {TEXT_SEC};
    --Dash-Text-Disabled: {TEXT_MUTED};
    --Dash-Shading-Strong: rgba(0,0,0,0.15);
    --Dash-Shading-Weak: rgba(0,0,0,0.07);
}}
/* Slider marks and tooltips */
.rc-slider-mark-text {{ color: {TEXT_SEC} !important; font-size: 10px !important; }}
.rc-slider-mark-text-active {{ color: {TEXT_PRI} !important; }}
.rc-slider-dot {{ border-color: {_rgba(BRAND_BLUE, 0.5)} !important; background-color: {BG_SURFACE} !important; }}
.rc-slider-dot-active {{ border-color: {BRAND_BLUE} !important; }}
.rc-slider-track {{ background-color: {BRAND_BLUE} !important; }}
.rc-slider-handle {{ border-color: {BRAND_BLUE} !important; background-color: {BRAND_BLUE} !important; }}
.rc-slider-tooltip-inner {{ background-color: {BG_SURFACE} !important; color: {TEXT_PRI} !important; border: 1px solid {BORDER} !important; }}
/* Dash 4 dropdown components (.Select-* is Dash 2 dead code — replaced below) */
.dash-dropdown, .dash-dropdown-trigger, .dash-dropdown-value {{
    color: {TEXT_PRI} !important;
}}
.dash-dropdown-placeholder {{ color: {TEXT_MUTED} !important; }}
.dash-dropdown-content {{
    background: {BG_SURFACE} !important;
    border-color: {BORDER} !important;
}}
.dash-options-list-option {{
    background: {BG_SURFACE} !important;
    color: {TEXT_SEC} !important;
}}
.dash-options-list-option:not(:has(input[disabled])):hover,
.dash-options-list-option:not(:has(input[disabled])):focus-within {{
    color: {BRAND_BLUE} !important;
}}
.dash-dropdown-search-container {{
    background: {BG_INPUT} !important;
    border-color: {BORDER} !important;
}}
/* Checklist items */
.form-check-label {{ color: {TEXT_SEC} !important; font-size: 12px; }}
.form-check-input {{ background-color: {BG_INPUT} !important; border-color: {_rgba(BRAND_BLUE, 0.4)} !important; }}
.form-check-input:checked {{ background-color: {BRAND_BLUE} !important; border-color: {BRAND_BLUE} !important; }}
/* Textarea and all input types */
textarea, input[type="text"], input[type="number"],
input[type="search"], input[type="email"] {{
    background-color: {BG_INPUT} !important;
    color: {TEXT_PRI} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px;
    font-family: Inter, system-ui, sans-serif;
    font-size: 13px;
    padding: 8px 12px;
    min-height: 40px;
}}
textarea:focus, input[type="text"]:focus,
input[type="number"]:focus, input[type="search"]:focus {{
    border-color: {BRAND_BLUE} !important;
    outline: none;
    box-shadow: 0 0 0 2px {_rgba(BRAND_BLUE, 0.2)};
}}
/* Number input spinner hide on firefox */
input[type="number"] {{ -moz-appearance: textfield; }}
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {{ -webkit-appearance: none; margin: 0; }}
/* Dash 4 dropdown search input */
.dash-dropdown-search-container input {{
    color: {TEXT_PRI} !important;
    background: transparent !important;
}}
/* Dash 4 input container */
.dash-input-container {{ color: {TEXT_PRI} !important; }}
/* DateRangePicker / SingleDatePicker */
.DateInput, .DateInput_input {{
    background-color: {BG_INPUT} !important;
    color: {TEXT_PRI} !important;
    font-family: Inter, system-ui, sans-serif;
    font-size: 12px;
}}
.DateInput_input::placeholder {{ color: {TEXT_MUTED} !important; }}
.DateRangePickerInput, .SingleDatePickerInput {{
    background-color: {BG_INPUT} !important;
    border-color: {BORDER} !important;
}}
.DayPicker, .CalendarMonthGrid, .CalendarMonth,
.DayPickerNavigation_button, .DayPicker_weekHeader {{
    background-color: {BG_SURFACE} !important;
    color: {TEXT_PRI} !important;
}}
.CalendarMonth_caption {{ color: {TEXT_PRI} !important; }}
.CalendarDay__default {{
    background-color: {BG_SURFACE} !important;
    color: {TEXT_SEC} !important;
    border-color: {_rgba(BRAND_BLUE, 0.12)} !important;
}}
.CalendarDay__default:hover {{
    background-color: {BG_CARD} !important;
    color: {TEXT_PRI} !important;
}}
.CalendarDay__selected, .CalendarDay__selected:hover {{
    background-color: {BRAND_BLUE} !important;
    color: #fff !important;
    border-color: {BRAND_BLUE} !important;
}}
.CalendarDay__selected_span {{
    background-color: {_rgba(BRAND_BLUE, 0.3)} !important;
    color: {TEXT_PRI} !important;
}}
.DayPickerNavigation_button__default {{
    background-color: {BG_INPUT} !important;
    border-color: {BORDER} !important;
    color: {TEXT_SEC} !important;
}}
.DateInput_fang {{ display: none; }}
/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {BG_PAGE}; }}
::-webkit-scrollbar-thumb {{ background: {_rgba(BRAND_BLUE, 0.35)}; border-radius: 3px; }}
/* Nav / brand bar */
.insightera-navbar {{
    background: linear-gradient(135deg, {_rgba(BRAND_BLUE, 0.15)} 0%, {_rgba(BRAND_PURPLE, 0.10)} 100%);
    border-bottom: 1px solid {BORDER_LIT};
    padding: 10px 28px;
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 20px;
}}
.insightera-brand {{ font-size: 1.25rem; font-weight: 700; color: {TEXT_PRI}; letter-spacing: -0.3px; }}
.insightera-tagline {{ font-size: 0.75rem; color: {TEXT_MUTED}; }}
/* Dash DataTable dark override */
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td,
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {{
    background-color: {BG_CARD} !important;
    color: {TEXT_SEC} !important;
    border-color: {BORDER} !important;
}}
/* DataTable pagination inputs — page number box */
.dash-table-container .page-number input[type="number"],
.dash-table-container input.current-page,
.dash-table-container .current-page input,
.previous-next-container input[type="number"],
.dash-table-container input[type="number"] {{
    background-color: {BG_INPUT} !important;
    color: {TEXT_PRI} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    min-height: unset !important;
    width: 60px !important;
    text-align: center !important;
}}
/* Badge */
.filter-badge {{
    display: inline-block; background: {_rgba(BRAND_BLUE, 0.18)};
    color: {BRAND_BLUE}; border: 1px solid {_rgba(BRAND_BLUE, 0.3)};
    border-radius: 20px; padding: 2px 10px; font-size: 11px; font-weight: 500;
}}
/* Theme toggle button */
.theme-toggle-btn {{
    background: transparent; border: 1px solid {_rgba(BRAND_BLUE, 0.35)};
    border-radius: 8px; color: {TEXT_SEC}; padding: 4px 14px;
    cursor: pointer; font-size: 16px; line-height: 1; transition: all 0.2s;
    margin-left: auto;
}}
.theme-toggle-btn:hover {{ border-color: {BRAND_BLUE}; color: {BRAND_BLUE}; }}
/* ── Light mode overrides (same as default — default IS light) ──────── */
body.light-mode, html.light-mode {{
    background-color: {BG_PAGE} !important;
    color: {TEXT_PRI} !important;
}}
.light-mode .analytics-card {{
    background-color: {BG_CARD} !important;
    border-color: {BORDER} !important;
}}
.light-mode .insightera-navbar {{
    background: linear-gradient(135deg, {_rgba(BRAND_BLUE, 0.08)} 0%, {_rgba(BRAND_PURPLE, 0.05)} 100%) !important;
    border-bottom-color: {BORDER_LIT} !important;
}}
.light-mode h4, .light-mode h6 {{ color: {TEXT_PRI} !important; }}
.light-mode p {{ color: {TEXT_SEC} !important; }}
.light-mode label, .light-mode span {{ color: {TEXT_SEC} !important; }}
.light-mode .filter-badge {{ background: {_rgba(BRAND_BLUE, 0.10)} !important; }}
.light-mode ::-webkit-scrollbar-track {{ background: {BG_SURFACE} !important; }}
/* Inputs + dropdowns in light mode → black text on white */
.light-mode textarea,
.light-mode input[type="text"],
.light-mode input[type="number"],
.light-mode input[type="search"] {{
    background-color: #FFFFFF !important;
    color: #1A2845 !important;
    border-color: rgba(75,99,245,0.30) !important;
}}
/* Dash 4 CSS variables for light mode */
.light-mode {{
    --Dash-Fill-Inverse-Strong: #FFFFFF;
    --Dash-Stroke-Strong: rgba(75,99,245,0.30);
    --Dash-Text-Primary: #1A2845;
    --Dash-Text-Strong: #1A2845;
    --Dash-Text-Weak: #3A5278;
    --Dash-Text-Disabled: #8B9ABD;
    --Dash-Fill-Interactive-Weak: rgba(75,99,245,0.08);
    --Dash-Shading-Strong: rgba(0,0,0,0.12);
    --Dash-Shading-Weak: rgba(0,0,0,0.05);
}}
.light-mode .dash-dropdown,
.light-mode .dash-dropdown-trigger,
.light-mode .dash-dropdown-value,
.light-mode .dash-dropdown-search-container input,
.light-mode .dash-input-container {{ color: #1A2845 !important; }}
.light-mode .dash-dropdown-content {{ background: #FFFFFF !important; }}
.light-mode .dash-options-list-option {{
    background: #FFFFFF !important;
    color: #1A2845 !important;
}}
.light-mode .dash-options-list-option:not(:has(input[disabled])):hover,
.light-mode .dash-options-list-option:not(:has(input[disabled])):focus-within {{
    color: #4B63F5 !important;
}}
.light-mode .dash-dropdown-search-container {{ background: #F5F7FF !important; }}
/* DataTable light mode */
.light-mode .dash-table-container .page-number input[type="number"],
.light-mode .dash-table-container input.current-page,
.light-mode .previous-next-container input[type="number"],
.light-mode .dash-table-container input[type="number"] {{
    background-color: #FFFFFF !important;
    color: #1A2845 !important;
    border-color: rgba(75,99,245,0.30) !important;
}}
.light-mode .dash-table-container .dash-spreadsheet-inner td,
.light-mode .dash-table-container .dash-spreadsheet-inner th {{
    background-color: #FFFFFF !important;
    color: #1A2845 !important;
    border-color: rgba(75,99,245,0.15) !important;
}}
/* DatePicker light mode */
.light-mode .DateInput, .light-mode .DateInput_input,
.light-mode .DateRangePickerInput, .light-mode .SingleDatePickerInput {{
    background-color: #FFFFFF !important;
    color: #1A2845 !important;
    border-color: rgba(75,99,245,0.30) !important;
}}
.light-mode .DayPicker, .light-mode .CalendarMonthGrid,
.light-mode .CalendarMonth, .light-mode .DayPicker_weekHeader {{
    background-color: #FFFFFF !important;
    color: #1A2845 !important;
}}
.light-mode .CalendarDay__default {{
    background-color: #F5F7FF !important;
    color: #1A2845 !important;
}}
.light-mode .CalendarMonth_caption {{ color: #1A2845 !important; }}
/* FullStory journey viewer styles */
.journey-profile-panel {{
    background: {BG_SURFACE}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 16px; height: 100%;
}}
.light-mode .journey-profile-panel {{
    background: #EEF2FF !important; border-color: rgba(75,99,245,0.18) !important;
}}
.journey-profile-label {{
    color: {TEXT_MUTED}; font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.5px; display: block; margin-bottom: 2px;
}}
.journey-profile-value {{
    color: {TEXT_PRI}; font-size: 0.88rem; font-weight: 600; margin-bottom: 10px;
}}
.light-mode .journey-profile-label {{ color: #8A9ABB !important; }}
.light-mode .journey-profile-value {{ color: #1A2845 !important; }}
/* ── Sticky navbar ─────────────────────────────────────── */
.insightera-navbar {{
    position: sticky !important; top: 0 !important;
    z-index: 1000 !important;
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
}}
/* ── Section quick-nav bar ─────────────────────────────── */
.section-nav-bar {{
    display: flex; overflow-x: auto; gap: 5px;
    padding: 5px 24px 7px;
    background: {BG_SURFACE}; border-bottom: 1px solid {BORDER};
    position: sticky; top: 52px; z-index: 999;
    scrollbar-width: none; -ms-overflow-style: none;
    margin-bottom: 16px;
}}
.section-nav-bar::-webkit-scrollbar {{ display: none; }}
.section-nav-link {{
    color: {TEXT_SEC}; text-decoration: none;
    font-size: 10.5px; padding: 3px 9px;
    border-radius: 10px; border: 1px solid {BORDER};
    white-space: nowrap; transition: all 0.15s;
    font-family: Inter, system-ui, sans-serif;
}}
.section-nav-link:hover {{
    color: {BRAND_BLUE}; border-color: {BRAND_BLUE};
    background: {_rgba(BRAND_BLUE, 0.08)}; text-decoration: none;
}}
/* ── Collapsible section header ─────────────────────────── */
.section-collapse-btn {{
    background: transparent; border: 1px solid {BORDER};
    border-radius: 6px; color: {TEXT_MUTED};
    padding: 2px 8px; font-size: 14px; font-weight: 700;
    cursor: pointer; line-height: 1; flex-shrink: 0;
    margin-left: 10px; margin-top: 4px; transition: all 0.15s;
}}
.section-collapse-btn:hover {{ border-color: {BRAND_BLUE}; color: {BRAND_BLUE}; }}
.section-header-row {{
    display: flex; align-items: flex-start;
    justify-content: space-between; margin-bottom: 0;
}}
/* ══ MAIN TABS ════════════════════════════════════════════ */
.main-nav-tabs {{
    border-bottom: 1px solid {_rgba(BRAND_BLUE, 0.20)} !important;
    margin-bottom: 20px;
    flex-wrap: nowrap;
    overflow-x: auto;
}}
.main-nav-tabs .nav-link {{
    color: {TEXT_SEC} !important; font-size: 13px; font-weight: 500;
    padding: 10px 18px;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 8px 8px 0 0;
    white-space: nowrap;
    transition: background 0.15s, color 0.15s;
}}
.main-nav-tabs .nav-link:hover {{
    color: {TEXT_PRI} !important;
    background: {_rgba(BRAND_BLUE, 0.10)} !important;
    text-decoration: none;
}}
.main-nav-tabs .nav-link.active {{
    color: #fff !important;
    background: {BRAND_BLUE} !important;
    border-bottom: 2px solid {BRAND_BLUE} !important;
    font-weight: 700;
    box-shadow: 0 2px 8px {_rgba(BRAND_BLUE, 0.30)};
}}
.tab-pane-pad {{ padding-top: 4px; }}
/* ── KPI alert badge ──────────────────────────────────── */
.kpi-alert {{ display: inline-block; border-radius: 4px; padding: 1px 6px;
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.3px; margin-bottom: 3px; }}
/* ── Recommendation card section ────────────────────── */
.rec-card {{ transition: border-color 0.15s; }}
.rec-card:hover {{ border-color: {BRAND_BLUE} !important; }}
/* ══════════════════════════════════════════════════════
   GRID ALIGNMENT — equal-height cards within every row */
.row > [class*="col"] {{ display: flex; flex-direction: column; }}
.analytics-card {{ flex: 1 1 auto; }}
/* ══ DARK MODE overrides ════════════════════════════════ */
body.dark-mode {{
    background-color: #060917 !important;
    color: #E8EDFF !important;
    --Dash-Fill-Inverse-Strong: #0A0F22;
    --Dash-Stroke-Strong: rgba(75,99,245,0.16);
    --Dash-Stroke-Weak: rgba(75,99,245,0.08);
    --Dash-Text-Primary: #E8EDFF;
    --Dash-Text-Strong: #E8EDFF;
    --Dash-Text-Weak: #8BA2D3;
    --Dash-Text-Disabled: #455278;
    --Dash-Fill-Interactive-Strong: #4B63F5;
    --Dash-Fill-Interactive-Weak: rgba(75,99,245,0.12);
    --Dash-Shading-Strong: rgba(0,0,0,0.6);
    --Dash-Shading-Weak: rgba(0,0,0,0.3);
}}
.dark-mode .analytics-card {{ background-color: #0C1228 !important; border-color: rgba(75,99,245,0.16) !important; }}
.dark-mode .insightera-navbar {{
    background: linear-gradient(135deg, rgba(75,99,245,0.15) 0%, rgba(147,51,234,0.10) 100%) !important;
    border-bottom-color: rgba(75,99,245,0.40) !important;
}}
.dark-mode h4, .dark-mode h5, .dark-mode h6 {{ color: #E8EDFF !important; }}
.dark-mode p {{ color: #8BA2D3 !important; }}
.dark-mode label {{ color: #8BA2D3 !important; }}
.dark-mode .filter-badge {{ background: rgba(75,99,245,0.18) !important; color: #4B63F5 !important; }}
.dark-mode ::-webkit-scrollbar-track {{ background: #060917 !important; }}
.dark-mode ::-webkit-scrollbar-thumb {{ background: rgba(75,99,245,0.35) !important; }}
.dark-mode textarea, .dark-mode input[type="text"],
.dark-mode input[type="number"], .dark-mode input[type="search"],
.dark-mode input[type="email"] {{
    background-color: #0A0F22 !important; color: #E8EDFF !important;
    border-color: rgba(75,99,245,0.16) !important;
}}
.dark-mode .dash-dropdown, .dark-mode .dash-dropdown-trigger,
.dark-mode .dash-dropdown-value, .dark-mode .dash-input-container {{ color: #E8EDFF !important; }}
.dark-mode .dash-dropdown-placeholder {{ color: #455278 !important; }}
.dark-mode .dash-dropdown-content {{ background: #111C3A !important; border-color: rgba(75,99,245,0.16) !important; }}
.dark-mode .dash-options-list-option {{ background: #111C3A !important; color: #8BA2D3 !important; }}
.dark-mode .dash-options-list-option:not(:has(input[disabled])):hover,
.dark-mode .dash-options-list-option:not(:has(input[disabled])):focus-within {{ color: #4B63F5 !important; }}
.dark-mode .dash-dropdown-search-container {{ background: #0A0F22 !important; border-color: rgba(75,99,245,0.16) !important; }}
.dark-mode .dash-dropdown-search-container input {{ color: #E8EDFF !important; background: transparent !important; }}
.dark-mode .form-check-label {{ color: #8BA2D3 !important; }}
.dark-mode .form-check-input {{ background-color: #0A0F22 !important; border-color: rgba(75,99,245,0.4) !important; }}
.dark-mode .form-check-input:checked {{ background-color: #4B63F5 !important; border-color: #4B63F5 !important; }}
.dark-mode .dash-table-container .dash-spreadsheet-inner td,
.dark-mode .dash-table-container .dash-spreadsheet-inner th {{
    background-color: #0C1228 !important; color: #8BA2D3 !important; border-color: rgba(75,99,245,0.16) !important;
}}
.dark-mode .dash-table-container input[type="number"],
.dark-mode .previous-next-container input[type="number"] {{
    background-color: #0A0F22 !important; color: #E8EDFF !important; border-color: rgba(75,99,245,0.16) !important;
}}
.dark-mode .DateInput, .dark-mode .DateInput_input {{ background-color: #0A0F22 !important; color: #E8EDFF !important; }}
.dark-mode .DateRangePickerInput, .dark-mode .SingleDatePickerInput {{ background-color: #0A0F22 !important; border-color: rgba(75,99,245,0.16) !important; }}
.dark-mode .DayPicker, .dark-mode .CalendarMonthGrid, .dark-mode .CalendarMonth,
.dark-mode .DayPicker_weekHeader {{ background-color: #111C3A !important; color: #E8EDFF !important; }}
.dark-mode .CalendarDay__default {{ background-color: #111C3A !important; color: #8BA2D3 !important; border-color: rgba(75,99,245,0.12) !important; }}
.dark-mode .CalendarMonth_caption {{ color: #E8EDFF !important; }}
.dark-mode .CalendarDay__selected, .dark-mode .CalendarDay__selected:hover {{ background-color: #4B63F5 !important; color: #fff !important; }}
.dark-mode .journey-profile-panel {{ background: #111C3A !important; border-color: rgba(75,99,245,0.16) !important; }}
.dark-mode .journey-profile-label {{ color: #455278 !important; }}
.dark-mode .journey-profile-value {{ color: #E8EDFF !important; }}
.dark-mode .main-nav-tabs {{ border-bottom-color: rgba(75,99,245,0.20) !important; }}
.dark-mode .main-nav-tabs .nav-link {{ color: #8BA2D3 !important; }}
.dark-mode .main-nav-tabs .nav-link:hover {{ color: #E8EDFF !important; background: rgba(75,99,245,0.10) !important; }}
.dark-mode .main-nav-tabs .nav-link.active {{ color: #fff !important; background: #4B63F5 !important; border-bottom-color: #4B63F5 !important; font-weight: 700; box-shadow: 0 2px 8px rgba(75,99,245,0.35); }}
.dark-mode .rc-slider-mark-text {{ color: #8BA2D3 !important; }}
.dark-mode .rc-slider-dot {{ border-color: rgba(75,99,245,0.5) !important; background-color: #111C3A !important; }}
.dark-mode .rc-slider-track {{ background-color: #4B63F5 !important; }}
.dark-mode .rc-slider-handle {{ border-color: #4B63F5 !important; background-color: #4B63F5 !important; }}
"""


# ══════════════════════════════════════════════════════════════
#  DASH APP
# ══════════════════════════════════════════════════════════════
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Insightera · User Intelligence",
)
server = app.server  # exposed for gunicorn

app.index_string = f'''<!DOCTYPE html>
<html>
<head>
{{%metas%}}
<title>{{%title%}}</title>
{{%favicon%}}
{{%css%}}
<style>
{CUSTOM_CSS}
</style>
</head>
<body>
{{%app_entry%}}
<footer>
{{%config%}}
{{%scripts%}}
{{%renderer%}}
</footer>
</body>
</html>'''


# ══════════════════════════════════════════════════════════════
#  NEW: GITHUB-STYLE EVENT CALENDAR
# ══════════════════════════════════════════════════════════════
def build_event_calendar():
    """GitHub-style 52-week × 7-day event volume heatmap."""
    ud = events_raw[["timestamp"]].copy()
    ud["date"] = ud["timestamp"].dt.normalize()
    daily = ud.groupby("date").size().reset_index(name="count")

    end_date   = daily["date"].max()
    start_date = end_date - pd.Timedelta(weeks=52)
    all_dates  = pd.date_range(start=start_date, end=end_date, freq="D")
    daily = daily.set_index("date").reindex(all_dates, fill_value=0).reset_index()
    daily.columns = ["date", "count"]

    daily["week"] = (daily["date"] - daily["date"].min()).dt.days // 7
    daily["dow"]  = daily["date"].dt.dayofweek   # 0=Mon, 6=Sun

    n_weeks = int(daily["week"].max()) + 1
    z     = np.zeros((7, n_weeks))
    hover = [["" for _ in range(n_weeks)] for _ in range(7)]

    for _, row in daily.iterrows():
        w, d = int(row["week"]), int(row["dow"])
        if w < n_weeks:
            z[d, w] = row["count"]
            hover[d][w] = f"{row['date'].strftime('%d %b %Y')}: {int(row['count']):,} events"

    # Month label positions
    week_starts = [daily["date"].min() + pd.Timedelta(weeks=i) for i in range(n_weeks)]
    month_ticks, month_texts = [], []
    prev_m = None
    for i, d in enumerate(week_starts):
        if d.month != prev_m:
            month_ticks.append(i)
            month_texts.append(d.strftime("%b"))
            prev_m = d.month

    fig = go.Figure(go.Heatmap(
        z=z,
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        colorscale=[
            [0.00, BG_SURFACE],
            [0.01, _rgba(BRAND_BLUE, 0.20)],
            [0.30, _rgba(BRAND_BLUE, 0.55)],
            [0.70, BRAND_BLUE],
            [1.00, BRAND_PURPLE],
        ],
        showscale=False,
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        **_cdefaults(height=160, margin=dict(l=42, r=14, t=24, b=16)),
    )
    fig.update_yaxes(
        tickvals=list(range(7)),
        ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        tickfont=dict(size=9, color=TEXT_MUTED),
        gridcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(
        tickvals=month_ticks,
        ticktext=month_texts,
        tickfont=dict(size=9, color=TEXT_MUTED),
        gridcolor="rgba(0,0,0,0)",
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  NEW: LTV EARLY-SIGNAL PREDICTION CHART
# ══════════════════════════════════════════════════════════════
def build_ltv_early_signals():
    """
    Bar + line chart: Avg LTV vs number of events in first 7 days.
    Shows how early engagement predicts lifetime revenue.
    """
    up_ = user_profiles[["first_event", "ltv"]].copy()
    ev  = events[["user_id", "timestamp"]].copy()
    ev  = ev.join(up_[["first_event"]], on="user_id")
    ev  = ev[ev["timestamp"].notna() & ev["first_event"].notna()].copy()
    ev["days_since_start"] = (ev["timestamp"] - ev["first_event"]).dt.days
    d7  = ev[ev["days_since_start"] <= 7].groupby("user_id").size().rename("d7_events")

    up_ = up_.join(d7, how="left")
    up_["d7_events"] = up_["d7_events"].fillna(0)

    bins   = [0, 1, 5, 10, 20, 50, np.inf]
    labels = ["0", "1–4", "5–9", "10–19", "20–49", "50+"]
    up_["d7_bin"] = pd.cut(up_["d7_events"], bins=bins, labels=labels, right=False)

    stats = up_.groupby("d7_bin", observed=True).agg(
        avg_ltv=("ltv", "mean"),
        n_users=("ltv", "count"),
    ).reset_index()

    n = len(stats)
    colors = [_rgba(BRAND_BLUE, 0.30 + 0.70 * i / max(n - 1, 1)) for i in range(n)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=stats["d7_bin"].astype(str), y=stats["avg_ltv"], name="Avg LTV",
        marker_color=colors,
        text=[f"${v:.0f}" for v in stats["avg_ltv"]],
        textposition="outside", textfont=dict(size=9, color=TEXT_SEC),
        hovertemplate="D7 Events: %{x}<br>Avg LTV: $%{y:.2f}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=stats["d7_bin"].astype(str), y=stats["n_users"], name="# Users",
        mode="lines+markers",
        line=dict(color=BRAND_AMBER, width=2, dash="dot"),
        marker=dict(size=5),
        hovertemplate="D7 Events: %{x}<br>Users: %{y:,}<extra></extra>",
    ), secondary_y=True)
    fig.update_layout(**_cdefaults(height=300), xaxis_title="Events in First 7 Days after Signup")
    fig.update_yaxes(title_text="Avg LTV ($)", tickprefix="$", secondary_y=False,
                     tickfont=dict(color=TEXT_SEC, size=10))
    fig.update_yaxes(title_text="# Users", secondary_y=True,
                     tickfont=dict(color=BRAND_AMBER, size=10))
    return fig


# ══════════════════════════════════════════════════════════════
#  NEW: TTV DISTRIBUTION HISTOGRAM
# ══════════════════════════════════════════════════════════════
def build_ttv_histogram():
    """
    Histogram of days-from-signup to first payment.
    Reveals whether most users convert quickly or have a long tail.
    """
    pay_ev    = events[events["event_name"] == "payment_success"]
    first_pay = pay_ev.groupby("user_id")["timestamp"].min()
    first_ev  = user_profiles["first_event"]
    days_to_pay = ((first_pay - first_ev).dt.days).dropna()
    days_to_pay = days_to_pay[days_to_pay >= 0]

    if days_to_pay.empty:
        return _empty_fig("No payment data")

    p50 = float(days_to_pay.median())
    p75 = float(days_to_pay.quantile(0.75))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=days_to_pay.clip(upper=90),
        nbinsx=30,
        name="Users",
        marker_color=_rgba(BRAND_TEAL, 0.72),
        marker_line=dict(color=_rgba(BRAND_TEAL, 0.9), width=1),
        hovertemplate="Days: %{x}<br>Users: %{y:,}<extra></extra>",
    ))
    fig.add_vline(x=p50, line=dict(color=BRAND_TEAL, width=2, dash="dot"),
                  annotation_text=f"p50: {p50:.0f}d",
                  annotation_font=dict(color=BRAND_TEAL, size=10),
                  annotation_position="top right")
    fig.add_vline(x=p75, line=dict(color=BRAND_AMBER, width=2, dash="dot"),
                  annotation_text=f"p75: {p75:.0f}d",
                  annotation_font=dict(color=BRAND_AMBER, size=10),
                  annotation_position="top right")
    fig.update_layout(
        **_cdefaults(height=300,
                     xaxis_title="Days from Signup to First Payment (capped at 90)",
                     yaxis_title="# Users"),
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  NEW: RULE-BASED RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
def _compute_recommendations():
    """
    Generate a prioritised list of rule-based PM recommendations.
    Returns list of dicts: {priority, title, desc, metric, href}.
    """
    recs = []
    up_  = user_profiles
    total = max(len(up_), 1)

    activation_rate = up_["completed_onboarding"].mean() * 100
    conversion_rate = up_["converted"].mean() * 100
    churn_pct       = up_["churned"].mean() * 100
    last_ratio      = dau_mau_df["ratio"].iloc[-1] if len(dau_mau_df) else 0

    # — Activation
    if activation_rate < 30:
        recs.append({"priority": "critical",
            "title": "Critically low onboarding completion",
            "desc": f"Only {activation_rate:.0f}% of users complete onboarding. "
                    "Review the Conversion Funnel for the biggest drop-off step "
                    "and simplify the first three required actions.",
            "metric": f"{activation_rate:.0f}% activation",
            "href": "#sec-wrapper-conversion"})
    elif activation_rate < 60:
        recs.append({"priority": "high",
            "title": "Onboarding completion below target",
            "desc": f"{activation_rate:.0f}% activation rate. "
                    "The Aha Moment chart shows which feature most separates "
                    "retained from churned users — add it to onboarding.",
            "metric": f"{activation_rate:.0f}% activation",
            "href": "#sec-wrapper-activation"})

    # — DAU/MAU
    if last_ratio < 0.10:
        recs.append({"priority": "critical",
            "title": "Very low engagement depth (DAU/MAU < 10%)",
            "desc": f"DAU/MAU is {last_ratio:.2f}. Most monthly users are not returning daily. "
                    "Investigate stickiness patterns in §10 Value & Retention Health.",
            "metric": f"{last_ratio:.2f} DAU/MAU",
            "href": "#sec-wrapper-value"})
    elif last_ratio < 0.20:
        recs.append({"priority": "medium",
            "title": "Engagement depth below healthy threshold",
            "desc": f"DAU/MAU of {last_ratio:.2f} (target: >20%). "
                    "Cluster analysis in §5 can reveal which user segments drive return visits.",
            "metric": f"{last_ratio:.2f} DAU/MAU",
            "href": "#sec-wrapper-value"})

    # — Churn
    if churn_pct > 40:
        recs.append({"priority": "critical",
            "title": "High churn rate",
            "desc": f"{churn_pct:.0f}% of users have churned. "
                    "See Churn Risk Deep-Dive for the at-risk list "
                    "and the 'What Did Churned Users Miss?' behaviour comparison.",
            "metric": f"{churn_pct:.0f}% churned",
            "href": "#sec-wrapper-churn"})
    elif churn_pct > 25:
        recs.append({"priority": "high",
            "title": "Elevated churn rate",
            "desc": f"{churn_pct:.0f}% churn. "
                    "The biggest behavioural gaps between active and churned users "
                    "are shown in §4 — address the top gap first.",
            "metric": f"{churn_pct:.0f}% churned",
            "href": "#sec-wrapper-churn"})

    # — Conversion
    if conversion_rate < 5:
        recs.append({"priority": "high",
            "title": "Low paid conversion rate",
            "desc": f"Only {conversion_rate:.1f}% of users convert to paying. "
                    "The Conversion Lift chart shows which events are highest-signal "
                    "predictors of payment — focus on driving those actions earlier.",
            "metric": f"{conversion_rate:.1f}% conversion",
            "href": "#sec-wrapper-conversion"})

    # — Revenue concentration
    pay_ = up_[up_["total_revenue"] > 0]["total_revenue"]
    if len(pay_) > 10:
        top10_rev = pay_.nlargest(max(1, len(pay_) // 10)).sum()
        top10_pct = top10_rev / pay_.sum() * 100
        if top10_pct > 80:
            recs.append({"priority": "medium",
                "title": "Revenue concentrated in top 10% of payers",
                "desc": f"Top 10% of payers generate {top10_pct:.0f}% of revenue — "
                        "a fragile base. Focus on moving Mid Revenue users up-tier "
                        "via targeted feature campaigns.",
                "metric": f"{top10_pct:.0f}% in top 10%",
                "href": "#sec-wrapper-revenue"})

    # — Feature adoption
    feat_events = [("view_feature_A","Feature A"), ("view_feature_B","Feature B"),
                   ("view_feature_C","Feature C"), ("view_feature_D","Feature D")]
    for fe, fname in feat_events:
        adopters   = events[events["event_name"] == fe]["user_id"].nunique()
        adopt_pct  = adopters / total * 100
        if adopt_pct < 15:
            recs.append({"priority": "medium",
                "title": f"Low adoption of {fname}",
                "desc": f"Only {adopt_pct:.0f}% of users have used {fname}. "
                        "If this feature drives LTV (see Feature → Revenue Impact), "
                        "consider surfacing it earlier in onboarding.",
                "metric": f"{adopt_pct:.0f}% adoption",
                "href": "#sec-wrapper-activation"})
            break   # one feature rec at a time

    priority_order = {"critical": 0, "high": 1, "medium": 2}
    recs.sort(key=lambda r: priority_order.get(r["priority"], 99))
    # Inject target tab based on section href
    _SEC_TO_TAB = {
        "sec-wrapper-conversion": "conversion", "sec-wrapper-activation": "conversion",
        "sec-wrapper-revenue": "conversion", "sec-wrapper-churn": "churn",
        "sec-wrapper-value": "engagement", "sec-wrapper-kpi": "overview",
        "sec-wrapper-health": "overview", "sec-wrapper-opportunity": "overview",
        "sec-wrapper-goals": "overview",
    }
    for r in recs:
        sec = r.get("href", "#").lstrip("#")
        r["tab"] = _SEC_TO_TAB.get(sec, "overview")
    return recs


def build_recommendations_section(recs):
    """Render prioritised recommendation cards."""
    if not recs:
        return html.Div()

    priority_cfg = {
        "critical": (COL_LO,      "⚠ Critical"),
        "high":     (BRAND_AMBER, "↑ High"),
        "medium":   (BRAND_BLUE,  "→ Medium"),
    }
    # Build target map for clientside callback
    rec_targets = []
    cards = []
    for i, rec in enumerate(recs):
        color, badge = priority_cfg.get(rec["priority"], (TEXT_SEC, "—"))
        rec_targets.append({"tab": rec.get("tab", "overview"),
                            "section": rec.get("href", "#").lstrip("#")})
        cards.append(dbc.Col(html.Div([
            html.Div([
                html.Span(badge, style={
                    "backgroundColor": _rgba(color, 0.15),
                    "border": f"1px solid {_rgba(color, 0.40)}",
                    "color": color, "borderRadius": "12px",
                    "padding": "2px 8px", "fontSize": "9px", "fontWeight": "700",
                    "textTransform": "uppercase", "letterSpacing": "0.4px",
                    "marginBottom": "6px", "display": "inline-block",
                }),
                html.Span(rec.get("metric", ""), style={
                    "color": TEXT_MUTED, "fontSize": "9px", "marginLeft": "7px",
                }),
            ]),
            html.Div(rec["title"], style={
                "color": TEXT_PRI, "fontWeight": "700",
                "fontSize": "0.82rem", "marginBottom": "5px", "lineHeight": "1.25",
            }),
            html.Div(rec["desc"], style={
                "color": TEXT_SEC, "fontSize": "11px", "lineHeight": "1.45",
                "marginBottom": "8px",
            }),
            html.Button("→ Go to section",
                id={"type": "rec-go", "index": i},
                style={"background": "none", "border": "none", "cursor": "pointer",
                       "color": color, "fontSize": "10px", "padding": "0",
                       "textDecoration": "none", "fontWeight": "600"}),
        ], className="rec-card", style={
            "backgroundColor": BG_SURFACE,
            "border": f"1px solid {BORDER}",
            "borderLeft": f"3px solid {color}",
            "borderRadius": "10px",
            "padding": "12px 14px",
            "height": "100%",
        }), md=4, sm=6, className="mb-2"))

    return html.Div([
        html.Div([
            html.H4("Recommendations", style=DIVIDER_H),
            html.P(
                "Rule-based insights generated from your current metrics. "
                "Sorted by priority. Click any card to jump to the relevant section.",
                style=DIVIDER_SUB,
            ),
        ]),
        dbc.Row(cards),
        dcc.Store(id="rec-targets-store", data=rec_targets),
    ], style={"marginBottom": "20px"})


# Pre-build static charts
_sankey_default       = build_rete_sankey()
_bridge_fig           = build_conversion_lift()
_decliners_fig        = build_conversion_decliners()
_matched_lift_fig     = build_engagement_matched_lift()
_neg_impact_fig       = build_negative_signal_impact()
_ltv_fig         = build_ltv_distribution()
_radar_fig       = build_segment_radar()
_seg_kpi_fig     = build_revenue_segment_kpis()
_journey_cmp     = build_revenue_journey_comparison()
_onboard_fig     = build_onboarding_funnel()
_cluster_fig, _cluster_summary, _cluster_user_map_raw, _cluster_k_scores = build_clustering(DEFAULT_CLUSTER_FEATS)
# Rename cluster keys from "Cluster 1" → archetype names
_cluster_archetypes = _generate_cluster_archetypes(_cluster_summary, DEFAULT_CLUSTER_FEATS)
_cluster_user_map = {}
_raw_keys = list(_cluster_user_map_raw.keys())
for _i, (_aname, _adesc) in enumerate(_cluster_archetypes):
    _clabel = _raw_keys[_i] if _i < len(_raw_keys) else f"Cluster {_i + 1}"
    _cluster_user_map[f"{_aname} ({_clabel})"] = _cluster_user_map_raw.get(_clabel, [])
_friction_fig         = build_friction_score()
_friction_breakdown   = build_friction_breakdown()
_friction_seg_cmp     = build_friction_segment_comparison()
_comeback_fig         = build_comeback_loops()
_dropoff_fig          = build_dropoff_risk_map()
_opp_matrix_fig  = build_opportunity_matrix()
_kpi_row         = build_kpi_row(30)   # default: compare vs prior month
# New consolidated charts
print("Building new chart figures …")
_trans_table     = build_transition_table()       # replaces heatmap
_avg_time_fig    = build_avg_time_between()
_step_dist_fig   = build_step_distribution()
_self_loop_fig   = build_self_loop_chart()
_cohort_ret_fig  = build_cohort_retention()       # heatmap now
_mrr_fig         = build_mrr_trend()
_feat_adopt_fig  = build_feature_adoption()
_ttv_fig         = build_time_to_value()          # callback-driven; this is default
_activity_hmap   = build_activity_heatmap()
_rev_conc_fig    = build_revenue_concentration()
_sess_dur_fig    = build_session_duration()
_activation_fig  = build_activation_health()
_stickiness_fig  = build_stickiness_trend()
_churn_beh_fig        = build_churn_behavior_comparison()
_churn_clust_fig      = build_churn_vs_clusters()
_at_risk_table        = build_at_risk_table()

# SHAP churn prediction model
print("Training churn prediction model …")
_shap_fig, _shap_auc, _shap_records, _churn_clf, _churn_feat_cols, _shap_sv, _shap_X_sample = _build_churn_model()
_shap_feature_table = build_shap_feature_table(_shap_records)

# ── New features: churn intelligence & dashboard enhancements ────────────
print("Computing churn probabilities & health scores …")
_health_scores     = _compute_health_scores()
_health_dist_fig   = build_health_score_distribution(_health_scores)
_churn_probs       = _compute_churn_probabilities(_churn_clf, _churn_feat_cols)
_churn_prob_hist   = build_churn_prob_histogram(_churn_probs)
_next_best_actions = _compute_next_best_actions(_churn_clf, _churn_feat_cols, _shap_sv, _shap_X_sample)
_survival_fig, _median_survival = build_survival_curve()
_cohort_cmp_fig, _cohort_cmp_rows = build_churn_cohort_comparison()
_winback_records   = _compute_winback_scores()
_winback_tbl       = build_winback_table(_winback_records)
_DAU_MAU_RATIO     = dau_mau_df["ratio"].iloc[-1] if len(dau_mau_df) else 0
_goal_tracking     = build_goal_tracking()
_feature_funnel_fig = build_feature_adoption_funnel()
_rev_impact_fig, _rev_impact_rows = build_revenue_impact_attribution()

# New improvement charts
print("Building improvement charts …")
_event_calendar_fig  = build_event_calendar()
_ltv_signals_fig     = build_ltv_early_signals()
_ttv_hist_fig        = build_ttv_histogram()
_recommendations     = _compute_recommendations()
_recs_section        = build_recommendations_section(_recommendations)

# PM Intelligence — pre-build
print("Building PM intelligence figures …")
_health_trends_row    = build_product_health_trends()
_aha_fig              = build_aha_moment_chart()
_adopt_timing_fig     = build_adoption_timing_chart()
_feat_ltv_fig         = build_feature_ltv_impact()
_seg_funnel_fig       = build_segmented_funnel_chart("all")

# Churn KPI summary values
_at_risk_count        = _CHURN_RISK["at_risk_count"]
_active_count         = _CHURN_RISK["active_count"]
_risk_pct             = round(_at_risk_count / max(_active_count, 1) * 100, 1)
_never_onboarded_pct  = _CHURN_RISK["never_onboarded_pct"]
_post_paid_count      = _CHURN_RISK["post_paid_count"]
_post_paid_pct        = _CHURN_RISK["post_paid_pct"]

# Date range bounds (for filter panel)
_DATE_MIN = events_raw["timestamp"].min().date()
_DATE_MAX = events_raw["timestamp"].max().date()

# ── Pre-compute values for per-tab KPI strips ───────────────────────────────
_yesterday_cutoff  = pd.Timestamp(_DATE_MAX)
_dau_val           = int(events[events["timestamp"].dt.date == _DATE_MAX]["user_id"].nunique())
_conv_pct          = round(user_profiles["converted"].mean() * 100, 1)
_rev_total         = int(user_profiles["ltv"].sum())
_avg_ltv_val       = round(user_profiles[user_profiles["converted"] == 1]["ltv"].mean(), 0) \
                     if user_profiles["converted"].sum() > 0 else 0
_act_pct_val       = round(user_profiles["completed_onboarding"].mean() * 100, 1)
_ob_count_val      = int(user_profiles["completed_onboarding"].sum())
_epd_avg_val       = round(user_profiles["events_per_day"].mean(), 1)
_active_days_avg   = round(user_profiles["active_days"].mean(), 1)
_median_ttv_days   = (round(user_profiles["median_ttv"].dropna().median(), 1)
                      if "median_ttv" in user_profiles.columns else "—")


def _mini_kpi_card(label, value, sub):
    """Compact 3-line KPI tile for section tab headers."""
    return html.Div([
        html.Div(label, style={
            "color": TEXT_MUTED, "fontSize": "0.65rem",
            "textTransform": "uppercase", "letterSpacing": "0.5px",
            "marginBottom": "2px",
        }),
        html.Div(str(value), style={
            "color": TEXT_PRI, "fontSize": "1.35rem",
            "fontWeight": "700", "lineHeight": "1.15",
            "marginBottom": "2px",
        }),
        html.Div(sub, style={"color": TEXT_MUTED, "fontSize": "0.68rem"}),
    ], style={
        "backgroundColor": BG_SURFACE,
        "border": f"1px solid {BORDER}",
        "borderRadius": "10px",
        "padding": "10px 14px",
    })


_TAB_KPI_STRIPS = {
    "conversion": [
        ("Conversion %",  f"{_conv_pct}%",          "paid / signed up"),
        ("Total Revenue", f"${_rev_total:,}",        "all-time LTV sum"),
        ("Avg LTV",       f"${_avg_ltv_val:,.0f}",   "per paying user"),
    ],
    "engagement": [
        ("Total Users",  f"{total_users:,}",          "all-time signups"),
        ("Events / Day", str(_epd_avg_val),            "avg per user"),
        ("Active Days",  str(_active_days_avg),        "avg user lifetime"),
    ],
    "retention": [
        ("Activation %",  f"{_act_pct_val}%",          "completed onboarding"),
        ("Onboarded",     f"{_ob_count_val:,}",         "users"),
        ("Median TTV",    f"{_median_ttv_days}d",       "days to first payment"),
    ],
    "churn": [
        ("Churn Risk",   f"{_risk_pct}%",              "of active base"),
        ("At-Risk",      f"{_at_risk_count:,}",        "users flagged"),
    ],
}


def _tab_kpi_strip(section):
    """Compact KPI card row pinned to top of each section tab."""
    cards = _TAB_KPI_STRIPS.get(section, [])
    if not cards:
        return html.Div()
    return dbc.Row(
        [dbc.Col(_mini_kpi_card(lbl, val, sub), md=3, sm=6, className="mb-3")
         for lbl, val, sub in cards],
        className="g-3",
        style={"marginBottom": "4px"},
    )


# ── Custom user filter panel
def _filter_panel():
    return html.Div([
        # Row 1: Date range
        dbc.Row([
            dbc.Col([
                html.Label("Date Range",
                           style={"color": TEXT_MUTED, "fontSize": "0.7rem",
                                  "textTransform": "uppercase", "letterSpacing": "0.5px",
                                  "marginBottom": "6px", "display": "block"}),
                dcc.DatePickerRange(
                    id="date-range-picker",
                    min_date_allowed=_DATE_MIN,
                    max_date_allowed=_DATE_MAX,
                    start_date=_DATE_MIN,
                    end_date=_DATE_MAX,
                    display_format="DD MMM YYYY",
                    style={"backgroundColor": BG_INPUT},
                    className="dark-datepicker",
                ),
                html.Div("Applied to: All charts across all tabs",
                         style={"color": TEXT_MUTED, "fontSize": "0.65rem",
                                "marginTop": "4px"}),
            ], md=5),
            dbc.Col([
                html.Label("Segment Filter",
                           style={"color": TEXT_MUTED, "fontSize": "0.7rem",
                                  "textTransform": "uppercase", "letterSpacing": "0.5px",
                                  "marginBottom": "6px", "display": "block"}),
                dcc.Dropdown(
                    id="segment-dropdown",
                    options=SEGMENTS,
                    value="all",
                    clearable=False,
                    style={"backgroundColor": BG_INPUT, "color": TEXT_PRI,
                           "border": f"1px solid {BORDER}", "borderRadius": "8px"},
                ),
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Br(),
                    dbc.Button("Apply Filter", id="apply-filter-btn", color="primary",
                               size="sm", className="me-2",
                               style={"backgroundColor": BRAND_BLUE,
                                      "borderColor": BRAND_BLUE}),
                    dbc.Button("Reset", id="reset-filter-btn", color="secondary",
                               size="sm",
                               style={"backgroundColor": BG_SURFACE,
                                      "borderColor": BORDER}),
                ]),
                html.Div(id="filter-badge-text",
                         style={"marginTop": "8px", "fontSize": "11px",
                                "color": TEXT_MUTED}),
            ], md=2),
        ], className="g-3 mb-2"),
        # Row 2: Custom IDs
        dbc.Row([
            dbc.Col([
                html.Label("Custom User IDs (comma-separated integers)",
                           style={"color": TEXT_MUTED, "fontSize": "0.7rem",
                                  "textTransform": "uppercase", "letterSpacing": "0.5px",
                                  "marginBottom": "6px", "display": "block"}),
                dcc.Textarea(
                    id="custom-ids-input",
                    placeholder="e.g. 1, 42, 107, 3304 …",
                    rows=2,
                    style={"width": "100%", "backgroundColor": BG_INPUT,
                           "color": TEXT_PRI, "border": f"1px solid {BORDER}",
                           "borderRadius": "8px", "resize": "vertical",
                           "fontFamily": "Inter, monospace", "fontSize": "12px",
                           "padding": "8px 12px"},
                ),
            ], md=10),
        ], className="g-3"),
        dcc.Store(id="user-filter-store",
                  data={"mode": "all", "segment": "all", "user_ids": [],
                        "date_start": str(_DATE_MIN), "date_end": str(_DATE_MAX)}),
    ], style={**CARD_STYLE, "marginBottom": "20px"}, className="analytics-card")


# ── Layout
app.layout = dbc.Container([

    dcc.Store(id="theme-store", data="light"),

    # ── Brand bar ──────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("◆ ", style={"color": BRAND_BLUE, "fontSize": "1.1rem"}),
            html.Span("Insightera", style={"fontWeight": "700", "color": TEXT_PRI,
                                           "fontSize": "1.2rem"}),
            html.Span(" Analytics", style={"color": BRAND_TEAL,
                                            "fontSize": "1.2rem", "fontWeight": "300"}),
        ]),
        html.Div([
            html.Span("User Intelligence Platform · Powered by Retentioneering",
                      style={"color": TEXT_MUTED, "fontSize": "0.72rem"}),
            html.Span(f"  ·  Data as of {_DATE_MAX:%d %b %Y}",
                      style={"color": TEXT_MUTED, "fontSize": "0.65rem",
                             "marginLeft": "6px"}),
        ]),
        html.Div([
            html.Button("🌙", id="theme-toggle-btn", n_clicks=0,
                        className="theme-toggle-btn",
                        title="Toggle light / dark mode"),
            dbc.Button("⚙ Customize", id="customize-btn", size="sm", outline=True,
                       style={"borderColor": BORDER, "color": TEXT_MUTED,
                              "fontSize": "11px", "padding": "4px 10px",
                              "borderRadius": "6px", "marginLeft": "8px"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], className="insightera-navbar"),


    # ── Customize drawer ───────────────────────────────────────────────────
    dbc.Collapse(id="customize-drawer", is_open=False, children=[
        html.Div([
            html.Div([
                html.Span("KPI Metrics", style={"color": TEXT_PRI, "fontWeight": "600",
                                                 "fontSize": "0.8rem",
                                                 "textTransform": "uppercase",
                                                 "letterSpacing": "0.5px",
                                                 "marginBottom": "8px",
                                                 "display": "block"}),
                # Group headers + checklists for each card group
                html.Div([
                    html.Span("KPI Cards", style={"color": TEXT_MUTED, "fontSize": "10px",
                                                   "textTransform": "uppercase",
                                                   "letterSpacing": "0.4px",
                                                   "display": "block", "marginBottom": "4px"}),
                    dcc.Checklist(
                        id="section-checklist",
                        options=[{"label": f"  {label}", "value": cid}
                                 for cid, label in ALL_KPI_CARDS],
                        value=[cid for cid, _ in ALL_KPI_CARDS],
                        inputStyle={"marginRight": "6px", "accentColor": BRAND_BLUE},
                        labelStyle={"display": "inline-block", "marginRight": "16px",
                                    "marginBottom": "5px", "fontSize": "12px",
                                    "color": TEXT_SEC, "cursor": "pointer"},
                    ),
                ]),
            ], style={"padding": "14px 18px", "backgroundColor": BG_SURFACE,
                      "border": f"1px solid {BORDER}", "borderRadius": "10px",
                      "marginBottom": "12px"}),
        ]),
    ]),

    # Hidden stores
    dcc.Store(id="selected-at-risk-user"),
    dcc.Store(id="section-visibility", storage_type="local", data=None),

    # ── Recommendations ────────────────────────────────────────────────────
    html.Div(id="recs-section-wrapper", children=_recs_section),

    # ── Custom user filter (persistent above all tabs) ─────────────────────
    _filter_panel(),

    dbc.Tabs(id="main-tabs", active_tab="overview", className="main-nav-tabs",
             persistence=True, persistence_type="session", children=[

    # ══════════════════════════════════════════════════════════════
    #  TAB 1 — KPI OVERVIEW
    # ══════════════════════════════════════════════════════════════
    dbc.Tab(label="KPI Overview", tab_id="overview", className="tab-pane-pad", children=[
    html.Div([
        html.Div([
            html.Span("Customize Cards", style={
                "color": TEXT_MUTED, "fontSize": "0.67rem",
                "textTransform": "uppercase", "letterSpacing": "0.5px",
            }),
            dcc.RadioItems(
                id="period-window",
                options=[
                    {"label": " 1 Month",  "value": 30},
                    {"label": " 2 Months", "value": 60},
                    {"label": " Quarter",  "value": 90},
                ],
                value=30, inline=True,
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "14px", "fontSize": "12px",
                            "color": TEXT_SEC},
            ),
        ], style={
            "display": "flex", "alignItems": "center", "gap": "10px",
            "justifyContent": "flex-start", "marginBottom": "10px",
        }),
        html.Div(id="kpi-row-content", children=_kpi_row),
    ], id="sec-wrapper-kpi"),
    html.Div(id="sec-wrapper-health", children=_health_trends_row),

    # ── Improvement Opportunity Matrix ────────────────────────────────────
    _section_header("Improvement Opportunity Matrix",
        "Priority-ranked view of where product improvements will have the most impact."),
    html.Div([
    _card(
        "Priority Matrix — Friction × Traffic",
        "",
        [
            _opp_summary(),
            dbc.Row([
                dbc.Col([
                    html.Label("Y-axis:", style={
                        "color": TEXT_MUTED, "fontSize": "0.72rem",
                        "textTransform": "uppercase", "letterSpacing": "0.5px",
                        "lineHeight": "32px", "paddingTop": "2px"}),
                ], width="auto"),
                dbc.Col(dcc.RadioItems(
                    id="opp-y-axis",
                    options=[{"label": " # Users Affected", "value": "users"},
                             {"label": " LTV at Risk ($)", "value": "ltv"},
                             {"label": " Est. Conv Impact (pp)", "value": "impact"}],
                    value="users",
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "18px", "fontSize": "12px",
                                "color": TEXT_SEC},
                )),
            ], className="align-items-center mb-3"),
            dcc.Loading(
                dcc.Graph(id="opp-matrix-chart", figure=_opp_matrix_fig,
                          config={"displayModeBar": False}),
                type="dot", color=BRAND_BLUE,
            ),
            html.Div([
                html.H6("Priority Table — Top 20 Events",
                        style={**SECTION_H, "marginTop": "20px", "display": "inline-block",
                               "marginRight": "12px"}),
                dbc.Button("Export CSV", id="export-opp-btn", size="sm",
                           style={"backgroundColor": _rgba(BRAND_TEAL, 0.15),
                                  "border": f"1px solid {_rgba(BRAND_TEAL, 0.4)}",
                                  "color": BRAND_TEAL, "fontSize": "11px",
                                  "padding": "2px 12px", "verticalAlign": "middle"}),
                dcc.Download(id="download-opp-csv"),
            ]),
            html.P(id="opp-table-subtitle",
                   children="Sorted by Priority Score = Friction × log(Traffic) × (1 + Drop-off/2). "
                   "Click column headers to sort.",
                   style={**SECTION_SUB, "marginBottom": "12px"}),
            html.Div(id="opp-table-div", children=[build_opportunity_table()]),
            html.Div([
                html.P([
                    html.Strong("vs FullStory: ", style={"color": TEXT_PRI}),
                    "FullStory surfaces rage clicks and dead clicks per session. "
                    "This matrix goes further — it quantifies ",
                    html.Em("systemic"), " friction across all users and ranks by "
                    "impact potential. ",
                    html.Span("Friction Score", style={"color": COL_LO}),
                    " ≈ FullStory frustration signals. ",
                    html.Span("Priority Score", style={"color": COL_MID}),
                    " adds traffic × drop-off weighting FullStory lacks.",
                ], style={"fontSize": "0.75rem", "color": TEXT_MUTED,
                          "borderTop": f"1px solid {BORDER}",
                          "paddingTop": "12px", "marginTop": "16px",
                          "marginBottom": "0"}),
            ]),
        ],
    ),
    ], id="sec-wrapper-opportunity"),

    # ── Goal Tracking / North Star Metrics ────────────────────────
    _section_header("Goal Tracking — North Star Metrics",
        "Progress toward key product targets. Green = on track, red = needs attention."),
    html.Div([
        html.Div([
            html.Button("\u2699 Set Targets", id="goal-targets-toggle-btn",
                n_clicks=0,
                style={"background": "none", "border": f"1px solid {BORDER}",
                       "borderRadius": "6px", "color": TEXT_MUTED,
                       "fontSize": "0.65rem", "padding": "4px 10px",
                       "cursor": "pointer", "letterSpacing": "0.3px",
                       "textTransform": "uppercase"}),
        ], style={"marginBottom": "6px"}),
        dbc.Collapse(id="goal-targets-collapse", is_open=False, children=[
            dbc.Row([
                *[dbc.Col(html.Div([
                    html.Div(label, style={"color": TEXT_MUTED, "fontSize": "0.55rem",
                        "textTransform": "uppercase", "letterSpacing": "0.3px",
                        "marginBottom": "2px"}),
                    dcc.Input(
                        id=f"goal-target-{mid}", type="number",
                        value=default, debounce=True,
                        style={"width": "80px", "fontSize": "11px",
                               "backgroundColor": BG_INPUT, "color": TEXT_PRI,
                               "border": f"1px solid {BORDER}",
                               "borderRadius": "6px", "padding": "3px 8px"},
                    ),
                ]), width="auto") for mid, label, default in GOAL_DEFAULTS],
            ], className="mb-3 align-items-end g-2"),
        ]),
        html.Div(id="goal-cards-content", children=[_goal_tracking]),
    ], id="sec-wrapper-goals"),

    # ── User Health Score Distribution ────────────────────────────
    _section_header("User Health Score Distribution",
        "Composite score (0–100) from engagement, feature adoption, monetisation, and recency. "
        "Overlap between active and churned users shows your at-risk zone."),
    html.Div([
    dbc.Row([
        dbc.Col(_card("Health Score — Active vs Churned",
            "25 pts each: engagement (events/day, sessions), feature adoption (breadth, onboarding), "
            "monetisation (LTV, converted), recency (days since last activity). "
            "Overlap region = users likely to churn without intervention.",
            [dcc.Graph(id="health-dist-fig", figure=_health_dist_fig, config={"displayModeBar": False})]), md=12),
    ]),
    ], id="sec-wrapper-health-score"),
    _ai_tab_panel("overview"),
    ]),  # end KPI Overview tab

    # ══════════════════════════════════════════════════════════════
    #  TAB 2 — CONVERSION
    # ══════════════════════════════════════════════════════════════
    dbc.Tab(label="Conversion", tab_id="conversion", className="tab-pane-pad", children=[
    _tab_kpi_strip("conversion"),
    _section_header("Conversion Intelligence",
        "What events predict payment? Which journey stages lose the most users?"),
    html.Div([
    dbc.Row([
        dbc.Col(_card(
            "Conversion Lift by Event (positive signals only)",
            "Lift = P(paid | touched event) ÷ overall conversion rate. "
            "Bars above 1.0x correlate with paying; below 1.0x = weak predictor. "
            "Excludes payment_success / plan_upgrade (tautological) and "
            "error / payment_failed / rage_click / contact_support (reverse-causal "
            "— see the two charts below for the correct analysis).",
            [dcc.Graph(id="bridge-fig", figure=_bridge_fig, config={"displayModeBar": False})],
        ), md=7),
        dbc.Col(_card(
            "Conversion Funnel — by Segment",
            "Drop-off at each stage. 'All' = standard funnel. "
            "'By Segment' = compare revenue segments side-by-side. "
            "'By Cluster' = compare clusters (run clustering first).",
            [
                dbc.Row([
                    dbc.Col(
                        dcc.RadioItems(
                            id="funnel-seg-mode",
                            options=[
                                {"label": " All Users",         "value": "all"},
                                {"label": " By Rev. Segment",   "value": "segment"},
                                {"label": " By Cluster",        "value": "cluster"},
                            ],
                            value="all", inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "14px", "fontSize": "11px",
                                        "color": TEXT_SEC},
                        ), md=12,
                    ),
                ], className="mb-2"),
                dcc.Loading(dcc.Graph(id="funnel-chart", config={"displayModeBar": False}),
                            type="dot", color=BRAND_BLUE),
            ],
        ), md=5),
    ]),
    ], id="sec-wrapper-conversion"),

    # ── Negative-event impact (decliners) + confounder-controlled lift ────
    _section_header("Negative Signals & Confounder-Controlled Lift",
        "Friction events (errors, rage clicks, support contacts) often look "
        "like they boost conversion in naïve correlation because only engaged "
        "users encounter them. These two charts remove that illusion."),
    html.Div([
    dbc.Row([
        dbc.Col(_card(
            "Conversion Decliners — long-term retention by friction exposure",
            "For each friction event, users are bucketed by how often they "
            "encountered it (None / 1–2x / 3+x). A widening drop in retention "
            "from None → High indicates a real retention tax. A flat line means "
            "the event is incidental. Dashed line = overall retention.",
            [dcc.Graph(id="decliners-fig", figure=_decliners_fig,
                       config={"displayModeBar": False})],
        ), md=6),
        dbc.Col(_card(
            "Engagement-Matched Lift (confounder-controlled)",
            "Industry pattern (propensity stratification / habit-matched "
            "cohorts): split users into Low / Mid / High engagement tiers, then "
            "compute each event's conversion lift within each tier and average. "
            "If the matched bar stays ≫1, the signal survives the engagement "
            "confound. If it collapses to ≈1, the naïve lift was just a proxy "
            "for 'engaged users do everything'.",
            [dcc.Graph(id="matched-lift-fig", figure=_matched_lift_fig,
                       config={"displayModeBar": False})],
        ), md=6),
    ]),
    dbc.Row([
        dbc.Col(_card(
            "Negative Signal Impact on Conversion (fed into Priority Matrix)",
            "Engagement-matched percentage-point impact of each friction "
            "event. Red bars = event lowers conversion rate (relative to "
            "equally engaged peers who didn't experience it). The amber "
            "diamond shows the retention \u0394 for comparison. "
            "This exact value is fed into the Priority Matrix's score as an "
            "impact-boost multiplier (\u22125 pp conversion loss \u2192 \u00d72 priority "
            "weight), and is now available as the 'Est. Conv Impact (pp)' "
            "y-axis mode on that matrix.",
            [dcc.Graph(id="neg-impact-fig", figure=_neg_impact_fig,
                       config={"displayModeBar": False})],
        ), md=12),
    ]),
    ], id="sec-wrapper-conversion-confound"),

    _section_header("Activation & Aha Moment",
        "Which features drive long-term retention? "
        "The 'aha moment' is the event that — when performed earliest — "
        "most strongly separates retained users from churned ones."),
    html.Div([
    dbc.Row([
        dbc.Col(_card(
            "Aha Moment: Feature Adoption Timing → Retention",
            "Retention rate by feature and adoption window (composite flag: events, activity, feature breadth — not literal D30 survival). "
            "Earliest adoption (≤ Day 3) vs Never Adopted shows the maximum aha-moment gap. "
            "Dotted line = all-user baseline. Correlation only — not causal.",
            [dcc.Graph(id="aha-fig", figure=_aha_fig, config={"displayModeBar": False})],
        ), md=8),
        dbc.Col(_card(
            "When Retained Users Discover Features",
            "Median day of first use per feature — for retained users only. "
            "Features adopted in the first 1–3 days are your strongest onboarding hooks. "
            "Day 3 / 7 / 14 reference lines shown.",
            [dcc.Graph(id="adopt-timing-fig", figure=_adopt_timing_fig, config={"displayModeBar": False})],
        ), md=4),
    ]),
    dbc.Row([
        dbc.Col(_card(
            "Feature → Revenue Impact",
            "Avg LTV for feature adopters vs non-adopters. "
            "Delta annotation = revenue uplift per user. "
            "Hover to see total LTV upside if all non-adopters could be converted.",
            [dcc.Graph(id="feat-ltv-fig", figure=_feat_ltv_fig, config={"displayModeBar": False})],
        ), md=12),
    ]),
    dbc.Row([
        dbc.Col(_card(
            "Onboarding Speed",
            "Median days from first event to completing each onboarding step. "
            "Use this to identify the slowest step in your activation sequence "
            "and optimise it to reduce time-to-value.",
            [dcc.Graph(id="onboard-fig", figure=_onboard_fig, config={"displayModeBar": False})],
        ), md=12),
    ]),
    ], id="sec-wrapper-activation"),
    _section_header("Revenue Patterns",
        "How do high-revenue users differ from free users? "
        "Use these patterns to identify features worth doubling down on."),
    html.Div([
    dbc.Row([
        dbc.Col(_card(
            "Feature Engagement by Revenue Segment",
            "Heatmap: % of each segment who ever used each feature. "
            "Dark cells = high adoption. Gap between High Revenue and Free = monetisation signal.",
            [dcc.Graph(id="radar-fig", figure=_radar_fig, config={"displayModeBar": False})],
        ), md=6),
        dbc.Col(_card(
            "Journey Comparison: High Revenue vs Free",
            "What % of each segment reaches each stage. "
            "Largest gap = biggest conversion lever for PMs.",
            [dcc.Graph(id="journey-cmp-fig", figure=_journey_cmp, config={"displayModeBar": False})],
        ), md=6),
    ]),
    dbc.Row([
        dbc.Col(_card(
            "Segment KPIs",
            "Avg sessions, onboarding, features, checkout conversion, events/day per segment.",
            [dcc.Graph(id="seg-kpi-fig", figure=_seg_kpi_fig, config={"displayModeBar": False})],
        ), md=8),
        dbc.Col(_card(
            "LTV Distribution",
            "Revenue distribution among paying users by quartile.",
            [dcc.Graph(id="ltv-fig", figure=_ltv_fig, config={"displayModeBar": False})],
        ), md=4),
    ]),
    dbc.Row([
        dbc.Col(_card(
            "MRR Trend",
            "Monthly recurring revenue from invoices. "
            "Orange line = month-over-month growth %.",
            [dcc.Graph(id="mrr-fig", figure=_mrr_fig, config={"displayModeBar": False})],
        ), md=6),
        dbc.Col(_card(
            "Feature Adoption Curve",
            "Cumulative % of users who have used each feature by day N after signup. "
            "Shows how quickly users discover the product's core value. "
            "Responds to the segment filter above.",
            [dcc.Loading(
                dcc.Graph(id="feat-adopt-chart", figure=_feat_adopt_fig,
                          config={"displayModeBar": False}),
                type="dot", color=BRAND_BLUE,
            )],
        ), md=6),
    ]),
    dbc.Row([
        dbc.Col(_card(
            "Time-to-Value (TTV)",
            "CDF of days from signup to first occurrence of your chosen value event. "
            "p50/p75 lines show where half and three-quarters of users reached it. "
            "Responds to the segment filter above.",
            [
                dcc.Dropdown(
                    id="ttv-event-select",
                    options=TTV_EVENT_OPTIONS,
                    value="payment_success",
                    clearable=False,
                    style={"backgroundColor": BG_INPUT, "color": TEXT_PRI,
                           "border": f"1px solid {BORDER}", "borderRadius": "8px",
                           "marginBottom": "10px", "fontSize": "12px"},
                ),
                dcc.Loading(dcc.Graph(id="ttv-chart", config={"displayModeBar": False}),
                            type="dot", color=BRAND_BLUE),
            ],
        ), md=6),
        dbc.Col(_card(
            "Revenue Concentration (Pareto)",
            "Lorenz curve: what share of total revenue comes from your top X% of users. "
            "The further the curve bends below the diagonal, the more concentrated revenue is.",
            [dcc.Graph(id="rev-conc-fig", figure=_rev_conc_fig, config={"displayModeBar": False})],
        ), md=6),
    ]),
    dbc.Row([
        dbc.Col(_card(
            "Early Engagement → LTV",
            "Users binned by events in first 7 days after signup. "
            "Bar = avg LTV, line = number of users in each bin. "
            "Shows whether high early engagement predicts higher lifetime revenue.",
            [dcc.Graph(id="ltv-signals-fig", figure=_ltv_signals_fig, config={"displayModeBar": False})],
        ), md=6),
        dbc.Col(_card(
            "Time-to-Payment Distribution",
            "Histogram of days from signup to first payment (capped at 90 days). "
            "A right-skewed distribution means most conversions happen fast. "
            "p50 and p75 reference lines shown.",
            [dcc.Graph(id="ttv-hist-fig", figure=_ttv_hist_fig, config={"displayModeBar": False})],
        ), md=6),
    ]),
    ], id="sec-wrapper-revenue"),
    _ai_tab_panel("conversion"),
    ]),  # end Conversion tab


    # ══════════════════════════════════════════════════════════════
    #  TAB 3 — ENGAGEMENT
    # ══════════════════════════════════════════════════════════════
    dbc.Tab(label="Engagement", tab_id="engagement", className="tab-pane-pad", children=[
    _tab_kpi_strip("engagement"),
    _section_header("User Journey Flow",
        "Step-by-step flow powered by retentioneering. "
        "Use the segment filter above to compare any user cohort."),
    html.Div([
    build_journey_insight_cards(),
    _card(
        "Step Sankey — Event Flow",
        "Each column = one step in the journey. "
        "Generic events (click, page_view, session) are excluded for clarity. "
        "Adjust steps/threshold below.",
        [
            dbc.Row([
                dbc.Col([
                    html.Label("Max Steps", style={"color": TEXT_MUTED,
                                "fontSize": "0.72rem", "display": "block",
                                "marginBottom": "4px"}),
                    dcc.Slider(id="rete-max-steps", min=4, max=12, step=1, value=8,
                               marks={i: {"label": str(i),
                                          "style": {"color": TEXT_SEC}} for i in range(4, 13)},
                               tooltip={"placement": "bottom",
                                        "style": {"backgroundColor": BG_SURFACE,
                                                  "color": TEXT_PRI}}),
                ], md=4),
                dbc.Col([
                    html.Label("Threshold (min % of users to show event)",
                               style={"color": TEXT_MUTED, "fontSize": "0.72rem",
                                      "display": "block", "marginBottom": "4px"}),
                    dcc.Slider(id="rete-threshold", min=0.01, max=0.15, step=0.01, value=0.03,
                               marks={v: {"label": f"{v:.0%}",
                                          "style": {"color": TEXT_SEC}}
                                      for v in [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]},
                               tooltip={"placement": "bottom",
                                        "style": {"backgroundColor": BG_SURFACE,
                                                  "color": TEXT_PRI}}),
                ], md=8),
            ], className="mb-3"),
            dcc.Loading(dcc.Graph(id="rete-sankey", figure=_sankey_default,
                                  config={"displayModeBar": False}),
                        type="dot", color=BRAND_BLUE),
        ],
    ),
    dbc.Row([
        dbc.Col([
            dbc.Button("▶ Show Transition Table", id="toggle-trans-table-btn", size="sm",
                style={"backgroundColor": "transparent", "border": f"1px solid {BORDER}",
                       "color": TEXT_MUTED, "fontSize": "11px", "padding": "4px 12px",
                       "borderRadius": "6px", "marginBottom": "8px", "cursor": "pointer"},
                n_clicks=0),
            dbc.Collapse(id="trans-table-collapse", is_open=False, children=[
                _card("Transition Probability Table",
                    "Top event-to-event transitions ranked by count. "
                    "Session boundaries and self-loops excluded. Sort or filter any column.",
                    [html.Div(id="transition-table-div", children=_trans_table)]),
            ]),
        ], md=12),
    ]),
    dbc.Row([
        dbc.Col(_card("Post-Signup Event Distribution",
            "Which events occur at each of the first 10 steps after signup. "
            "Shows how the event mix shifts as users progress through your product.",
            [dcc.Graph(id="step-dist-fig", figure=_step_dist_fig, config={"displayModeBar": False})]), md=12),
    ]),
    ], id="sec-wrapper-journey"),
    _section_header("Engagement Deep Dive",
        "Time spent between actions, where users get stuck, and event timing patterns."),
    html.Div([
    dbc.Row([
        dbc.Col(_card(
            "Avg Time Between Transitions",
            "Time between top event pairs. Within Session = same 30-min session only "
            "(minutes); All Transitions = includes cross-session gaps (hours).",
            [
                dbc.Row([dbc.Col(dcc.RadioItems(id="avg-time-scope",
                    options=[{"label": " Within Session", "value": "within"},
                             {"label": " All Transitions", "value": "all"}],
                    value="within", inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "16px", "fontSize": "12px",
                                "color": TEXT_SEC}), md=12)], className="mb-2"),
                dcc.Loading(dcc.Graph(id="avg-time-fig", figure=_avg_time_fig,
                              config={"displayModeBar": False}),
                    type="dot", color=BRAND_BLUE),
            ],
        ), md=6),
        dbc.Col(_card("Self-Loops — Where Users Get Stuck",
            "Events where users repeat the same action consecutively. "
            "High repeat counts signal UX confusion or missing affordances.",
            [dcc.Graph(id="self-loop-fig", figure=_self_loop_fig, config={"displayModeBar": False})]), md=6),
    ]),
    dbc.Row([
        dbc.Col(_card("Daily Event Volume Calendar",
            "GitHub-style activity calendar: daily event count over the last 52 weeks. "
            "Reveals seasonal patterns, release spikes, and growth trends at a glance.",
            [dcc.Graph(id="event-calendar-fig", figure=_event_calendar_fig, config={"displayModeBar": False})]), md=12),
    ]),
    dbc.Row([
        dbc.Col(_card("Activity Heatmap — When Users Are Active",
            "Event volume by hour of day and day of week. "
            "Use this to time product communications, A/B tests, and feature releases.",
            [dcc.Graph(id="activity-hmap-fig", figure=_activity_hmap, config={"displayModeBar": False})]), md=6),
        dbc.Col(_card("Session Duration Distribution",
            "How long do user sessions last (capped at 120 min)? "
            "A bimodal distribution often signals two distinct user behaviours. "
            "Responds to the segment filter above.",
            [dcc.Loading(dcc.Graph(id="sess-dur-chart", figure=_sess_dur_fig,
                          config={"displayModeBar": False}), type="dot", color=BRAND_BLUE)]), md=6),
    ]),
    ], id="sec-wrapper-engagement"),
    _section_header("Friction & Comeback Hotspots",
        "Where do users struggle, loop back, or abandon? "
        "Friction Score = composite of self-loops, rage clicks, errors, and drop-offs."),
    html.Div([
    _card("Friction Score by Event",
        "Composite score per event: self-loop (30%) + rage click (30%) + error (20%) + drop-off (20%). "
        "Higher = more user pain. Select multiple segments to compare side-by-side.",
        [
            dbc.Row([dbc.Col([
                html.Label("Segments to Compare", style={"color": TEXT_MUTED, "fontSize": "0.72rem",
                    "textTransform": "uppercase", "letterSpacing": "0.5px",
                    "marginBottom": "6px", "display": "block"}),
                dcc.Dropdown(id="friction-group", options=SEGMENTS, value=["all"],
                    multi=True, clearable=False,
                    style={"backgroundColor": BG_INPUT, "color": TEXT_PRI,
                           "border": f"1px solid {BORDER}", "borderRadius": "8px"}),
            ], md=6)], className="mb-3"),
            dcc.Loading(dcc.Graph(id="friction-chart", figure=_friction_fig,
                config={"displayModeBar": False}), type="dot", color=BRAND_BLUE),
        ],
    ),
    dbc.Row([
        dbc.Col(_card("Friction Signal Breakdown",
            "For the top 12 highest-friction events: what percentage of the friction score "
            "comes from each signal (drop-off, rage click, error, self-loop). "
            "Use this to diagnose the root cause of friction at each event.",
            [dcc.Graph(id="friction-breakdown-fig", figure=_friction_breakdown, config={"displayModeBar": False})]), md=6),
        dbc.Col(_card("Friction by Revenue Segment",
            "Top 8 friction events broken down by revenue segment. "
            "If friction is highest for Free users, it blocks conversion. "
            "If highest for High Revenue users, it threatens retention.",
            [dcc.Graph(id="friction-seg-cmp-fig", figure=_friction_seg_cmp, config={"displayModeBar": False})]), md=6),
    ]),
    dbc.Row([
        dbc.Col(_card("Comeback Loops",
            "Events users return to after leaving — indicating incomplete tasks or "
            "repeated confusion. Bar = absolute count; diamond = comeback rate % of users who hit this event.",
            [dcc.Graph(id="comeback-fig", figure=_comeback_fig, config={"displayModeBar": False})]), md=6),
        dbc.Col(_card("Drop-off Risk Map",
            "Where users permanently stop progressing through the product journey. "
            "Blue = users who reached this stage. Red = users who stopped here. "
            "Orange line = % who never progressed further.",
            [dcc.Graph(id="dropoff-fig", figure=_dropoff_fig, config={"displayModeBar": False})]), md=6),
    ]),
    ], id="sec-wrapper-friction"),
    _section_header("User Clustering",
        "KMeans + PCA. Select any mix of financial and product-usage features. "
        "Clusters reveal natural user archetypes — compare their revenue, friction, and engagement."),
    html.Div([
    _card("Interactive Cluster Explorer",
        "Choose features → run clustering → inspect each cluster's average profile below. "
        "Bubble size = LTV. Axes = first 2 PCA components.",
        [
            dbc.Row([
                dbc.Col([
                    html.Label("Select Features", style={"color": TEXT_MUTED, "fontSize": "0.72rem",
                        "textTransform": "uppercase", "letterSpacing": "0.5px",
                        "display": "block", "marginBottom": "8px"}),
                    dcc.Checklist(id="cluster-features", options=CLUSTER_FEATURES,
                        value=DEFAULT_CLUSTER_FEATS, className="cluster-checklist",
                        inputStyle={"marginRight": "6px", "accentColor": BRAND_BLUE},
                        labelStyle={"display": "flex", "alignItems": "center",
                                    "color": TEXT_SEC, "fontSize": "12px",
                                    "marginBottom": "6px", "cursor": "pointer"},
                        style={"backgroundColor": BG_INPUT, "border": f"1px solid {BORDER}",
                               "borderRadius": "10px", "padding": "12px 14px",
                               "maxHeight": "340px", "overflowY": "auto"}),
                    html.Div([
                        html.Label("Number of Clusters", style={"color": TEXT_MUTED, "fontSize": "0.7rem",
                            "textTransform": "uppercase", "letterSpacing": "0.5px",
                            "display": "block", "marginTop": "14px", "marginBottom": "6px"}),
                        dcc.Dropdown(id="cluster-n-select",
                            options=[{"label": "Auto (best k)", "value": "auto"}] +
                                    [{"label": f"k = {k}", "value": k} for k in range(2, 9)],
                            value="auto", clearable=False,
                            style={"backgroundColor": BG_INPUT, "color": TEXT_PRI,
                                   "border": f"1px solid {BORDER}", "borderRadius": "8px",
                                   "fontSize": "12px"}),
                        dbc.Button("Run Clustering", id="run-cluster-btn", color="primary", size="sm",
                            style={"backgroundColor": BRAND_BLUE, "borderColor": BRAND_BLUE,
                                   "marginTop": "10px", "width": "100%"}),
                        html.Div(id="cluster-score-display", style={"marginTop": "10px"}),
                    ]),
                    html.Hr(style={"borderColor": BORDER, "margin": "14px 0"}),
                    html.Label("Export cluster users", style={"color": TEXT_MUTED, "fontSize": "0.7rem",
                        "textTransform": "uppercase", "letterSpacing": "0.5px",
                        "display": "block", "marginBottom": "6px"}),
                    dcc.Dropdown(id="cluster-export-select", options=[], placeholder="Select cluster…",
                        clearable=True,
                        style={"backgroundColor": BG_INPUT, "color": TEXT_PRI,
                               "border": f"1px solid {BORDER}", "borderRadius": "8px",
                               "fontSize": "12px"}),
                    html.Div([
                        dbc.Button("Download User IDs", id="export-cluster-btn", size="sm",
                            style={"backgroundColor": _rgba(BRAND_TEAL, 0.15),
                                   "border": f"1px solid {_rgba(BRAND_TEAL, 0.4)}",
                                   "color": BRAND_TEAL, "marginTop": "8px", "width": "100%"}),
                        dcc.Download(id="download-cluster-csv"),
                    ]),
                    dcc.Store(id="cluster-user-store",
                              data=json.dumps({k: list(v) for k, v in _cluster_user_map.items()})),
                ], md=3),
                dbc.Col([
                    dcc.Loading(dcc.Graph(id="cluster-scatter", figure=_cluster_fig,
                        config={"displayModeBar": False}), type="dot", color=BRAND_BLUE),
                ], md=9),
            ], className="g-3"),
            html.Div(id="cluster-summary-div", children=[
                html.H6("Cluster Summary", style={**SECTION_H, "marginTop": "16px"}),
                html.P("Mean values per cluster across key metrics.", style=SECTION_SUB),
                DataTable(
                    data=_cluster_summary.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in _cluster_summary.columns],
                    style_table={"overflowX": "auto", "borderRadius": "8px",
                                 "border": f"1px solid {BORDER}"},
                    style_header={"backgroundColor": BG_SURFACE, "color": TEXT_PRI,
                                  "fontWeight": "600", "fontSize": "11px",
                                  "border": f"1px solid {BORDER}",
                                  "fontFamily": "Inter, system-ui, sans-serif"},
                    style_data={"backgroundColor": BG_CARD, "color": TEXT_SEC,
                                "fontSize": "11px",
                                "border": f"1px solid {_rgba(BRAND_BLUE, 0.08)}",
                                "fontFamily": "Inter, system-ui, sans-serif"},
                    style_data_conditional=[{"if": {"row_index": "odd"},
                                             "backgroundColor": BG_SURFACE}],
                ),
            ]),
            html.Div(id="cluster-archetypes-div", style={"marginTop": "16px"}),
        ],
    ),
    ], id="sec-wrapper-clustering"),

    # ── Individual User Journey ────────────────────────────────────────────
    _section_header("Individual User Journey",
        "Session-by-session event replay — inspired by FullStory. "
        "Each row = one session; each dot = one event, coloured by category. "
        "Select any row in the At-Risk Users table (Churn tab) to auto-load that user here."),
    html.Div([
    _card(
        "User Journey Viewer",
        "Enter a user ID and click Show Journey. Use the category checkboxes to focus on "
        "specific event types. The event log table shows the raw sequence below the chart.",
        [
            dbc.Row([
                dbc.Col([
                    html.Label("User ID", style={
                        "color": TEXT_MUTED, "fontSize": "0.7rem",
                        "textTransform": "uppercase", "letterSpacing": "0.5px",
                        "display": "block", "marginBottom": "4px"}),
                    dcc.Input(
                        id="journey-user-id",
                        type="number",
                        placeholder="e.g. 42",
                        debounce=False,
                        style={"backgroundColor": BG_INPUT, "color": TEXT_PRI,
                               "border": f"1px solid {BORDER_LIT}",
                               "borderRadius": "8px",
                               "padding": "10px 16px", "fontSize": "16px",
                               "fontWeight": "600", "letterSpacing": "0.5px",
                               "width": "100%", "height": "46px"},
                    ),
                ], md=2),
                dbc.Col([
                    html.Label("\u00a0", style={"display": "block", "marginBottom": "4px",
                                               "fontSize": "0.7rem"}),
                    dbc.Button("Show Journey", id="show-journey-btn", color="primary",
                               style={"backgroundColor": BRAND_BLUE,
                                      "borderColor": BRAND_BLUE,
                                      "width": "100%", "height": "46px"}),
                ], md=2),
                dbc.Col([
                    html.Div(id="journey-user-info",
                             style={"color": TEXT_MUTED, "fontSize": "0.78rem",
                                    "paddingTop": "8px"}),
                ], md=8),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Show categories:", style={"color": TEXT_MUTED,
                               "fontSize": "0.7rem", "textTransform": "uppercase",
                               "letterSpacing": "0.5px", "marginRight": "10px"}),
                    dcc.Checklist(
                        id="journey-cat-filter",
                        options=[{"label": f" {c}", "value": c}
                                 for c in sorted(CATEGORY_COLORS.keys())],
                        value=list(CATEGORY_COLORS.keys()),
                        inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "14px", "fontSize": "12px",
                                    "color": TEXT_SEC},
                    ),
                ]),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Div(id="journey-profile-panel",
                             style={"minHeight": "300px"}),
                ], md=3),
                dbc.Col([
                    dcc.Loading(
                        dcc.Graph(
                            id="journey-viewer-chart",
                            figure=_empty_fig("Enter a user ID above and click Show Journey"),
                            config={"displayModeBar": True,
                                    "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
                        ),
                        type="dot", color=BRAND_BLUE,
                    ),
                ], md=9),
            ], className="g-3 mb-3"),
            html.H6("Event Log", style={**SECTION_H, "marginTop": "8px",
                                         "marginBottom": "8px"}),
            dcc.Loading(
                DataTable(
                    id="journey-event-table",
                    columns=[],
                    data=[],
                    page_size=20,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "backgroundColor": BG_CARD, "color": TEXT_SEC,
                        "border": f"1px solid {BORDER}", "fontSize": "11px",
                        "padding": "6px 10px", "fontFamily": "Inter, monospace",
                        "textAlign": "left",
                    },
                    style_header={
                        "backgroundColor": BG_SURFACE, "color": TEXT_PRI,
                        "fontWeight": "600", "border": f"1px solid {BORDER}",
                    },
                    style_data_conditional=[
                        {"if": {"column_id": "Category", "filter_query": '{Category} = "friction"'},
                         "color": COL_LO},
                        {"if": {"column_id": "Category", "filter_query": '{Category} = "monetization"'},
                         "color": BRAND_AMBER},
                        {"if": {"column_id": "Category", "filter_query": '{Category} = "onboarding"'},
                         "color": BRAND_BLUE},
                        {"if": {"column_id": "Category", "filter_query": '{Category} = "engagement"'},
                         "color": BRAND_TEAL},
                    ],
                ),
                type="dot", color=BRAND_BLUE,
            ),
        ],
    ),
    ], id="sec-wrapper-individual"),

    # ── Feature Adoption Funnel ───────────────────────────────────
    _section_header("Feature Adoption Funnel",
        "How deeply are users engaging with each product feature? "
        "Discovered (1+) → Tried (3+) → Adopted (7+) → Power User (15+)."),
    html.Div([
    dbc.Row([
        dbc.Col(_card("Feature Adoption Stages",
            "Each bar group shows a product feature. Stages show progressive depth of usage. "
            "Large drop-offs between stages reveal features with poor stickiness — "
            "users discover them but don't form habits.",
            [dcc.Graph(id="feature-funnel-fig", figure=_feature_funnel_fig, config={"displayModeBar": False})]), md=12),
    ]),
    ], id="sec-wrapper-feature-funnel"),
    _ai_tab_panel("engagement"),
    ]),  # end Engagement tab

    # ══════════════════════════════════════════════════════════════
    #  TAB 4 — RETENTION
    # ══════════════════════════════════════════════════════════════
    dbc.Tab(label="Retention", tab_id="retention", className="tab-pane-pad", children=[
    _tab_kpi_strip("retention"),
    _section_header("Cohort Retention",
        "What percentage of each signup cohort is still active over time?"),
    html.Div([
    dbc.Row([
        dbc.Col(_card("Cohort Retention Heatmap",
            "Each cell = % of the cohort's users active in that period. "
            "Darker = higher retention. Choose a period roll below.",
            [
                dbc.Row([
                    dbc.Col(html.Label("Period roll:", style={"color": TEXT_MUTED,
                        "fontSize": "0.72rem", "textTransform": "uppercase",
                        "letterSpacing": "0.5px", "lineHeight": "32px",
                        "paddingTop": "2px"}), width="auto"),
                    dbc.Col(dcc.RadioItems(id="cohort-roll-mode",
                        options=[{"label": " 7-day", "value": "7d"},
                                 {"label": " 14-day", "value": "14d"},
                                 {"label": " Monthly", "value": "monthly"}],
                        value="7d", inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "18px", "fontSize": "12px",
                                    "color": TEXT_SEC})),
                ], className="align-items-center mb-3"),
                dcc.Loading(dcc.Graph(id="cohort-retention-heatmap",
                    figure=_cohort_ret_fig, config={"displayModeBar": False}),
                    type="dot", color=BRAND_BLUE),
            ]
        ), md=12),
    ]),
    ], id="sec-wrapper-retention"),
    _section_header("Value & Retention Health",
        "Are users reaching value quickly enough? Are they coming back? "
        "Activation health per cohort and stickiness over time."),
    html.Div([
    dbc.Row([
        dbc.Col(_card("Activation Health by Cohort",
            "Bar = % of each signup cohort who completed onboarding. "
            "Line = conversion rate to paying. "
            "Declining bars signal onboarding regressions.",
            [dcc.Graph(id="activation-fig", figure=_activation_fig, config={"displayModeBar": False})]), md=6),
        dbc.Col(_card("Stickiness Trend (DAU/MAU)",
            "7-day rolling DAU/MAU ratio — the standard engagement health metric. "
            "Above 20% = healthy. Dotted line = 30-day active users (scale right). "
            "Flat or rising stickiness = users are returning for value.",
            [dcc.Graph(id="stickiness-fig", figure=_stickiness_fig, config={"displayModeBar": False})]), md=6),
    ]),
    ], id="sec-wrapper-value"),

    # ── Revenue Impact Attribution ────────────────────────────────
    _section_header("Revenue Impact Attribution",
        "Which friction points cost the most revenue? Stacked bars show at-risk LTV (potential loss) "
        "plus already-lost revenue from churned users who hit each event."),
    html.Div([
    dbc.Row([
        dbc.Col(_card("Revenue at Risk by Friction Event",
            "Revenue at Risk = avg active LTV × at-risk users who hit this event. "
            "Revenue Lost = actual LTV of churned users who experienced this friction. "
            "Fix the top bars first for maximum revenue recovery.",
            [dcc.Graph(id="rev-impact-fig", figure=_rev_impact_fig, config={"displayModeBar": False})]), md=12),
    ]),
    ], id="sec-wrapper-rev-impact"),
    _ai_tab_panel("retention"),
    ]),  # end Retention tab

    # ══════════════════════════════════════════════════════════════
    #  TAB 5 — CHURN
    # ══════════════════════════════════════════════════════════════
    dbc.Tab(label="Churn", tab_id="churn", className="tab-pane-pad", children=[
    # Download components (invisible)
    dcc.Download(id="download-churned-csv"),
    _tab_kpi_strip("churn"),
    html.Div([
    dbc.Row([
        dbc.Col(_card("Churn Breakdown",
            "Active vs churned. Within churned: never paid vs post-payment churn.",
            [dcc.Loading(dcc.Graph(id="churn-donut-chart",
                config={"displayModeBar": False}), type="dot", color=BRAND_BLUE)]), md=5),
        dbc.Col(_card("Cohort Timeline",
            "Users joined per month (stacked: active vs churned). "
            "Orange line = churn rate for that cohort.",
            [dcc.Loading(dcc.Graph(id="cohort-chart",
                config={"displayModeBar": False}), type="dot", color=BRAND_BLUE)]), md=7),
    ]),
    dbc.Row([
        dbc.Col(dbc.Button(
            "📥 Download Churned Users CSV",
            id="download-churned-btn",
            color="outline-danger",
            size="sm",
            style={"fontSize": "12px", "marginBottom": "6px"},
        ), width="auto"),
        dbc.Col(html.Span(
            f"{_CHURN_RISK['churned_count']:,} churned users · all profile columns included",
            style={"color": TEXT_MUTED, "fontSize": "11px", "lineHeight": "30px"},
        ), width="auto"),
    ], align="center", className="mb-2"),
    _section_header("Churn Risk Deep-Dive",
        f"{_at_risk_count:,} active users ({_risk_pct:.1f}% of active base) show 2+ early warning "
        "signals (below-median engagement AND low feature adoption). Explore root causes and who "
        "to prioritise for intervention."),
    dbc.Row([
        dbc.Col(_card("At-Risk Active Users",
            "Active users with 2+ of: below-median events, below-median active days, "
            "fewer than 2 features used. High-confidence churn risk.",
            [html.Div([
                html.Span(f"{_at_risk_count:,}", style={"fontSize": "2.2rem",
                    "fontWeight": "700", "color": BRAND_AMBER}),
                html.Span(f"  of {_active_count:,} active  ({_risk_pct:.1f}%)",
                          style={"color": TEXT_SEC, "fontSize": "0.85rem"}),
            ])]), md=4),
        dbc.Col(_card("Never Completed Onboarding",
            "Share of churned users who left before finishing onboarding — "
            "the clearest early-stage churn signal.",
            [html.Div([
                html.Span(f"{_never_onboarded_pct:.1f}%", style={"fontSize": "2.2rem",
                    "fontWeight": "700", "color": COL_LO}),
                html.Span(" of churned users",
                          style={"color": TEXT_SEC, "fontSize": "0.85rem"}),
            ])]), md=4),
        dbc.Col(_card("Churned Post-Payment",
            "Users who paid at least once then churned — highest-value segment to win back "
            "with targeted re-engagement.",
            [html.Div([
                html.Span(f"{_post_paid_count:,}", style={"fontSize": "2.2rem",
                    "fontWeight": "700", "color": COL_MID}),
                html.Span(f"  users  ({_post_paid_pct:.1f}% of churned)",
                          style={"color": TEXT_SEC, "fontSize": "0.85rem"}),
            ])]), md=4),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(_card("Early-Action Churn Predictors",
            "Actions taken within a user's first N events that most strongly separate retained "
            "users from churned ones. Positive gap = retention signal (do more of this). "
            "Negative gap = churn signal (users stuck here tend to leave).",
            [
                dbc.Row([
                    dbc.Col(html.Span("Analyse first",
                                      style={"color": TEXT_MUTED, "fontSize": "12px",
                                             "lineHeight": "28px"}), width="auto"),
                    dbc.Col(dcc.RadioItems(
                        id="churn-predictor-n",
                        options=[{"label": f" {n}", "value": n} for n in [3, 5, 10, 15, 20]],
                        value=5,
                        inline=True,
                        inputStyle={"marginRight": "3px"},
                        labelStyle={"marginRight": "14px", "color": TEXT_SEC,
                                    "fontSize": "12px", "cursor": "pointer"},
                    ), width="auto"),
                    dbc.Col(html.Span("events per user",
                                      style={"color": TEXT_MUTED, "fontSize": "12px",
                                             "lineHeight": "28px"}), width="auto"),
                ], align="center", className="mb-2"),
                build_churn_predictor_table(),
            ]), md=12),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(_card("What Did Churned Users Miss?",
            "% of each segment who completed key product actions. "
            "Gaps between Active and Churned bars reveal root causes — "
            "biggest gap = highest-leverage fix.",
            [dcc.Graph(id="churn-beh-fig", figure=_churn_beh_fig, config={"displayModeBar": False})]), md=12),
    ]),
    _section_header("ML Churn Prediction — SHAP Feature Importance",
        f"RandomForest classifier (AUC = {_shap_auc:.3f}) trained on all user profile features. "
        "SHAP values reveal which features most influence the model's churn prediction and in which "
        "direction. Red bars push toward churn; green bars push toward retention."),
    dbc.Row([
        dbc.Col(_card("SHAP Feature Importance — What Drives Churn vs Retention",
            "Features ranked by mean |SHAP value| — the most predictive signals for churn. "
            "Colour indicates direction: red features increase churn probability when high, "
            "green features increase retention probability when high.",
            [dcc.Graph(id="shap-fig", figure=_shap_fig, config={"displayModeBar": False})]), md=6),
        dbc.Col(_card("Feature Breakdown — Active vs Churned Averages",
            "Compare the average value of each top feature between active and churned users. "
            "SHAP importance tells you *how much* the model relied on each feature; "
            "the averages reveal *why* — where the biggest behavioural gaps are.",
            [_shap_feature_table]), md=6),
    ]),
    dbc.Row([
        dbc.Col(_card("Churn Rate by User Cluster",
            "Active vs churned split within each cluster from the Engagement tab. "
            "Clusters with high churn rates share predictable behavioural profiles. "
            "Re-run clustering to refresh.",
            [
                dbc.Row([dbc.Col([
                    html.Label("Show clusters", style={"color": TEXT_MUTED,
                        "fontSize": "0.65rem", "textTransform": "uppercase",
                        "letterSpacing": "0.5px", "display": "block",
                        "marginBottom": "3px"}),
                    dcc.Dropdown(
                        id="churn-cluster-filter",
                        options=[{"label": "All clusters", "value": "all"}],
                        value="all", clearable=False,
                        style={"fontSize": "12px"},
                    ),
                ], md=6)], className="mb-2"),
                dcc.Loading(dcc.Graph(id="churn-cluster-chart", figure=_churn_clust_fig,
                    config={"displayModeBar": False}), type="dot", color=BRAND_BLUE),
            ]), md=5),
        dbc.Col(_card("At-Risk Users — Prioritised for Intervention",
            f"Active users with 2+ early warning signals, sorted by risk level then "
            f"days of inactivity. Top {min(_at_risk_count, 200):,} shown.",
            [
                dbc.Row([
                    dbc.Col([
                        html.Label("Risk level", style={"color": TEXT_MUTED,
                            "fontSize": "0.65rem", "textTransform": "uppercase",
                            "letterSpacing": "0.5px", "display": "block",
                            "marginBottom": "3px"}),
                        dcc.Dropdown(
                            id="at-risk-risk-filter",
                            options=_AT_RISK_RISK_OPTIONS,
                            value="all",
                            clearable=False,
                            style={"fontSize": "12px"},
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label("Recommended action", style={"color": TEXT_MUTED,
                            "fontSize": "0.65rem", "textTransform": "uppercase",
                            "letterSpacing": "0.5px", "display": "block",
                            "marginBottom": "3px"}),
                        dcc.Dropdown(
                            id="at-risk-action-filter",
                            options=_AT_RISK_ACTION_OPTIONS,
                            value="all",
                            clearable=False,
                            style={"fontSize": "12px"},
                        ),
                    ], md=8),
                ], className="mb-2"),
                _at_risk_table,
            ]), md=7),
    ]),
    ], id="sec-wrapper-churn"),

    # ── Churn Probability Distribution ────────────────────────────
    _section_header("Per-User Churn Probability",
        "ML-predicted churn probability for every active user (0–100%). "
        "Users in the red zone (>60%) are prime candidates for immediate intervention."),
    html.Div([
    dbc.Row([
        dbc.Col(_card("Churn Probability Distribution — Active Users",
            "Histogram of predicted churn probabilities from the RandomForest model. "
            "Green zone (0–30%) = safe. Yellow (30–60%) = monitor. Red (60–100%) = intervene now.",
            [dcc.Graph(id="churn-prob-fig", figure=_churn_prob_hist, config={"displayModeBar": False})]), md=6),
        dbc.Col(_card("Time-to-Churn Survival Curve",
            "Kaplan-Meier survival estimate: probability a user is still active after N days. "
            + (f"Median survival = {_median_survival:.0f} days — after this point, >50% have churned. "
               if _median_survival is not None else "Survival stays above 50% across the observation window. ")
            + "Steepest drops indicate the most dangerous retention windows.",
            [dcc.Graph(id="survival-fig", figure=_survival_fig, config={"displayModeBar": False})]), md=6),
    ]),
    ], id="sec-wrapper-churn-prob"),

    # ── Cohort Comparison: Recent vs Older Churners ──────────────
    _section_header("Churn Cohort Comparison",
        "Are recent churners different from older ones? Shifting patterns reveal "
        "whether your product changes are helping or hurting retention."),
    html.Div([
    dbc.Row([
        dbc.Col(_card("Recent vs Older Churners vs Active Users",
            "Compares average metrics across 3 groups: users who churned in the last 14 days, "
            "users who churned earlier, and currently active users. "
            "If recent churners look worse than older ones, churn is accelerating.",
            [dcc.Graph(id="cohort-cmp-fig", figure=_cohort_cmp_fig, config={"displayModeBar": False})]), md=12),
    ]),
    ], id="sec-wrapper-cohort-cmp"),

    # ── Win-Back Candidates ──────────────────────────────────────
    _section_header("Win-Back Candidates",
        f"Churned users scored by recoverability (0–100). {len(_winback_records):,} churned users ranked. "
        "Focus re-engagement campaigns on High-tier users — they had the most engagement "
        "and revenue before leaving."),
    html.Div([
    dbc.Row([
        dbc.Col(_card("Win-Back Priority Table",
            "Composite score from: pre-churn engagement (sessions, events), recency of churn, "
            "monetisation history (LTV, converted), feature adoption (breadth, onboarding). "
            "Higher score = easier to win back. Click column headers to sort.",
            [_winback_tbl]), md=12),
    ]),
    ], id="sec-wrapper-winback"),
    _ai_tab_panel("churn"),
    ]),  # end Churn tab

    # ══════════════════════════════════════════════════════════════
    #  TAB 6 — SETTINGS
    # ══════════════════════════════════════════════════════════════
    dbc.Tab(label="⚙ Settings", tab_id="settings",
            label_style={"color": TEXT_MUTED},
            className="tab-pane-pad", children=[
        _section_header("Settings", "Configure API keys and data sources."),

        # ── API key ─────────────────────────────────────────────
        html.Div([
            html.H6("Anthropic API Key",
                    style={"color": TEXT_PRI, "fontWeight": "600",
                           "marginBottom": "6px"}),
            html.P("Required for the AI Analyst tab and AI-assisted ETL. "
                   "Stored in memory for this session only.",
                   style={"color": TEXT_MUTED, "fontSize": "0.82rem",
                          "marginBottom": "10px"}),
            dbc.InputGroup([
                dbc.Input(id="settings-api-key", type="password",
                    placeholder="sk-ant-…  (leave blank to keep current)",
                    style={"backgroundColor": BG_INPUT, "color": TEXT_PRI,
                           "border": f"1px solid {BORDER}",
                           "fontSize": "0.85rem"}),
                dbc.Button("Save Key", id="settings-api-key-save",
                    color="primary", style={"fontWeight": "600"}),
            ]),
            html.Div(id="settings-api-key-status", style={"marginTop": "8px"}),
            html.Small(
                "Set ANTHROPIC_API_KEY env var before startup for persistence.",
                style={"color": TEXT_MUTED, "fontSize": "0.75rem"},
            ),
        ], style={"padding": "8px 8px 24px", "maxWidth": "620px"}),

        # ── AI-assisted ETL ──────────────────────────────────────
        html.Hr(style={"borderColor": BORDER, "margin": "18px 0"}),
        html.Div([
            html.H6([html.Span("\u2728 ", style={"color": BRAND_PURPLE}),
                     "AI-assisted ETL — connect any data source"],
                    style={"color": TEXT_PRI, "fontWeight": "600",
                           "marginBottom": "6px"}),
            html.P([
                "Upload a CSV (or describe a SQL connection) and Claude will ",
                "read a sample of your data, infer how to map each event to ",
                "Insightera's canonical schema, and produce a reusable YAML ",
                "mapping. Review the mapping below, edit if needed, then apply.",
            ], style={"color": TEXT_MUTED, "fontSize": "0.82rem",
                      "marginBottom": "10px"}),

            # Step 1: choose source
            html.Div([
                html.Label("Source type",
                           style={"color": TEXT_MUTED, "fontSize": "0.72rem",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px",
                                  "marginBottom": "6px", "display": "block"}),
                dcc.RadioItems(
                    id="etl-source-type",
                    options=[
                        {"label": " CSV upload", "value": "csv"},
                        {"label": " SQL (SQLAlchemy URL)", "value": "sql"},
                    ],
                    value="csv", inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "18px", "fontSize": "12px",
                                "color": TEXT_SEC},
                ),
            ], style={"marginBottom": "10px"}),

            # CSV upload
            html.Div(id="etl-csv-upload-wrapper", children=[
                dcc.Upload(id="etl-csv-upload",
                    children=html.Div([
                        "Drag & drop CSV or ",
                        html.A("browse files",
                               style={"color": BRAND_BLUE, "cursor": "pointer"}),
                    ]),
                    multiple=False, accept=".csv",
                    style={"width": "100%", "height": "54px",
                           "lineHeight": "54px", "borderWidth": "1px",
                           "borderStyle": "dashed",
                           "borderColor": _rgba(BRAND_BLUE, 0.45),
                           "borderRadius": "8px", "textAlign": "center",
                           "backgroundColor": _rgba(BRAND_BLUE, 0.03),
                           "color": TEXT_SEC, "fontSize": "0.85rem",
                           "marginBottom": "8px"}),
            ]),

            # SQL input
            html.Div(id="etl-sql-wrapper", children=[
                dbc.InputGroup([
                    dbc.Input(id="etl-sql-url",
                        placeholder="sqlite:///mydata.sqlite  or  postgresql://user:pass@host/db",
                        style={"backgroundColor": BG_INPUT, "color": TEXT_PRI,
                               "border": f"1px solid {BORDER}",
                               "fontSize": "0.85rem"}),
                    dbc.Input(id="etl-sql-table",
                        placeholder="events table", value="events",
                        style={"backgroundColor": BG_INPUT, "color": TEXT_PRI,
                               "border": f"1px solid {BORDER}",
                               "fontSize": "0.85rem", "maxWidth": "160px"}),
                ]),
            ], style={"display": "none", "marginBottom": "8px"}),

            # Hints
            html.Label("Hints for the AI mapper (optional)",
                       style={"color": TEXT_MUTED, "fontSize": "0.72rem",
                              "textTransform": "uppercase",
                              "letterSpacing": "0.5px",
                              "marginTop": "10px", "marginBottom": "4px",
                              "display": "block"}),
            dbc.Textarea(id="etl-hints",
                placeholder="e.g. 'This is Segment data. Our signup event is called user_registered.'",
                style={"backgroundColor": BG_INPUT, "color": TEXT_PRI,
                       "border": f"1px solid {BORDER}",
                       "fontSize": "0.82rem", "height": "60px",
                       "marginBottom": "10px"}),

            # Action row
            html.Div([
                dbc.Button([html.Span("\u2728 "), "Suggest mapping with AI"],
                    id="etl-infer-btn", color="primary",
                    style={"fontWeight": "600", "marginRight": "10px"}),
                dbc.Button("Apply mapping & reload dashboard",
                    id="etl-apply-btn", color="success",
                    disabled=True, style={"fontWeight": "600"}),
            ]),

            # Probe + status
            dbc.Spinner(html.Div(id="etl-probe-info",
                style={"marginTop": "12px", "fontSize": "0.85rem"}),
                size="sm", color="primary"),

            # Proposed-mapping editor
            html.Div([
                html.Label("Proposed mapping (editable YAML)",
                    style={"color": TEXT_MUTED, "fontSize": "0.72rem",
                           "textTransform": "uppercase",
                           "letterSpacing": "0.5px",
                           "marginTop": "14px", "marginBottom": "4px",
                           "display": "block"}),
                dbc.Textarea(id="etl-mapping-yaml",
                    style={"backgroundColor": BG_INPUT, "color": TEXT_PRI,
                           "border": f"1px solid {BORDER}",
                           "fontSize": "0.78rem", "height": "280px",
                           "fontFamily": "monospace"}),
                html.Div(id="etl-mapping-validation",
                         style={"marginTop": "8px", "fontSize": "0.82rem"}),
            ], id="etl-mapping-editor-wrapper", style={"display": "none"}),

            # Apply result
            html.Div(id="etl-apply-status", style={"marginTop": "10px"}),

            # Hidden stores
            dcc.Store(id="etl-uploaded-csv-store"),
            dcc.Store(id="etl-probe-store"),
        ], style={"padding": "8px 8px 24px", "maxWidth": "900px"}),
    ]),

    ]),  # end dbc.Tabs

    # ── Bottom tab navigation ────────────────────────────────────
    html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                *[html.Button(
                    label, id=f"bottom-tab-{tid}",
                    n_clicks=0,
                    style={"background": "none", "border": f"1px solid {BORDER}",
                           "borderRadius": "20px", "color": TEXT_SEC,
                           "fontSize": "0.72rem", "padding": "6px 18px",
                           "cursor": "pointer", "fontWeight": "500",
                           "letterSpacing": "0.2px",
                           "transition": "all 0.2s"},
                ) for tid, label in [
                    ("overview", "KPI Overview"), ("conversion", "Conversion"),
                    ("engagement", "Engagement"), ("retention", "Retention"),
                    ("churn", "Churn"), ("settings", "⚙ Settings"),
                ]],
            ], style={"display": "flex", "gap": "8px", "justifyContent": "center",
                      "flexWrap": "wrap"}),
            width=12),
        ]),
    ], style={"padding": "20px 0 10px", "borderTop": f"1px solid {BORDER}",
              "marginTop": "30px"}),

], fluid=True, style={
    "backgroundColor": BG_PAGE,
    "minHeight": "100vh",
    "paddingBottom": "60px",
    "paddingLeft": "0",
    "paddingRight": "0",
})


# ══════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════

# ── User filter store
@callback(
    Output("user-filter-store", "data"),
    Output("filter-badge-text", "children"),
    Input("apply-filter-btn", "n_clicks"),
    Input("reset-filter-btn", "n_clicks"),
    State("segment-dropdown", "value"),
    State("custom-ids-input", "value"),
    State("date-range-picker", "start_date"),
    State("date-range-picker", "end_date"),
    prevent_initial_call=True,
)
def update_user_store(apply_clicks, reset_clicks, segment, custom_text,
                      date_start, date_end):
    from dash import ctx
    d_start = str(date_start) if date_start else str(_DATE_MIN)
    d_end   = str(date_end)   if date_end   else str(_DATE_MAX)
    if ctx.triggered_id == "reset-filter-btn":
        return ({"mode": "all", "segment": "all", "user_ids": [],
                 "date_start": str(_DATE_MIN), "date_end": str(_DATE_MAX)},
                "Showing all users")

    custom_ids = []
    if custom_text and custom_text.strip():
        parts = custom_text.replace("\n", ",").split(",")
        custom_ids = [p.strip() for p in parts if p.strip().isdigit()]

    if custom_ids:
        badge = html.Span(
            [html.Span(f"{len(custom_ids):,} custom users", className="filter-badge")],
        )
        return {"mode": "custom", "segment": segment, "user_ids": custom_ids,
                "date_start": d_start, "date_end": d_end}, badge
    else:
        seg_label = next((s["label"] for s in SEGMENTS if s["value"] == segment), segment)
        n = len(get_user_ids(segment))
        badge = html.Span(
            [html.Span(f"{n:,} users · {seg_label}", className="filter-badge")]
        )
        return {"mode": "segment", "segment": segment, "user_ids": [],
                "date_start": d_start, "date_end": d_end}, badge


# ── Helper: filter events by date range from store
def _date_filtered_events(store, base_events=None):
    src = base_events if base_events is not None else events
    if not store:
        return src
    d_start = store.get("date_start")
    d_end   = store.get("date_end")
    if d_start and d_end:
        try:
            ts_start = pd.Timestamp(d_start)
            ts_end   = pd.Timestamp(d_end) + pd.Timedelta(days=1)
            return src[(src["timestamp"] >= ts_start) & (src["timestamp"] < ts_end)]
        except Exception:
            pass
    return src


import threading
from collections import OrderedDict

_filter_lock = threading.Lock()


def _store_key(store):
    """Canonical hashable key for a filter store. Returns None for unfiltered."""
    if not store:
        return None
    mode = store.get("mode", "all")
    segment = store.get("segment", "all")
    ids = tuple(sorted(store.get("user_ids") or []))
    d_s = store.get("date_start", str(_DATE_MIN))
    d_e = store.get("date_end", str(_DATE_MAX))
    unfiltered = (mode == "all" and d_s == str(_DATE_MIN) and d_e == str(_DATE_MAX))
    if unfiltered:
        return None
    return (mode, segment, ids, d_s, d_e)


# ── Filter data cache ────────────────────────────────────────────────
# Caches the triple (events, user_profiles, events_raw) per filter key.
_FILTER_DATA_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_FILTER_DATA_CACHE_MAX = 8


def _resolve_filter(store):
    """Return (filtered_events, filtered_profiles, filtered_events_raw) from store.
    Results are cached per canonical filter key (LRU, 8 entries)."""
    key = _store_key(store)
    if key is None:
        return events, user_profiles, events_raw
    cached = _FILTER_DATA_CACHE.get(key)
    if cached is not None:
        _FILTER_DATA_CACHE.move_to_end(key)
        return cached
    ev, up, ev_raw = events, user_profiles, events_raw
    uids = get_user_ids(store.get("segment", "all"), store.get("user_ids"))
    up = up.loc[up.index.intersection(uids)]
    ev_raw = ev_raw[ev_raw["user_id"].isin(uids)]
    ev = ev[ev["user_id"].isin(uids)]
    d_start = store.get("date_start")
    d_end = store.get("date_end")
    if d_start and d_end:
        try:
            ts_s = pd.Timestamp(d_start)
            ts_e = pd.Timestamp(d_end) + pd.Timedelta(days=1)
            ev_raw = ev_raw[(ev_raw["timestamp"] >= ts_s) & (ev_raw["timestamp"] < ts_e)]
            ev = ev[(ev["timestamp"] >= ts_s) & (ev["timestamp"] < ts_e)]
        except Exception:
            pass
    result = (ev, up, ev_raw)
    _FILTER_DATA_CACHE[key] = result
    if len(_FILTER_DATA_CACHE) > _FILTER_DATA_CACHE_MAX:
        _FILTER_DATA_CACHE.popitem(last=False)
    return result


# ── Generic per-figure cache: (callback_name, filter_key) → figure ───
_FIG_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_FIG_CACHE_MAX = 200


def _fig_cache_get(name, store):
    key = (name, _store_key(store))
    val = _FIG_CACHE.get(key)
    if val is not None:
        _FIG_CACHE.move_to_end(key)
    return val


def _fig_cache_put(name, store, fig):
    key = (name, _store_key(store))
    _FIG_CACHE[key] = fig
    if len(_FIG_CACHE) > _FIG_CACHE_MAX:
        _FIG_CACHE.popitem(last=False)


def _clear_fig_cache():
    _FIG_CACHE.clear()
    _FILTER_DATA_CACHE.clear()


def _with_filtered_globals(fn, store, *args, **kwargs):
    """Call a chart builder with module globals temporarily swapped to filtered data.
    Thread-safe via lock. Falls through to unfiltered call if store is empty."""
    if _store_key(store) is None:
        return fn(*args, **kwargs)
    ev_f, up_f, ev_raw_f = _resolve_filter(store)
    global events, user_profiles, events_raw
    global _CHURN_RISK, _comeback_dict, _cluster_user_map, _churn_probs
    with _filter_lock:
        saved = (events, user_profiles, events_raw,
                 _CHURN_RISK, _comeback_dict, _cluster_user_map, _churn_probs)
        try:
            events, user_profiles, events_raw = ev_f, up_f, ev_raw_f
            # Rebuild derived caches so charts that depend on them reflect filter
            try:
                _CHURN_RISK = _churn_risk_precompute()
            except Exception:
                pass
            try:
                _comeback_dict = _comeback_counts()
            except Exception:
                pass
            # Cluster map: restrict to the users present in filter
            try:
                uid_set = set(up_f.index.tolist())
                _cluster_user_map = {k: [u for u in v if u in uid_set]
                                     for k, v in _cluster_user_map.items()}
            except Exception:
                pass
            try:
                _churn_probs = _churn_probs.loc[_churn_probs.index.intersection(up_f.index)]
            except Exception:
                pass
            return fn(*args, **kwargs)
        finally:
            (events, user_profiles, events_raw,
             _CHURN_RISK, _comeback_dict, _cluster_user_map, _churn_probs) = saved


def _cached_figure(name, store, build_fn):
    """Lookup or compute a figure for (name, store). Builds via build_fn() on miss."""
    cached = _fig_cache_get(name, store)
    if cached is not None:
        return cached
    fig = build_fn()
    _fig_cache_put(name, store, fig)
    return fig


# ── Sankey (responds to sliders + user filter + date range)
@callback(
    Output("rete-sankey", "figure"),
    Input("rete-max-steps", "value"),
    Input("rete-threshold", "value"),
    Input("user-filter-store", "data"),
)
def update_sankey(max_steps, threshold, store):
    uid = None
    if store and store.get("mode") != "all":
        uid = get_user_ids(store.get("segment", "all"), store.get("user_ids"))
    # Date filter: rebuild Eventstream on date-sliced events if date range changed
    d_start = (store or {}).get("date_start", str(_DATE_MIN))
    d_end   = (store or {}).get("date_end", str(_DATE_MAX))
    if d_start == str(_DATE_MIN) and d_end == str(_DATE_MAX):
        return build_rete_sankey(max_steps=max_steps, threshold=threshold, user_ids=uid)
    # Rebuild on filtered events
    evts_f = _date_filtered_events(store)
    if uid is not None:
        evts_f = evts_f[evts_f["user_id"].isin(uid)]
    if evts_f.empty:
        return _empty_fig("No events in selected date range")
    rdf = (evts_f[["user_id", "event_name", "timestamp"]]
           .rename(columns={"event_name": "event"})
           .sort_values(["user_id", "timestamp"])
           .reset_index(drop=True))
    try:
        es_f = Eventstream(rdf)
        return build_rete_sankey(max_steps=max_steps, threshold=threshold, eventstream=es_f)
    except Exception:
        return build_rete_sankey(max_steps=max_steps, threshold=threshold, user_ids=uid)


# ── Friction score (multi-segment comparison)
@callback(
    Output("friction-chart", "figure"),
    Input("friction-group", "value"),
    Input("user-filter-store", "data"),
)
def update_friction(segments, store):
    uid = None
    if store and store.get("mode") == "custom" and store.get("user_ids"):
        uid = get_user_ids(custom_ids=store["user_ids"])
    if not segments:
        segments = ["all"]
    if isinstance(segments, str):
        segments = [segments]
    return build_friction_score(segments=segments, user_ids=uid)


# ── Silhouette quality helper
def _score_badge(score: float) -> html.Span:
    if score >= 0.50:
        color, label = COL_HI,      "Excellent"
    elif score >= 0.35:
        color, label = BRAND_TEAL,  "Good"
    elif score >= 0.20:
        color, label = BRAND_AMBER, "Fair"
    else:
        color, label = COL_LO,      "Weak"
    return html.Span(
        f"{label}  {score:.3f}",
        style={"backgroundColor": _rgba(color, 0.15),
               "border": f"1px solid {_rgba(color, 0.45)}",
               "color": color, "borderRadius": "20px",
               "padding": "2px 10px", "fontSize": "11px", "fontWeight": "600"},
    )


def _k_score_chips(k_scores: dict, active_k: int) -> html.Div:
    """Row of per-k score chips; the active k is highlighted."""
    chips = []
    for k, sc in sorted(k_scores.items()):
        is_active = k == active_k
        bg = _rgba(BRAND_BLUE, 0.25) if is_active else _rgba(BRAND_BLUE, 0.06)
        border = BRAND_BLUE if is_active else _rgba(BRAND_BLUE, 0.2)
        text_col = TEXT_PRI if is_active else TEXT_MUTED
        chips.append(html.Span(
            f"k{k}: {sc:.2f}",
            style={"backgroundColor": bg, "border": f"1px solid {border}",
                   "color": text_col, "borderRadius": "6px",
                   "padding": "2px 7px", "fontSize": "10px",
                   "marginRight": "4px", "display": "inline-block",
                   "fontWeight": "700" if is_active else "400"},
        ))
    return html.Div(chips, style={"marginTop": "6px", "flexWrap": "wrap"})


# ── Clustering
@callback(
    Output("cluster-scatter", "figure"),
    Output("cluster-summary-div", "children"),
    Output("cluster-user-store", "data"),
    Output("cluster-export-select", "options"),
    Output("cluster-score-display", "children"),
    Output("churn-cluster-chart", "figure"),
    Output("cluster-archetypes-div", "children"),
    Output("churn-cluster-filter", "options"),
    Output("churn-cluster-filter", "value"),
    Input("run-cluster-btn", "n_clicks"),
    State("cluster-features", "value"),
    State("cluster-n-select", "value"),
    State("user-filter-store", "data"),
    prevent_initial_call=False,
)
def update_clustering(n_clicks, features, n_k, store):
    uid = None
    if store and store.get("mode") != "all":
        uid = get_user_ids(store.get("segment", "all"), store.get("user_ids"))

    n_override = None if (n_k is None or n_k == "auto") else int(n_k)
    feat_list = features or DEFAULT_CLUSTER_FEATS
    fig, summary, cluster_user_map, k_scores = build_clustering(
        feat_list, uid, n_clusters_override=n_override)
    table = DataTable(
        data=summary.to_dict("records"),
        columns=[{"name": c, "id": c} for c in summary.columns],
        style_table={"overflowX": "auto", "borderRadius": "8px",
                     "border": f"1px solid {BORDER}"},
        style_header={"backgroundColor": BG_SURFACE, "color": TEXT_PRI,
                      "fontWeight": "600", "fontSize": "11px",
                      "border": f"1px solid {BORDER}",
                      "fontFamily": "Inter, system-ui, sans-serif"},
        style_data={"backgroundColor": BG_CARD, "color": TEXT_SEC,
                    "fontSize": "11px",
                    "border": f"1px solid {_rgba(BRAND_BLUE, 0.08)}",
                    "fontFamily": "Inter, system-ui, sans-serif"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": BG_SURFACE}
        ],
    )
    children = [
        html.H6("Cluster Summary", style={**SECTION_H, "marginTop": "16px"}),
        html.P("Mean values per cluster. Bubble size = LTV. "
               "Compare clusters to understand natural user archetypes.",
               style=SECTION_SUB),
        table,
    ]
    store_data = json.dumps({k: list(v) for k, v in cluster_user_map.items()})
    export_opts = [{"label": f"{k} ({len(v):,} users)", "value": k}
                   for k, v in cluster_user_map.items()]

    # Score display: quality badge + per-k chips
    active_k = n_override if n_override else (
        max(k_scores, key=k_scores.get) if k_scores else 2)
    active_score = k_scores.get(active_k, 0.0)
    score_display = html.Div([
        html.Div([
            html.Span("Silhouette: ", style={"color": TEXT_MUTED, "fontSize": "11px"}),
            _score_badge(active_score),
        ], style={"marginBottom": "5px"}),
        _k_score_chips(k_scores, active_k),
        html.P("Score: 1 = perfect clusters, 0 = overlapping. "
               ">0.35 is generally meaningful.",
               style={"color": TEXT_MUTED, "fontSize": "9px", "marginTop": "5px",
                      "marginBottom": "0"}),
    ]) if k_scores else html.Div()

    # ── Archetype cards — generate names FIRST so we can rename cluster_user_map
    archetypes = _generate_cluster_archetypes(summary, feat_list)
    archetype_cards = []
    palette = [BRAND_BLUE, BRAND_TEAL, BRAND_PURPLE, BRAND_AMBER,
               COL_HI, "#F06292", "#4DD0E1"]

    # Rename cluster_user_map keys from "Cluster 1" → archetype name
    renamed_map = {}
    raw_keys = list(cluster_user_map.keys())
    for i, (arch_name, arch_desc) in enumerate(archetypes):
        cluster_label = raw_keys[i] if i < len(raw_keys) else f"Cluster {i + 1}"
        display_name = f"{arch_name} ({cluster_label})"
        uids = cluster_user_map.get(cluster_label, [])
        renamed_map[display_name] = uids
        color = palette[i % len(palette)]
        archetype_cards.append(
            dbc.Col(html.Div([
                html.Div(cluster_label,
                         style={"color": TEXT_MUTED, "fontSize": "9px",
                                "textTransform": "uppercase", "letterSpacing": "0.6px",
                                "marginBottom": "2px"}),
                html.Div(arch_name,
                         style={"color": color, "fontSize": "1.0rem",
                                "fontWeight": "700", "lineHeight": "1.2",
                                "marginBottom": "4px"}),
                html.Div(arch_desc,
                         style={"color": TEXT_SEC, "fontSize": "11px",
                                "marginBottom": "4px"}),
                html.Div(f"{len(uids):,} users",
                         style={"color": TEXT_MUTED, "fontSize": "10px",
                                "fontStyle": "italic"}),
            ], style={
                "backgroundColor": BG_SURFACE,
                "border": f"1px solid {BORDER}",
                "borderLeft": f"3px solid {color}",
                "borderRadius": "8px",
                "padding": "12px 14px",
            }), md=3, sm=6, className="mb-2")
        )
    archetype_div = html.Div([
        html.H6("User Archetypes", style={**SECTION_H, "marginTop": "0"}),
        html.P("Plain-English descriptions of each cluster based on dominant signals.",
               style={**SECTION_SUB, "marginBottom": "10px"}),
        dbc.Row(archetype_cards),
    ]) if archetype_cards else html.Div()

    # Use renamed map for churn chart and dropdown
    churn_clust_fig = build_churn_vs_clusters(renamed_map)

    cluster_filter_opts = [{"label": "All clusters", "value": "all"}] + [
        {"label": k, "value": k} for k in renamed_map.keys()
    ]

    # Store renamed map so the filter callback can use it
    store_data = json.dumps({k: list(v) for k, v in renamed_map.items()})

    return (fig, children, store_data, export_opts, score_display,
            churn_clust_fig, archetype_div, cluster_filter_opts, "all")


# ── Churn cluster filter dropdown
@callback(
    Output("churn-cluster-chart", "figure", allow_duplicate=True),
    Input("churn-cluster-filter", "value"),
    State("cluster-user-store", "data"),
    prevent_initial_call=True,
)
def filter_churn_clusters(selected, store_json):
    if not store_json:
        return build_churn_vs_clusters(show_cluster=selected)
    cum = json.loads(store_json)
    return build_churn_vs_clusters(cluster_user_map=cum, show_cluster=selected)


# ── Goal tracking: update when user changes targets
@callback(
    Output("goal-cards-content", "children"),
    [Input(f"goal-target-{mid}", "value") for mid, _, _ in GOAL_DEFAULTS],
    prevent_initial_call=True,
)
def update_goal_targets(*target_vals):
    keys = [mid for mid, _, _ in GOAL_DEFAULTS]
    targets = {}
    for k, v in zip(keys, target_vals):
        if v is not None and v > 0:
            targets[k] = float(v)
    return [build_goal_tracking(targets)]


# ── Goal tracking: toggle target inputs visibility
@callback(
    Output("goal-targets-collapse", "is_open"),
    Input("goal-targets-toggle-btn", "n_clicks"),
    State("goal-targets-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_goal_targets(n, is_open):
    return not is_open


# ── Bottom tab navigation: switch main tabs + scroll to top
_BOTTOM_TAB_IDS = ["overview", "conversion", "engagement", "retention", "churn", "settings"]

@callback(
    Output("main-tabs", "active_tab"),
    [Input(f"bottom-tab-{tid}", "n_clicks") for tid in _BOTTOM_TAB_IDS],
    prevent_initial_call=True,
)
def bottom_tab_switch(*clicks):
    from dash import ctx
    triggered = ctx.triggered_id
    for tid in _BOTTOM_TAB_IDS:
        if triggered == f"bottom-tab-{tid}":
            return tid
    return dash.no_update


app.clientside_callback(
    "function() { window.scrollTo({top: 0, behavior: 'smooth'}); return window.dash_clientside.no_update; }",
    Output("main-tabs", "className"),
    [Input(f"bottom-tab-{tid}", "n_clicks") for tid in _BOTTOM_TAB_IDS],
    prevent_initial_call=True,
)


# ── Recommendation "Go to section" → switch tab + scroll to section
@callback(
    Output("main-tabs", "active_tab", allow_duplicate=True),
    Input({"type": "rec-go", "index": dash.ALL}, "n_clicks"),
    State("rec-targets-store", "data"),
    prevent_initial_call=True,
)
def rec_go_to_section(clicks, targets):
    if not any(clicks):
        return no_update
    triggered = ctx.triggered_id
    if isinstance(triggered, dict):
        idx = triggered.get("index", 0)
        if targets and idx < len(targets):
            return targets[idx]["tab"]
    return no_update

app.clientside_callback(
    """
    function() {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || ctx.triggered.length === 0)
            return window.dash_clientside.no_update;
        var prop_id = ctx.triggered[0].prop_id;
        var m = prop_id.match(/"index":(\\d+)/);
        if (!m) return window.dash_clientside.no_update;
        var idx = parseInt(m[1]);
        var targets = arguments[arguments.length - 1];
        if (targets && targets[idx]) {
            var secId = targets[idx].section;
            setTimeout(function() {
                var el = document.getElementById(secId);
                if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});
            }, 400);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("rec-targets-store", "data"),
    [Input({"type": "rec-go", "index": i}, "n_clicks") for i in range(6)],
    State("rec-targets-store", "data"),
    prevent_initial_call=True,
)


# ── Funnel responds to user filter + segmentation mode
@callback(
    Output("funnel-chart", "figure"),
    Input("user-filter-store", "data"),
    Input("funnel-seg-mode", "value"),
)
def update_funnel(store, seg_mode):
    mode = seg_mode or "all"
    uid = None
    if store and store.get("mode") != "all":
        uid = get_user_ids(store.get("segment", "all"), store.get("user_ids"))
    # Segmented mode overrides simple uid filter
    if mode in ("segment", "cluster"):
        return build_segmented_funnel_chart(mode)
    return build_conversion_funnel(uid)


# ── Churn donut responds to user filter
@callback(
    Output("churn-donut-chart", "figure"),
    Input("user-filter-store", "data"),
)
def update_churn_donut(store):
    uid = None
    if store and store.get("mode") != "all":
        uid = get_user_ids(store.get("segment", "all"), store.get("user_ids"))
    return build_churn_donut(uid)


# ── Cohort timeline responds to user filter
@callback(
    Output("cohort-chart", "figure"),
    Input("user-filter-store", "data"),
)
def update_cohort(store):
    uid = None
    if store and store.get("mode") != "all":
        uid = get_user_ids(store.get("segment", "all"), store.get("user_ids"))
    return build_cohort_timeline(uid)


# ── Individual user journey viewer (FullStory-like)
@callback(
    Output("journey-viewer-chart", "figure"),
    Output("journey-user-info", "children"),
    Output("journey-profile-panel", "children"),
    Output("journey-event-table", "data"),
    Output("journey-event-table", "columns"),
    Input("show-journey-btn", "n_clicks"),
    State("journey-user-id", "value"),
    State("journey-cat-filter", "value"),
    prevent_initial_call=True,
)
def update_journey(n_clicks, user_id, cat_filter):
    empty = (_empty_fig("Enter a user ID above"), "", html.Div(), [], [])
    if not user_id:
        return empty
    uid = int(user_id)
    cats = cat_filter or list(CATEGORY_COLORS.keys())

    fig     = build_user_journey_viewer(uid, include_categories=cats)
    profile = build_journey_profile(uid)
    tbl_data, tbl_cols = build_journey_event_table(uid, include_categories=cats)

    up_row = user_profiles[user_profiles.index == uid]
    if up_row.empty:
        info = html.Span(f"User {uid} not found", style={"color": COL_LO})
    else:
        r = up_row.iloc[0]
        ltv_val = f"${r['ltv']:.0f}" if r.get("ltv", 0) > 0 else "$0"
        info = html.Span([
            html.Span(f"LTV {ltv_val} · ", style={"color": BRAND_TEAL}),
            html.Span("Churned · " if r.get("churned") else "Active · ",
                      style={"color": COL_LO if r.get("churned") else COL_HI}),
            html.Span("Onboarded " if r.get("completed_onboarding") else "Not onboarded ",
                      style={"color": TEXT_SEC}),
            html.Span(f"· {int(r.get('active_days', 0))} active days",
                      style={"color": TEXT_MUTED}),
        ])
    return fig, info, profile, tbl_data, tbl_cols


# ── CSV export for opportunity matrix
@callback(
    Output("download-opp-csv", "data"),
    Input("export-opp-btn", "n_clicks"),
    prevent_initial_call=True,
)
def export_opp_csv(n_clicks):
    return dcc.send_data_frame(opp_df.to_csv, "opportunity_matrix.csv", index=False)


# ── Cohort retention heatmap (roll mode toggle)
@callback(
    Output("cohort-retention-heatmap", "figure"),
    Input("cohort-roll-mode", "value"),
)
def update_cohort_retention_heatmap(roll_mode):
    return build_cohort_retention_heatmap(roll_mode or "7d")


# ── Time-to-Value (TTV event dropdown + user filter)
@callback(
    Output("ttv-chart", "figure"),
    Input("ttv-event-select", "value"),
    Input("user-filter-store", "data"),
)
def update_ttv(value_event, store):
    uid = None
    if store and store.get("mode") != "all":
        uid = get_user_ids(store.get("segment", "all"), store.get("user_ids"))
    return build_time_to_value(value_event or "payment_success", user_ids=uid)


# ── Opportunity matrix y-axis toggle + segment filter
@callback(
    Output("opp-matrix-chart", "figure"),
    Output("opp-table-div", "children"),
    Output("opp-table-subtitle", "children"),
    Input("opp-y-axis", "value"),
    Input("user-filter-store", "data"),
)
def update_opp_matrix(y_mode, store):
    mode = y_mode or "users"
    subtitle = (
        "Sorted by LTV at Risk — estimated revenue lost from users who drop off at each event. "
        "Click column headers to re-sort."
        if mode == "ltv" else
        "Sorted by Priority Score = Friction × log(Traffic) × (1 + Drop-off/2). "
        "Click column headers to sort."
    )
    def _build():
        ev_f, up_f, _ = _resolve_filter(store)
        filtered_opp = _opp_matrix_data(_ev=ev_f, _up=up_f)
        return build_opportunity_matrix(mode, _opp=filtered_opp)
    fig = _cached_figure(f"opp-matrix-{mode}", store, _build)
    return fig, [build_opportunity_table(mode)], subtitle


# ── Cluster CSV export
@callback(
    Output("download-cluster-csv", "data"),
    Input("export-cluster-btn", "n_clicks"),
    State("cluster-user-store", "data"),
    State("cluster-export-select", "value"),
    prevent_initial_call=True,
)
def export_cluster_csv(n_clicks, store_data, cluster_name):
    if not cluster_name or not store_data:
        return dash.no_update
    cluster_map = json.loads(store_data)
    user_ids = cluster_map.get(cluster_name, [])
    df_out = pd.DataFrame({"user_id": user_ids})
    safe_name = cluster_name.lower().replace(" ", "_")
    return dcc.send_data_frame(df_out.to_csv, f"{safe_name}_user_ids.csv", index=False)


# ── Feature adoption responds to user filter
@callback(
    Output("feat-adopt-chart", "figure"),
    Input("user-filter-store", "data"),
)
def update_feature_adoption(store):
    uid = None
    if store and store.get("mode") != "all":
        uid = get_user_ids(store.get("segment", "all"), store.get("user_ids"))
    return build_feature_adoption(user_ids=uid)


# ── Session duration responds to user filter
@callback(
    Output("sess-dur-chart", "figure"),
    Input("user-filter-store", "data"),
)
def update_session_duration(store):
    uid = None
    if store and store.get("mode") != "all":
        uid = get_user_ids(store.get("segment", "all"), store.get("user_ids"))
    return build_session_duration(user_ids=uid)


# ── At-risk table — filter by risk + action dropdowns
@callback(
    Output("at-risk-table", "data"),
    Input("at-risk-risk-filter",   "value"),
    Input("at-risk-action-filter", "value"),
)
def filter_at_risk_table(risk_val, action_val):
    data = list(_AT_RISK_RECORDS)
    if risk_val and risk_val != "all":
        data = [r for r in data if r.get("Risk") == risk_val]
    if action_val and action_val != "all":
        data = [r for r in data if r.get("Recommended Action") == action_val]
    return data


# ── Download churned users CSV
@callback(
    Output("download-churned-csv", "data"),
    Input("download-churned-btn",  "n_clicks"),
    prevent_initial_call=True,
)
def download_churned_csv(n_clicks):
    df = _CHURN_RISK["churned_df"].copy().reset_index()
    df.rename(columns={"index": "user_id"}, inplace=True, errors="ignore")
    if "index" in df.columns:
        df = df.drop(columns=["index"])
    return dcc.send_data_frame(df.to_csv, "churned_users.csv", index=False)


# ── Churn predictor table — update on N change
@callback(
    Output("churn-predictor-table", "data"),
    Input("churn-predictor-n", "value"),
)
def update_churn_predictor(n):
    return _churn_predictor_data(n or 5)


# ── Light / dark theme toggle (clientside for instant response)
app.clientside_callback(
    """
    function(n_clicks, current) {
        if (!n_clicks) return current || 'light';
        var newTheme = (current === 'light') ? 'dark' : 'light';

        var dark = {
            bgPage:  '#060917', bgCard:  '#0C1228',
            font:    '#8BA2D3', grid:    'rgba(75,99,245,0.08)',
            zero:    'rgba(75,99,245,0.12)',
        };
        var light = {
            bgPage:  '#E8F0FE', bgCard:  '#F4F8FF',
            font:    '#2D4A7A', grid:    'rgba(59,82,217,0.15)',
            zero:    'rgba(59,82,217,0.25)',
        };
        var t = (newTheme === 'dark') ? dark : light;

        /* Toggle .dark-mode class — default CSS is already light */
        if (newTheme === 'dark') {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
        document.body.style.backgroundColor = t.bgPage;

        /* Update all Plotly chart backgrounds */
        document.querySelectorAll('.js-plotly-plot').forEach(function(el) {
            try {
                Plotly.relayout(el, {
                    paper_bgcolor: t.bgCard,
                    plot_bgcolor:  t.bgPage,
                    'font.color':  t.font,
                    'xaxis.gridcolor':      t.grid,
                    'yaxis.gridcolor':      t.grid,
                    'xaxis.zerolinecolor':  t.zero,
                    'yaxis.zerolinecolor':  t.zero,
                    'xaxis.tickfont.color': t.font,
                    'yaxis.tickfont.color': t.font,
                    'legend.font.color':    t.font,
                });
            } catch(e) {}
        });

        /* Update toggle button icon */
        var btn = document.getElementById('theme-toggle-btn');
        if (btn) btn.textContent = (newTheme === 'dark') ? '☀' : '🌙';

        return newTheme;
    }
    """,
    Output("theme-store", "data"),
    Input("theme-toggle-btn", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)


# ── Avg time between transitions scope toggle
@callback(
    Output("avg-time-fig", "figure"),
    Input("avg-time-scope", "value"),
)
def update_avg_time(scope):
    return build_avg_time_between(within_session=(scope == "within"))


# ── Period window change or date filter → rebuild KPI row with updated data
@callback(
    Output("kpi-row-content", "children"),
    Input("period-window", "value"),
    Input("user-filter-store", "data"),
)
def update_kpi_row(window, filter_store):
    days = int(window) if window else 30
    _up = None
    if filter_store:
        d_start = pd.Timestamp(filter_store.get("date_start", str(_DATE_MIN)))
        d_end   = pd.Timestamp(filter_store.get("date_end",   str(_DATE_MAX))) + pd.Timedelta(days=1)
        ev_f = events_raw[(events_raw["timestamp"] >= d_start) & (events_raw["timestamp"] < d_end)]
        if not ev_f.empty and len(ev_f) < len(events_raw):
            from etl import build_user_profiles
            fin = data.get("user_financials", pd.DataFrame())
            _up = build_user_profiles(ev_f, fin)
    return build_kpi_row(window_days=days, _up=_up)


# ── Period window or date filter → rebuild Product Health Trends sparklines
@callback(
    Output("sec-wrapper-health", "children"),
    Input("user-filter-store", "data"),
    Input("period-window", "value"),
)
def update_health_trends(filter_store, window):
    _up, _ev = None, None
    days = int(window) if window else 30
    if filter_store:
        d_start = pd.Timestamp(filter_store.get("date_start", str(_DATE_MIN)))
        d_end   = pd.Timestamp(filter_store.get("date_end",   str(_DATE_MAX))) + pd.Timedelta(days=1)
        ev_f = events_raw[(events_raw["timestamp"] >= d_start) & (events_raw["timestamp"] < d_end)]
        if not ev_f.empty and len(ev_f) < len(events_raw):
            from etl import build_user_profiles
            fin = data.get("user_financials", pd.DataFrame())
            _up = build_user_profiles(ev_f, fin)
            _ev = ev_f
    return build_product_health_trends(_up=_up, _ev=_ev, window_days=days)


# ── At-risk table row selection → populate journey viewer user ID
@callback(
    Output("selected-at-risk-user", "data"),
    Input("at-risk-table", "selected_rows"),
    State("at-risk-table", "data"),
    prevent_initial_call=True,
)
def link_at_risk_to_journey(selected_rows, table_data):
    if not selected_rows or not table_data:
        return dash.no_update
    row = table_data[selected_rows[0]]
    user_id = row.get("User ID") or row.get("user_id")
    if user_id is None:
        return dash.no_update
    return {"user_id": int(user_id)}


# ── Auto-load journey viewer when at-risk row is selected
@callback(
    Output("journey-user-id", "value"),
    Output("show-journey-btn", "n_clicks"),
    Input("selected-at-risk-user", "data"),
    State("show-journey-btn", "n_clicks"),
    prevent_initial_call=True,
)
def load_journey_from_at_risk(store_data, current_clicks):
    if not store_data or "user_id" not in store_data:
        return dash.no_update, dash.no_update
    user_id = store_data["user_id"]
    # Increment n_clicks to trigger the journey callback
    new_clicks = (current_clicks or 0) + 1
    return user_id, new_clicks


# ── Transition table show/hide toggle
@callback(
    Output("trans-table-collapse", "is_open"),
    Output("toggle-trans-table-btn", "children"),
    Input("toggle-trans-table-btn", "n_clicks"),
    State("trans-table-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_trans_table(n, is_open):
    if is_open:
        return False, "▶ Show Transition Table"
    return True, "▼ Hide Transition Table"


# ── Customize drawer toggle
@callback(
    Output("customize-drawer", "is_open"),
    Input("customize-btn", "n_clicks"),
    State("customize-drawer", "is_open"),
    prevent_initial_call=True,
)
def toggle_customize_drawer(n, is_open):
    return not is_open


# ── Checklist → section-visibility store
@callback(
    Output("section-visibility", "data"),
    Input("section-checklist", "value"),
)
def save_section_visibility(checked):
    return checked  # list of visible card IDs


# ── Apply visibility to each individual KPI / Health / Period card
@callback(
    [Output(cid, "style") for cid, _ in ALL_KPI_CARDS],
    Input("section-visibility", "data"),
)
def apply_kpi_card_visibility(checked):
    # None means "first load / all visible"
    visible = set(checked) if checked is not None else {cid for cid, _ in ALL_KPI_CARDS}
    return [
        {} if cid in visible else {"display": "none"}
        for cid, _ in ALL_KPI_CARDS
    ]



# ── Collapsible sections (pattern-matching callback)
from dash import MATCH as _MATCH

@callback(
    Output({"type": "sec-content", "index": _MATCH}, "style"),
    Output({"type": "sec-toggle",  "index": _MATCH}, "children"),
    Input({"type": "sec-toggle",   "index": _MATCH}, "n_clicks"),
    State({"type": "sec-content",  "index": _MATCH}, "style"),
    prevent_initial_call=True,
)
def toggle_section_collapse(n_clicks, current_style):
    """Show/hide section body when the ⊟/⊞ button is clicked."""
    is_hidden = bool(current_style) and current_style.get("display") == "none"
    if is_hidden:
        return {}, "⊟"
    return {"display": "none"}, "⊞"


# ── Global segment filter → rebuild static charts ────────────────────────────

# KPI Overview: health score distribution
@callback(
    Output("health-dist-fig", "figure"),
    Input("user-filter-store", "data"),
)
def update_health_dist(store):
    if _store_key(store) is None: return _health_dist_fig
    def _build():
        _, up_f, _ = _resolve_filter(store)
        scores = _compute_health_scores(_up=up_f)
        return build_health_score_distribution(scores, _up=up_f)
    return _cached_figure("health-dist-fig", store, _build)

# Conversion: bridge lift chart + onboarding funnel
@callback(Output("bridge-fig", "figure"), Input("user-filter-store", "data"))
def update_bridge(store):
    if _store_key(store) is None: return _bridge_fig
    def _build():
        ev_f, up_f, _ = _resolve_filter(store)
        return build_conversion_lift(_ev=ev_f, _up=up_f)
    return _cached_figure("bridge-fig", store, _build)

@callback(Output("onboard-fig", "figure"), Input("user-filter-store", "data"))
def update_onboard(store):
    if _store_key(store) is None: return _onboard_fig
    def _build():
        ev_f, up_f, _ = _resolve_filter(store)
        return build_onboarding_funnel(_ev=ev_f, _up=up_f)
    return _cached_figure("onboard-fig", store, _build)

# Conversion: decliners (negative-event retention) + engagement-matched lift
@callback(Output("decliners-fig", "figure"), Input("user-filter-store", "data"))
def update_decliners(store):
    if _store_key(store) is None: return _decliners_fig
    def _build():
        ev_f, up_f, _ = _resolve_filter(store)
        return build_conversion_decliners(_ev=ev_f, _up=up_f)
    return _cached_figure("decliners-fig", store, _build)

@callback(Output("matched-lift-fig", "figure"), Input("user-filter-store", "data"))
def update_matched_lift(store):
    if _store_key(store) is None: return _matched_lift_fig
    def _build():
        ev_f, up_f, _ = _resolve_filter(store)
        return build_engagement_matched_lift(_ev=ev_f, _up=up_f)
    return _cached_figure("matched-lift-fig", store, _build)

@callback(Output("neg-impact-fig", "figure"), Input("user-filter-store", "data"))
def update_neg_impact(store):
    if _store_key(store) is None: return _neg_impact_fig
    def _build():
        ev_f, up_f, _ = _resolve_filter(store)
        return build_negative_signal_impact(_ev=ev_f, _up=up_f)
    return _cached_figure("neg-impact-fig", store, _build)

# Engagement: self loops, activity heatmap, friction breakdown
@callback(Output("self-loop-fig", "figure"), Input("user-filter-store", "data"))
def update_self_loops(store):
    if _store_key(store) is None: return _self_loop_fig
    def _build():
        ev_f, _, _ = _resolve_filter(store)
        if store and store.get("mode") != "all":
            uids = set(ev_f["user_id"].unique())
            filtered_loops = {k: v for k, v in self_loops.items()
                              if any(uid in uids for uid in v.get("user_ids", []))} if isinstance(self_loops, dict) else self_loops
            if isinstance(self_loops, pd.DataFrame):
                filtered_loops = self_loops[self_loops["user_id"].isin(uids)] if "user_id" in self_loops.columns else self_loops
            return build_self_loop_chart(_loops=filtered_loops)
        return build_self_loop_chart()
    return _cached_figure("self-loop-fig", store, _build)

@callback(Output("activity-hmap-fig", "figure"), Input("user-filter-store", "data"))
def update_activity_hmap(store):
    if _store_key(store) is None: return _activity_hmap
    def _build():
        _, _, ev_raw_f = _resolve_filter(store)
        return build_activity_heatmap(_ev_raw=ev_raw_f)
    return _cached_figure("activity-hmap-fig", store, _build)

@callback(Output("friction-breakdown-fig", "figure"), Input("user-filter-store", "data"))
def update_friction_breakdown(store):
    if _store_key(store) is None: return _friction_breakdown
    def _build():
        ev_f, _, _ = _resolve_filter(store)
        return build_friction_breakdown(_ev=ev_f)
    return _cached_figure("friction-breakdown-fig", store, _build)

# Retention: cohort heatmap + stickiness
@callback(
    Output("cohort-retention-heatmap", "figure", allow_duplicate=True),
    Input("user-filter-store", "data"),
    prevent_initial_call=True,
)
def update_cohort_retention_filter(store):
    if _store_key(store) is None: return _cohort_ret_fig
    def _build():
        _, up_f, ev_raw_f = _resolve_filter(store)
        return build_cohort_retention(_up=up_f, _ev_raw=ev_raw_f)
    return _cached_figure("cohort-retention-heatmap", store, _build)

@callback(Output("stickiness-fig", "figure"), Input("user-filter-store", "data"))
def update_stickiness(store):
    if _store_key(store) is None: return _stickiness_fig
    def _build():
        _, _, ev_raw_f = _resolve_filter(store)
        dau_f = _compute_dau_mau(ev_raw_f)
        return build_stickiness_trend(_dau=dau_f)
    return _cached_figure("stickiness-fig", store, _build)

# Churn: behaviour comparison
@callback(Output("churn-beh-fig", "figure"), Input("user-filter-store", "data"))
def update_churn_beh(store):
    if _store_key(store) is None: return _churn_beh_fig
    def _build():
        _, up_f, _ = _resolve_filter(store)
        return build_churn_behavior_comparison(_up=up_f)
    return _cached_figure("churn-beh-fig", store, _build)


# Recommendations banner — recompute on filter change
@callback(Output("recs-section-wrapper", "children"),
          Input("user-filter-store", "data"))
def update_recs_section(store):
    if _store_key(store) is None:
        return _recs_section
    cached = _fig_cache_get("recs-section", store)
    if cached is not None:
        return cached
    recs = _with_filtered_globals(_compute_recommendations, store)
    section = build_recommendations_section(recs)
    _fig_cache_put("recs-section", store, section)
    return section

# ── Remaining charts: use _with_filtered_globals to swap module data ─────────

# Conversion tab
@callback(Output("aha-fig", "figure"), Input("user-filter-store", "data"))
def _flt_aha(s):
    if _store_key(s) is None: return _aha_fig
    return _cached_figure("aha-fig", s, lambda: _with_filtered_globals(build_aha_moment_chart, s))

@callback(Output("adopt-timing-fig", "figure"), Input("user-filter-store", "data"))
def _flt_adopt_timing(s):
    if _store_key(s) is None: return _adopt_timing_fig
    return _cached_figure("adopt-timing-fig", s, lambda: _with_filtered_globals(build_adoption_timing_chart, s))

@callback(Output("feat-ltv-fig", "figure"), Input("user-filter-store", "data"))
def _flt_feat_ltv(s):
    if _store_key(s) is None: return _feat_ltv_fig
    return _cached_figure("feat-ltv-fig", s, lambda: _with_filtered_globals(build_feature_ltv_impact, s))

@callback(Output("radar-fig", "figure"), Input("user-filter-store", "data"))
def _flt_radar(s):
    if _store_key(s) is None: return _radar_fig
    return _cached_figure("radar-fig", s, lambda: _with_filtered_globals(build_segment_radar, s))

@callback(Output("journey-cmp-fig", "figure"), Input("user-filter-store", "data"))
def _flt_journey_cmp(s):
    if _store_key(s) is None: return _journey_cmp
    return _cached_figure("journey-cmp-fig", s, lambda: _with_filtered_globals(build_revenue_journey_comparison, s))

@callback(Output("seg-kpi-fig", "figure"), Input("user-filter-store", "data"))
def _flt_seg_kpi(s):
    if _store_key(s) is None: return _seg_kpi_fig
    return _cached_figure("seg-kpi-fig", s, lambda: _with_filtered_globals(build_revenue_segment_kpis, s))

@callback(Output("ltv-fig", "figure"), Input("user-filter-store", "data"))
def _flt_ltv(s):
    if _store_key(s) is None: return _ltv_fig
    return _cached_figure("ltv-fig", s, lambda: _with_filtered_globals(build_ltv_distribution, s))

@callback(Output("mrr-fig", "figure"), Input("user-filter-store", "data"))
def _flt_mrr(s):
    if _store_key(s) is None: return _mrr_fig
    return _cached_figure("mrr-fig", s, lambda: _with_filtered_globals(build_mrr_trend, s))

@callback(Output("rev-conc-fig", "figure"), Input("user-filter-store", "data"))
def _flt_rev_conc(s):
    if _store_key(s) is None: return _rev_conc_fig
    return _cached_figure("rev-conc-fig", s, lambda: _with_filtered_globals(build_revenue_concentration, s))

@callback(Output("ltv-signals-fig", "figure"), Input("user-filter-store", "data"))
def _flt_ltv_signals(s):
    if _store_key(s) is None: return _ltv_signals_fig
    return _cached_figure("ltv-signals-fig", s, lambda: _with_filtered_globals(build_ltv_early_signals, s))

@callback(Output("ttv-hist-fig", "figure"), Input("user-filter-store", "data"))
def _flt_ttv_hist(s):
    if _store_key(s) is None: return _ttv_hist_fig
    return _cached_figure("ttv-hist-fig", s, lambda: _with_filtered_globals(build_ttv_histogram, s))

# Engagement tab
@callback(Output("step-dist-fig", "figure"), Input("user-filter-store", "data"))
def _flt_step_dist(s):
    if _store_key(s) is None: return _step_dist_fig
    return _cached_figure("step-dist-fig", s, lambda: _with_filtered_globals(build_step_distribution, s))

@callback(Output("event-calendar-fig", "figure"), Input("user-filter-store", "data"))
def _flt_event_cal(s):
    if _store_key(s) is None: return _event_calendar_fig
    return _cached_figure("event-calendar-fig", s, lambda: _with_filtered_globals(build_event_calendar, s))

@callback(Output("friction-seg-cmp-fig", "figure"), Input("user-filter-store", "data"))
def _flt_fric_seg(s):
    if _store_key(s) is None: return _friction_seg_cmp
    return _cached_figure("friction-seg-cmp-fig", s, lambda: _with_filtered_globals(build_friction_segment_comparison, s))

@callback(Output("comeback-fig", "figure"), Input("user-filter-store", "data"))
def _flt_comeback(s):
    if _store_key(s) is None: return _comeback_fig
    return _cached_figure("comeback-fig", s, lambda: _with_filtered_globals(build_comeback_loops, s))

@callback(Output("dropoff-fig", "figure"), Input("user-filter-store", "data"))
def _flt_dropoff(s):
    if _store_key(s) is None: return _dropoff_fig
    return _cached_figure("dropoff-fig", s, lambda: _with_filtered_globals(build_dropoff_risk_map, s))

@callback(Output("feature-funnel-fig", "figure"), Input("user-filter-store", "data"))
def _flt_feat_funnel(s):
    if _store_key(s) is None: return _feature_funnel_fig
    return _cached_figure("feature-funnel-fig", s, lambda: _with_filtered_globals(build_feature_adoption_funnel, s))

# Retention tab
@callback(Output("activation-fig", "figure"), Input("user-filter-store", "data"))
def _flt_activation(s):
    if _store_key(s) is None: return _activation_fig
    return _cached_figure("activation-fig", s, lambda: _with_filtered_globals(build_activation_health, s))

@callback(Output("rev-impact-fig", "figure"), Input("user-filter-store", "data"))
def _flt_rev_impact(s):
    if _store_key(s) is None: return _rev_impact_fig
    return _cached_figure("rev-impact-fig", s, lambda: _with_filtered_globals(lambda: build_revenue_impact_attribution()[0], s))

# Churn tab
@callback(Output("shap-fig", "figure"), Input("user-filter-store", "data"))
def _flt_shap(s):
    if _store_key(s) is None: return _shap_fig
    return _cached_figure("shap-fig", s, lambda: _with_filtered_globals(lambda: _build_churn_model()[0], s))

@callback(Output("churn-prob-fig", "figure"), Input("user-filter-store", "data"))
def _flt_churn_prob(s):
    if _store_key(s) is None: return _churn_prob_hist
    def _build():
        if s and s.get("mode") != "all":
            try:
                return _with_filtered_globals(
                    lambda: build_churn_prob_histogram(_compute_churn_probabilities(_churn_clf, _churn_feat_cols)), s)
            except Exception:
                return _churn_prob_hist
        return _churn_prob_hist
    return _cached_figure("churn-prob-fig", s, _build)

@callback(Output("survival-fig", "figure"), Input("user-filter-store", "data"))
def _flt_survival(s):
    if _store_key(s) is None: return _survival_fig
    return _cached_figure("survival-fig", s, lambda: _with_filtered_globals(lambda: build_survival_curve()[0], s))

@callback(Output("cohort-cmp-fig", "figure"), Input("user-filter-store", "data"))
def _flt_cohort_cmp(s):
    if _store_key(s) is None: return _cohort_cmp_fig
    return _cached_figure("cohort-cmp-fig", s, lambda: _with_filtered_globals(lambda: build_churn_cohort_comparison()[0], s))


# ── Per-tab AI panels: toggle, auto-summarise, follow-up chat ────────────────

def _summary_block(tab_key: str, store) -> object:
    """Render the 'Key Findings' summary for a tab using the filtered context."""
    if not _AI_AVAILABLE or not os.environ.get("ANTHROPIC_API_KEY"):
        return dbc.Alert(
            "Set your ANTHROPIC_API_KEY in Settings to enable AI insights.",
            color="info", style={"fontSize": "0.82rem"})
    label = _filter_label_from_store(store)
    header_tail = f"  \u2022  filter: {label}" if label else ""
    try:
        insight = generate_tab_insight(tab_key, _get_data_context(store))
    except Exception as e:
        return html.Span(f"\u26a0 {e}", style={"color": COL_LO, "fontSize": "0.82rem"})
    return html.Div([
        html.Div(
            [html.Span("Key Findings"), html.Span(header_tail, style={
                "color": TEXT_MUTED, "fontWeight": "500", "textTransform": "none",
                "letterSpacing": "0", "marginLeft": "8px"})],
            style={"fontWeight": "700", "fontSize": "0.78rem",
                   "color": BRAND_PURPLE, "marginBottom": "6px",
                   "textTransform": "uppercase", "letterSpacing": "0.4px"}),
        dcc.Markdown(insight, style={"fontSize": "0.84rem",
                                     "color": TEXT_SEC, "lineHeight": "1.5"}),
    ])


def _make_ai_panel_callbacks(tab_key: str):
    """Register collapse toggle + auto-summary + chat callbacks for one tab."""

    # Toggle panel open/closed AND (re)generate the summary when opened or when
    # the filter changes while panel is open.
    @callback(
        Output(f"ai-panel-collapse-{tab_key}", "is_open"),
        Output(f"ai-panel-summary-{tab_key}", "children"),
        Input(f"ai-panel-toggle-{tab_key}", "n_clicks"),
        Input("user-filter-store", "data"),
        State(f"ai-panel-collapse-{tab_key}", "is_open"),
        State(f"ai-panel-summary-{tab_key}", "children"),
        prevent_initial_call=True,
    )
    def toggle_and_summarise(_n, store, is_open, existing_summary):
        trigger = ctx.triggered_id
        # Case 1: user toggled the panel button
        if trigger == f"ai-panel-toggle-{tab_key}":
            new_open = not is_open
            if new_open:
                return new_open, _summary_block(tab_key, store)
            return new_open, existing_summary
        # Case 2: filter changed — only refresh if panel is already open
        if trigger == "user-filter-store" and is_open:
            return is_open, _summary_block(tab_key, store)
        return is_open, no_update

    # Chat follow-up — uses the filter-aware context (via system prompt)
    @callback(
        Output(f"ai-panel-store-{tab_key}", "data"),
        Output(f"ai-panel-history-{tab_key}", "children"),
        Output(f"ai-panel-input-{tab_key}", "value"),
        Input(f"ai-panel-send-{tab_key}", "n_clicks"),
        State(f"ai-panel-input-{tab_key}", "value"),
        State(f"ai-panel-store-{tab_key}", "data"),
        State("user-filter-store", "data"),
        prevent_initial_call=True,
    )
    def panel_chat(n_clicks, user_input, history, store):
        question = (user_input or "").strip()
        if not question:
            return no_update, no_update, no_update
        if not _AI_AVAILABLE:
            return no_update, [_chat_bubble("assistant", "\u26a0 ai_utils not available.")], ""
        tab_hint = _TAB_PROMPTS.get(tab_key, tab_key)
        scoped_q = f"(Focus on {tab_hint}.) {question}"
        try:
            reply = ask_claude(scoped_q, _get_data_context(store), history or None)
        except ValueError as e:
            reply = f"\u26a0 {e}"
        except Exception as e:
            reply = f"\u26a0 Error: {e}"
        new_history = list(history or [])
        new_history.append({"role": "user", "content": question})
        new_history.append({"role": "assistant", "content": reply})
        bubbles = [_chat_bubble(m["role"], m["content"]) for m in new_history]
        return new_history, bubbles, ""

for _tk in ["overview", "conversion", "engagement", "retention", "churn"]:
    _make_ai_panel_callbacks(_tk)


# ── Settings: save API key ────────────────────────────────────────────────────
@callback(
    Output("settings-api-key-status", "children"),
    Input("settings-api-key-save", "n_clicks"),
    State("settings-api-key", "value"),
    prevent_initial_call=True,
)
def handle_api_key_save(n_clicks, api_key):
    if not api_key or not api_key.strip():
        return dbc.Alert("No key entered.", color="warning", dismissable=True)
    os.environ["ANTHROPIC_API_KEY"] = api_key.strip()
    _invalidate_ai_caches()
    if _AI_AVAILABLE:
        try:
            import ai_utils as _au
            _au.reset_client()
        except Exception:
            pass
    return dbc.Alert("API key saved for this session.", color="success", dismissable=True)


# ══════════════════════════════════════════════════════════════
#  AI-ASSISTED ETL CALLBACKS
# ══════════════════════════════════════════════════════════════

# Toggle visibility of CSV-upload vs SQL-config block based on source type
@callback(
    Output("etl-csv-upload-wrapper", "style"),
    Output("etl-sql-wrapper", "style"),
    Input("etl-source-type", "value"),
)
def _etl_toggle_source_ui(source_type):
    show = {"marginBottom": "8px"}
    hide = {"display": "none", "marginBottom": "8px"}
    if source_type == "sql":
        return hide, show
    return show, hide


def _make_probe_info(probe, source_label) -> object:
    """Render a compact summary of what the connector sees."""
    top_events = probe.event_name_samples[:8]
    rows = [html.Tr([html.Td(name, style={"fontFamily": "monospace",
                                          "fontSize": "0.78rem"}),
                     html.Td(f"{cnt:,}", style={"textAlign": "right",
                                                "fontSize": "0.78rem"})])
            for name, cnt in top_events]
    return html.Div([
        html.Div([
            html.Strong(source_label),
            html.Span(f"  \u2022  {probe.total_events:,} events",
                      style={"color": TEXT_MUTED, "marginLeft": "6px"})
            if probe.total_events else "",
        ]),
        html.Div(f"Columns: {', '.join(probe.event_columns)}",
                 style={"color": TEXT_MUTED, "fontSize": "0.78rem",
                        "marginTop": "4px"}),
        html.Table(rows, style={"marginTop": "8px", "fontSize": "0.82rem"}),
    ])


def _build_probe_from_upload(contents, filename) -> tuple:
    """Decode a browser-uploaded CSV and probe it via DataFrameConnector.
    Returns (probe, connector, err_or_None)."""
    if not contents:
        return None, None, "No file provided"
    import base64, io
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.BytesIO(decoded))
    except Exception as e:
        return None, None, f"Could not parse CSV: {e}"
    connector = DataFrameConnector(df)
    return connector.probe(), connector, None


# Infer a mapping via AI
@callback(
    Output("etl-probe-info", "children"),
    Output("etl-probe-store", "data"),
    Output("etl-mapping-yaml", "value"),
    Output("etl-mapping-editor-wrapper", "style"),
    Output("etl-apply-btn", "disabled"),
    Output("etl-uploaded-csv-store", "data"),
    Input("etl-infer-btn", "n_clicks"),
    State("etl-source-type", "value"),
    State("etl-csv-upload", "contents"),
    State("etl-csv-upload", "filename"),
    State("etl-sql-url", "value"),
    State("etl-sql-table", "value"),
    State("etl-hints", "value"),
    prevent_initial_call=True,
)
def etl_infer_mapping(n, source_type, csv_contents, csv_filename,
                      sql_url, sql_table, hints):
    if not n:
        return no_update, no_update, no_update, no_update, no_update, no_update
    if not _AI_AVAILABLE or not os.environ.get("ANTHROPIC_API_KEY"):
        return (dbc.Alert("Set ANTHROPIC_API_KEY first (above) to enable AI mapping.",
                          color="warning"),
                no_update, "", {"display": "none"}, True, no_update)

    # Build a connector + probe for the chosen source
    uploaded_csv = no_update
    if source_type == "csv":
        if not csv_contents:
            return (dbc.Alert("Upload a CSV first.", color="warning"),
                    no_update, "", {"display": "none"}, True, no_update)
        probe, connector, err = _build_probe_from_upload(csv_contents, csv_filename)
        if err:
            return (dbc.Alert(err, color="danger"),
                    no_update, "", {"display": "none"}, True, no_update)
        source_label = csv_filename or "Uploaded CSV"
        uploaded_csv = csv_contents  # persist so Apply can re-read
    elif source_type == "sql":
        if not sql_url or not sql_url.strip():
            return (dbc.Alert("Enter a SQL URL first.", color="warning"),
                    no_update, "", {"display": "none"}, True, no_update)
        try:
            connector = SQLConnector(sql_url.strip(),
                                     events_table=(sql_table or "events").strip())
            probe = connector.probe()
        except Exception as e:
            return (dbc.Alert(f"SQL connection failed: {e}", color="danger"),
                    no_update, "", {"display": "none"}, True, no_update)
        source_label = f"SQL: {sql_url[:40]}…"
    else:
        return no_update, no_update, no_update, no_update, no_update, no_update

    # Ask Claude to propose a mapping
    try:
        yaml_text = infer_data_mapping(
            probe_summary=probe.to_summary(),
            canonical_events=sorted(CANONICAL_EVENTS),
            extra_hints=(hints or ""),
        )
    except Exception as e:
        return (dbc.Alert(f"AI mapping failed: {e}", color="danger"),
                no_update, "", {"display": "none"}, True, no_update)

    probe_info = _make_probe_info(probe, source_label)
    # Persist the source so the Apply step can reproduce it
    probe_payload = {
        "source_type": source_type,
        "sql_url": (sql_url or "").strip() if source_type == "sql" else None,
        "sql_table": (sql_table or "events").strip() if source_type == "sql" else None,
        "filename": csv_filename if source_type == "csv" else None,
    }
    return (probe_info, probe_payload, yaml_text,
            {"display": "block"}, False, uploaded_csv)


# Live validation of the edited YAML
@callback(
    Output("etl-mapping-validation", "children"),
    Input("etl-mapping-yaml", "value"),
    prevent_initial_call=True,
)
def etl_validate_mapping(yaml_text):
    if not yaml_text or not yaml_text.strip():
        return html.Span("(empty)", style={"color": TEXT_MUTED})
    try:
        import yaml as _yaml
        d = _yaml.safe_load(yaml_text)
        mapping = DataMapping.from_dict(d)
    except Exception as e:
        return html.Span(f"\u26a0 YAML parse error: {e}",
                         style={"color": COL_LO})
    errors = mapping.validate()
    if errors:
        return html.Div([html.Span("\u26a0 Issues:",
                                    style={"color": COL_LO, "fontWeight": "600"}),
                         html.Ul([html.Li(e) for e in errors],
                                 style={"color": COL_LO, "marginTop": "4px",
                                        "marginBottom": "0"})])
    n_canonical = len(mapping.event_map)
    n_raw = sum(len(v) for v in mapping.event_map.values())
    return html.Span(
        f"\u2713 Valid. {n_canonical} canonical events, "
        f"{n_raw} raw events mapped.",
        style={"color": COL_HI})


# Apply the reviewed mapping and reload the dashboard data
@callback(
    Output("etl-apply-status", "children"),
    Input("etl-apply-btn", "n_clicks"),
    State("etl-mapping-yaml", "value"),
    State("etl-probe-store", "data"),
    State("etl-uploaded-csv-store", "data"),
    prevent_initial_call=True,
)
def etl_apply_mapping(n, yaml_text, probe_payload, uploaded_csv):
    if not n:
        return no_update
    if not yaml_text or not probe_payload:
        return dbc.Alert("Nothing to apply.", color="warning")
    try:
        import yaml as _yaml
        mapping = DataMapping.from_dict(_yaml.safe_load(yaml_text))
    except Exception as e:
        return dbc.Alert(f"YAML parse error: {e}", color="danger")
    errors = mapping.validate()
    if errors:
        return dbc.Alert(
            [html.Strong("Fix validation errors first:"),
             html.Ul([html.Li(e) for e in errors])],
            color="danger")

    # Build the connector and run ETL
    try:
        if probe_payload["source_type"] == "csv":
            if not uploaded_csv:
                return dbc.Alert(
                    "Uploaded CSV not in memory (try re-uploading).",
                    color="warning")
            import base64, io
            _, content_string = uploaded_csv.split(",", 1)
            df = pd.read_csv(io.BytesIO(base64.b64decode(content_string)))
            connector = DataFrameConnector(df)
        else:
            connector = SQLConnector(
                probe_payload["sql_url"],
                events_table=probe_payload["sql_table"] or "events")

        new_data = run_etl_from_connector(connector, mapping=mapping)
    except Exception as e:
        return dbc.Alert(f"ETL failed: {e}", color="danger")

    # Persist the mapping YAML so it's reusable next session
    try:
        cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "configs")
        os.makedirs(cfg_dir, exist_ok=True)
        slug = (mapping.source_name or "custom").replace(" ", "_").lower()
        save_path = os.path.join(cfg_dir, f"mapping_{slug}.yaml")
        save_mapping(mapping, save_path)
    except Exception:
        save_path = None

    # Swap globals in a thread-safe way — matches _with_filtered_globals style
    global events, events_raw, user_profiles, data
    global opp_df, _CHURN_RISK, _comeback_dict, dau_mau_df
    with _filter_lock:
        data = new_data
        events_raw = new_data["events"]
        GENERIC_SET = {"page_view", "click", "session_start", "session_end"}
        events = events_raw[~events_raw["event_name"].isin(GENERIC_SET)].copy()
        user_profiles = new_data["user_profiles"]
        # Invalidate caches so charts recompute on next interaction
        _FIG_CACHE.clear()
        _FILTER_DATA_CACHE.clear()
        _DATA_CONTEXT_CACHE.clear()
        # Rebuild heavy derived caches
        try:
            opp_df = _opp_matrix_data()
        except Exception:
            pass
        try:
            _CHURN_RISK = _churn_risk_precompute()
        except Exception:
            pass
        try:
            _comeback_dict = _comeback_counts()
        except Exception:
            pass
        try:
            dau_mau_df = _compute_dau_mau()
        except Exception:
            pass

    cov = coverage_report(
        new_data["events"].rename(columns={"event_name": "event_name"}),
        mapping)
    saved_msg = (f" Saved mapping to configs/{os.path.basename(save_path)}."
                 if save_path else "")
    return dbc.Alert(
        [html.Strong("\u2713 Data replaced successfully."),
         html.Div(f"{len(events):,} meaningful events \u2022 "
                  f"{len(user_profiles):,} users \u2022 "
                  f"mapping coverage: {cov.get('mapped_pct', 0)}%"),
         html.Small(
             "Refresh the page or switch tabs to see updated charts." + saved_msg,
             style={"color": TEXT_MUTED, "display": "block", "marginTop": "6px"}),
         ],
        color="success", dismissable=True)


# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  Dashboard → http://127.0.0.1:8050\n")
    app.run(debug=False, port=8050)
