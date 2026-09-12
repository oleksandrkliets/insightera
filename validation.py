"""
Data validation for the canonical layer.

Why this exists:
    A mapping can be structurally valid and still be semantically wrong — right
    columns, plausible types, completely incorrect numbers. Structural checks
    alone won't catch a mapping that folded two events into one and produced a
    340% conversion rate. Five layers, each catching what the others cannot:

      1. structural     — required columns, dtypes, null rates
      2. semantic       — values that are individually implausible
      3. plausibility   — aggregate metrics that are physically impossible
      4. reconciliation — our totals vs what the source system itself reports
      5. referential    — joins actually resolve

Design principle — warn, don't reject:
    Customer data is always messy. Rigid validation means a broken dashboard on
    day one, which is worse than a dashboard with visible caveats. Only genuinely
    unusable data BLOCKs; everything else surfaces as a warning the user can see.

Usage:
    report = validate_events(events_df)
    if report.blocked:
        raise ValueError(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
#  SEVERITY
# ══════════════════════════════════════════════════════════════════════════════

BLOCK = "BLOCK"   # data is unusable — refuse to commit the mapping
WARN = "WARN"     # usable but degraded — apply, but surface prominently
INFO = "INFO"     # worth knowing, no action needed

_SEVERITY_ORDER = {BLOCK: 0, WARN: 1, INFO: 2}


@dataclass
class Finding:
    """A single validation result."""
    severity: str
    layer: str
    check: str
    message: str
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity}] {self.layer}/{self.check}: {self.message}"


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)

    def add(self, severity: str, layer: str, check: str,
            message: str, **detail: Any) -> None:
        self.findings.append(Finding(severity, layer, check, message, detail))

    @property
    def blocked(self) -> bool:
        return any(f.severity == BLOCK for f in self.findings)

    def by_severity(self, severity: str) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    @property
    def counts(self) -> dict[str, int]:
        return {s: len(self.by_severity(s)) for s in (BLOCK, WARN, INFO)}

    def sorted_findings(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: _SEVERITY_ORDER[f.severity])

    def summary(self) -> str:
        if not self.findings:
            return "All validation checks passed."
        c = self.counts
        head = f"{c[BLOCK]} blocking, {c[WARN]} warnings, {c[INFO]} notes"
        lines = [head] + [f"  {f}" for f in self.sorted_findings()]
        return "\n".join(lines)

    def merge(self, other: "ValidationReport") -> "ValidationReport":
        return ValidationReport(self.findings + other.findings)

    def to_dict(self) -> dict:
        """Serialisable form, for the Data Health Report UI."""
        return {
            "blocked": self.blocked,
            "counts": self.counts,
            "findings": [
                {"severity": f.severity, "layer": f.layer, "check": f.check,
                 "message": f.message, "detail": f.detail}
                for f in self.sorted_findings()
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — STRUCTURAL CONTRACTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FieldContract:
    name: str
    kind: str                              # "int" | "float" | "datetime" | "string" | "bool"
    required: bool = True
    max_null_pct: float = 0.0              # tolerated null percentage
    allowed_values: Optional[set] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def check_kind(self, s: pd.Series) -> bool:
        if self.kind == "int":
            return pd.api.types.is_integer_dtype(s)
        if self.kind == "float":
            return pd.api.types.is_numeric_dtype(s)
        if self.kind == "datetime":
            return pd.api.types.is_datetime64_any_dtype(s)
        if self.kind == "bool":
            return pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s)
        return True  # "string" — object dtype covers most cases


EVENTS_CONTRACT: list[FieldContract] = [
    FieldContract("user_id",    "int",      required=True,  max_null_pct=0.0),
    FieldContract("timestamp",  "datetime", required=True,  max_null_pct=0.0),
    FieldContract("event_name", "string",   required=True,  max_null_pct=0.0),
]

PROFILES_CONTRACT: list[FieldContract] = [
    FieldContract("total_events",    "float", max_null_pct=0.0, min_value=0),
    FieldContract("active_days",     "float", max_null_pct=0.0, min_value=0),
    FieldContract("first_event",     "datetime"),
    FieldContract("last_event",      "datetime"),
    FieldContract("converted",       "bool"),
    FieldContract("churned",         "bool"),
    FieldContract("revenue_to_date", "float", max_null_pct=0.0, min_value=0),
]


def check_structure(df: pd.DataFrame, contract: list[FieldContract],
                    label: str) -> ValidationReport:
    """Layer 1 — required columns present, correct types, nulls within tolerance."""
    r = ValidationReport()

    if df.empty:
        r.add(BLOCK, "structural", "empty", f"{label} contains no rows.")
        return r

    for fc in contract:
        if fc.name not in df.columns:
            sev = BLOCK if fc.required else WARN
            r.add(sev, "structural", "missing_column",
                  f"{label} is missing required column '{fc.name}'.",
                  column=fc.name)
            continue

        s = df[fc.name]

        if not fc.check_kind(s):
            r.add(WARN, "structural", "dtype",
                  f"'{fc.name}' is {s.dtype}, expected {fc.kind}.",
                  column=fc.name, actual=str(s.dtype), expected=fc.kind)

        null_pct = float(s.isna().mean() * 100)
        if null_pct > fc.max_null_pct:
            sev = BLOCK if (fc.required and null_pct > 50) else WARN
            r.add(sev, "structural", "nulls",
                  f"'{fc.name}' is {null_pct:.1f}% null "
                  f"(tolerated: {fc.max_null_pct}%).",
                  column=fc.name, null_pct=round(null_pct, 2))

        if fc.allowed_values is not None:
            bad = set(s.dropna().unique()) - fc.allowed_values
            if bad:
                pct = float(s.isin(list(bad)).mean() * 100)
                r.add(WARN, "structural", "unexpected_values",
                      f"'{fc.name}' has {len(bad)} value(s) outside the "
                      f"allowed set ({pct:.1f}% of rows).",
                      column=fc.name, examples=sorted(map(str, bad))[:10])

        if fc.min_value is not None and pd.api.types.is_numeric_dtype(s):
            n_below = int((s < fc.min_value).sum())
            if n_below:
                r.add(WARN, "structural", "below_min",
                      f"'{fc.name}' has {n_below:,} value(s) below "
                      f"{fc.min_value}.", column=fc.name, count=n_below)

    return r


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — SEMANTIC
# ══════════════════════════════════════════════════════════════════════════════

def check_semantics(events: pd.DataFrame,
                    canonical_events: Optional[set] = None) -> ValidationReport:
    """Layer 2 — individual values that are implausible in context."""
    r = ValidationReport()
    if events.empty or "timestamp" not in events.columns:
        return r

    ts = events["timestamp"]
    now = pd.Timestamp.utcnow().tz_localize(None)

    # Future timestamps — a small number is clock skew, a lot is a parsing bug
    future_pct = float((ts > now + pd.Timedelta(days=1)).mean() * 100)
    if future_pct > 0:
        sev = BLOCK if future_pct > 5 else WARN
        r.add(sev, "semantic", "future_timestamps",
              f"{future_pct:.2f}% of events are dated in the future — "
              f"likely a timestamp unit or timezone parsing error.",
              future_pct=round(future_pct, 3))

    # Implausibly old — usually epoch-zero from a failed parse
    ancient_pct = float((ts < pd.Timestamp("2000-01-01")).mean() * 100)
    if ancient_pct > 0:
        sev = BLOCK if ancient_pct > 5 else WARN
        r.add(sev, "semantic", "ancient_timestamps",
              f"{ancient_pct:.2f}% of events predate 2000 — "
              f"likely epoch-zero from a failed parse.",
              ancient_pct=round(ancient_pct, 3))

    # Timezone consistency
    if pd.api.types.is_datetime64_any_dtype(ts) and ts.dt.tz is not None:
        r.add(INFO, "semantic", "timezone",
              f"Timestamps are timezone-aware ({ts.dt.tz}). The pipeline "
              f"expects UTC-normalised naive timestamps.")

    # Load-bearing canonical events. This matters more than the raw unmapped
    # percentage: a customer with many bespoke events we don't need is fine,
    # but losing the events the charts are built on is not. Each group needs at
    # least one member present for the dashboard to mean anything.
    if "event_name" in events.columns:
        present = set(events["event_name"].unique())
        required_groups = {
            "user arrival": {"signup", "session_start"},
            "engagement": {"view_dashboard", "view_feature_A", "view_feature_B",
                           "view_feature_C", "view_feature_D", "page_view",
                           "click", "invite_team"},
        }
        missing = [label for label, opts in required_groups.items()
                   if not (opts & present)]
        if missing:
            r.add(BLOCK, "semantic", "missing_critical_events",
                  f"No events mapped for: {', '.join(missing)}. The dashboard "
                  f"cannot produce meaningful analysis without these.",
                  missing_groups=missing)

    # Canonical coverage
    if canonical_events and "event_name" in events.columns:
        counts = events["event_name"].value_counts()
        unmapped = [str(n) for n in counts.index if n not in canonical_events]
        if unmapped:
            unmapped_pct = float(
                events["event_name"].isin(unmapped).mean() * 100)
            sev = (BLOCK if unmapped_pct > 80
                   else WARN if unmapped_pct > 20 else INFO)
            r.add(sev, "semantic", "unmapped_events",
                  f"{len(unmapped)} event name(s) are not canonical, covering "
                  f"{unmapped_pct:.1f}% of rows.",
                  unmapped_pct=round(unmapped_pct, 2),
                  examples=unmapped[:10])

    # user_id type — the silent join-killer
    if "user_id" in events.columns:
        uid = events["user_id"].dropna()
        if not uid.empty and not (
            pd.api.types.is_integer_dtype(uid)
            or pd.api.types.is_string_dtype(uid)
        ):
            r.add(WARN, "semantic", "user_id_type",
                  f"user_id has dtype {uid.dtype}; mixed or float ids break "
                  f"joins against finance and CRM sources silently.",
                  dtype=str(uid.dtype))

    return r


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — STATISTICAL PLAUSIBILITY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlausibilityRule:
    metric: str
    predicate: Callable[[float], bool]
    describe: str
    severity: str = WARN


PLAUSIBILITY_RULES: list[PlausibilityRule] = [
    PlausibilityRule("conversion_rate_pct", lambda v: 0 <= v <= 100,
                     "must be between 0 and 100%", BLOCK),
    PlausibilityRule("churn_rate_pct", lambda v: 0 <= v <= 100,
                     "must be between 0 and 100%", BLOCK),
    PlausibilityRule("events_per_user", lambda v: 0 < v < 100_000,
                     "outside a plausible range"),
    PlausibilityRule("duplicate_pct", lambda v: v < 5.0,
                     "duplicate event rate above 5%"),
    PlausibilityRule("null_user_id_pct", lambda v: v < 1.0,
                     "more than 1% of events have no user"),
    PlausibilityRule("single_user_share_pct", lambda v: v < 50.0,
                     "one user accounts for over half of all events"),
    # A cohort comparison needs both sides populated. With, say, 2 retained
    # users out of 15,000, every "Active %" is 0, 50 or 100 — arithmetically
    # correct and analytically worthless. Flag it so the numbers are read with
    # the right scepticism rather than taken at face value.
    PlausibilityRule("smaller_churn_cohort", lambda v: v >= 30,
                     "too few users on one side of the churn line for cohort "
                     "comparisons or churn modelling to be meaningful"),
]


def compute_metrics(events: pd.DataFrame,
                    profiles: Optional[pd.DataFrame] = None) -> dict[str, float]:
    """Aggregate metrics the plausibility layer reasons about."""
    m: dict[str, float] = {}
    if events.empty:
        return m

    n = len(events)
    if "user_id" in events.columns:
        m["null_user_id_pct"] = float(events["user_id"].isna().mean() * 100)
        nu = events["user_id"].nunique()
        m["events_per_user"] = n / max(nu, 1)
        top = events["user_id"].value_counts()
        if len(top):
            m["single_user_share_pct"] = float(top.iloc[0] / n * 100)

    key = [c for c in ("user_id", "timestamp", "event_name")
           if c in events.columns]
    if key:
        m["duplicate_pct"] = float(events.duplicated(subset=key).mean() * 100)

    if profiles is not None and not profiles.empty:
        if "converted" in profiles.columns:
            m["conversion_rate_pct"] = float(profiles["converted"].mean() * 100)
        if "churned" in profiles.columns:
            m["churn_rate_pct"] = float(profiles["churned"].mean() * 100)
            n_churn = int(profiles["churned"].sum())
            m["smaller_churn_cohort"] = float(
                min(n_churn, len(profiles) - n_churn))

    return m


def check_plausibility(metrics: dict[str, float]) -> ValidationReport:
    """Layer 3 — aggregate figures that are physically impossible.

    Catches the failure mode structural checks cannot: a mapping that produced
    the right shape but the wrong numbers.
    """
    r = ValidationReport()
    for rule in PLAUSIBILITY_RULES:
        if rule.metric not in metrics:
            continue
        value = metrics[rule.metric]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if not rule.predicate(value):
            r.add(rule.severity, "plausibility", rule.metric,
                  f"{rule.metric} = {value:,.2f} — {rule.describe}.",
                  value=round(float(value), 4))
    return r


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 4 — RECONCILIATION
# ══════════════════════════════════════════════════════════════════════════════

def reconcile(computed: float, source_reported: float, label: str,
              tolerance: float = 0.02) -> ValidationReport:
    """Layer 4 — compare our figure against what the source system reports.

    The highest-value check available. A mapping can pass every other layer and
    still be wrong; only the source's own totals reveal it. If your dashboard
    says $84k MRR and Stripe says $97k, you want to find that before a customer
    screenshots it.
    """
    r = ValidationReport()
    if source_reported in (None, 0) or pd.isna(source_reported):
        r.add(INFO, "reconciliation", label,
              f"No source-reported figure available for {label}; skipped.")
        return r

    delta = abs(computed - source_reported) / abs(source_reported)
    if delta > tolerance:
        sev = BLOCK if delta > 0.25 else WARN
        r.add(sev, "reconciliation", label,
              f"{label}: computed {computed:,.2f} vs source-reported "
              f"{source_reported:,.2f} ({delta * 100:.1f}% apart).",
              computed=computed, reported=source_reported,
              delta_pct=round(delta * 100, 2))
    else:
        r.add(INFO, "reconciliation", label,
              f"{label} reconciles within {tolerance * 100:.0f}% "
              f"({delta * 100:.2f}% apart).",
              delta_pct=round(delta * 100, 3))
    return r


def reconcile_row_count(ingested: int, source_count: int) -> ValidationReport:
    """Did we ingest everything the source said it had?"""
    return reconcile(float(ingested), float(source_count),
                     "row_count", tolerance=0.001)


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 5 — REFERENTIAL INTEGRITY
# ══════════════════════════════════════════════════════════════════════════════

def check_referential(events: pd.DataFrame,
                      profiles: pd.DataFrame) -> ValidationReport:
    """Layer 5 — do the joins actually resolve?"""
    r = ValidationReport()
    if events.empty or profiles is None or profiles.empty:
        return r
    if "user_id" not in events.columns:
        return r

    event_users = set(events["user_id"].dropna().unique())
    profile_users = set(profiles.index)

    orphans = event_users - profile_users
    if orphans:
        pct = len(orphans) / max(len(event_users), 1) * 100
        sev = BLOCK if pct > 50 else WARN
        r.add(sev, "referential", "orphan_events",
              f"{len(orphans):,} user(s) appear in events but have no profile "
              f"({pct:.1f}% of users).", orphan_count=len(orphans))

    empty_profiles = profile_users - event_users
    if empty_profiles:
        r.add(INFO, "referential", "profiles_without_events",
              f"{len(empty_profiles):,} profile(s) have no matching events.",
              count=len(empty_profiles))

    return r


def check_join_rate(left_ids: set, right_ids: set, label: str,
                    warn_below: float = 80.0) -> ValidationReport:
    """Match rate between two id sets — e.g. product users ↔ Stripe customers.

    A silent all-NaN join (int ids vs 'cus_xxx' strings) shows up here as a 0%
    match rate rather than as quietly missing revenue.
    """
    r = ValidationReport()
    if not left_ids:
        return r
    matched = len(left_ids & right_ids)
    rate = matched / len(left_ids) * 100
    if rate == 0:
        r.add(BLOCK, "referential", f"join_{label}",
              f"{label}: no ids matched at all — the two sides are probably "
              f"different id types.", match_rate_pct=0.0)
    elif rate < warn_below:
        r.add(WARN, "referential", f"join_{label}",
              f"{label}: only {rate:.1f}% of ids matched "
              f"({matched:,} of {len(left_ids):,}).",
              match_rate_pct=round(rate, 2))
    else:
        r.add(INFO, "referential", f"join_{label}",
              f"{label}: {rate:.1f}% matched.",
              match_rate_pct=round(rate, 2))
    return r


# ══════════════════════════════════════════════════════════════════════════════
#  TOP-LEVEL ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def validate_pipeline(events: pd.DataFrame,
                      profiles: Optional[pd.DataFrame] = None,
                      canonical_events: Optional[set] = None,
                      source_row_count: Optional[int] = None
                      ) -> ValidationReport:
    """Run every layer and return one merged report."""
    report = check_structure(events, EVENTS_CONTRACT, "events")
    report = report.merge(check_semantics(events, canonical_events))

    if profiles is not None and not profiles.empty:
        report = report.merge(
            check_structure(profiles, PROFILES_CONTRACT, "user_profiles"))
        report = report.merge(check_referential(events, profiles))

    report = report.merge(
        check_plausibility(compute_metrics(events, profiles)))

    if source_row_count is not None:
        report = report.merge(reconcile_row_count(len(events), source_row_count))

    return report
