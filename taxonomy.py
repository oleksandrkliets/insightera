"""
Canonical event taxonomy + mapping engine.

Why this exists:
    The original dashboard hard-coded event names like `signup`, `payment_success`,
    `rage_click`, `view_feature_A/B/C/D` throughout charts and analyses. Any
    customer whose events are called something else would need code changes.

    This module defines the canonical schema the dashboard expects. A
    per-source mapping (YAML or dict) declares how a customer's raw events
    and columns translate to canonical names. The mapping is applied before
    ETL, so the rest of the codebase keeps working with canonical names.

Usage:
    from taxonomy import CANONICAL_EVENTS, apply_mapping, load_mapping

    mapping = load_mapping("configs/my_source.yaml")
    normalised_events = apply_mapping(raw_events, mapping)
    normalised_users  = apply_user_mapping(raw_users, mapping)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:  # graceful fallback: JSON works too
    _HAS_YAML = False


# ══════════════════════════════════════════════════════════════════════════════
#  CANONICAL SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

# These are the canonical event names the dashboard analytics/charts expect.
# A mapping translates customer-specific event names TO these.
CANONICAL_EVENTS: set[str] = {
    # Onboarding / activation
    "signup",
    "onboarding_step1",
    "onboarding_step2",

    # Conversion funnel
    "view_pricing",
    "start_checkout",
    "payment_success",
    "payment_failed",
    "plan_upgrade",

    # Core product engagement
    "view_dashboard",
    "view_feature_A",
    "view_feature_B",
    "view_feature_C",
    "view_feature_D",
    "invite_team",

    # Friction signals
    "rage_click",
    "error",
    "contact_support",

    # Session / generic (treated as noise in most analyses)
    "page_view",
    "click",
    "session_start",
    "session_end",
}

# Canonical event-role groups (semantic roles used by analyses).
# These are derived from the event set — charts can target roles OR specific
# events. `CONVERSION_NEGATIVE_EVENTS` in dashboard_flows.py mirrors FRICTION.
EVENT_ROLES: dict[str, set[str]] = {
    "ACTIVATION":          {"signup", "onboarding_step1", "onboarding_step2"},
    "CONVERSION_INTENT":   {"view_pricing", "start_checkout"},
    "CONVERSION_SUCCESS":  {"payment_success", "plan_upgrade"},
    "CONVERSION_FAILURE":  {"payment_failed"},
    "FEATURE_ENGAGEMENT":  {"view_feature_A", "view_feature_B", "view_feature_C",
                            "view_feature_D", "view_dashboard", "invite_team"},
    "FRICTION":            {"rage_click", "error", "contact_support"},
    "SESSION":             {"session_start", "session_end", "page_view", "click"},
}

# Required columns in the normalised events DataFrame
CANONICAL_EVENT_COLUMNS: list[str] = ["user_id", "event_name", "timestamp"]

# Optional columns — carried through if the source provides them
OPTIONAL_EVENT_COLUMNS: list[str] = ["session_id", "properties",
                                     "platform", "country"]

# Required columns in the normalised users DataFrame (if provided separately)
CANONICAL_USER_COLUMNS: list[str] = ["user_id"]

# Optional user columns
OPTIONAL_USER_COLUMNS: list[str] = ["signup_at", "email", "country",
                                    "plan", "referrer"]


# ══════════════════════════════════════════════════════════════════════════════
#  MAPPING FORMAT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataMapping:
    """Per-source mapping from raw data to the canonical schema.

    A mapping is typically loaded from a YAML file per customer / data source.
    It has three parts:

    1. `events_columns`: which raw column holds each canonical column.
       Example: {"user_id": "distinct_id", "timestamp": "time"}

    2. `event_map`: which raw event names map to each canonical event.
       A single raw event can map to only one canonical event.
       Example: {"payment_success": ["SubscriptionCreated", "TrialConverted"]}

    3. `users_columns`: same as events_columns but for the users table
       (optional — if the source has a separate users dimension table).

    `unmapped_events` is a list of raw events the mapper recognised but
    intentionally dropped (noise, deprecated, etc).
    """
    source_name: str = "unknown"
    description: str = ""
    events_columns: dict[str, str] = field(default_factory=dict)
    event_map: dict[str, list[str]] = field(default_factory=dict)
    users_columns: dict[str, str] = field(default_factory=dict)
    unmapped_events: list[str] = field(default_factory=list)
    # Free-form notes from the mapper (e.g. AI-generated rationale)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "description": self.description,
            "events_columns": self.events_columns,
            "event_map": self.event_map,
            "users_columns": self.users_columns,
            "unmapped_events": self.unmapped_events,
            "notes": self.notes,
        }

    def to_yaml(self) -> str:
        d = self.to_dict()
        if _HAS_YAML:
            return yaml.safe_dump(d, sort_keys=False, default_flow_style=False)
        import json
        return json.dumps(d, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "DataMapping":
        return cls(
            source_name=d.get("source_name", "unknown"),
            description=d.get("description", ""),
            events_columns=dict(d.get("events_columns", {})),
            event_map={k: list(v) for k, v in d.get("event_map", {}).items()},
            users_columns=dict(d.get("users_columns", {})),
            unmapped_events=list(d.get("unmapped_events", [])),
            notes=d.get("notes", ""),
        )

    def validate(self) -> list[str]:
        """Return a list of validation errors. Empty = mapping is valid."""
        errors: list[str] = []
        # Events_columns must cover the required canonical columns
        missing_cols = [c for c in CANONICAL_EVENT_COLUMNS
                        if c not in self.events_columns]
        if missing_cols:
            errors.append(
                f"events_columns is missing required mappings for: "
                f"{', '.join(missing_cols)}")
        # All canonical event names in event_map must be valid
        bad_events = [e for e in self.event_map if e not in CANONICAL_EVENTS]
        if bad_events:
            errors.append(
                f"event_map contains non-canonical events: "
                f"{', '.join(bad_events)} — see CANONICAL_EVENTS")
        # No raw event can map to more than one canonical event
        seen: dict[str, str] = {}
        for canon, raws in self.event_map.items():
            for r in raws:
                if r in seen and seen[r] != canon:
                    errors.append(
                        f"raw event '{r}' is mapped to both "
                        f"'{seen[r]}' and '{canon}'")
                seen[r] = canon
        return errors


def identity_mapping() -> DataMapping:
    """A mapping that assumes source data already uses canonical names.
    Useful when no mapping is configured (e.g. the existing CSV dataset)."""
    return DataMapping(
        source_name="identity",
        description="No transformation — source already uses canonical names",
        events_columns={c: c for c in CANONICAL_EVENT_COLUMNS},
        event_map={e: [e] for e in CANONICAL_EVENTS},
        users_columns={c: c for c in CANONICAL_USER_COLUMNS},
        notes="Auto-generated identity mapping",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD / SAVE
# ══════════════════════════════════════════════════════════════════════════════

def load_mapping(path: str) -> DataMapping:
    """Load a mapping from YAML or JSON."""
    with open(path) as f:
        text = f.read()
    if path.endswith(".yaml") or path.endswith(".yml"):
        if not _HAS_YAML:
            raise RuntimeError("PyYAML not installed; cannot load .yaml mapping")
        d = yaml.safe_load(text)
    else:  # .json
        import json
        d = json.loads(text)
    return DataMapping.from_dict(d)


def save_mapping(mapping: DataMapping, path: str) -> None:
    """Save a mapping to YAML (or JSON if pyyaml is absent)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(mapping.to_yaml())


# ══════════════════════════════════════════════════════════════════════════════
#  APPLY MAPPING TO DATA
# ══════════════════════════════════════════════════════════════════════════════

def apply_mapping(
    raw_events: pd.DataFrame,
    mapping: DataMapping,
    drop_unmapped: bool = False,
) -> pd.DataFrame:
    """Apply a mapping to a raw events DataFrame.

    Steps:
      1. Rename source columns to canonical column names (`user_id`,
         `event_name`, `timestamp`, plus any optional ones present).
      2. Translate source event names to canonical event names via `event_map`.
      3. Unmapped raw events are kept with their original name (or dropped
         if `drop_unmapped=True`) so you can see them in QA before deciding.

    Returns a DataFrame with at least `user_id`, `event_name`, `timestamp`.
    """
    errors = mapping.validate()
    if errors:
        raise ValueError(f"invalid mapping: {errors}")

    df = raw_events.copy()

    # 1) Column rename: the YAML is canonical→source; we want the inverse.
    rename = {src: canon for canon, src in mapping.events_columns.items()
              if src in df.columns and src != canon}
    if rename:
        df = df.rename(columns=rename)

    # Coerce timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df[df["timestamp"].notna()]

    # 2) Event-name translation
    # Build reverse map: raw_name → canonical_name
    reverse_event_map: dict[str, str] = {}
    for canon, raw_names in mapping.event_map.items():
        for raw in raw_names:
            reverse_event_map[raw] = canon

    if reverse_event_map and "event_name" in df.columns:
        mapped_mask = df["event_name"].isin(reverse_event_map)
        df.loc[mapped_mask, "event_name"] = df.loc[mapped_mask, "event_name"].map(
            reverse_event_map)
        if drop_unmapped:
            df = df[mapped_mask]

    # Drop rows with missing required fields
    required = [c for c in CANONICAL_EVENT_COLUMNS if c in df.columns]
    if required:
        df = df.dropna(subset=required)

    return df.reset_index(drop=True)


def apply_user_mapping(
    raw_users: pd.DataFrame,
    mapping: DataMapping,
) -> pd.DataFrame:
    """Rename user-table columns according to mapping.users_columns."""
    df = raw_users.copy()
    rename = {src: canon for canon, src in mapping.users_columns.items()
              if src in df.columns and src != canon}
    if rename:
        df = df.rename(columns=rename)
    if "signup_at" in df.columns:
        df["signup_at"] = pd.to_datetime(df["signup_at"], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  COVERAGE REPORT — helpful for QA after mapping
# ══════════════════════════════════════════════════════════════════════════════

def coverage_report(
    raw_events: pd.DataFrame,
    mapping: DataMapping,
    event_col: Optional[str] = None,
) -> dict:
    """Summarise what the mapping covers on this dataset.

    Returns:
      {
        'total_events': N,
        'mapped_events': M,
        'mapped_pct': float,
        'per_canonical': {canon_name: count},
        'unmapped_event_names': [list of raw names with no mapping],
        'canonical_not_seen':   [canonical events the mapping claims exist
                                 but no raw events produced them],
      }
    """
    if event_col is None:
        event_col = mapping.events_columns.get("event_name", "event_name")
    if event_col not in raw_events.columns:
        return {"error": f"column '{event_col}' not in data"}

    reverse_event_map: dict[str, str] = {}
    for canon, raw_names in mapping.event_map.items():
        for raw in raw_names:
            reverse_event_map[raw] = canon

    counts = raw_events[event_col].value_counts()
    total = int(counts.sum())
    mapped_mask = raw_events[event_col].isin(reverse_event_map)
    mapped = int(mapped_mask.sum())

    per_canonical: dict[str, int] = {}
    for raw_name, cnt in counts.items():
        canon = reverse_event_map.get(raw_name)
        if canon:
            per_canonical[canon] = per_canonical.get(canon, 0) + int(cnt)

    unmapped_names = [str(n) for n in counts.index if n not in reverse_event_map]
    canonical_not_seen = sorted([
        c for c in mapping.event_map.keys()
        if c not in per_canonical
    ])

    return {
        "total_events": total,
        "mapped_events": mapped,
        "mapped_pct": round(mapped / total * 100, 1) if total else 0,
        "per_canonical": per_canonical,
        "unmapped_event_names": unmapped_names[:50],
        "canonical_not_seen": canonical_not_seen,
    }
