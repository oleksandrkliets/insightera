"""
Data-source connector layer.

A `Connector` is a small adapter that knows how to read raw events (and
optionally raw users) from one kind of source. It returns a raw DataFrame —
no normalisation. Normalisation happens downstream in `taxonomy.apply_mapping`.

Built-in connectors:
    • CSVConnector    — one CSV of events (+ optional users CSV)
    • SQLConnector    — SQLAlchemy URL + table names (SQLite, Postgres, ...)
    • DataFrameConnector — pass pre-loaded DataFrames (useful for tests / uploads)

Adding a new connector:
    1. Subclass `Connector`
    2. Implement `read_events(self, since=None)` → returns a DataFrame
    3. (Optional) implement `read_users()`, `read_financials()`, `probe()`

The `probe()` method returns a small sample of the data + the set of
event names + column names + row counts. It's what the AI mapper uses to
infer a mapping without pulling the whole dataset into context.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd


@dataclass
class SourceProbe:
    """Lightweight description of a data source, used to infer mappings."""
    event_columns: list[str]
    user_columns: list[str]
    event_name_samples: list[tuple[str, int]]  # (raw_event_name, count)
    events_sample_rows: pd.DataFrame            # first ~5 rows
    users_sample_rows: Optional[pd.DataFrame]   # first ~5 rows, or None
    total_events: Optional[int] = None

    def to_summary(self, max_events: int = 40) -> str:
        """Compact text summary — what gets sent to the LLM for mapping inference."""
        ev_lines = "\n".join(
            f"  {name}: {cnt:,}"
            for name, cnt in self.event_name_samples[:max_events]
        )
        col_lines = ", ".join(self.event_columns)
        user_lines = (", ".join(self.user_columns)
                      if self.user_columns else "(no separate users table)")
        total = f"{self.total_events:,}" if self.total_events is not None else "?"
        return (
            f"EVENTS TABLE COLUMNS: {col_lines}\n"
            f"USERS TABLE COLUMNS: {user_lines}\n"
            f"TOTAL EVENTS: {total}\n\n"
            f"UNIQUE EVENT NAMES (top {min(len(self.event_name_samples), max_events)}, "
            f"by frequency):\n{ev_lines}\n"
        )


class Connector(ABC):
    """Base class every data-source connector extends."""

    source_name: str = "generic"

    @abstractmethod
    def read_events(self) -> pd.DataFrame:
        """Return the raw events DataFrame (no normalisation)."""
        ...

    def read_users(self) -> Optional[pd.DataFrame]:
        """Return a raw users DataFrame, or None if unavailable."""
        return None

    def read_financials(self) -> Optional[dict]:
        """Return optional financial tables (Stripe-style). Default: none."""
        return None

    def probe(self, max_event_names: int = 60,
              sample_rows: int = 5) -> SourceProbe:
        """Cheap metadata pull for mapping inference.

        Default implementation reads the full events table then slices;
        subclasses should override if they can push the sampling down
        (e.g. SQL LIMIT) for efficiency.
        """
        events = self.read_events()
        event_cols = list(events.columns)
        # Heuristic for picking the event-name column for sampling
        candidates = [c for c in ["event_name", "event", "name", "type"]
                      if c in event_cols]
        event_col = candidates[0] if candidates else event_cols[0]
        counts = events[event_col].value_counts().head(max_event_names)
        users = self.read_users()
        user_cols = list(users.columns) if users is not None else []
        return SourceProbe(
            event_columns=event_cols,
            user_columns=user_cols,
            event_name_samples=[(str(k), int(v)) for k, v in counts.items()],
            events_sample_rows=events.head(sample_rows).copy(),
            users_sample_rows=(users.head(sample_rows).copy()
                               if users is not None else None),
            total_events=len(events),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  BUILT-IN CONNECTORS
# ══════════════════════════════════════════════════════════════════════════════

class CSVConnector(Connector):
    """Connector for one (or two) CSV files on disk.

    Minimal footprint — no dependencies beyond pandas.
    """
    source_name = "csv"

    def __init__(self, events_path: str,
                 users_path: Optional[str] = None,
                 read_csv_kwargs: Optional[dict] = None):
        if not os.path.exists(events_path):
            raise FileNotFoundError(events_path)
        self.events_path = events_path
        self.users_path = users_path
        self._csv_kw = read_csv_kwargs or {}

    def read_events(self) -> pd.DataFrame:
        return pd.read_csv(self.events_path, **self._csv_kw)

    def read_users(self) -> Optional[pd.DataFrame]:
        if self.users_path and os.path.exists(self.users_path):
            return pd.read_csv(self.users_path, **self._csv_kw)
        return None


class DataFrameConnector(Connector):
    """Pass pre-loaded DataFrames directly (useful for tests and for
    bytes-uploaded-through-the-UI flows that already materialise a DataFrame)."""
    source_name = "dataframe"

    def __init__(self, events: pd.DataFrame,
                 users: Optional[pd.DataFrame] = None,
                 financials: Optional[dict] = None):
        self._events = events
        self._users = users
        self._fin = financials

    def read_events(self) -> pd.DataFrame:
        return self._events

    def read_users(self) -> Optional[pd.DataFrame]:
        return self._users

    def read_financials(self) -> Optional[dict]:
        return self._fin


class SQLConnector(Connector):
    """SQLAlchemy-backed connector (SQLite / Postgres / MySQL / BigQuery-via-sqlalchemy).

    Pass an SQLAlchemy URL and the table names. This is a thin wrapper —
    downstream analytics still happen in pandas.
    """
    source_name = "sql"

    def __init__(self, db_url: str,
                 events_table: str = "events",
                 users_table: Optional[str] = None,
                 events_where: Optional[str] = None):
        from sqlalchemy import create_engine
        self.db_url = db_url
        self.events_table = events_table
        self.users_table = users_table
        self.events_where = events_where
        self._engine = create_engine(db_url)

    def read_events(self) -> pd.DataFrame:
        q = f"SELECT * FROM {self.events_table}"
        if self.events_where:
            q += f" WHERE {self.events_where}"
        return pd.read_sql(q, self._engine)

    def read_users(self) -> Optional[pd.DataFrame]:
        if not self.users_table:
            return None
        q = f"SELECT * FROM {self.users_table}"
        try:
            return pd.read_sql(q, self._engine)
        except Exception:
            return None

    def probe(self, max_event_names: int = 60,
              sample_rows: int = 5) -> SourceProbe:
        """Efficient probe via SQL LIMIT and aggregation."""
        from sqlalchemy import text
        with self._engine.connect() as cx:
            # Try to introspect the events table efficiently
            try:
                cnt = cx.execute(
                    text(f"SELECT COUNT(*) FROM {self.events_table}")
                ).scalar()
            except Exception:
                cnt = None
            head = pd.read_sql(
                f"SELECT * FROM {self.events_table} LIMIT {sample_rows}",
                cx)
            event_cols = list(head.columns)
            candidates = [c for c in ["event_name", "event", "name", "type"]
                          if c in event_cols]
            ev_col = candidates[0] if candidates else event_cols[0]
            try:
                freq = pd.read_sql(
                    f"SELECT {ev_col} AS name, COUNT(*) AS n "
                    f"FROM {self.events_table} GROUP BY {ev_col} "
                    f"ORDER BY n DESC LIMIT {max_event_names}",
                    cx)
                samples = list(zip(freq["name"].astype(str), freq["n"].astype(int)))
            except Exception:
                samples = []
            users_head = None
            user_cols: list[str] = []
            if self.users_table:
                try:
                    users_head = pd.read_sql(
                        f"SELECT * FROM {self.users_table} LIMIT {sample_rows}",
                        cx)
                    user_cols = list(users_head.columns)
                except Exception:
                    pass
        return SourceProbe(
            event_columns=event_cols,
            user_columns=user_cols,
            event_name_samples=samples,
            events_sample_rows=head,
            users_sample_rows=users_head,
            total_events=int(cnt) if cnt is not None else None,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN BASE CLASSES
# ══════════════════════════════════════════════════════════════════════════════
#
# Finance, advertising and email data is not event-shaped. It is
# metric-shaped: "this channel spent $X and got Y clicks on date D". Forcing
# that through the event interface means every connector fights the
# abstraction, so each domain gets its own contract.
#
# Subclasses implement only the raw pull and the platform's own quirks. The
# output shape is fixed here, which is what makes the fourth ads connector
# cheap after the first one.

# Columns every AdsConnector must return
AD_METRIC_COLUMNS = [
    "date", "channel", "campaign_id", "campaign_name",
    "spend", "impressions", "clicks", "conversions",
    "conversion_value", "currency",
]

EMAIL_METRIC_COLUMNS = [
    "date", "campaign_id", "campaign_name", "sent", "delivered",
    "opened", "clicked", "unsubscribed", "bounced",
]


class AdsConnector(Connector):
    """Base for Google / Meta / LinkedIn / TikTok Ads.

    Note on conversions: platform-reported conversion counts use that
    platform's own attribution model and will not agree with first-party
    revenue. Return them, but they are labelled as platform-attributed
    downstream and never used as the revenue source of truth.
    """
    source_name = "ads"
    channel_name = "unknown"

    @abstractmethod
    def read_performance(self, since: "date", until: "date") -> pd.DataFrame:
        """Daily campaign-level metrics conforming to AD_METRIC_COLUMNS."""
        ...

    def read_events(self) -> pd.DataFrame:
        """Ads sources carry no user-level events."""
        return pd.DataFrame(columns=["user_id", "timestamp", "event_name"])

    def normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Guarantee the canonical shape regardless of what the API returned."""
        out = df.copy()
        for col in AD_METRIC_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        out["channel"] = out["channel"].fillna(self.channel_name)
        if "date" in out:
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
        for num in ("spend", "impressions", "clicks", "conversions",
                    "conversion_value"):
            out[num] = pd.to_numeric(out[num], errors="coerce").fillna(0)
        return out[AD_METRIC_COLUMNS]

    def probe(self, max_event_names: int = 60,
              sample_rows: int = 5) -> SourceProbe:
        from datetime import date as _date, timedelta as _td
        sample = self.read_performance(_date.today() - _td(days=7),
                                       _date.today())
        return SourceProbe(
            event_columns=list(sample.columns),
            user_columns=[],
            event_name_samples=[(self.channel_name, len(sample))],
            events_sample_rows=sample.head(sample_rows),
            users_sample_rows=None,
            total_events=len(sample),
        )


class FinanceConnector(Connector):
    """Base for Stripe / QuickBooks / Chargebee.

    Currency matters here in a way it does not elsewhere: amounts arrive in the
    account's own currency, and reporting them as USD without conversion is a
    silent, material error.
    """
    source_name = "finance"

    @abstractmethod
    def read_invoices(self, since: Optional["date"] = None) -> pd.DataFrame: ...

    def read_subscriptions(self, since: Optional["date"] = None
                           ) -> pd.DataFrame:
        return pd.DataFrame()

    def read_charges(self, since: Optional["date"] = None) -> pd.DataFrame:
        return pd.DataFrame()

    def read_expenses(self, since: Optional["date"] = None) -> pd.DataFrame:
        """Only accounting platforms have these; billing platforms do not."""
        return pd.DataFrame()

    def reported_total(self, metric: str,
                       since: "date", until: "date") -> Optional[float]:
        """The platform's own figure for a metric, used for reconciliation.

        Returning None means "no independent figure available" and the
        reconciliation check is skipped rather than failed.
        """
        return None

    def read_events(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["user_id", "timestamp", "event_name"])

    def read_financials(self) -> Optional[dict]:
        return {
            "invoices": self.read_invoices(),
            "subscriptions": self.read_subscriptions(),
            "charges": self.read_charges(),
        }


class EmailMarketingConnector(Connector):
    """Base for Mailchimp / Klaviyo."""
    source_name = "email"

    @abstractmethod
    def read_campaign_metrics(self, since: Optional["date"] = None
                              ) -> pd.DataFrame:
        """Campaign metrics conforming to EMAIL_METRIC_COLUMNS."""
        ...

    def read_events(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["user_id", "timestamp", "event_name"])

    def normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in EMAIL_METRIC_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        if "date" in out:
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
        for num in ("sent", "delivered", "opened", "clicked",
                    "unsubscribed", "bounced"):
            out[num] = pd.to_numeric(out[num], errors="coerce").fillna(0)
        return out[EMAIL_METRIC_COLUMNS]


# ══════════════════════════════════════════════════════════════════════════════
#  REGISTRY — lets the UI list available connector kinds
# ══════════════════════════════════════════════════════════════════════════════

CONNECTOR_REGISTRY: dict[str, type[Connector]] = {
    "csv":       CSVConnector,
    "sql":       SQLConnector,
    "dataframe": DataFrameConnector,
}


def register_connector(name: str, cls: type[Connector]) -> None:
    """Third-party connectors can self-register via this hook."""
    CONNECTOR_REGISTRY[name] = cls
