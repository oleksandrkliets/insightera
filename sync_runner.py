"""
Scheduled incremental sync.

Every connector needs the same four things, and none of them are interesting
enough to reimplement ten times:

  • a cursor, so each run fetches only what appeared since the last one
  • backoff, because every platform rate-limits differently and all of them do
  • a log, because a silent failure is worse than a loud one
  • isolation, so one broken connector doesn't stop the other nine

This is deliberately a single scheduled process rather than Airflow or Dagster.
At the scale this needs to work at, a cron entry and a lock file is the correct
amount of infrastructure.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

# ══════════════════════════════════════════════════════════════════════════════
#  CURSORS
# ══════════════════════════════════════════════════════════════════════════════

class CursorStore:
    """Remembers how far each (workspace, connector) pair has read.

    Without this, every sync refetches the entire history — which works on a
    demo dataset and falls over on a real one.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".state", "cursors.json")
        self._lock = threading.Lock()
        self._cursors: dict[str, str] = {}
        self._load()

    @staticmethod
    def _key(workspace_id: str, connector: str, resource: str = "") -> str:
        return f"{workspace_id}::{connector}::{resource}" if resource \
            else f"{workspace_id}::{connector}"

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self._cursors = json.load(f)
            except Exception:
                self._cursors = {}

    def _persist(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._cursors, f, indent=2)
        os.replace(tmp, self.path)

    def get(self, workspace_id: str, connector: str,
            resource: str = "", default: Optional[str] = None) -> Optional[str]:
        return self._cursors.get(self._key(workspace_id, connector, resource),
                                 default)

    def set(self, workspace_id: str, connector: str, value: str,
            resource: str = "") -> None:
        with self._lock:
            self._cursors[self._key(workspace_id, connector, resource)] = value
            self._persist()

    def clear(self, workspace_id: str, connector: str = "") -> int:
        """Reset cursors to force a full re-read."""
        prefix = f"{workspace_id}::{connector}" if connector \
            else f"{workspace_id}::"
        with self._lock:
            keys = [k for k in self._cursors if k.startswith(prefix)]
            for k in keys:
                del self._cursors[k]
            if keys:
                self._persist()
        return len(keys)


# ══════════════════════════════════════════════════════════════════════════════
#  RETRY
# ══════════════════════════════════════════════════════════════════════════════

class RateLimitError(Exception):
    """Raised by a connector when the platform asks us to slow down.

    Carries `retry_after` when the platform tells us how long to wait, so we
    honour their number instead of guessing.
    """
    def __init__(self, message: str = "", retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


def with_retry(fn: Callable, *args, attempts: int = 4,
               base_delay: float = 1.0, max_delay: float = 60.0, **kwargs):
    """Exponential backoff with jitter. Honours a platform-supplied delay."""
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            last = e
            if i == attempts - 1:
                break
            delay = e.retry_after if e.retry_after is not None \
                else min(base_delay * (2 ** i), max_delay)
            time.sleep(delay + random.uniform(0, 0.5))
        except Exception as e:
            last = e
            if i == attempts - 1:
                break
            time.sleep(min(base_delay * (2 ** i), max_delay)
                       + random.uniform(0, 0.5))
    raise last  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
#  SYNC LOG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SyncRecord:
    workspace_id: str
    connector: str
    status: str                    # "ok" | "error" | "rate_limited" | "skipped"
    started_at: str
    duration_sec: float
    rows: int = 0
    error: Optional[str] = None
    cursor_advanced_to: Optional[str] = None


class SyncLog:
    """Recent sync outcomes — feeds the Data Health Report."""

    def __init__(self, path: Optional[str] = None, keep: int = 500):
        self.path = path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".state", "sync_log.json")
        self.keep = keep
        self._lock = threading.Lock()
        self._records: list[dict] = []
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self._records = json.load(f)
            except Exception:
                self._records = []

    def record(self, rec: SyncRecord) -> None:
        with self._lock:
            self._records.insert(0, asdict(rec))
            self._records = self._records[: self.keep]
            os.makedirs(os.path.dirname(os.path.abspath(self.path)),
                        exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._records, f, indent=2)
            os.replace(tmp, self.path)

    def recent(self, workspace_id: Optional[str] = None,
               limit: int = 50) -> list[dict]:
        rows = self._records
        if workspace_id:
            rows = [r for r in rows if r["workspace_id"] == workspace_id]
        return rows[:limit]

    def health(self, workspace_id: str) -> list[dict]:
        """Latest outcome per connector — what the UI shows as sync status."""
        seen: dict[str, dict] = {}
        for r in self._records:
            if r["workspace_id"] != workspace_id:
                continue
            seen.setdefault(r["connector"], r)
        return list(seen.values())


# ══════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SyncJob:
    workspace_id: str
    connector_name: str
    run: Callable[[Optional[str]], Any]   # run(cursor) -> (rows, new_cursor)
    interval_minutes: int = 60
    enabled: bool = True


class SyncRunner:
    """Runs registered jobs, one workspace-connector pair at a time.

    A connector raising does not stop the others — its failure is logged
    against that connection and the run continues.
    """

    def __init__(self, cursors: Optional[CursorStore] = None,
                 log: Optional[SyncLog] = None,
                 on_result: Optional[Callable[[SyncRecord], None]] = None):
        self.cursors = cursors or CursorStore()
        self.log = log or SyncLog()
        self.jobs: list[SyncJob] = []
        self.on_result = on_result

    def register(self, job: SyncJob) -> None:
        self.jobs.append(job)

    def _due(self, job: SyncJob) -> bool:
        if not job.enabled:
            return False
        last = [r for r in self.log.recent(job.workspace_id, limit=200)
                if r["connector"] == job.connector_name and r["status"] == "ok"]
        if not last:
            return True
        started = datetime.fromisoformat(last[0]["started_at"])
        return datetime.now(timezone.utc) - started >= timedelta(
            minutes=job.interval_minutes)

    def run_job(self, job: SyncJob) -> SyncRecord:
        started = datetime.now(timezone.utc)
        t0 = time.time()
        cursor = self.cursors.get(job.workspace_id, job.connector_name)
        try:
            rows, new_cursor = with_retry(job.run, cursor)
            if new_cursor:
                self.cursors.set(job.workspace_id, job.connector_name,
                                 str(new_cursor))
            rec = SyncRecord(
                job.workspace_id, job.connector_name, "ok",
                started.isoformat(), round(time.time() - t0, 2),
                rows=int(rows or 0),
                cursor_advanced_to=str(new_cursor) if new_cursor else None)
        except RateLimitError as e:
            rec = SyncRecord(job.workspace_id, job.connector_name,
                             "rate_limited", started.isoformat(),
                             round(time.time() - t0, 2), error=str(e))
        except Exception as e:
            rec = SyncRecord(job.workspace_id, job.connector_name, "error",
                             started.isoformat(), round(time.time() - t0, 2),
                             error=f"{type(e).__name__}: {e}")
            traceback.print_exc()

        self.log.record(rec)
        if self.on_result:
            try:
                self.on_result(rec)
            except Exception:
                pass
        return rec

    def run_due(self) -> list[SyncRecord]:
        """One pass over every registered job. Call this from cron."""
        out: list[SyncRecord] = []
        for job in self.jobs:
            if not self._due(job):
                continue
            out.append(self.run_job(job))
        return out

    def run_all(self) -> list[SyncRecord]:
        """Ignore schedules and run everything — manual 'Sync now'."""
        return [self.run_job(j) for j in self.jobs if j.enabled]


if __name__ == "__main__":                             # pragma: no cover
    runner = SyncRunner()
    results = runner.run_due()
    print(f"{len(results)} job(s) ran")
    for r in results:
        print(f"  {r.connector}: {r.status} "
              f"({r.rows:,} rows in {r.duration_sec}s)"
              + (f" — {r.error}" if r.error else ""))
