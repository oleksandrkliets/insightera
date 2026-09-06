"""
Self-healing mapping engine.

The problem this solves:
    Asking an LLM to map an unfamiliar schema in one shot gets it right most of
    the time and subtly wrong the rest. "Subtly wrong" is the dangerous case —
    a mapping that produces a working dashboard showing incorrect numbers.

    So don't trust the first proposal. Validate it against real data, and when
    it fails, hand the failures back and ask for a correction. Repeat until it
    passes or we've tried enough times to escalate to a human.

The second problem — cost:
    Most accounts look like accounts you've already seen. A default HubSpot
    pipeline is a default HubSpot pipeline. Fingerprint the source's structure,
    and if it matches a mapping that already passed validation, reuse it
    directly: no AI call, no review, no waiting.

    This is what makes customer #30 near-instant while customer #1 took a
    review cycle.

Flow:
    probe → fingerprint → template hit?  ─yes→ apply immediately
                              │
                              no
                              ↓
                     AI proposes mapping
                              ↓
                     validate against real data ─pass→ save as template
                              │
                             fail
                              ↓
                  feed findings back to the AI  ──┐
                              ↑                   │
                              └───────────────────┘
                          (up to MAX_ATTEMPTS, then escalate)
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from taxonomy import (CANONICAL_EVENTS, DataMapping, apply_mapping,
                      load_mapping, save_mapping)
from validation import BLOCK, WARN, ValidationReport, validate_pipeline

MAX_ATTEMPTS = 3


# ══════════════════════════════════════════════════════════════════════════════
#  FINGERPRINTING
# ══════════════════════════════════════════════════════════════════════════════

def fingerprint_probe(probe, platform: str = "generic") -> str:
    """A stable signature of a source's *structure*, not its contents.

    Two customers with the same column layout and broadly the same event
    vocabulary produce the same fingerprint, so the second one can reuse the
    first one's reviewed mapping.

    Deliberately excludes row counts and specific values — those differ between
    accounts that are structurally identical.
    """
    cols = sorted(probe.event_columns or [])
    user_cols = sorted(probe.user_columns or [])
    # Event vocabulary, not frequency. Cap it so one noisy account with
    # thousands of ad-hoc event names still matches its siblings.
    events = sorted({str(name) for name, _ in (probe.event_name_samples or [])})[:40]
    payload = json.dumps(
        {"platform": platform, "cols": cols, "user_cols": user_cols,
         "events": events},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPLATE LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MappingTemplate:
    fingerprint: str
    platform: str
    mapping: DataMapping
    times_reused: int = 0
    origin: str = "ai"          # "ai" | "human" | "builtin"

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "platform": self.platform,
            "times_reused": self.times_reused,
            "origin": self.origin,
            "mapping": self.mapping.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MappingTemplate":
        return cls(
            fingerprint=d["fingerprint"],
            platform=d.get("platform", "generic"),
            mapping=DataMapping.from_dict(d["mapping"]),
            times_reused=int(d.get("times_reused", 0)),
            origin=d.get("origin", "ai"),
        )


class MappingTemplateLibrary:
    """Mappings that already passed validation, keyed by structural fingerprint.

    Persisted as one JSON file. This is deliberately simple — it becomes a
    database table when there's more than one server process, but the interface
    stays the same.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "configs", "mapping_templates.json")
        self._templates: dict[str, MappingTemplate] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                raw = json.load(f)
            for fp, d in raw.items():
                self._templates[fp] = MappingTemplate.from_dict(d)
        except Exception:
            self._templates = {}          # a corrupt cache must not break ingest

    def _persist(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({fp: t.to_dict() for fp, t in self._templates.items()},
                      f, indent=2)

    def find(self, fingerprint: str) -> Optional[MappingTemplate]:
        return self._templates.get(fingerprint)

    def save(self, fingerprint: str, platform: str, mapping: DataMapping,
             origin: str = "ai") -> MappingTemplate:
        existing = self._templates.get(fingerprint)
        reuse = existing.times_reused if existing else 0
        t = MappingTemplate(fingerprint, platform, mapping, reuse, origin)
        self._templates[fingerprint] = t
        self._persist()
        return t

    def record_reuse(self, fingerprint: str) -> None:
        t = self._templates.get(fingerprint)
        if t:
            t.times_reused += 1
            self._persist()

    def __len__(self) -> int:
        return len(self._templates)

    def stats(self) -> dict:
        return {
            "templates": len(self._templates),
            "total_reuses": sum(t.times_reused for t in self._templates.values()),
            "by_platform": {
                p: sum(1 for t in self._templates.values() if t.platform == p)
                for p in {t.platform for t in self._templates.values()}
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MappingResult:
    mapping: Optional[DataMapping]
    report: Optional[ValidationReport]
    source: str                    # "template" | "ai" | "failed"
    attempts: int = 0
    fingerprint: str = ""
    needs_review: bool = False
    history: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.mapping is not None and not self.needs_review

    def explain(self) -> str:
        if self.source == "template":
            return (f"Reused a previously validated mapping "
                    f"(fingerprint {self.fingerprint}). No AI call needed.")
        if self.source == "ai" and not self.needs_review:
            n = self.attempts
            return (f"AI produced a valid mapping in {n} "
                    f"attempt{'s' if n != 1 else ''}.")
        return (f"Could not produce a valid mapping in {self.attempts} "
                f"attempts — human review required.")


# ══════════════════════════════════════════════════════════════════════════════
#  THE LOOP
# ══════════════════════════════════════════════════════════════════════════════

def _findings_as_feedback(report: ValidationReport) -> str:
    """Turn validation failures into correction instructions for the model."""
    lines = []
    for f in report.sorted_findings():
        if f.severity in (BLOCK, WARN):
            lines.append(f"- [{f.severity}] {f.check}: {f.message}")
    return "\n".join(lines) if lines else "- (no specific findings)"


def resolve_mapping(
    probe,
    raw_events: pd.DataFrame,
    platform: str = "generic",
    library: Optional[MappingTemplateLibrary] = None,
    infer_fn: Optional[Callable] = None,
    build_profiles_fn: Optional[Callable] = None,
    extra_hints: str = "",
    max_attempts: int = MAX_ATTEMPTS,
) -> MappingResult:
    """Produce a validated mapping for a source, reusing or self-correcting.

    `infer_fn(probe_summary, canonical_events, extra_hints) -> yaml str` is
    injected so this module stays testable without network access.
    """
    library = library if library is not None else MappingTemplateLibrary()
    fp = fingerprint_probe(probe, platform)
    history: list[str] = []

    # ── 1. Known structure? Reuse it. ────────────────────────────────────
    template = library.find(fp)
    if template is not None:
        try:
            mapped = apply_mapping(raw_events, template.mapping)
            report = _validate(mapped, build_profiles_fn)
            if not report.blocked:
                library.record_reuse(fp)
                history.append(f"template hit ({fp}) — validated, reused")
                return MappingResult(template.mapping, report, "template",
                                     0, fp, False, history)
            history.append(
                f"template hit ({fp}) but failed validation — re-inferring")
        except Exception as e:
            history.append(f"template hit ({fp}) but could not apply: {e}")

    # ── 2. No usable template — ask the AI, then check its work. ─────────
    if infer_fn is None:
        from ai_utils import infer_data_mapping as infer_fn  # noqa: PLC0415

    feedback = ""
    last_report: Optional[ValidationReport] = None

    for attempt in range(1, max_attempts + 1):
        hints = extra_hints
        if feedback:
            hints = (
                f"{extra_hints}\n\n"
                f"Your previous mapping attempt failed validation. Fix these "
                f"specific problems and return a corrected mapping:\n{feedback}"
            ).strip()

        try:
            yaml_text = infer_fn(
                probe_summary=probe.to_summary(),
                canonical_events=sorted(CANONICAL_EVENTS),
                extra_hints=hints,
            )
            mapping = _parse_mapping(yaml_text)
        except Exception as e:
            history.append(f"attempt {attempt}: proposal unusable — {e}")
            feedback = f"- Your output could not be parsed: {e}"
            continue

        errors = mapping.validate()
        if errors:
            history.append(f"attempt {attempt}: mapping self-check failed")
            feedback = "\n".join(f"- {e}" for e in errors)
            continue

        try:
            mapped = apply_mapping(raw_events, mapping)
        except Exception as e:
            history.append(f"attempt {attempt}: mapping could not be applied — {e}")
            feedback = f"- Applying the mapping raised: {e}"
            continue

        report = _validate(mapped, build_profiles_fn)
        last_report = report

        if not report.blocked:
            library.save(fp, platform, mapping, origin="ai")
            history.append(f"attempt {attempt}: passed validation — saved as template")
            return MappingResult(mapping, report, "ai", attempt, fp, False, history)

        history.append(f"attempt {attempt}: failed validation")
        feedback = _findings_as_feedback(report)

    # ── 3. Out of attempts — hand it to a human. ─────────────────────────
    history.append(f"exhausted {max_attempts} attempts — escalating")
    return MappingResult(None, last_report, "failed", max_attempts, fp,
                         True, history)


def _parse_mapping(yaml_text: str) -> DataMapping:
    try:
        import yaml
        return DataMapping.from_dict(yaml.safe_load(yaml_text))
    except ImportError:
        return DataMapping.from_dict(json.loads(yaml_text))


def _validate(mapped_events: pd.DataFrame,
              build_profiles_fn: Optional[Callable]) -> ValidationReport:
    profiles = None
    if build_profiles_fn is not None:
        try:
            profiles = build_profiles_fn(mapped_events)
        except Exception:
            profiles = None
    return validate_pipeline(mapped_events, profiles,
                             canonical_events=CANONICAL_EVENTS)
