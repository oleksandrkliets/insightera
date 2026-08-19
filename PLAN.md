# Insightera — Build Plan

> **Read this first every session. Update the Status block last.**

---

## 📍 STATUS — edit this every session

```
Current week:     Week 1
Last session:     — (not started)
Next task:        W1.S2 — Fix active_days calculation
Blocked on:       nothing
```

**Session log** (append one line per session, newest at top):

```
2026-08-05  W1.S1  Submitted ad-platform approvals — DONE / IN PROGRESS / NOT STARTED
```

---

## How I work on this

| Rule | Why |
|---|---|
| One task per session, decided the night before | Never open the laptop wondering what to do |
| Read this file first, update it last | Working memory between sessions |
| Every task fits in ≤2 sessions | If bigger, split it before starting |
| End every session with code committed and running | Never leave a broken tree overnight |
| Friday = no new code. Test, fix, update plan | Otherwise debt compounds invisibly |
| If a task blows past 2 sessions → cut scope, don't grind | Grinding is how months disappear |

**Session shape (2 hrs):**
1. (5 min) Read Status block above
2. (10 min) `git pull`, run the app, confirm it works
3. (90 min) The one task
4. (10 min) Commit + push
5. (5 min) Update Status block, write tomorrow's task

---

## MONTH 1 — Foundation
*No new connectors. This month makes every future connector ~5 hrs instead of ~14.*

### Week 1 — Correctness + start the approval clocks

- [ ] **W1.S1 — Submit all platform approvals** ⏰ *do this first, clocks run for months*
  - Google Ads: Standard access on developer token
  - Meta: App Review for `ads_read` Advanced Access + Business Verification
  - LinkedIn: Marketing Developer Platform application
  - TikTok: app registration + Business Verification
  - QuickBooks: Production keys
  - **Done when:** all 5 submitted, confirmation emails saved

- [ ] **W1.S2 — Fix `active_days`**
  - Currently: `(last_event - first_event)` = total span. A user active on day 1 and day 30 shows 30 active days.
  - Change to: count of distinct calendar dates with events
  - Also fixes `events_per_day` (inflated denominator)
  - **Done when:** `active_days` ≤ span for every user; spot-check 3 users manually

- [ ] **W1.S3 — Fix `churned`**
  - Currently: heuristic (2 of 3: low events / short span / few features). Calls a user who signed up yesterday "churned".
  - Change to: `days_since_last > 14` for users past trial
  - **Done when:** churn rate is plausible; a brand-new active user is NOT churned

- [ ] **W1.S4 — Rename `ltv` → `revenue_to_date`; fix "now"**
  - `ltv` is just historical revenue, not a projection — the name lies
  - Replace `df["timestamp"].max()` with `pd.Timestamp.now(tz="UTC")` as "now"
  - **Done when:** no chart labels say LTV for a to-date figure

- [ ] **W1.S5 — Timezone normalization**
  - All timestamps → UTC at ingest, timezone-aware
  - **Done when:** `events["timestamp"].dt.tz` is not None

### Week 2 — Validation contracts

- [ ] **W2.S1 — `FieldContract` + `EVENTS_CONTRACT`** in `taxonomy.py`
  - **Done when:** contract defined for user_id / timestamp / event_name
- [ ] **W2.S2 — Layer 1+2: structural + semantic validation**
  - Required cols, dtypes, nulls, canonical event names, plausible date range
  - **Done when:** running on good data → 0 findings; on broken fixture → findings
- [ ] **W2.S3 — Layer 3: statistical plausibility**
  - conversion 0-100%, churn 0-100%, events/user < 100k, future_ts < 0.1%, dupes < 5%
  - **Done when:** a deliberately double-mapped event trips the conversion rule
- [ ] **W2.S4 — Layer 4: reconciliation**
  - Computed row count vs source `COUNT(*)`; later Stripe MRR vs Stripe's own total
  - **Done when:** mismatch > 2% produces a finding
- [ ] **W2.S5 — `ValidationReport` + BLOCK/WARN/INFO gating**
  - BLOCK = refuse mapping · WARN = apply + flag · INFO = log
  - **Done when:** a BLOCK-level failure prevents `apply_mapping` from committing

### Week 3 — Self-healing loop ⭐ *the differentiator*

- [ ] **W3.S1-S2 — Retry loop**
  - AI proposes mapping → validate → on failure feed findings back to Claude → re-propose
  - Max 3 attempts, then escalate to human review
  - **Done when:** a deliberately confusing schema self-corrects within 3 attempts
- [ ] **W3.S3 — `MappingTemplateLibrary`** — fingerprint + store approved mappings
- [ ] **W3.S4 — Template lookup before AI call** (skip the AI entirely on a fingerprint match)
  - **Done when:** second connect of the same schema shape needs 0 AI calls
- [ ] **W3.S5 — End-to-end test on broken fixture data**

### Week 4 — Shared connector infrastructure

- [ ] **W4.S1-S2 — `OAuthTokenStore`** — encrypted at rest, auto-refresh, server-side only
- [ ] **W4.S3 — 🔒 SECURITY FIX: DB credentials out of `dcc.Store`**
  - Currently `dashboard_flows.py:8985` puts the full DB URL *including password* into a client-side store → readable in the browser DOM
  - Move to server-side dict keyed by opaque `connection_id`
  - **Done when:** no credential appears anywhere in browser DOM/devtools
- [ ] **W4.S4 — `sync_runner.py` + cursor state table** (incremental syncs)
- [ ] **W4.S5 — Base classes:** `FinanceConnector`, `AdsConnector`, `EmailMarketingConnector`

> **🚦 GATE (end of W4):** Does the self-healing loop fix a broken mapping on its own?
> If no → stop, fix it. Everything downstream depends on this.

---

## MONTH 2 — Connectors (no approval gates)

### Week 5 — Stripe
- [ ] S1-S2 — Auth + pull customers, subscriptions, invoices, charges, refunds
- [ ] S3 — Incremental sync via `created` cursors
- [ ] S4 — Map into `CANONICAL_FINANCE_OBJECTS`
- [ ] S5 — Reconcile computed MRR vs Stripe's reported balance
- **Done when:** your MRR figure matches the Stripe dashboard within 2%

### Week 6 — Database derivation engine ⭐ *works for ~95% of customers, zero instrumentation*
- [ ] S1 — `derivations.yaml` config schema
- [ ] S2-S3 — SQL generator (UNION ALL across entity tables → canonical event stream)
- [ ] S4 — AI proposes derivations from a database schema
- [ ] S5 — Test against a realistic app-DB fixture (users / sessions / projects / subscriptions)
- **Done when:** a plain Postgres app DB with no event tracking produces a working dashboard

### Week 7 — PostHog + HubSpot
- [ ] S1-S2 — PostHog (REST API + key)
- [ ] S3-S5 — HubSpot (OAuth; map custom properties + pipeline stages to canonical)

### Week 8 — Mailchimp + Klaviyo + buffer
- [ ] S1-S2 — Mailchimp
- [ ] S3-S4 — Klaviyo
- [ ] S5 — Buffer / catch-up

> **🚦 GATE (end of W8):** Can you demo 6 connectors against someone else's real data?
> If no → cut connectors, not quality. 4 solid beats 6 flaky.

---

## MONTH 3 — Make it sellable

### Week 9 — Multi-tenancy
- [ ] S1-S2 — Workspace model + per-workspace data isolation
- [ ] S3-S4 — Remove `_with_filtered_globals` global-swapping (unsafe with >1 user)
- [ ] S5 — Test two workspaces side by side, confirm no data bleed

### Week 10 — Auth + deploy
- [ ] S1-S2 — Clerk auth (don't build your own)
- [ ] S3-S4 — Deploy to Render (Dockerfile exists)
- [ ] S5 — Custom domain + HTTPS
- **Done when:** a real URL a stranger can log into

### Week 11 — Onboarding flow
- [ ] S1-S2 — Source picker screen
- [ ] S3 — AI mapping review screen
- [ ] S4 — "First dashboard" moment
- [ ] S5 — Time it end to end
- **Done when:** signup → populated dashboard in under 10 minutes

### Week 12 — Data Health Report + "only what they need"
- [ ] S1-S2 — Data Health Report UI (coverage %, sync status, quality flags)
- [ ] S3-S4 — **Cut the dashboard down**: one hero KPI + 5-6 charts per tab, rest behind "show more"
- [ ] S5 — Visual polish: kill colored left-borders, tighten type, add loading skeletons

> **🚦 GATE (end of W12):** Can a stranger sign up and see their data in <10 min?
> If no → fix onboarding before building anything else.

---

## MONTH 4 — Ads + first customer

### Week 13 — Google Ads
- [ ] MCC account linking flow + GAQL queries + PMax handling

### Week 14 — Meta Ads
- [ ] OAuth + Marketing API + attribution-window handling

### Week 15 — GA4 + Segment
- [ ] S1-S3 — GA4 via BigQuery export (flatten nested `event_params`)
- [ ] S4-S5 — Segment webhook receiver

### Week 16 — Ship
- [ ] S1-S2 — Docs + help pages
- [ ] S3 — Status page + Sentry
- [ ] S4-S5 — **Onboard customer #1**

> **🚦 GATE (W13):** Have you talked to 20 prospects yet?
> If no → **stop building for two weeks and sell.** This is the gate you'll most want to skip.

---

## If I fall behind — cut in this order

1. LinkedIn + TikTok Ads (smallest share)
2. Segment (GA4 + PostHog cover most of it)
3. Mailchimp / Klaviyo (not core to the product+finance thesis)
4. GA4 (most painful build per unit of value)

**Never cut:** validation · self-healing loop · onboarding · deployment.
Those are what make it a product instead of a script.

---

## Reference — what already exists

| File | Contains |
|---|---|
| `dashboard_flows.py` | Dashboard UI, ~40 charts, filters, AI panels, Settings + AI-ETL flow |
| `etl.py` | Pipeline: profiles, transitions, rage-click detection, Stripe financials |
| `taxonomy.py` | `CANONICAL_EVENTS`, `DataMapping`, `apply_mapping`, `coverage_report` |
| `connectors.py` | `Connector` ABC, `CSVConnector`, `SQLConnector`, `DataFrameConnector`, `SourceProbe` |
| `ai_utils.py` | Claude chat, tab insights, `infer_data_mapping`, retry/backoff |
| `configs/` | Mapping YAML files |

**Known issues to fix (tracked above):**
- 🔒 DB credentials leak into client-side `dcc.Store` → W4.S3
- `churned` / `active_days` / `ltv` are misleading → W1
- No timezone handling → W1.S5
- No multi-tenancy (global swapping) → W9

---

## Realistic expectation

16 weeks is the **optimistic** path with no bad weeks. Plan for **5-6 months**.
Debugging eats sessions unpredictably; every connector has one surprise in it.

That's fine. What kills projects like this isn't slow progress — it's *undirected* progress.
