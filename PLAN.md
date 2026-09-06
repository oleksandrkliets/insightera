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
2026-08-05  W1.S1  Submitted early-stage API applications — DONE / IN PROGRESS / NOT STARTED
```

---

## 🔑 API ACCESS TIMELINE

Not every platform accepts an application before a working integration exists.
Submitting too early gets you rejected; submitting too late blocks the build.
Full justification copy lives in [`docs/API_ACCESS_APPLICATION.md`](docs/API_ACCESS_APPLICATION.md).

### Submit NOW — Week 1 (nothing to demonstrate yet)

| Platform | What to apply for | Why it can go now | Status |
|---|---|---|---|
| **Google Ads** | Developer token, **Basic** access | Token issued on application; Standard upgrade comes later | ☐ |
| **TikTok** | Business Verification | Company-level, independent of any app | ☐ |
| **QuickBooks** | Sandbox keys | Self-serve, immediate | ☐ |
| **HubSpot** | Developer account + app registration | No review needed for unlisted apps | ☐ |
| **GA4** | GCP project + service account | No review for the service-account approach | ☐ |

### No application needed at all

| Platform | Access method |
|---|---|
| **Stripe** | Customer pastes a restricted API key |
| **PostHog** | Customer pastes an API key |
| **Mailchimp / Klaviyo** | Customer pastes an API key |
| **Segment** | Customer adds a webhook destination |
| **Databases / warehouses** | Customer supplies a read-only connection |

### Submit AFTER Week 10 — needs hosted app + privacy policy live

| Platform | What to apply for | Blocker | Status |
|---|---|---|---|
| **LinkedIn** | Marketing Developer Platform | Detailed use-case review; may request a demo | ☐ |

### Submit AFTER the connector is built and demonstrable

| Platform | What to apply for | Blocker | Target | Status |
|---|---|---|---|---|
| **Meta** | App Review, `ads_read` Advanced Access + Business Verification | Requires working flow, reviewer test account, screencast, live privacy policy | W14 | ☐ |
| **QuickBooks** | Production keys | Requires working OAuth flow | after Stripe/QBO built | ☐ |
| **Google Ads** | **Standard** access upgrade | Requires demonstrating the tool in use | W13 | ☐ |

### Hard prerequisites for the later applications

- [ ] Hosted app at a public URL — *W10*
- [ ] Privacy policy published at a public URL — *W10.S5*
- [ ] Terms of service published — *W10.S5*
- [ ] Token encryption at rest — *W4.S1*
- [ ] Per-customer data isolation — *W9*
- [ ] Data deletion path implemented + documented — *W9.S5*
- [ ] Reviewer test account with credentials — *W14*
- [ ] Screencast: connect → sync → view — *W14*

> ⚠️ **TikTok note:** business verification is worth starting now (company-level,
> takes calendar time), but the *connector* is low priority — TikTok ad spend in
> B2B SaaS is near zero. It sits in the cut list, not the 16 weeks.
> If your first customers turn out to be ecommerce, this ordering flips.

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

- [ ] **W1.S1 — Submit the early-stage API applications** ⏰ *~45 min, clocks run for weeks*
  - Google Ads: developer token, **Basic** access
  - TikTok: Business Verification (company docs — see below)
  - QuickBooks: sandbox keys
  - HubSpot: developer account + app registration
  - GA4: GCP project + service account
  - ⚠️ **Do NOT apply for Meta, LinkedIn, or QuickBooks production yet** — they
    require a hosted app and a working flow. See the API Access Timeline above.
  - Have ready for TikTok verification: certificate of incorporation, business
    address matching it, a live company website
  - **Done when:** all 5 submitted, confirmation emails saved to a folder

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
- [ ] S3-S4 — Remove `_with_filtered_globals` global-swapping (`dashboard_flows.py:7680`, unsafe with >1 user)
- [ ] S5 — **Data deletion path** — disconnect a source → tokens deleted, data purged
  - 🔑 *Required by Meta and QuickBooks applications*
- **Done when:** two workspaces run side by side with no data bleed

### Week 10 — Auth + deploy + unblock LinkedIn
- [ ] S1-S2 — Clerk auth (don't build your own)
- [ ] S3-S4 — Deploy to Render (Dockerfile exists) + custom domain + HTTPS
- [ ] S5 — **Publish privacy policy + terms of service** 🔑
  - Substance already drafted in `docs/API_ACCESS_APPLICATION.md` §5
  - Must name: data collected, sub-processors, retention, deletion path
- [ ] **🔑 SUBMIT: LinkedIn Marketing Developer Platform** — now unblocked
- **Done when:** a real URL a stranger can log into, with policy pages live

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
- [ ] S1-S2 — MCC account linking flow (customer approves link in their own Ads UI)
- [ ] S3-S4 — Campaign-level GAQL query (works across all campaign types)
  - Remember: `cost_micros` ÷ 1,000,000. Getting this wrong reports spend 1,000,000× high.
  - Store `advertising_channel_type`; warn where PMax limits available breakdowns
- [ ] S5 — **🔑 SUBMIT: Google Ads Standard access upgrade** — you can now demonstrate the tool

### Week 14 — Meta Ads + submit the blocked applications
- [ ] S1-S3 — OAuth + Marketing API insights + attribution-window handling
  - Label Meta-reported conversions as Meta-attributed; use Stripe for revenue truth
- [ ] S4 — Record the reviewer screencast: connect → sync → view → disconnect
- [ ] S5 — **🔑 SUBMIT: Meta App Review** (`ads_read` Advanced Access + Business Verification)
- [ ] S5 — **🔑 SUBMIT: QuickBooks production keys** (OAuth flow now demonstrable)
- **Done when:** all remaining applications submitted, test account credentials documented

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

1. **TikTok Ads** — near-zero B2B SaaS spend (keep the business verification, skip the connector)
2. **LinkedIn Ads** — longest approval, modest share
3. **Segment** — GA4 + PostHog cover most of the same ground
4. **Mailchimp / Klaviyo** — not core to the product+finance thesis
5. **GA4** — most painful build per unit of value (nested `event_params` flattening)

**Never cut:** validation · self-healing loop · onboarding · deployment.
Those are what make it a product instead of a script.

> If your first customers are **ecommerce rather than B2B SaaS**, this ordering
> inverts — TikTok and Klaviyo jump up, LinkedIn drops further. One more reason
> to run the customer conversations before Week 13.

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
| `docs/API_ACCESS_APPLICATION.md` | Platform application copy — endpoints, data handling, compliance |

**Known issues — all verified against the code 2026-08-05:**

| Issue | Location | Fixed in |
|---|---|---|
| 🔒 DB credentials (incl. password) written to client-side store | `dashboard_flows.py:8985` → `dcc.Store` at `:7492` | W4.S3 |
| `active_days` = total span, not distinct active dates | `etl.py:343` | W1.S2 |
| `churned` = 3-signal heuristic, flags new users as churned | `etl.py:393-394` | W1.S3 |
| `ltv` is revenue-to-date, not a projection — name misleads | `etl.py:410` | W1.S4 |
| "now" = last event in data, not actual now | `etl.py:381` | W1.S4 |
| No timezone handling anywhere (0 occurrences of `tz=`) | `etl.py` | W1.S5 |
| Global-swapping blocks multi-tenancy | `dashboard_flows.py:7680` | W9 |

*Line numbers drift as you edit — grep the symbol if a reference goes stale.*

---

## Realistic expectation

16 weeks is the **optimistic** path with no bad weeks. Plan for **5-6 months**.
Debugging eats sessions unpredictably; every connector has one surprise in it.

That's fine. What kills projects like this isn't slow progress — it's *undirected* progress.
