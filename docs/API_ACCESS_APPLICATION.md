# Insightera — API Access Application Documentation

> **Working document.** Sections are written to be copy-pasted into platform
> application forms. Fields marked `⟨FILL IN⟩` need your details before submission.
> Sections marked ⚠️ describe things not yet built — see the readiness checklist
> at the end before applying to that platform.

---

## 1. Company & contact details

| Field | Value |
|---|---|
| Legal entity name | ⟨FILL IN⟩ |
| Trading name | Insightera |
| Registered address | ⟨FILL IN⟩ |
| Country of incorporation | ⟨FILL IN⟩ |
| Website | ⟨FILL IN — e.g. https://insightera.io⟩ |
| Privacy policy URL | ⟨FILL IN — required by Meta, LinkedIn, QuickBooks⟩ |
| Terms of service URL | ⟨FILL IN⟩ |
| Technical contact | Oleksandr Kliets — ⟨email⟩ |
| Data protection contact | ⟨FILL IN⟩ |
| Hosting provider / region | ⟨FILL IN — e.g. Render, EU-Frankfurt⟩ |

---

## 2. Product overview

**One-line description**

Insightera is a business analytics platform that unifies product usage,
financial, and marketing data into a single reporting layer for
software-as-a-service companies.

**Full description** *(suitable for most application forms)*

Insightera is a reporting and analytics tool used by product, growth, and
finance teams at software companies. It connects to a customer's existing
business systems — their billing provider, CRM, advertising accounts, and
product analytics — and produces unified reporting on customer acquisition
cost, conversion rates, retention, churn risk, and revenue attribution by
marketing channel.

The platform is read-only with respect to connected advertising and CRM
accounts. It does not create, modify, pause, or manage advertising campaigns,
budgets, bids, audiences, or creative assets. Its sole function with respect to
advertising platforms is to retrieve historical performance metrics
(spend, impressions, clicks) so those figures can be combined with the
customer's own revenue data to calculate acquisition cost and payback period.

Each customer connects only their own accounts, and views only their own data.

**Who uses it**

Employees of the account-holding company — typically:

- Product managers analysing feature adoption and activation
- Growth and marketing leads measuring channel efficiency
- Founders and finance leads tracking revenue retention and unit economics
- Customer success teams identifying accounts at risk of churn

**What it is not**

- Not an ad-management or bid-automation tool
- Not a data broker — data is never sold, shared, or aggregated across customers
- Not an audience-building or targeting tool
- Not consumer-facing — all users are employees of the connecting business

---

## 3. Architecture and data flow

```
Customer's own systems          Insightera
─────────────────────           ──────────────────────────────────
Billing (Stripe)         ──┐
Advertising accounts     ──┤
CRM (HubSpot)            ──┼──▶  Connector layer (read-only API calls)
Product analytics        ──┤            │
Customer database        ──┘            ▼
                                 Schema mapping to canonical model
                                        │
                                        ▼
                                 Validation & quality checks
                                        │
                                        ▼
                                 Per-customer isolated storage
                                        │
                                        ▼
                                 Dashboard (customer's own users only)
```

**Processing model**

1. A customer authorises a connection to one of their own accounts.
2. Insightera makes scheduled read-only API calls to retrieve historical data.
3. Retrieved records are mapped to an internal canonical schema.
4. Automated validation checks run before data is committed.
5. Aggregated reporting is presented to that customer's users only.

**Sync frequency:** hourly at most; typically daily for advertising and
financial data. Data is retrieved incrementally using date or cursor
watermarks so that each sync retrieves only records created or modified since
the previous successful sync.

---

## 4. Per-platform API usage

### 4.1 Google Ads API

**Access requested:** Standard access on developer token

**Tool classification:** Reporting/analytics tool. Insightera does not perform
campaign management operations.

**How the API is used**

Insightera issues read-only GAQL queries against the `campaign` resource to
retrieve historical performance metrics. The complete set of fields requested:

```sql
SELECT
  segments.date,
  campaign.id,
  campaign.name,
  campaign.status,
  campaign.advertising_channel_type,
  metrics.cost_micros,
  metrics.impressions,
  metrics.clicks,
  metrics.conversions,
  metrics.conversions_value
FROM campaign
WHERE segments.date BETWEEN @start_date AND @end_date
```

**Purpose:** these metrics are combined with the customer's own revenue data
(from their billing provider) to calculate customer acquisition cost, return on
ad spend, and payback period by campaign and channel.

**Operations performed:** `GoogleAdsService.SearchStream` (read) only.
No mutate operations of any kind are issued.

**Account linkage:** customers link their own Google Ads account to
Insightera's manager account (MCC). Each customer approves this link from
within their own Google Ads interface. Insightera accesses only accounts that
have been explicitly linked by their owner.

**Campaign types supported:** all types. Queries operate at campaign level,
which is available across Search, Performance Max, Display, Demand Gen, Video,
and Shopping campaigns. Where a campaign type does not expose a given
breakdown (for example, keyword-level data is not available for Performance
Max campaigns), the interface indicates this to the user rather than
presenting incomplete data as complete.

**Estimated call volume:** approximately 2–5 API calls per connected account
per day (one incremental query per sync, plus retries). With ⟨N⟩ connected
accounts, this is approximately ⟨N × 5⟩ calls per day.

**Required Minimum Functionality:** Insightera operates as a reporting tool.
It presents campaign performance metrics retrieved from the API within a
reporting interface, alongside the customer's own first-party revenue data.

---

### 4.2 Meta Marketing API ⚠️

**Permissions requested:** `ads_read`, `business_management` (read)

**How the API is used**

Read-only retrieval of advertising insights from ad accounts the customer owns
or administers:

| Endpoint | Purpose |
|---|---|
| `/act_{ad_account_id}/insights` | Spend, impressions, clicks, reach by campaign and date |
| `/act_{ad_account_id}/campaigns` | Campaign names and IDs for labelling |
| `/me/adaccounts` | List accounts the authorising user can access, so they can select which to connect |

**Purpose:** to calculate acquisition cost and channel efficiency by combining
Meta advertising spend with the customer's own revenue records.

**Operations performed:** GET requests only. No campaign creation, editing,
budget changes, audience creation, or creative management.

**Data retained:** aggregated daily metrics per campaign (date, campaign ID,
campaign name, spend, impressions, clicks). No user-level data, no audience
data, no creative assets, and no personally identifiable information from Meta
is retrieved or stored.

**Attribution handling:** Meta-reported conversion figures are labelled in the
interface as Meta-attributed and are presented separately from the customer's
own first-party revenue data, to avoid conflating two different attribution
models.

---

### 4.3 LinkedIn Marketing API ⚠️

**Access requested:** Marketing Developer Platform — Advertising API (read)

**How the API is used**

| Endpoint | Purpose |
|---|---|
| `/adAnalytics` | Spend, impressions, clicks by campaign and date |
| `/adCampaignsV2` | Campaign metadata for labelling |
| `/adAccountsV2` | Enumerate accounts the authorising user can access |

**Purpose:** identical to the above — combining LinkedIn advertising spend with
first-party revenue to compute acquisition cost and channel return.

**Operations performed:** read only. No campaign management, no audience
creation, no lead retrieval, no messaging.

**Estimated call volume:** approximately 5–10 calls per connected account per
day, well within standard rate limits.

---

### 4.4 TikTok Marketing API ⚠️

**Access requested:** Reporting (read)

**How the API is used**

Read-only retrieval of `/report/integrated/get/` for daily campaign-level
spend, impressions, and clicks. Campaign metadata retrieved from
`/campaign/get/` for labelling. No campaign management operations.

---

### 4.5 QuickBooks Online API ⚠️

**Access requested:** Production keys

**How the API is used**

Read-only retrieval of accounting records to support revenue and expense
reporting alongside product data:

| Entity | Purpose |
|---|---|
| `Invoice` | Revenue recognition and billing history |
| `Payment` | Cash receipts |
| `Purchase` / `Bill` | Expense categorisation for margin analysis |
| `Account` | Chart of accounts, for mapping customer-defined accounts to canonical categories |
| `Customer` | Entity names, for joining to product usage records |

**Operations performed:** read only. Insightera does not create, modify, or
delete any accounting record, and does not post transactions.

---

### 4.6 Stripe API

**Access method:** customer-supplied restricted API key, read-only scopes

**Resources accessed:** `Customer`, `Subscription`, `Invoice`, `Charge`,
`Refund`, `Dispute` — all read operations.

**Purpose:** revenue, MRR, churn, and lifetime value reporting; and to serve as
the first-party source of truth for conversion and revenue attribution.

**Operations performed:** read only. No charges, refunds, subscription
changes, or customer modifications are ever issued.

---

## 5. Data handling

### What is stored

| Category | Examples | Retention |
|---|---|---|
| Aggregated advertising metrics | date, campaign ID/name, spend, impressions, clicks | For the contracted term of the customer's subscription |
| Financial records | invoices, subscriptions, charges (amounts, dates, status) | Same |
| Product usage events | event name, timestamp, pseudonymous user identifier | Same |
| CRM records | company name, deal stage, deal value | Same |
| OAuth tokens | access and refresh tokens per connection | Until the connection is removed |

### What is not stored

- Advertising creative assets, audience definitions, or targeting criteria
- Payment card numbers or bank account details (Insightera never receives these)
- Advertising platform user-level or cookie-level data
- Data from any account the customer has not explicitly connected

### Isolation

Each customer's data is stored in a logically isolated workspace. Data is
never combined, aggregated, benchmarked, or otherwise processed across
customers. No customer can access another customer's data.

### Encryption

- In transit: TLS 1.2 or higher for all API calls and all user access
- At rest: OAuth tokens and connection credentials encrypted at rest;
  encryption keys held in a managed secrets store separate from the
  application database

### Deletion

Customers may disconnect any data source at any time from the application
settings. On disconnection, stored OAuth tokens for that source are deleted
immediately and data retrieved from that source is deleted within 30 days.

On account termination, all customer data is deleted within 30 days.

Deletion requests may also be made to ⟨FILL IN — data protection contact⟩ and
are actioned within 30 days.

### Sub-processors

| Sub-processor | Purpose |
|---|---|
| ⟨FILL IN — e.g. Render⟩ | Application hosting |
| ⟨FILL IN — e.g. Clerk⟩ | User authentication |
| Anthropic | AI-generated written summaries of aggregated metrics |

**Note on AI processing:** Insightera uses a large language model to generate
plain-language summaries of a customer's aggregated metrics and to assist in
mapping source data schemas to its internal model. Only aggregated statistics
and schema metadata (column names, event names) are sent for this purpose.
Advertising platform data sent for this purpose is limited to aggregated
campaign-level figures. No personally identifiable information is included.

---

## 6. User authorisation flow

1. A user signs in to Insightera with their own credentials.
2. From the Settings screen, they select the platform they wish to connect.
3. They are redirected to that platform's own authorisation screen, hosted by
   the platform, where they sign in with their own credentials and review the
   requested permissions.
4. On approval, the platform returns an authorisation code, which Insightera
   exchanges for access and refresh tokens.
5. Tokens are encrypted and stored server-side. They are never transmitted to
   the browser or exposed in client-side code.
6. The user selects which of their accounts to include.
7. Scheduled read-only syncs begin.

Users may revoke access at any time, either from within Insightera or from the
platform's own application settings.

---

## 7. Security practices

- All access over HTTPS; TLS 1.2 minimum
- Credentials stored encrypted at rest; never written to logs or client-side storage
- Least-privilege scopes requested — read-only in every case
- Access tokens automatically refreshed; expired credentials surfaced to the
  customer for reconnection rather than retried indefinitely
- Application error monitoring with alerting on failed syncs
- Dependency vulnerability scanning in the build pipeline
- ⟨FILL IN — any additional practices: MFA, access review cadence, backup policy⟩

---

## 8. Reviewer testing instructions

*(Meta and QuickBooks require step-by-step instructions and often a screencast.)*

1. Navigate to ⟨FILL IN — application URL⟩
2. Sign in with the test credentials provided: ⟨FILL IN⟩
3. Open the **Settings** tab
4. Under **Data sources**, select ⟨platform⟩
5. Click **Connect** — you will be redirected to ⟨platform⟩'s authorisation screen
6. Approve the requested read permissions
7. Select an ad account from the list presented
8. The connection status will show **Connected**, and an initial sync will run
9. Open the **Conversion** tab — retrieved spend figures appear in the
   acquisition-cost reporting alongside the customer's own revenue data
10. To revoke: return to **Settings** and click **Disconnect**

⚠️ **This section cannot be completed until the relevant connector and a hosted
environment exist.** See readiness checklist below.

---

## 9. Readiness checklist before applying

Not all platforms accept applications before a working integration exists. This
affects the order you should apply in.

| Platform | Can apply before building? | What it requires |
|---|---|---|
| **Google Ads** | ✅ Basic token yes | Standard access typically requires demonstrating the tool. Apply for the token now, upgrade later. |
| **TikTok** | ✅ Business verification yes | Verification is company-level, independent of the app |
| **LinkedIn** | ⚠️ Partly | Application asks detailed use-case questions; a demo may be requested |
| **Meta** | ❌ No | Requires a working app, a testable flow, a screencast, and a live privacy policy URL |
| **QuickBooks** | ❌ Not for production | Sandbox access is immediate; production keys need a working OAuth flow |

**Still needed before Meta and QuickBooks production applications:**

- [ ] Hosted application at a public URL *(plan: Week 10)*
- [ ] Privacy policy published at a public URL
- [ ] Terms of service published
- [ ] Working OAuth flow for the platform in question
- [ ] Test account credentials for reviewers
- [ ] Screencast of the connect → sync → view flow
- [ ] Token encryption at rest implemented *(plan: Week 4)*
- [ ] Per-customer data isolation implemented *(plan: Week 9)*
- [ ] Data deletion path implemented and documented

**Recommended application order:**

1. **Now:** Google Ads developer token (basic), TikTok business verification,
   QuickBooks sandbox
2. **After Week 10** (hosted, with privacy policy): LinkedIn MDP
3. **After the relevant connector is built and demonstrable:** Meta App Review,
   QuickBooks production keys, Google Ads Standard access upgrade

---

## 10. Statements of compliance

The following are true of Insightera's use of advertising platform APIs:

- Data is retrieved read-only. No campaign, budget, bid, audience, or creative
  is ever created, modified, paused, or deleted.
- Data is accessed only from accounts the account owner has explicitly
  authorised, and only for the duration of that authorisation.
- Data is not sold, licensed, shared with third parties, or used for any
  purpose other than presenting reporting to the customer who authorised the
  connection.
- Data is not combined across customers, and is not used to build benchmarks,
  audiences, models, or derived products.
- Data is not used for advertising targeting, audience building, or
  re-identification of individuals.
- Platform-reported figures are presented in the interface as originating from
  that platform, and are not conflated with figures from other sources.
- Access is revocable by the customer at any time, from within Insightera or
  from the platform's own settings.

---

*Document version 1.0 — ⟨date⟩*
