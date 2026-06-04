# Skill Definition: Weekly Phishing-Feed Triage

**Skill ID:** `phishield-skill-phishing-feed-triage-weekly`  
**Version:** 1.0  
**Owner:** Phishield SOC / Threat Intelligence  
**Review cadence:** Quarterly

---

## 1. Purpose

Codifies the repeatable procedure for triaging inbound phishing indicators from threat-intelligence feeds on a weekly basis. Ensures consistent prioritisation, deduplication, and escalation of actionable indicators before they are pushed to detection infrastructure or shared with customers.

---

## 2. Trigger Conditions

This skill activates when **any** of the following conditions are met:

| # | Condition | Notes |
|---|-----------|-------|
| T1 | Scheduled weekly triage window opens (Monday 08:00 SAST) | Calendar-driven; primary trigger |
| T2 | Feed volume spike — inbound indicator count exceeds 2× the 4-week rolling average | Automated alert from feed aggregator |
| T3 | High-severity indicator flagged by upstream OSINT partner outside the normal window | Ad-hoc; treat as priority-1 triage |
| T4 | Customer reports a suspected phishing artefact that matches a known feed source | Incident-driven; merge into next triage cycle or open emergency lane |

---

## 3. Inputs

- Raw indicator export from feed aggregator (CSV or STIX 2.1 bundle)
- Previous week's triage log (for deduplication baseline)
- Customer sector mapping (maps indicators to affected verticals: banking, retail, legal, tech)
- Threat-actor TTP reference (MITRE ATT&CK, internal playbooks)

---

## 4. Step-by-Step Procedure

### Step 1 — Ingest & Normalise

1. Pull the weekly export from the feed aggregator (TAXII pull or manual download).
2. Normalise all indicators to a common schema: `{type, value, first_seen, last_seen, confidence, source_feed, tags}`.
3. Deduplicate against the previous 30-day indicator store. Flag duplicates as `KNOWN`; new entries as `NEW`.
4. Record raw count: `total_ingested`, `known`, `net_new`.

### Step 2 — Automated Pre-Scoring

1. Run the indicator batch through the automated pre-scorer:
   - **Confidence score** (0–100) from feed provider metadata.
   - **Recency score**: indicators seen within the last 7 days score higher.
   - **Sector relevance**: cross-reference customer sector mapping; indicators targeting active customer verticals receive a +20 relevance bonus.
2. Assign a composite **Triage Priority** label:
   - `P1` — confidence ≥ 80 AND sector-relevant
   - `P2` — confidence 50–79 OR sector-relevant but not both
   - `P3` — confidence < 50 AND not sector-relevant
3. Output: prioritised indicator list sorted P1 → P3.

### Step 3 — Analyst Review (P1 and P2 only)

1. For each P1 indicator:
   - Pivot on the indicator in threat-intel platform (VirusTotal, Shodan, internal graph).
   - Confirm active infrastructure (DNS resolves, URL returns content, domain registered recently).
   - Identify associated campaign or threat actor if possible.
   - Classify: `CONFIRMED_ACTIVE`, `STALE`, or `FALSE_POSITIVE`.
2. For each P2 indicator:
   - Perform lightweight pivot (VirusTotal lookup minimum).
   - Classify: `PLAUSIBLE`, `STALE`, or `FALSE_POSITIVE`.
3. P3 indicators are bulk-archived without manual review unless an analyst flags one during P1/P2 work.

### Step 4 — Escalation Decision

| Classification | Action |
|----------------|--------|
| `CONFIRMED_ACTIVE` (P1) | Escalate immediately — open threat-intel ticket, push block rule to SIEM/firewall within 4 hours |
| `CONFIRMED_ACTIVE` (P2) | Add to weekly block-rule batch; notify affected customer verticals |
| `PLAUSIBLE` | Add to watchlist; re-evaluate next weekly cycle |
| `STALE` | Archive; no action |
| `FALSE_POSITIVE` | Archive; flag source feed for quality review if FP rate > 10% |

### Step 5 — Customer Notification

1. Draft sector-specific advisory for any `CONFIRMED_ACTIVE` P1 indicators affecting customer verticals.
2. Advisory must include: indicator value (sanitised — defang URLs/IPs), indicator type, observed TTPs, recommended defensive action.
3. Route advisory through account manager for delivery, or push directly to customer portal if SLA requires < 24-hour notification.

### Step 6 — Documentation & Metrics

1. Update the weekly triage log:
   - `total_ingested`, `net_new`, `P1_count`, `P2_count`, `P3_count`
   - `confirmed_active`, `stale`, `false_positive` counts
   - Block rules pushed, advisories sent
2. If FP rate for any single feed exceeds 10% over 4 consecutive weeks, raise a feed-quality review ticket.
3. Archive the full indicator batch with triage decisions for audit trail (POPIA-compliant retention: 3 years).

---

## 5. Outputs

- Updated indicator store (deduplicated, classified)
- Block-rule batch (submitted to SIEM/firewall change pipeline)
- Customer advisories (where applicable)
- Weekly triage log entry
- Feed-quality review tickets (where triggered)

---

## 6. Worked Examples

### Example A — Banking-Sector Credential-Harvesting Campaign

**Context:** Monday triage window opens. Feed aggregator exports 340 indicators. Pre-scorer flags 12 as P1.

**Indicator under review:**  
`hxxps://secure-absa-login[.]co[.]za/verify` — phishing URL, confidence 91, first seen 3 days ago, tagged `credential-harvesting`, `banking`.

**Step 1 — Ingest & Normalise:**  
URL normalised to schema. Marked `NEW` (not in 30-day store). Net-new count incremented.

**Step 2 — Pre-Scoring:**  
Confidence 91 → high. Recency: 3 days → high. Sector: banking vertical matches 4 active Phishield banking customers → +20 bonus. Composite score: P1.

**Step 3 — Analyst Review:**  
- DNS resolves to a VPS in Eastern Europe (AS registered 6 days ago).  
- URL returns a convincing clone of a South African banking login page.  
- VirusTotal: 14/90 vendors flag as phishing.  
- No prior campaign association found; likely opportunistic actor.  
- Classification: `CONFIRMED_ACTIVE`.

**Step 4 — Escalation:**  
Threat-intel ticket opened. Block rule pushed to SIEM and perimeter firewall within 2 hours. Affected banking customers notified via advisory (see Step 5).

**Step 5 — Advisory:**  
> **Indicator:** `hxxps://secure-absa-login[.]co[.]za/verify` (defanged)  
> **Type:** Phishing URL — credential harvesting  
> **TTPs:** T1566.002 (Spearphishing Link), T1078 (Valid Accounts)  
> **Action:** Block at proxy/firewall; add to email gateway URL filter; brief end-users on unsolicited banking login prompts.

**Outcome:** Indicator blocked across 4 banking customers within SLA. No confirmed victim impact reported.

---

### Example B — Low-Confidence Retail Smishing Indicators

**Context:** Same Monday triage batch. Pre-scorer flags 28 indicators as P2, including a cluster of 9 SMS sender IDs tagged `smishing`, `retail`.

**Indicators under review:**  
Sender IDs: `+27-XXXX-XXXX` (×9), confidence 55, first seen 10 days ago, tagged `smishing`, `retail`, `parcel-delivery-lure`.

**Step 1 — Ingest & Normalise:**  
All 9 sender IDs normalised. 3 marked `KNOWN` (seen 18 days ago, previously classified `PLAUSIBLE`). 6 marked `NEW`.

**Step 2 — Pre-Scoring:**  
Confidence 55 → moderate. Recency: 10 days → moderate. Sector: retail vertical matches 2 active Phishield retail customers → +20 bonus. Composite: P2.

**Step 3 — Analyst Review (lightweight):**  
- VirusTotal lookup: 2/90 vendors flag; inconclusive.  
- No active infrastructure to pivot on (SMS sender IDs, not URLs).  
- 3 `KNOWN` indicators were `PLAUSIBLE` last cycle — no new victim reports received.  
- Classification: `PLAUSIBLE` (all 9).

**Step 4 — Escalation:**  
No immediate block rule. All 9 added/updated on watchlist. Flagged for re-evaluation next Monday.

**Step 5 — Advisory:**  
No customer advisory issued (threshold not met — `PLAUSIBLE` only). Internal note added to retail customer account files: monitor for end-user reports of parcel-delivery SMS lures.

**Step 6 — Feed Quality Note:**  
Source feed for this cluster has a 12% FP rate over the past 3 weeks. If it exceeds 10% again next week (4 consecutive weeks), a feed-quality review ticket will be raised automatically.

**Outcome:** Indicators on watchlist. No escalation. Feed quality being monitored.

---

## 7. Skill Maintenance

- **Quarterly review:** Validate scoring thresholds against actual FP/TP rates from the prior quarter.
- **Feed onboarding:** Any new feed source must be shadow-run for 4 weeks before contributing to P1 scoring.
- **Escalation SLAs:** P1 block rules within 4 hours; P2 block rules within 48 hours; customer advisories within 24 hours of `CONFIRMED_ACTIVE` classification.
