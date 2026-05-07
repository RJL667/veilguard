# Phishing Incident Report

## Summary
- **Incident ID:**
- **Date Detected:**
- **Severity:** [Critical / High / Medium / Low]
- **Status:** [Open / Contained / Closed]
- **Analyst:**
- **Affected Users:**
- **Detection Source:** [SIEM / User Report / Email Gateway / Threat Intel]
- **Escalation Trigger:**
- **Initial Severity Rationale:**

Brief description of the incident (2–3 sentences: what happened, who was targeted, current status).

---

## IOCs
| Type | Value | Context | First Seen (UTC) | Blocked |
|------|-------|---------|------------------|---------|
| Domain | | | | |
| IP Address | | | | |
| URL | | | | |
| Sender Email | | | | |
| File Hash (SHA256) | | | | |
| Email Subject Line | | | | |

---

## Blast Radius
- Total recipients of phishing email:
- Users who clicked / interacted:
- Credentials submitted (confirmed / suspected):
- Systems accessed post-compromise:
- Data exfiltration confirmed (Y/N):

---

## Timeline
| Timestamp (UTC) | Event | Actor |
|-----------------|-------|-------|
| | Phishing email delivered to mailbox | Threat Actor |
| | User interaction detected (click/open) | Victim |
| | Alert triggered in SIEM | Automated |
| | Analyst assigned, investigation opened | SOC |
| | IOCs identified and extracted | Analyst |
| | Containment actions executed | SOC |
| | Affected accounts remediated | SOC |
| | Incident closed | Analyst |

---

## Recommendations
| Action | Owner | Priority | Status | Completed (UTC) |
|--------|-------|----------|--------|-----------------|
| Block IOCs at email gateway | | Immediate | | |
| Block IOCs at proxy/firewall | | Immediate | | |
| Reset affected credentials | | Short-term | | |
| MFA re-enrollment | | Short-term | | |
| Update phishing awareness training | | Long-term | | |
| Tune SIEM detection rules | | Long-term | | |
