# Phishing Trends & Threat Landscape — 2025 Research Brief

**Prepared by:** Veilguard Research Agent  
**Scope:** Current phishing techniques, threat actor tactics, and defensive recommendations relevant to South African SMEs  
**Note:** Web research bridge was offline during this task. Claims marked [unverified] could not be confirmed against live sources. All other claims are drawn from established cybersecurity knowledge current to early 2025.

---

## 1. Executive Summary

Phishing remains the leading initial-access vector for cybercriminals targeting SMEs globally and in South Africa. The 2024–2025 period has seen a marked shift toward AI-assisted lure generation, multi-channel attacks (email + SMS + voice), and abuse of legitimate cloud infrastructure to bypass reputation-based filters. South African organisations face additional exposure from targeted Business Email Compromise (BEC) campaigns exploiting local banking relationships and SARS (South African Revenue Service) impersonation.

---

## 2. Key Phishing Trends (2024–2025)

### 2.1 AI-Generated Lures

Generative AI tools have dramatically lowered the barrier to producing grammatically correct, contextually convincing phishing emails. Historically, poor grammar and spelling were reliable detection signals; that signal is now largely unreliable. [unverified: specific percentage increase in AI-generated phishing volume]

Key characteristics:
- Personalised salutations and body text drawn from OSINT (LinkedIn, company websites)
- Tone-matched to the target organisation's communication style
- Multilingual lures including Afrikaans and Zulu variants targeting South African recipients [unverified]

### 2.2 Adversary-in-the-Middle (AiTM) Phishing

AiTM frameworks (e.g., Evilginx2, Modlishka, Muraena) proxy authentication sessions in real time, capturing session cookies post-MFA. This bypasses TOTP and push-notification MFA entirely. Targets are typically Microsoft 365 and Google Workspace users — both dominant in the South African SME market.

Indicators:
- Login page hosted on a lookalike domain (e.g., `login.microsoftonline-secure[.]com`)
- Valid TLS certificate (Let's Encrypt or purchased)
- Redirect chain through a legitimate CDN (Cloudflare, Fastly) to obscure origin

### 2.3 QR Code Phishing ("Quishing")

QR codes embedded in PDF attachments or email bodies redirect victims to credential-harvesting pages. Email security gateways that scan URLs in message bodies do not parse QR image content, creating a blind spot.

Detection gap: Most SEGs (Secure Email Gateways) as of 2024 do not perform optical character recognition on embedded images by default. [unverified: vendor-specific coverage rates]

### 2.4 Multi-Channel / Hybrid Attacks

Attackers combine:
1. An initial email establishing context (e.g., "your invoice is attached")
2. A follow-up SMS or WhatsApp message impersonating the same sender
3. A vishing (voice phishing) call to pressure the victim into completing the action

This pattern is particularly effective against finance and HR staff who handle payment authorisations.

### 2.5 Abuse of Legitimate Cloud Services

Phishing lures increasingly originate from or link to:
- SharePoint Online and OneDrive (Microsoft)
- Google Drive and Google Sites
- Dropbox, DocuSign, Adobe Sign
- Notion, Canva, and other SaaS platforms

Because these domains carry high reputation scores, URL-reputation filters pass them. The malicious redirect is one or two hops downstream.

### 2.6 South Africa–Specific Vectors

- **SARS impersonation:** Tax refund and penalty notices spoofing `@sars.gov.za` sender domains. Peak volume aligns with filing seasons (February–March, August–September). [unverified: 2024 volume statistics]
- **FNB / Standard Bank / Absa / Nedbank impersonation:** OTP-harvesting pages mimicking South African banking portals. Often delivered via SMS (smishing) with a link to an AiTM proxy.
- **Load-shedding themed lures:** Eskom and municipal utility impersonation offering "prepaid credit" or "outage compensation" — a South Africa–specific social engineering hook. [unverified: active campaign status in 2025]
- **CIPC / Companies and Intellectual Property Commission lures:** Fake compliance notices targeting business owners. [unverified]

---

## 3. Threat Actor Profiles (Relevant to SA SMEs)

### 3.1 TA2541 (Aviation/Logistics Focus)
A financially motivated actor known for targeting logistics and transport companies with commodity RATs (AsyncRAT, NetWire). South African logistics firms are plausible targets given regional trade volumes. [unverified: confirmed SA targeting]

### 3.2 BEC Syndicates (West African Origin)
Well-documented BEC operations, some with South African operational nodes, target CFOs and accounts-payable staff. Tactics include:
- CEO fraud (impersonating the MD/CEO to instruct wire transfers)
- Vendor impersonation (hijacking supplier email threads)
- Payroll diversion (HR impersonation to redirect salary payments)

### 3.3 Ransomware Initial Access Brokers (IABs)
Phishing is the primary delivery mechanism for IABs selling access to ransomware-as-a-service (RaaS) affiliates. South African healthcare, legal, and retail SMEs have appeared in leak sites associated with LockBit, BlackCat/ALPHV, and Medusa affiliates. [unverified: current active campaigns post-LockBit disruption]

---

## 4. Technical Indicators of Compromise (IOCs) — Generic Patterns

The following are pattern-level IOCs, not live threat-intel feeds. Validate against current feeds before actioning.

| Pattern | Description |
|---|---|
| Lookalike domains | Homoglyph substitution (e.g., `rn` → `m`), TLD swap (`.co.za` → `.co-za[.]com`) |
| Newly registered domains | Registration age < 30 days combined with MX records and valid TLS |
| Mismatched `From` / `Reply-To` | Legitimate display name, attacker-controlled reply address |
| HTML smuggling | Base64-encoded payload assembled client-side via JavaScript to bypass gateway inspection |
| Redirector chains | Legitimate URL → open redirect → phishing page (e.g., Google AMP, Bing redirect) |

---

## 5. Defensive Recommendations for South African SMEs

### 5.1 Email Security Controls
- Deploy DMARC (policy: `reject`), DKIM, and SPF on all owned domains — including parked domains that send no mail (set SPF `v=spf1 -all`, DMARC `p=reject`)
- Enable anti-impersonation controls in Microsoft 365 Defender or Google Workspace Advanced Protection
- Configure SEG to detonate URLs in a sandbox, not just check reputation at delivery time
- Enable QR-code scanning in SEG if the vendor supports it; otherwise consider a compensating control (user awareness)

### 5.2 Identity & Authentication
- Enforce phishing-resistant MFA (FIDO2/WebAuthn hardware keys or passkeys) for all privileged accounts and internet-facing services
- For accounts where FIDO2 is not yet deployed, use Microsoft Authenticator number-matching or similar push-upgrade to reduce AiTM effectiveness
- Implement Conditional Access policies restricting authentication to compliant, managed devices

### 5.3 User Awareness
- Conduct simulated phishing exercises quarterly, with immediate just-in-time training for clickers
- Train staff specifically on QR code lures, vishing follow-ups, and the "legitimate cloud service" pattern
- Establish a clear, low-friction internal reporting channel (e.g., a "Report Phish" button in Outlook/Gmail)

### 5.4 Incident Response Readiness
- Pre-define the playbook for a compromised Microsoft 365 / Google Workspace account: token revocation, session invalidation, audit log preservation
- Ensure audit logging (Unified Audit Log in M365, Admin Audit in Google Workspace) is enabled and retained for at least 90 days
- Maintain an out-of-band communication channel (Signal group or similar) for IR coordination in case primary email is compromised

### 5.5 South Africa–Specific Guidance
- Register defensive domain variants of your primary domain (e.g., `company-za[.]com`, `companyza[.]co.za`) to prevent lookalike registration by threat actors
- Monitor CIPC filings for your company name — attackers sometimes register shell companies with similar names to add legitimacy to BEC lures
- Subscribe to the SABRIC (South African Banking Risk Information Centre) threat alerts if operating in financial services [unverified: current subscription availability]

---

## 6. Detection Rules (SIEM / EDR)

### 6.1 Sigma Rule — HTML Smuggling via Base64 in Email Attachment

```yaml
title: HTML Smuggling — Base64 Payload in HTML Attachment
status: experimental
description: Detects HTML attachments containing large Base64 blobs typical of HTML smuggling payloads
logsource:
    category: email
detection:
    selection:
        attachment_extension: '.html'
        attachment_content|contains:
            - 'atob('
            - 'fromCharCode'
            - 'createElement'
    condition: selection
falsepositives:
    - Legitimate HTML newsletters with embedded images
level: medium
```

### 6.2 Sigma Rule — AiTM Session Cookie Theft Indicator

```yaml
title: Suspicious OAuth Token Refresh from New ASN
status: experimental
description: Detects OAuth token refresh from an ASN not previously seen for this user — potential AiTM session hijack
logsource:
    product: azure
    service: signin
detection:
    selection:
        ResultType: 0
        TokenIssuerType: 'AzureAD'
    filter_known_asn:
        # Populate with baseline ASNs for your organisation
        NetworkLocationDetails|contains: 'knownNetwork'
    condition: selection and not filter_known_asn
falsepositives:
    - Legitimate travel, VPN changes
level: high
```

---

## 7. Limitations & Caveats

- The web research bridge was unavailable during this task. All statistics and vendor-specific claims are marked [unverified] and must be validated against current threat-intel feeds (e.g., Proofpoint State of the Phish, Verizon DBIR, SABRIC Annual Crime Stats) before use in client-facing materials.
- IOC patterns are generic; live IOC feeds (VirusTotal, MISP, OpenPhish) should be consulted for current indicators.
- Threat actor attribution is based on publicly available reporting current to early 2025 and may not reflect post-disruption rebranding (e.g., post-LockBit takedown affiliate migration).

---

## 8. References (Pending Verification)

1. Proofpoint — *State of the Phish 2024* — https://www.proofpoint.com/us/resources/threat-reports/state-of-phish [unverified: live link]
2. Verizon — *2024 Data Breach Investigations Report* — https://www.verizon.com/business/resources/reports/dbir/ [unverified: live link]
3. Microsoft — *Digital Defense Report 2024* — https://www.microsoft.com/en-us/security/security-insider/microsoft-digital-defense-report-2024 [unverified: live link]
4. SABRIC — *Annual Crime Stats* — https://www.sabric.co.za [unverified: live link]
5. MITRE ATT&CK — Phishing (T1566) — https://attack.mitre.org/techniques/T1566/
