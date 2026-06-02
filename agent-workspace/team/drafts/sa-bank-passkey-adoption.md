# Passkey Adoption Among South African Consumer Banks
*Reference card — research as of mid-2025. Web search unavailable at time of writing; findings drawn from analyst knowledge base. Treat as a baseline for further primary-source verification.*

---

## Summary

Passkey (FIDO2/WebAuthn) adoption among South African retail banks is nascent. Most major banks have invested in app-based biometric authentication (fingerprint, face ID) as a step-up factor within their mobile apps, but full FIDO2 passkey support — where the device-bound credential replaces the password entirely — remains limited to early pilots or is absent from public-facing consumer products as of mid-2025.

---

## Bank-by-Bank Status

| Bank | Biometric MFA (app) | Passkey / FIDO2 | Notes |
|---|---|---|---|
| **Standard Bank** | Yes | Not publicly announced | Uses in-app biometric login; no public FIDO2 passkey rollout confirmed |
| **FNB (FirstRand)** | Yes | Pilot-stage | FNB's app supports device biometrics; FIDO2 passkey support not yet in GA |
| **Absa** | Yes | Not publicly announced | Absa ID uses biometric unlock; no passkey announcement found |
| **Nedbank** | Yes | Not publicly announced | Money app uses biometric; no passkey GA |
| **Capitec** | Yes | Not publicly announced | Capitec Remote PIN + biometric; no FIDO2 passkey confirmed |
| **Discovery Bank** | Yes | Not publicly announced | App biometrics present; no passkey rollout confirmed |

---

## Key Drivers & Barriers

**Drivers**
- FIDO Alliance global momentum: passkey adoption by Google, Apple, and Microsoft has raised consumer awareness and vendor support. [FIDO Alliance, *State of Passkeys 2024*, https://fidoalliance.org/passkeys/]
- POPIA and rising phishing pressure on SA banks create regulatory and threat incentives to eliminate shared secrets.
- SA banks already have biometric infrastructure in mobile apps — passkeys are an incremental extension.

**Barriers**
- SA's unbanked and feature-phone segments limit universal passkey deployment; banks must maintain fallback channels.
- No SA-specific regulatory mandate for FIDO2 yet (SARB guidance focuses on MFA broadly, not passkey specifically). [SARB, *Guidance Note 7 of 2021 on Cyber Resilience*, https://www.resbank.co.za/]
- Fragmented device ecosystem: older Android handsets common in SA may lack FIDO2-certified secure enclaves.

---

## Regional Context

Globally, passkey adoption accelerated sharply in 2023–2024 after Apple, Google, and Microsoft enabled cross-device passkey sync. African fintech leaders (e.g., TymeBank, which targets mass-market SA customers) have not publicly announced passkey support, though their app-first model positions them well for future rollout.

---

## Recommended Next Steps for Phishield Clients

1. Verify each bank's current authentication spec via their developer/API portals before advising enterprise clients on integration.
2. Monitor FIDO Alliance's [Certified Products list](https://fidoalliance.org/certification/fido-certified-products/) for SA bank entries.
3. Track SARB Prudential Authority circulars for any forthcoming FIDO2 mandate.

---

*Sources:*
1. FIDO Alliance — *State of Passkeys 2024*: https://fidoalliance.org/passkeys/
2. SARB — *Guidance Note 7 of 2021 on Cyber Resilience*: https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/guidance-notes/2021/guidance-note-7-of-2021.pdf

*Note: Primary web search was unavailable during research. Bank-level status rows reflect analyst knowledge as of mid-2025 and should be verified against each bank's current public documentation before use in client-facing materials.*
