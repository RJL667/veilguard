# Passkey Adoption Among South African Consumer Banks

*Reference card — research as of mid-2025.*

---

## Summary

Passkey (FIDO2/WebAuthn) adoption among South African retail banks is nascent. All major banks have deployed app-based biometric authentication (fingerprint, face ID) as a step-up factor within their mobile apps, but full FIDO2 passkey support — where a device-bound credential replaces the password entirely — remains limited to early pilots or is absent from public-facing consumer products as of mid-2025.

The FIDO Alliance's 2024 market report notes that financial services globally are the leading vertical for passkey deployment, yet emerging-market banks lag behind US and European peers by 12–18 months on average.[^1] South African banks follow this pattern: strong biometric MFA inside proprietary apps, but no confirmed general-availability passkey rollout across the Big Four.

---

## Bank-by-Bank Status

| Bank | Biometric MFA (app) | Passkey / FIDO2 | Notes |
|---|---|---|---|
| **Standard Bank** | Yes | Not publicly announced | In-app biometric login (fingerprint/face); no public FIDO2 passkey rollout confirmed as of mid-2025 |
| **FNB (FirstRand)** | Yes | Pilot / limited | FNB app supports device biometrics; FIDO2 passkey support referenced in developer sandbox docs but not in GA for retail customers[^2] |
| **Absa** | Yes | Not publicly announced | Absa ID uses biometric unlock; no passkey announcement found in public press releases |
| **Nedbank** | Yes | Not publicly announced | Nedbank Money app uses biometric step-up; no FIDO2 passkey GA announcement found |
| **Capitec** | Yes | Not publicly announced | Capitec Remote PIN + biometric; no passkey rollout confirmed |

---

## Key Observations

- **Biometric ≠ Passkey.** All five banks offer biometric unlock inside their proprietary apps. This is *not* FIDO2 passkey authentication — it is a local biometric gate on a session token, not a synced or device-bound WebAuthn credential.
- **Regulatory driver absent.** SARB's Guidance Note 2/2023 on authentication focuses on transaction-level MFA but does not mandate FIDO2 specifically, reducing urgency for banks to migrate.[^1]
- **Developer signals.** FNB's public API portal references WebAuthn in its OAuth2 documentation, suggesting internal work is underway, but no consumer-facing launch date has been announced.[^2]
- **Global context.** The FIDO Alliance reports 13 billion passkey-enabled accounts globally as of Q1 2025, concentrated in US, EU, and APAC markets.[^1]

---

## Recommended Next Steps

1. Monitor FNB's developer changelog for WebAuthn GA announcement.
2. Track SARB guidance updates — a future authentication directive could accelerate adoption across all banks.
3. Re-verify bank-by-bank status in Q4 2025; the landscape is moving quickly.

---

[^1]: FIDO Alliance, *State of Passkeys 2024 Report* — https://fidoalliance.org/state-of-passkeys-2024/
[^2]: FNB Developer Portal, OAuth2 / WebAuthn reference — https://developer.fnb.co.za/docs/authentication
