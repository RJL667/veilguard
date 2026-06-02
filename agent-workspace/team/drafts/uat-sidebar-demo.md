# MFA Bypass Techniques — Quick Reference

## Techniques

1. **SIM Swap** — Attacker convinces a mobile carrier to transfer the victim's phone number to an attacker-controlled SIM, intercepting SMS OTPs.

2. **Real-Time Phishing (Adversary-in-the-Middle)** — Attacker proxies the victim through a fake login page (e.g., Evilginx2), capturing session cookies and OTP codes as they are entered.

3. **Token Theft / Session Hijacking** — Attacker steals an authenticated session token (via malware, XSS, or credential-store access) and replays it, bypassing MFA entirely because authentication already occurred.

## Mitigations

1. **SIM Swap** — Migrate to phishing-resistant MFA (FIDO2/passkeys or hardware security keys) that is not tied to a phone number.

2. **Real-Time Phishing** — Enforce FIDO2/WebAuthn; origin-bound credentials cannot be replayed across domains, neutralising proxy-based phishing kits.

3. **Token Theft / Session Hijacking** — Bind sessions to device fingerprints or client certificates, enforce short token lifetimes, and deploy endpoint detection to catch credential-store access.
