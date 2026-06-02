# Credential Stuffing Defenses — Research Note

*Note: Web search was unavailable during research. This note draws on established industry guidance from OWASP and NIST, cited below.*

---

## What Is Credential Stuffing?

Credential stuffing is an automated attack in which adversaries replay large sets of username/password pairs — typically sourced from prior data breaches — against login endpoints. Unlike brute-force attacks, stuffing uses real credentials, so success rates are far higher (often 0.1–2% of tested pairs). At scale, even a 0.1% hit rate against a list of 100 million credentials yields 100,000 compromised accounts.

---

## Core Defenses

### 1. Multi-Factor Authentication (MFA)
MFA is the single most effective control. Even when an attacker has a valid password, a second factor (TOTP, push notification, hardware key) blocks account takeover. NIST SP 800-63B recommends phishing-resistant authenticators (FIDO2/WebAuthn) as the strongest option. [[NIST SP 800-63B](https://pages.nist.gov/800-63-3/sp800-63b.html)]

### 2. Breached Password Detection
Check submitted passwords against known-compromised credential databases at login and registration. The Have I Been Pwned (HIBP) Pwned Passwords API provides over 800 million compromised hashes via a k-anonymity model, so the plaintext password never leaves the client. OWASP explicitly recommends this control in its Credential Stuffing Prevention Cheat Sheet. [[OWASP Credential Stuffing Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Credential_Stuffing_Prevention_Cheat_Sheet.html)]

### 3. Bot Detection and Rate Limiting
- **CAPTCHA / invisible challenge** (e.g., hCaptcha, Google reCAPTCHA v3) at login to distinguish humans from automated clients.
- **IP-based rate limiting**: throttle or block IPs exceeding a threshold of failed attempts per minute.
- **Device fingerprinting**: flag logins from headless browsers or known datacenter IP ranges.
- **Velocity checks**: alert on a single IP attempting many distinct usernames, or a single username attempted from many IPs.

### 4. Anomaly-Based Detection
Monitor login telemetry for statistical anomalies: unusual geographic distribution of failures, sudden spikes in login volume, or high failure-to-success ratios. Feed these signals into a SIEM for automated blocking and analyst review.

### 5. Credential Rotation Notification
When a breach is detected (via threat-intel feeds or HIBP Enterprise), proactively force password resets for affected accounts and notify users. This limits the window of exposure even when stuffing has already begun.

---

## Priority Order for SMEs

| Priority | Control | Effort |
|----------|---------|--------|
| 1 | MFA (TOTP minimum, FIDO2 preferred) | Medium |
| 2 | Breached password check (HIBP API) | Low |
| 3 | Rate limiting + CAPTCHA | Low–Medium |
| 4 | Anomaly detection / SIEM alerting | Medium–High |

---

## References

1. NIST SP 800-63B — *Digital Identity Guidelines: Authentication and Lifecycle Management*. https://pages.nist.gov/800-63-3/sp800-63b.html
2. OWASP — *Credential Stuffing Prevention Cheat Sheet*. https://cheatsheetseries.owasp.org/cheatsheets/Credential_Stuffing_Prevention_Cheat_Sheet.html
