# SA Bank Passkey / Passwordless Login Support

_Researched: 2024 — Standard Bank, FNB, Absa_

---

## Method

Web searches targeting each bank's official documentation and the MyBroadband article "Kissing passwords goodbye — what South Africa's banks say" (https://mybroadband.co.za/news/banking/521487-kissing-passwords-goodbye-what-south-africas-banks-say.html), supplemented by Entersekt industry commentary on FIDO2/WebAuthn adoption in SA banking.

---

## Findings

**Standard Bank** — No passkey support as of 2024. Authentication relies on passwords + 2FA, with app-based biometrics (Face ID/Touch ID), QR codes, and its DigiME digital identity layer. The bank was in early evaluation of passkeys as of late 2023 but has made no public commitment to a rollout date.

**FNB** — No FIDO2/WebAuthn passkey support for consumer web banking. Current MFA stack includes Smart inContact push notifications and OTPs. Biometrics exist within the mobile app only. Hardware Personal Security Keys (PSK) are available for enterprise clients but are not standard passkeys.

**Absa** — No passkey support. Login to online banking still requires access account number, user number, and password/PIN. AbsaID Facial Biometrics is used for password resets and device linking, not as a primary passwordless login mechanism.

All three banks have acknowledged passkeys as a future direction but none has committed to a production timeline.

---

## Recommendation

None of the top three SA banks currently support passkeys for online banking login. Phishield clients asking about passwordless authentication should be advised that FIDO2/WebAuthn is not yet available on these platforms and that current best practice remains strong unique passwords combined with app-based MFA (push notifications or TOTP). Phishield should monitor MyBroadband and each bank's security blog for rollout announcements, as the technology is under active evaluation industry-wide. When passkeys do launch, client advisory materials should be prepared in advance.
