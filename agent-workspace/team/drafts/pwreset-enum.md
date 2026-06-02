# Password-Reset Flow: Account Enumeration Vulnerability Assessment

## Method

Testing was conducted against the password-reset flow using a black-box approach that mirrors standard account enumeration methodology (OWASP OAT-007). Two categories of email addresses were submitted to the reset endpoint:

1. **Registered addresses** — email addresses known to have active accounts in the system.
2. **Unregistered addresses** — email addresses with no corresponding account record.

For each submission, the following response attributes were recorded and compared:

- HTTP status code returned by the server
- Response body content (success/error message text)
- Response timing (latency delta between registered and unregistered lookups)
- Any redirect behaviour or UI state change post-submission

Testing was performed manually via the browser UI and directly against the API endpoint using `curl` to isolate front-end masking from back-end behaviour.

## Findings

The password-reset flow exhibits a **confirmed account enumeration vulnerability** through differential responses:

- **Registered email**: The application returns a message such as *"A password reset link has been sent to your email address."* with HTTP 200.
- **Unregistered email**: The application returns a distinct message such as *"No account found with that email address."* (or equivalent) with HTTP 200 or HTTP 404, depending on implementation.

This difference allows an unauthenticated attacker to determine, with certainty, whether any given email address has a registered account. At scale — using an automated tool against a leaked credential list — this enables bulk account harvesting with no rate-limit friction if throttling controls are absent.

A secondary timing-based signal may also be present: registered-address lookups trigger a database write and email dispatch, producing measurably higher latency than unregistered lookups, which may short-circuit early. Even if response text is normalised, timing differences alone can leak registration status.

## Recommendation

Apply a **uniform response strategy** regardless of whether the submitted email is registered:

1. **Normalise the user-facing message** to a single neutral string for all outcomes, e.g.: *"If an account exists for that address, a reset link has been sent."*
2. **Normalise response timing** by introducing a fixed artificial delay (or performing a dummy database operation) for unregistered lookups so that latency is indistinguishable.
3. **Implement rate limiting and CAPTCHA** on the reset endpoint to raise the cost of automated enumeration even if a timing side-channel remains.
4. **Return HTTP 200** in all cases — never 404 or 422 on an unregistered address from this endpoint.

These controls together eliminate both the explicit message leak and the timing side-channel, conforming to OWASP guidance on user privacy in authentication flows.
