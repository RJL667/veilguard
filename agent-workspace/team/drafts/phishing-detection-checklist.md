# Phishing-Kit Detection Checklist

> **Source:** Based on established phishing-kit tradecraft knowledge. Validate against the sibling researcher output when available.

---

## OVERVIEW

Modern phishing kits are packaged, reusable attack bundles that clone legitimate login portals and exfiltrate credentials in real time. Detection requires examining both network-layer indicators and page-level artefacts. The three techniques below represent the most prevalent patterns observed in commodity and targeted kits: reverse-proxy credential interception, adversary-in-the-browser JavaScript injection, and QR-code redirect lures. Each section lists concrete detection indicators that security teams can operationalize in SIEM rules, proxy logs, and endpoint telemetry. Defenders who apply these checks systematically can identify kit infrastructure before credentials are stolen.

---

## TECHNIQUE 1 — Reverse-Proxy Credential Interception

Open-source adversary-in-the-middle frameworks (such as Evilginx2 and comparable reverse-proxy phishing toolkits) sit between the victim and the real identity provider (IdP), relaying live sessions to harvest session cookies alongside passwords. The victim sees a valid TLS certificate for a lookalike domain, making visual inspection unreliable. Because the kit proxies a real site, multi-factor authentication tokens are captured in transit.

**Detection indicators:**
- Domain registered within the last 30 days with a free certificate (Let's Encrypt or ZeroSSL)
- Hostname follows patterns: `login-<brand>.<tld>`, `<brand>-secure.<tld>`, `<brand>-auth.<tld>`
- HTTP response headers include `X-Forwarded-For` or `Via` with unexpected upstream IPs
- Session cookie `SameSite` attribute absent or set to `None` on a third-party domain
- DNS record time-to-live unusually short (under 300 seconds), suggesting rapid infrastructure rotation

---

## TECHNIQUE 2 — Adversary-in-the-Browser JavaScript Injection

Kit operators inject malicious JavaScript into cloned pages to intercept form submissions before the browser sends them. Unlike reverse-proxy kits, the page may be hosted on a compromised legitimate site, making domain-age checks insufficient. The injected script exfiltrates keystrokes or form data to an attacker-controlled endpoint in real time.

**Detection indicators:**
- Inline `<script>` blocks that override `HTMLFormElement.prototype.submit` or attach `keydown`/`input` event listeners to password fields
- Outbound XHR or `fetch()` calls to a different origin than the visible domain immediately after credential entry
- Content Security Policy (CSP) header absent or set to `unsafe-inline`, permitting arbitrary script execution
- Page source contains obfuscated strings decoded at runtime (e.g., `atob()`, `eval()`, `String.fromCharCode()` chains)
- Endpoint telemetry shows browser process making unexpected POST requests to IP-literal URLs

---

## TECHNIQUE 3 — QR-Code Redirect Lures

QR-code phishing (quishing) bypasses email URL scanners by encoding the malicious URL in an image. Victims scan the code with a mobile device that lacks enterprise proxy inspection, landing on a credential-harvesting page. This technique is especially effective against organisations that have trained users to hover over links but not to scrutinise QR codes.

**Detection indicators:**
- Email contains an image attachment or inline image with no accompanying text URL, paired with urgency language ("verify your account", "action required")
- QR code destination resolves to a domain registered within 14 days or hosted on a bulletproof autonomous system
- Mobile device DNS logs show resolution of lookalike domains not seen in desktop traffic
- Redirect chain passes through a URL-shortening or QR-generation service before the final phishing page
- Landing page requests camera or notification permissions atypical for the spoofed brand

---

## QUICK CHECKS

Use these rapid triage steps when a suspicious URL or email is reported:

1. **WHOIS age** — flag any domain under 30 days old impersonating a known brand.
2. **Certificate transparency** — search crt.sh for the domain; unexpected SANs indicate kit reuse.
3. **Page source scan** — grep for `eval(`, `atob(`, `document.cookie` in the raw HTML.
4. **Header audit** — confirm CSP, `X-Frame-Options`, and `Referrer-Policy` match the legitimate site.
5. **QR decode** — paste any QR image into an offline decoder; submit the extracted URL to VirusTotal before visiting.
6. **Proxy log correlation** — search for POST requests to the suspicious domain within 60 seconds of the reported click time.
