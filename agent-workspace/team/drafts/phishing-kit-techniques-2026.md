# Phishing-Kit Detection Checklist — 3 Common 2026 Techniques

1. **Adversary-in-the-Middle (AiTM) reverse-proxy kits** (e.g., Evilginx3, Modlishka derivatives): relay live sessions through an attacker-controlled proxy to harvest session cookies and bypass MFA, detectable by mismatched TLS certificate issuers, anomalous redirect chains, and session tokens appearing on unexpected ASNs.

2. **HTML smuggling with obfuscated payload assembly**: malicious payloads are encoded as Base64 or JavaScript Blob objects inside benign-looking HTML attachments and assembled client-side to evade gateway scanning, detectable by the presence of `msSaveOrOpenBlob`, `URL.createObjectURL`, or large inline Base64 strings in email-attached HTML files.

3. **Phishing-as-a-Service (PhaaS) kit fingerprints** (e.g., Rockstar2FA, LabHost successors): commodity kits share reused infrastructure patterns — identical URI path structures (`/login`, `/verify`, `/checkpoint`), common anti-bot Cloudflare Turnstile abuse, and kit-specific cookie names or HTTP response headers — detectable via threat-intel feeds and passive DNS correlation against known kit templates.
