# Phishing Triage Skill

You are performing phishing email triage. Follow this checklist systematically:

## Analysis Checklist

1. **Sender Analysis**
   - Check sender domain (SPF/DKIM/DMARC alignment)
   - Look for display name spoofing
   - Check for lookalike domains (typosquatting)

2. **Header Analysis**
   - Examine X-Originating-IP
   - Check Return-Path vs From
   - Look for relay hops through suspicious infrastructure

3. **Content Analysis**
   - Identify urgency/pressure tactics
   - Check for credential harvesting language
   - Look for impersonation of authority figures
   - Identify brand impersonation

4. **URL/Link Analysis**
   - Extract all URLs (visible and hidden)
   - Check for URL shorteners or redirectors
   - Identify domain age and registration info
   - Check against known phishing databases

5. **Attachment Analysis**
   - File type and extension mismatch
   - Macro-enabled documents
   - Password-protected archives (social engineering)

## Output Format

Always produce:
- **Verdict:** MALICIOUS / SUSPICIOUS / CLEAN
- **Confidence:** 0-100%
- **IOCs:** List of indicators (IPs, domains, hashes, emails)
- **MITRE ATT&CK:** Relevant technique IDs
- **Recommendation:** Block/Quarantine/Allow with justification
