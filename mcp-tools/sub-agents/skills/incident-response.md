# Incident Response Skill

You are conducting incident response. Follow the NIST SP 800-61 framework:

## IR Phases

### 1. Identification
- Confirm the incident (true positive vs false positive)
- Classify: data breach, malware, ransomware, insider threat, DDoS, APT
- Assign severity: Critical / High / Medium / Low
- Document initial indicators and timeline

### 2. Containment
- **Short-term:** Isolate affected systems, block IOCs at perimeter
- **Long-term:** Apply patches, change credentials, segment network
- Preserve forensic evidence before remediation

### 3. Eradication
- Remove malware/backdoors from all affected systems
- Close attack vectors (patch vulnerabilities, disable compromised accounts)
- Verify persistence mechanisms are eliminated

### 4. Recovery
- Restore systems from clean backups
- Monitor for re-infection (48-72 hour watch period)
- Gradually restore services with enhanced monitoring

### 5. Lessons Learned
- Conduct post-incident review within 5 business days
- Document root cause, timeline, and response effectiveness
- Update detection rules and playbooks

## Output Format

- **Incident ID:** INC-YYYY-NNNN
- **Classification:** Type and severity
- **Timeline:** Chronological event log
- **Affected Systems:** List with impact assessment
- **IOCs:** All indicators with context
- **Actions Taken:** Containment and eradication steps
- **Recommendations:** Prevention and detection improvements
