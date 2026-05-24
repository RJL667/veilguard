# Threat Intelligence Skill

You are producing a structured threat intelligence report. Follow the Diamond Model and MITRE ATT&CK framework.

## Analysis Framework

### Diamond Model Elements
- **Adversary:** Threat actor/group identification, attribution confidence
- **Infrastructure:** C2 servers, domains, hosting providers, bulletproof hosts
- **Capability:** Malware families, tools, exploits, custom vs commodity
- **Victim:** Targeted sectors, geographies, organization profiles

### MITRE ATT&CK Mapping
Map all observed TTPs to MITRE ATT&CK techniques:
- **Initial Access:** T1566 (Phishing), T1190 (Exploit Public-Facing App), etc.
- **Execution:** T1059 (Command Scripting), T1204 (User Execution), etc.
- **Persistence:** T1053 (Scheduled Task), T1547 (Boot/Logon Autostart), etc.
- **Lateral Movement:** T1021 (Remote Services), T1570 (Lateral Tool Transfer)
- **Exfiltration:** T1048 (Exfil Over Alternative Protocol), T1567 (Exfil to Cloud)

### IOC Classification
Categorize all indicators:
- **Network:** IP addresses, domains, URLs, certificates
- **Host:** File hashes (MD5/SHA1/SHA256), file paths, registry keys
- **Email:** Sender addresses, subjects, attachment names
- **Behavioral:** Process trees, command lines, scheduled tasks

## Output Format

- **TLP Level:** WHITE / GREEN / AMBER / RED
- **Confidence:** Low / Medium / High
- **Actor Profile:** Name, aliases, motivation, sophistication
- **Campaign Summary:** Objectives, timeline, scope
- **TTPs:** MITRE ATT&CK technique table
- **IOCs:** Structured indicator list with context
- **Detection Rules:** Sigma/YARA rules if applicable
- **Recommendations:** Defensive actions prioritized by impact
