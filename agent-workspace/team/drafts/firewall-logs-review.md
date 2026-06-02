# Firewall Logs Review

**Prepared by:** Researcher Agent
**Task ID:** task-dc2020e18ee9
**Date:** 2025-05-28
**Log Period:** 2025-05-21 00:00 UTC – 2025-05-27 23:59 UTC (7 days)
**Log Source:** Phishield perimeter firewall — pf-edge-01 (Cape Town HQ)

---

## SUMMARY

Review of 7-day firewall log export from pf-edge-01 covering 2,847,312 total events. Of these, 412,884 (14.5%) were DENY/DROP actions. Three high-severity anomalies were identified: a sustained SSH brute-force campaign from a single ASN, anomalous outbound beaconing from an internal host, and a spike in ICMP sweep traffic consistent with network reconnaissance. Two medium-severity policy violations were also found — internal hosts bypassing the proxy via direct port-80 egress, and a stale ALLOW rule permitting RDP from a decommissioned IP range. Immediate remediation is required for the outbound beaconing event; the remaining findings require rule updates within 5 business days.

---

## KEY_FINDINGS

1. **SSH brute-force campaign (Critical):** Between 2025-05-22 03:14 UTC and 2025-05-22 09:47 UTC, source IP `185.220.101.47` (AS205100, F3 Netze e.V. — a known Tor exit node operator) generated 38,412 DENY events against `203.0.113.15:22` (external-facing jump host). Peak rate reached 112 attempts/minute at 04:30 UTC. The IP was not in the existing blocklist. A second source, `185.220.101.63` from the same ASN, contributed a further 9,204 attempts from 06:00 UTC onward, suggesting coordinated tooling.

2. **Anomalous outbound beaconing (Critical):** Internal host `10.10.4.88` (asset tag WS-JHB-047, Johannesburg branch) initiated 1,440 outbound TCP connections to `91.195.240.117:4444` over a 24-hour window on 2025-05-24, at near-exact 60-second intervals. The destination IP resolves to a bulletproof hosting provider (Serverius AS50673, Netherlands). Beacon regularity and non-standard port are consistent with Cobalt Strike or Metasploit listener traffic. Host was not on the approved egress allowlist.

3. **ICMP sweep / reconnaissance (High):** On 2025-05-25 between 11:02–11:19 UTC, source `45.142.212.100` (AS208091) sent ICMP echo-request packets to 254 sequential addresses in the `203.0.113.0/24` range. 17 minutes elapsed; 254 unique destination IPs probed. This is a textbook /24 ping sweep. The firewall dropped all packets per the default-deny inbound ICMP rule, but the source IP was not auto-blocked post-sweep.

---

## ANOMALIES

| # | Timestamp (UTC) | Source IP | Destination | Protocol/Port | Event Count | Severity |
|---|-----------------|-----------|-------------|---------------|-------------|----------|
| 1 | 2025-05-22 03:14–09:47 | 185.220.101.47 | 203.0.113.15:22 | TCP/22 | 38,412 DENY | Critical |
| 2 | 2025-05-22 06:00–09:47 | 185.220.101.63 | 203.0.113.15:22 | TCP/22 | 9,204 DENY | Critical |
| 3 | 2025-05-24 00:01–23:59 | 10.10.4.88 (int.) | 91.195.240.117:4444 | TCP/4444 | 1,440 ALLOW | Critical |
| 4 | 2025-05-25 11:02–11:19 | 45.142.212.100 | 203.0.113.0/24 | ICMP | 254 DROP | High |
| 5 | 2025-05-23 (all day) | 10.10.2.x range | Various:80 | TCP/80 direct | 3,871 ALLOW | Medium |
| 6 | Ongoing | Any | 10.0.0.0/8:3389 | TCP/3389 | 214 ALLOW | Medium |

**Anomaly 5 detail:** Fourteen internal hosts in the `10.10.2.0/24` subnet made direct outbound TCP/80 connections, bypassing the mandatory Squid proxy at `10.10.0.5:3128`. This violates the egress policy requiring all HTTP/HTTPS traffic to traverse the proxy for content inspection and logging.

**Anomaly 6 detail:** A legacy ALLOW rule (`rule-id: FW-0047`) permits RDP (TCP/3389) inbound from `10.0.50.0/24` — a subnet decommissioned in Q4 2024. The rule has not been removed and represents an unnecessary attack surface if that range is ever reassigned.

---

## RECOMMENDATIONS

1. **Immediate — Isolate WS-JHB-047:** Host `10.10.4.88` must be taken offline for forensic imaging. The 60-second beacon cadence to a bulletproof host on port 4444 is high-confidence C2 activity. Initiate IR playbook; check for lateral movement from this host across the Johannesburg VLAN.

2. **Immediate — Block AS205100 at border:** Add `185.220.101.0/24` and the full AS205100 prefix list to the inbound blocklist. Enable automatic ASN-level blocking for Tor exit node ranges using the Spamhaus DROP list feed, which is not currently subscribed.

3. **Within 24 hours — Add post-sweep auto-block rule:** Configure the firewall to auto-block any source IP that generates ICMP echo-requests to more than 10 unique destinations within a 60-second window. Apply a 72-hour block duration with alert to SOC.

4. **Within 5 days — Enforce proxy egress for TCP/80 and TCP/443:** Add an explicit DENY rule for direct outbound TCP/80 and TCP/443 from all internal subnets except the proxy host itself. Audit the 14 hosts in `10.10.2.0/24` for policy acknowledgement.

5. **Within 5 days — Remove stale RDP rule FW-0047:** Delete or disable `rule-id: FW-0047`. Conduct a full audit of all ALLOW rules referencing decommissioned subnets (`10.0.50.0/24`, `10.0.51.0/24`) and remove any that lack a current business justification.
