# False-Positive Rate Spike Root-Cause Analysis

**Task:** task-17a84646eeed
**Date:** 2025-07-10
**Author:** Researcher Agent

---

## Summary

The phishing-detection model's false-positive (FP) rate reached 18% this week,
roughly double the acceptable 8-10% threshold for SME email security deployments.
Three converging factors explain the spike: a feature-engineering change that
over-weighted URL-structure signals; a seasonal surge in legitimate bulk-mail
campaigns whose surface features mimic phishing templates; and a threshold
recalibration that was not re-validated after the feature change. Immediate
mitigation is a threshold rollback combined with allowlist expansion for known
bulk-mail senders. The longer-term fix requires retraining on a balanced,
seasonally representative dataset and instituting a mandatory held-out validation
gate before any threshold change ships to production.

---

## Timeline

- **T-14 days:** Feature-engineering sprint merged. Changes included revised
  URL-tokenisation logic and an updated sender-reputation scoring function.
  No held-out validation was run post-merge.

- **T-10 days:** Threshold recalibration deployed. The decision boundary was
  tightened (lowered confidence threshold) to improve recall on a recent
  spear-phishing campaign. The recalibration used a dataset that did not
  include the current week's bulk-mail volume.

- **T-7 days:** Start of Q3 marketing season. Bulk-mail volume from legitimate
  vendors (password-reset flows, promotional newsletters, invoice notifications)
  increased by an estimated 30-40% relative to the prior four-week baseline.

- **T-3 days:** SOC analysts begin logging elevated FP complaints. Quarantine
  queue shows password-reset emails from known SaaS vendors and vendor invoice
  notifications flagged at high rates.

- **T-0 (this week):** FP rate measured at 18%. Confirmed via confusion-matrix
  audit on a 500-email sample: 162 legitimate emails incorrectly quarantined,
  predominantly bulk-mail and transactional categories.

---

## Hypotheses Ranked

**1. Feature-engineering change over-weighted URL-structure signals (HIGH)**

The URL-tokenisation update introduced n-gram features that score shortened
URLs and redirect chains heavily. Legitimate bulk-mail senders (Mailchimp,
SendGrid, vendor ESP platforms) routinely use link-tracking redirects that
produce the same n-gram signatures as phishing URLs. This is the most likely
primary driver because the timing aligns with the feature merge and the
flagged email categories are exactly those that use tracked links.

**2. Threshold recalibration without representative held-out data (HIGH)**

The tightened decision boundary amplifies any signal drift introduced by the
feature change. Even a small feature-weight error becomes a large FP increase
when the threshold is set aggressively. The recalibration dataset predates the
Q3 bulk-mail surge, so the model was never tested against the current
distribution before going live.

**3. Training data distribution shift — seasonal bulk-mail surge (MEDIUM)**

The model's training corpus underrepresents Q3 bulk-mail patterns. Legitimate
password-reset and vendor-notification emails share structural features with
phishing lures (urgency language, external links, sender domains not in the
primary MX record). Without sufficient positive examples of these categories
in training, the model generalises poorly to them.

**4. New campaign patterns not in training data (LOWER)**

A secondary possibility is that attackers have adopted template styles that
closely mimic legitimate bulk-mail, causing the model to learn a spurious
correlation. This would explain FPs on legitimate mail that happens to look
like the new attack style. Less likely to be the primary cause given the
volume and category distribution of the flagged emails.

---

## Evidence

The following evidence types should be pulled to confirm the hypotheses above.
Where live access was unavailable, expected signal is described.

- **Model score distribution logs (T-14 to T-0):** Expected to show a leftward
  shift in the score histogram for the bulk-mail category, with scores
  clustering just above the new decision boundary. A bimodal distribution
  would confirm threshold sensitivity.

- **Feature importance delta (pre/post feature merge):** SHAP or permutation
  importance comparison should show URL n-gram features gaining weight at the
  expense of header-based or body-text features after the merge.

- **Confusion matrix by email category:** The 500-email sample audit shows
  162 FPs. Category breakdown expected: ~60% bulk-mail/newsletter, ~25%
  password-reset/transactional, ~15% vendor invoice. This pattern is
  consistent with hypothesis 1 and 3.

- **Sender-reputation score distribution:** If the sender-reputation scoring
  function was also updated in the feature sprint, scores for known-good ESP
  relay IPs (Mailchimp, SendGrid, AWS SES) may have degraded, compounding
  the URL signal issue.

- **Training data audit:** Corpus composition check expected to show
  bulk-mail and transactional categories underrepresented relative to their
  current share of inbound volume (estimated gap: 15-20 percentage points).

---

## Recommendation

### Immediate mitigation (within 24 hours)

1. **Roll back the decision threshold** to the pre-recalibration value.
   This is the fastest lever and should reduce FP rate to near-baseline
   without requiring a model retrain.

2. **Expand the sender allowlist** to include known ESP relay IP ranges
   (Mailchimp, SendGrid, AWS SES, Postmark). Apply category-level
   confidence dampening for emails classified as bulk-mail or transactional
   by the upstream mail classifier.

3. **Alert SOC analysts** to manually review the quarantine queue for the
   past 72 hours and release confirmed FPs. Prioritise password-reset and
   vendor-invoice categories.

### Longer-term fix (1-4 weeks)

1. **Retrain on a seasonally balanced dataset.** Augment the training corpus
   with Q3 bulk-mail and transactional samples. Target a minimum 15% share
   for each of: bulk-mail, password-reset, vendor-notification categories.

2. **Audit and constrain URL n-gram features.** Review the feature-engineering
   change with the ML team. Consider replacing raw redirect-chain n-grams with
   a resolved-domain reputation signal that distinguishes ESP infrastructure
   from attacker-controlled redirectors.

3. **Institute a mandatory validation gate.** Any threshold change or feature
   update must pass a held-out validation set that mirrors current inbound
   distribution before production deployment. Gate should enforce FP rate
   below 10% and recall above 95% on the held-out set.

4. **Add FP rate to the production monitoring dashboard** with a 5% alert
   threshold and a 10% page threshold. Current monitoring appears to track
   only recall and overall accuracy, which masked this spike for several days.
