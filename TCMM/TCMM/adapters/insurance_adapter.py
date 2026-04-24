"""Insurance domain adapter for TCMM.

Knows about: claims, policies, policyholders, underwriting, premiums,
adjusters, deductibles, coverage types, regulatory requirements.

Usage:
    from adapters.insurance_adapter import InsuranceAdapter
    adapter = InsuranceAdapter()
    dream_engine = DreamEngine(tcmm, domain_adapter=adapter)
"""
import re
from typing import Dict, List, Optional
from .domain_adapter import DomainAdapter


class InsuranceAdapter(DomainAdapter):
    """Insurance-specific TCMM adapter.

    Extracts policy numbers, claim IDs, coverage types. Classifies
    claim outcomes (settled, disputed, denied). Enforces approval
    limits and regulatory constraints.
    """

    @property
    def domain_name(self) -> str:
        return "insurance"

    # ── Entity extraction ─────────────────────────────────────────

    _ENTITY_PATTERNS = {
        "policy_number": re.compile(
            r"\b(?:POL|HO|AUTO|LIFE|COM)[-\s]?\d{4,10}\b", re.IGNORECASE
        ),
        "claim_number": re.compile(
            r"\b(?:CLM|CLAIM)[-\s]?\d{4,10}\b", re.IGNORECASE
        ),
        "coverage_type": re.compile(
            r"\b(?:homeowner'?s?|auto|life|health|commercial|liability|"
            r"umbrella|flood|fire|theft|comprehensive|collision|"
            r"uninsured\s+motorist|bodily\s+injury|property\s+damage)\s+"
            r"(?:insurance|policy|coverage|protection)\b",
            re.IGNORECASE,
        ),
        "dollar_amount": re.compile(
            r"\$[\d,]+(?:\.\d{2})?\b"
        ),
        "deductible": re.compile(
            r"\bdeductible\s+(?:of\s+)?\$?[\d,]+", re.IGNORECASE
        ),
        "premium": re.compile(
            r"\bpremium\s+(?:of\s+)?\$?[\d,]+", re.IGNORECASE
        ),
    }

    # Insurance-specific entity names that NLP might miss
    _DOMAIN_ENTITIES = {
        "adjuster", "underwriter", "policyholder", "insured",
        "beneficiary", "claimant", "broker", "agent",
        "actuary", "loss adjuster", "claims handler",
    }

    def extract_domain_entities(
        self, text: str, existing_entities: List[Dict]
    ) -> List[Dict]:
        additional = []
        text_lower = text.lower()
        existing_names = {e.get("name", "").lower() for e in existing_entities}

        # Pattern-based extraction
        for ent_type, pattern in self._ENTITY_PATTERNS.items():
            for match in pattern.finditer(text):
                name = match.group().strip()
                if name.lower() not in existing_names:
                    additional.append({
                        "name": name,
                        "type": ent_type,
                        "score": 0.95,
                        "source": "insurance_adapter",
                    })
                    existing_names.add(name.lower())

        # Domain role entities
        for role in self._DOMAIN_ENTITIES:
            if role in text_lower and role not in existing_names:
                additional.append({
                    "name": role.title(),
                    "type": "insurance_role",
                    "score": 0.85,
                    "source": "insurance_adapter",
                })

        return additional

    # ── Outcome classification ────────────────────────────────────

    _OUTCOME_PATTERNS = {
        # Positive outcomes
        "claim_settled": (
            re.compile(r"\bclaim\s+(?:was\s+)?settled\b", re.IGNORECASE),
            1.0,
        ),
        "claim_approved": (
            re.compile(r"\bclaim\s+(?:was\s+)?approved\b", re.IGNORECASE),
            1.0,
        ),
        "policy_renewed": (
            re.compile(r"\bpolicy\s+(?:was\s+)?renewed\b", re.IGNORECASE),
            0.8,
        ),
        "policy_issued": (
            re.compile(r"\bpolicy\s+(?:was\s+)?issued\b", re.IGNORECASE),
            1.0,
        ),
        # Negative outcomes
        "claim_denied": (
            re.compile(r"\bclaim\s+(?:was\s+)?denied\b", re.IGNORECASE),
            -0.8,
        ),
        "claim_disputed": (
            re.compile(r"\bclaim\s+(?:is\s+)?(?:in\s+)?dispute\b", re.IGNORECASE),
            -0.3,
        ),
        "policy_lapsed": (
            re.compile(r"\bpolicy\s+(?:has\s+)?lapsed\b", re.IGNORECASE),
            -0.7,
        ),
        "policy_cancelled": (
            re.compile(r"\bpolicy\s+(?:was\s+)?cancell?ed\b", re.IGNORECASE),
            -0.9,
        ),
        # Neutral/in-progress
        "under_review": (
            re.compile(r"\bunder\s+review\b|\bbeing\s+processed\b", re.IGNORECASE),
            0.0,
        ),
        "awaiting_documents": (
            re.compile(r"\bawaiting\s+(?:documents|information)\b", re.IGNORECASE),
            0.0,
        ),
    }

    def classify_outcome(self, text: str) -> Optional[Dict]:
        for outcome_type, (pattern, score) in self._OUTCOME_PATTERNS.items():
            if pattern.search(text):
                return {
                    "type": outcome_type,
                    "score": score,
                    "detail": text[:200],
                }
        return None

    # ── Constraints ───────────────────────────────────────────────

    _GLOBAL_CONSTRAINTS = [
        {
            "rule": "Claims over $50,000 require manager approval before settlement",
            "priority": "hard",
            "scope": "global",
        },
        {
            "rule": "All claim communications must include the claim reference number",
            "priority": "hard",
            "scope": "global",
        },
        {
            "rule": "Policy renewals should be initiated at least 30 days before expiry",
            "priority": "soft",
            "scope": "global",
        },
        {
            "rule": "Sensitive policyholder information (ID numbers, medical data) must not be shared externally",
            "priority": "hard",
            "scope": "global",
        },
        {
            "rule": "Prefer settling claims within 30 business days of filing",
            "priority": "soft",
            "scope": "global",
        },
    ]

    def get_constraints(self, entity_key: str = None) -> List[Dict]:
        if entity_key is None:
            return list(self._GLOBAL_CONSTRAINTS)
        # Entity-specific constraints could be loaded from a database
        # or policy document in a production implementation.
        return []

    # ── Action scoring ────────────────────────────────────────────

    def score_action(self, action_text: str, context: Dict) -> float:
        """Score how appropriate an action is in insurance context."""
        text_lower = action_text.lower()
        multiplier = 1.0

        # Boost actions that reference specific claim/policy numbers
        if re.search(r"\b(?:CLM|POL|CLAIM)[-\s]?\d+\b", action_text, re.IGNORECASE):
            multiplier *= 1.2

        # Boost follow-up actions
        if any(w in text_lower for w in ["follow up", "contact", "reach out", "schedule"]):
            multiplier *= 1.1

        # Penalize vague actions
        if any(w in text_lower for w in ["maybe", "possibly", "consider", "might"]):
            multiplier *= 0.8

        return multiplier

    def get_entity_patterns(self) -> Dict[str, str]:
        return {k: v.pattern for k, v in self._ENTITY_PATTERNS.items()}
