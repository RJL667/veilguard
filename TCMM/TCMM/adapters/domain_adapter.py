"""Domain adapter interface for TCMM.

Pluggable domain logic that makes TCMM domain-agnostic. The core
memory system handles entities, episodes, facts, and patterns. The
domain adapter tells it what counts as an entity, how to classify
outcomes, and what constraints apply in a specific domain.

Usage:
    adapter = InsuranceAdapter()
    dream_engine = DreamEngine(tcmm, domain_adapter=adapter)

The adapter is called at several points during the dream cycle:
  - Entity enrichment: extract_domain_entities()
  - Outcome classification: classify_outcome()
  - Constraint generation: get_constraints()
  - Action scoring: score_action()
"""
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class DomainAdapter(ABC):
    """Base class for domain-specific TCMM adapters.

    Subclass this for each domain (insurance, healthcare, legal, etc.)
    and implement the abstract methods. The core dream engine calls
    these at the appropriate points — no hardcoded domain logic in
    the engine itself.
    """

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Short identifier: 'insurance', 'healthcare', 'legal', etc."""
        ...

    def extract_domain_entities(
        self, text: str, existing_entities: List[Dict]
    ) -> List[Dict]:
        """Enhance entity extraction with domain-specific patterns.

        Called during the NLP enrichment pass. Takes raw text +
        entities already found by NLP, returns additional entities
        that domain knowledge can detect (policy numbers, claim IDs,
        medical codes, etc.).

        Args:
            text: raw user/assistant text
            existing_entities: list of {"name": str, "type": str, ...}

        Returns:
            list of additional entity dicts to merge in
        """
        return []

    def classify_outcome(self, text: str) -> Optional[Dict]:
        """Classify whether text describes an outcome of a prior action.

        Called during arc_state_sheet event classification. Returns
        None if the text is not an outcome, or a dict with:
          {"type": str, "score": float, "detail": str}

        Score: -1.0 (failed) to +1.0 (succeeded).

        Args:
            text: a distilled "you" statement from the LLM

        Returns:
            outcome dict or None
        """
        return None

    def get_constraints(self, entity_key: str = None) -> List[Dict]:
        """Return domain constraints that should be enforced.

        Called during _build_constraint_nodes in the dream cycle.
        Returns a list of constraint dicts:
          {"rule": str, "priority": "hard"|"soft", "scope": str}

        Args:
            entity_key: if provided, return constraints specific to
                        this entity. If None, return global constraints.

        Returns:
            list of constraint dicts
        """
        return []

    def score_action(self, action_text: str, context: Dict) -> float:
        """Score how appropriate an action is in the current context.

        Called during strategy emergence to evaluate whether a
        recommended action was domain-appropriate. Returns a
        multiplier (1.0 = neutral, >1.0 = boost, <1.0 = penalize).

        Args:
            action_text: the recommended action
            context: dict with entity, situation, constraints

        Returns:
            float multiplier
        """
        return 1.0

    def get_entity_patterns(self) -> Dict[str, str]:
        """Return regex patterns for domain-specific entity extraction.

        Keys are entity type names, values are regex patterns.
        Used by extract_domain_entities as the default implementation.

        Returns:
            {"policy_number": r"POL-\d{6}", ...}
        """
        return {}

    def __repr__(self):
        return f"{self.__class__.__name__}(domain={self.domain_name})"
