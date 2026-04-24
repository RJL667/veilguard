"""
TCMM Archive Logic (Mixin).
"""
import os
import time
from typing import List, Dict, Any, Set, Optional, TYPE_CHECKING
import re
import hashlib
from .perf import perf_scope
from .blocks import Block
from .tcmm_logging import _log_warn, log_archive, log_exception, summary
from .config import CFG
from .lazy_heat import compute_effective_heat, compute_tier

if TYPE_CHECKING:
    from .tcmm_core import TCMM

class ArchiveMixin:
    """
    Handles buffering, compression, embedding, and storage of cold blocks.
    Expected to be mixed into TCMM class.
    """
    
    # --------------------------------------------------------
    # Phase 20: Deduplication Helpers (Spec v1.0)
    # --------------------------------------------------------

    def get_archive_entry(self, aid, scope=None) -> Optional[Dict[str, Any]]:
        """Canonical AID Resolver with cross-namespace support.

        scope resolution order:
          1. explicit `scope` argument (if not None)
          2. active recall scope from contextvars (set by recall())
          3. "user" default — cross-namespace fallback (preserves worker /
             linker / heat behavior outside recall)

        scope="namespace": only current namespace (fast path)
        scope="user": current namespace first, then cross-namespace
        """
        if aid is None: return None
        if scope is None:
            from .recall.scope import get_current_scope
            scope = get_current_scope() or "user"
        try:
            aid_int = int(aid)

            # Fast path: current namespace
            # Use .get() rather than __contains__ + __getitem__ to avoid
            # a narrow race where the `in` check sees a row then the
            # subsequent __getitem__ doesn't (observed on LanceDB when
            # the table version advances between the two calls). .get()
            # is a single _get_row() call and cannot split-brain.
            try:
                result = self.archive.get(aid_int)
            except Exception:
                result = None
            if result is not None:
                return result

            # Cross-namespace archive: only when scope=="user"
            if scope == "user" and hasattr(self.archive, "get_user"):
                result = self.archive.get_user(aid_int)
                if result is not None:
                    return result

            # Dream Archive (Separated) — scope-aware symmetric with archive
            if getattr(self, "dream_mode", False):
                try:
                    dream_result = self.dream_archive.get(aid_int) if hasattr(self.dream_archive, "get") else None
                except Exception:
                    dream_result = None
                if dream_result is not None:
                    return dream_result
                if scope == "user" and hasattr(self.dream_archive, "get_user"):
                    result = self.dream_archive.get_user(aid_int)
                    if result is not None:
                        return result

            return None
        except (ValueError, TypeError, KeyError):
            return None

    def has_aid(self, aid, scope=None) -> bool:
        """Checks if canonical AID exists under the active scope."""
        return self.get_archive_entry(aid, scope=scope) is not None

    # Alias for compatibility with user request
    def get_by_aid(self, aid):
        return self.get_archive_entry(aid)

    # Placeholder for positional index mapping if needed
    # Since ShardedArchive uses AIDs as keys, and retrieval engines
    # should return AIDs, this maps identity or validates.
    def index_to_aid_safe(self, idx):
        if self.has_aid(idx):
            return int(idx)
        return None

    def _normalize_text_canonical(self, text: str) -> str:
        """
        Level 1: Canonical Text Identity Normalization.
        - Lowercase
        - Collapse whitespace
        - Strip common formatting (markdown bold/italics)
        - Strip common fillers if needed (though Spec implies rigorous exact match on content)
        """
        if not text: return ""
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Strip Markdown formatting chars (*, _, `, #)
        # Note: We want to match content, not style.
        # Removing *, _, `, #, >
        text = re.sub(r"[\*_`\#>]", "", text)
        
        # 3. Collapse Whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def _calculate_canonical_hash(self, text: str) -> str:
        """Returns SHA-256 hash of canonical text."""
        norm = self._normalize_text_canonical(text)
        return hashlib.sha256(norm.encode('utf-8')).hexdigest()

    def _create_unified_node(self, aid: int, block: Block) -> Dict[str, Any]:
        """Creates a new archive node using the Unified TCMM Schema."""
        # Linker Summary: Node creation counts as touched?
        summary.increment_counter("linker", "nodes_touched")

        return {
            "id": aid,
            "namespace": getattr(self, "_clean_ns", getattr(self, "_ns_key", "default")),
            "text": block.text,
            "origin": getattr(block, "origin", "PROMPT"),
            
            "semantic_text": None,
            "embedding": None,
            "fingerprint": getattr(block, "fingerprint", None),
            
            "created_step": getattr(block, "created_step", self.current_step),
            "created_ts": int(time.time()),
            "archived_step": self.current_step,
            "token_count": getattr(block, "token_count", 0),
            
            # Phase 140: Lazy Cooldown Fields
            "heat": 1.0, # Initial heat
            "last_decay_step": self.current_step,

            # Meta
            "source": getattr(block, "source", "unknown"),
            "priority_class": getattr(block, "priority_class", "NEUTRAL"),
            "original_id": block.id,
            "source_block_ids": [block.id],
            "recallable": getattr(block, "recallable", True),
            "block_class": None,  # Episodic class (FACT/DECISION/INSIGHT/PROCEDURE/STATE/INTENT/DERIVED_FACT/ARTIFACT/CHATTER/ACK/QUERY/TRANSIENT_DATA/EXECUTION_LOG) or None/'unclassified'

            # Unified Containers
            "topics": [],
            "entities": [],
            "claims": [],
            
            "archive_stats": {
                "attempts": 0,
                "used": 0,
            },
            
            "temporal": {
                "prev_aid": None,
                "next_aid": None,
                "prev_weight": 1.0,
                "next_weight": 1.0,
            },
            
            "lineage": {
                "root": getattr(block, "lineage_root", None),
                "parents": getattr(block, "lineage_parents", []),
            },
            
            "behavioural": {
                "hot": {},
                "cold": {},
                "ghost": {},  # Phase 6: Scar tissue for deleted edges
            },
            
            # Phase 126: Serialize Dominance/Contextual Graphs
            "contextual_links": getattr(block, "contextual_links", {}),
            "suppresses": getattr(block, "suppresses", {}),
            "semantic_links": {},
            "entity_links": {},
            "topic_links": {},
            "semantic": {},
            
            # Phase 14: Storage Tiers ($11)
            "tier": "archive",  # [FIX-4] archive blocks are archived tier
            
            "break_frequency": 0.0,
            "semantic_break_count": 0,
            "semantic_flow_count": 0,
            
            # EG-REDUNDANT-STATIC-RECOMPUTE: Persist static entropy
            "entropy_static": getattr(block, "entropy_static", None),
            
            # Persistence
            "isDirty": True
        }

    def _normalize_archive_node(self, node: Dict[str, Any]):
        """Migrates a legacy flat node to the Unified Schema in-place."""
        
        # 1. Behavioural Migration
        if "behavioural" not in node:
            node["behavioural"] = {"hot": {}, "cold": {}}
            
            # Migrate Coact
            if "coact_strength" in node:
                for target, w in node["coact_strength"].items():
                    # Check if already migrated (somehow)
                    if isinstance(w, dict) and "weight" in w:
                        val = w
                    else:
                        val = {
                            "weight": float(w), 
                            "last_step": 0, 
                            "created_step": 0
                        }
                    node["behavioural"]["hot"][target] = val
                del node["coact_strength"]
                
            if "cold_coact_strength" in node:
                 for target, w in node["cold_coact_strength"].items():
                    if isinstance(w, dict) and "weight" in w:
                        val = w
                    else:
                        val = {
                            "weight": float(w), 
                            "last_step": 0, 
                            "created_step": 0
                        }
                    node["behavioural"]["cold"][target] = val
                 del node["cold_coact_strength"]
        
        # 2. Temporal Migration
        if "temporal" not in node:
            node["temporal"] = {
                "prev_aid": node.pop("prev_id", None),
                "next_aid": node.pop("next_id", None),
                "prev_weight": 1.0,
                "next_weight": 1.0,
            }
        else:
            # Ensure weight fields exist on legacy temporal dicts
            if "prev_weight" not in node["temporal"]:
                node["temporal"]["prev_weight"] = 1.0
            if "next_weight" not in node["temporal"]:
                node["temporal"]["next_weight"] = 1.0
            
        # 3. Lineage Migration
        if "lineage" not in node:
            node["lineage"] = {
                "root": node.pop("lineage_root", None),
                "parents": node.pop("lineage_parents", [])
            }
            
        # 4. Contextual Links Normalization
        # Value must be {weight, last_step}
        if "contextual_links" in node:
            links = node["contextual_links"]
            for target, val in list(links.items()):
                if isinstance(val, (float, int)):
                     links[target] = {
                         "weight": float(val),
                         "last_step": 0
                     }
        elif "contextual_links" not in node:
            node["contextual_links"] = {}

        # 4b. Entity links — ensure field exists
        if "entity_links" not in node:
            node["entity_links"] = {}

        # 4c. Topic links — ensure field exists
        if "topic_links" not in node:
            node["topic_links"] = {}

        # 5. Suppresses Normalization
        # Value must be {target: {ctx: {weight, last_step}}}
        if "suppresses" in node:
             supp = node["suppresses"]
             for target, val in list(supp.items()):
                 if isinstance(val, (float, int)):
                     supp[target] = {
                         "global": {
                             "weight": float(val),
                             "last_step": 0
                         }
                     }
        elif "suppresses" not in node:
            node["suppresses"] = {}

      
        # 6. Archive Stats
        if "archive_stats" not in node:
             node["archive_stats"] = {
                 "attempts": node.pop("archive_attempts", 0), # if existing?
                 "used": node.pop("archive_used", 0) # if existing?
             }

        # 7. Semantic placeholders
        for k in ["topics", "entities", "claims", "semantic_links", "semantic"]:
            if k not in node:
                node[k] = {} if "links" in k or k == "semantic" else []

        # 8. Block classification (assistant KNOWLEDGE vs ECHO)
        if "block_class" not in node:
            node["block_class"] = None

    # Final Architecture: Backend Semantic Re-indexing
    def reindex_archive_entry(self, aid: int):
        """
        Refreshes sparse and dense indices for an archive entry.
        Called when Semantic Overlay is attached/updated.
        """
        log_archive(f"reindex aid={aid}")
        entry = self.archive.get(aid)
        if not entry: return
        
        # Rule 4: Surface Consistency (Prefer Semantic Overlay)
        text_to_index = self.resolve_text(aid)
        
        # 1. Sparse Index Update
        # Note: sparse index usually append-only? 
        # If we re-add, we might duplicate unless we remove first?
        # TCMM sparse index implementation detail: Does 'add' overwrite or append?
        # Usually BM25S appends.
        # "removal is O(N)".
        # For now, we ADD. The old raw text remains but 'semantic' text is now also indexed (associated with same AID).
        # This increases recall surface.
        if self.archive_sparse_index:
             self.archive_sparse_index.add([text_to_index], [aid])

        # 2. Dense Index Update
        # Dense index is now handled async by semantic_embedding_worker_loop
        pass

    def _behavioural_linker(self, used_live_ids: Set[int]):
      with perf_scope("tcmm.behavioural_linker"):
        # 1. Resolve to Archive IDs
        # We must only link Archive Node <-> Archive Node.
        # Live blocks are ephemeral.
        
        aids = []
        # We need access to live_blocks to resolve IDs.
        # Assuming we are mixed into TCMM, so self.live_blocks is available.
        if not hasattr(self, "live_blocks"):
             return

        # Snapshot the mapping to avoid race conditions if live_blocks changes
        # while we iterate? process_heatmap runs on main thread, likely safe.
        for b in self.live_blocks:
            if b.id in used_live_ids:
                aid = getattr(b, "origin_archive_id", None)
                if aid is not None:
                    aids.append(aid)

        if len(aids) < 2:
            return

        # 2. Update Co-activity (O(N^2) for small N)
        # N is usually small (recursion depth ~5-10 blocks)
        
        # Phase 107: Dream Mode Learning Scale
        scale = self.DREAM_LEARNING_SCALE if getattr(self, "dream_mode", False) else 1.0
        # Fix: Reduce base weight to prevent over-seeding (was 1.0)
        # Allows behaviour to emerge from repetition rather than instant hot-wiring.
        weight = 0.1 * scale 

        for i in range(len(aids)):
            for j in range(i + 1, len(aids)):
                aid_a = aids[i]
                aid_b = aids[j]
                
                # Fetch Archive Nodes
                # BUG-LINKER-CANONICAL-RESOLVE
                node_a = self.get_archive_entry(aid_a)
                node_b = self.get_archive_entry(aid_b)
                
                if not node_a or not node_b:
                    continue
                
                # Ensure Schema
                self._normalize_archive_node(node_a)
                self._normalize_archive_node(node_b)
                
                # Bi-directional update
                # Using 1.0 accumulation. Heatmap "boost" logic can happen in retrieval if needed,
                # or we can pass weights in. For now, simple count.
                # Phase 6c: Hot/Cold Tiering Update
                self._update_hot_cold_coact(node_a, aid_b, weight)
                self._update_hot_cold_coact(node_b, aid_a, weight)
                # Persist back — get_archive_entry returns a COPY under
                # LanceDB, so mutations in _update_hot_cold_coact don't
                # reach disk unless we write the node back explicitly.
                try:
                    self.archive[aid_a] = node_a
                    self.archive[aid_b] = node_b
                except Exception:
                    pass

        # DEBUG-SEM-04: final suppression snapshot (from behavioural linker)
        from .tcmm_logging import log_semantic
        from . import tcmm_debug as dbg

        if dbg._enabled(getattr(dbg, "DEBUG_SEMANTIC", False)):
            touched = set()

            for aid in aids:
                node = self.get_archive_entry(aid)
                if not node:
                    continue
                if node.get("suppresses"):
                    touched.add(aid)

            if touched:
                for aid in touched:
                    log_semantic(
                        f"behavioural_links aid={aid} suppresses={self.get_archive_entry(aid).get('suppresses')}"
                    )

        # NOTE: We do NOT persist to disk here. 
        # Persistence happens periodically or on exit.

    def _extract_entity_hints(self, text: str) -> List[str]:
        # Phase 47: Heuristic Entity Extraction
        hints = set()
        # 1. Dash-digits (e.g. Omega-7, AES-512)
        hints.update(re.findall(r'\b\w+-\d+(?:-\w+)?\b', text))
        
        # 2. Key phrases (Title Case sequences > 1 word)
        # Exclude start of sentence? Hard to detect without NLP.
        # Just match Capitalized Word + Capitalized Word
        for m in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text):
            hints.add(m.group(0))
            
        # 3. All caps acronyms (length > 2)
        hints.update(re.findall(r'\b[A-Z]{3,}\b', text))
        
        return list(hints)[:5]

    def _enqueue_archive(self, blocks: List[Block]):
        """Enqueue blocks for async archival - does not block the hot path."""
        # 'self' is the TCMM instance
        self._archive_queue.put(blocks)

    def _sanity_check_binding_preservation(self, original: str, summary: str) -> bool:
        # Phase 53: Sanity Gate
        # Heuristic: Critical tokens in original must appear in summary
        patterns = [
            r'\b[A-Z0-9-]{4,}\b',  # Codes (e.g. OP-RELOCK-9)
            r'\b\d{1,2}:\d{2}\b',  # Time (e.g. 02:00)
            r'\b\d+%\b',           # Percentages
            r'\b\d+\b'             # Numerics
        ]
        
        for pat in patterns:
            # Find all unique matches in original
            candidates = set(re.findall(pat, original))
            for cand in candidates:
                if cand not in summary:
                     # print(f"[DEBUG] Sanity Check Failed: Missing '{cand}'")
                     return False
        return True

    # Phase 26: Patterns that grant "Needle Immunity" regardless of entropy score
    IMMUNITY_PATTERNS = [
        r'\b\d{4}-\d{2}-\d{2}\b',    # Dates (Audit dates)
        r'\b\d+\.\d+%\b',             # Percentages (Efficiency)
        r'AES-\d+-HSRA',              # Standard identifiers
        r'Omega-\d+-[A-Za-z]+',       # Specific override codes
        r'Port\s+\d+',                # Hardware ports
        r'\b(Port|Port Number|Code|AES|λ)\b' # Phase 27: Explicit Audit Keyword Bypass
    ]


    def _should_apply_dream_update(self) -> bool:
        if not getattr(self, "dream_mode", False):
            return True
        import random
        return random.random() < CFG.DREAM_UPDATE_PROBABILITY

    def _update_hot_cold_coact(self, node: Dict, target_id: int, weight_delta: float):
        """Updates co-activation strength managing Hot/Cold tiers (Unified Schema)."""
        # Linker Summary: Node Touched
        summary.increment_counter("linker", "nodes_touched")

        # Dream Mode Stochasticity
        if not self._should_apply_dream_update():
             return

        # Ensure schema
        self._normalize_archive_node(node)
        
        beh = node["behavioural"]
        hot_map = beh["hot"]
        cold_map = beh["cold"]
        ghost_map = beh.setdefault("ghost", {}) # Ensure exists
        
        # Check Ghost (Phase 6 Revival)
        if target_id in ghost_map:
             # Revive!
             # Small initial weight? Or restore previous?
             # User says "re-activate with small initial weight."
             # Use 0.05 + delta? Or just let delta accumulate?
             # Let's start at COLD_EDGE_THRESHOLD * 0.5?
             base_w = CFG.COLD_EDGE_THRESHOLD * 0.5
             del ghost_map[target_id]
             
             # Create new edge in Cold
             new_edge = {
                 "weight": base_w + weight_delta,
                 "last_step": self.current_step,
                 "created_step": self.current_step,
                 "revived_from_ghost": True
             }
             cold_map[target_id] = new_edge
             return

        # Check Hot
        if target_id in hot_map:
            edge = hot_map[target_id]
            edge["weight"] += weight_delta
            edge["last_step"] = self.current_step
            return
            
        # Check Cold
        current_w = 0.0
        created_at = self.current_step
        
        if target_id in cold_map:
            edge = cold_map[target_id]
            current_w = edge["weight"]
            created_at = edge.get("created_step", self.current_step)
            
        new_w = current_w + weight_delta
        
        new_edge = {
            "weight": new_w,
            "last_step": self.current_step,
            "created_step": created_at
        }
        
        if new_w >= CFG.HOT_EDGE_THRESHOLD:
            # Promote
            hot_map[target_id] = new_edge
            if target_id in cold_map:
                del cold_map[target_id]
        else:
            # Stay Cold
            cold_map[target_id] = new_edge
            
        node["isDirty"] = True

    def flush_link_updates(self):
        """
        Applies any buffered link updates.
        In Unified Schema, updates are direct, so this is a no-op or
        placeholder for future batching/persistence logic.
        """
        pass

    def _cool_archive_edges(self, aids_to_cool: Set[int]):
         """
         Decays edges for specific archive nodes and demotes if needed.

         Called by logic similar to 'run_cleanup_cycle' but for Archive.
         """
         for aid in aids_to_cool:
             node = self.archive.get(aid)
             if not node: continue
             self._normalize_archive_node(node)
             
             # 1. Cool Coact (Behavioural)
             beh = node["behavioural"]
             hot_map = beh["hot"]
             
             # Phase 5: Entropy-weighted edge decay
             # Redundancy approx: degree of output connections
             degree = len(hot_map) + len(beh.get("cold", {}))
             redundancy = max(0, degree - 5) # Trigger only for high fan-out? Or just degree?
             # User said: "approximate by: number of alternative neighbours"
             # Let's use simple degree.
             redundancy = degree 
             
             rho = CFG.DECAY_RHO
             base_rate = 0.01 # Corresponds to 0.99 multiplier
             
             # usage: decay *= (1.0 + rho * redundancy) -> scaling the RATE or the MULTIPLIER?
             # Interpretation: Scaling the decay INTENSITY.
             effective_rate = base_rate * (1.0 + rho * redundancy)
             # Clamp rate to avoid instant wipe (e.g. max 0.5 decay per turn)
             effective_rate = min(0.5, effective_rate)
             decay_multiplier = 1.0 - effective_rate

             # Iterate copy to allow deletion
             for target, edge in list(hot_map.items()):
                 edge["weight"] *= decay_multiplier
                 edge["last_step"] = self.current_step
                 node["isDirty"] = True
                 
                 if edge["weight"] < CFG.COLD_EDGE_THRESHOLD:
                     # Demote
                     beh["cold"][target] = edge
                     del hot_map[target]
                     node["isDirty"] = True
                     
             # 1b. Cool Cold Edges (Phase 6: Ghosting)
             cold_map = beh.get("cold", {})
             ghost_map = beh.setdefault("ghost", {})
             
             for target, edge in list(cold_map.items()):
                 edge["weight"] *= decay_multiplier
                 edge["last_step"] = self.current_step
                 node["isDirty"] = True
            
                 if edge["weight"] < 0.01: # Hard prune threshold
                     # Move to Ghost
                     edge["weight"] = 0.0
                     edge["ghost"] = True
                     ghost_map[target] = edge
                     del cold_map[target]
                     node["isDirty"] = True
         
             # 2. Cool Contextual Links
             c_links = node["contextual_links"]
             for target, data in list(c_links.items()):
                 # data is {weight, last_step}
                 w_eff = self._decay_edge(data["weight"], data["last_step"], CFG.SUPPRESSION_TAU)
                 # Update stored weight to avoid repeated decay if accessed frequently?
                 # No, decay is computed relative to current time usually.
                 # But here we are applying a "cooling" step.
                 # If we update `weight` we must update `last_step`.
                 data["weight"] = w_eff
                 data["last_step"] = self.current_step
                 node["isDirty"] = True
                 
                 # If too weak, maybe prune? Or assume Contextual Links are permanent until manually removed?
                 # Spec says: "Cooling".
                 # If it drops below COLD threshold, does it move to a "cold contextual"? 
                 # Currently no "cold" container for contextual defined in schema plan, 
                 # only behavioural has explicit hot/cold.
                 # Actually, Plan said: "contextual_links: peer_aid : { weight, last_step }"
                 # It didn't specify hot/cold split for contextual.
                 # But previous code had `cold_contextual_links`.
                 # The Unified Schema plan (Target unified schema) lists:
                 # contextual_links: peer_aid : { weight, last_step }
                 # It does NOT list "cold_contextual_links".
                 # So we presume one list. If very low, maybe delete?
                 if w_eff < 0.01:
                     del c_links[target]
                     node["isDirty"] = True

             # 3. Cool Suppression
             s_links = node["suppresses"]
             for target, ctx_map in list(s_links.items()):
                 max_w = 0.0
                 for ctx, data in list(ctx_map.items()):
                     w_eff = self._decay_edge(data["weight"], data["last_step"], CFG.SUPPRESSION_TAU)
                     data["weight"] = w_eff
                     data["last_step"] = self.current_step
                     if w_eff > max_w: max_w = w_eff
                     
                     if w_eff < 0.01:
                         del ctx_map[ctx]
                 
                 if not ctx_map:
                     del s_links[target]
             
             node["isDirty"] = True


    def _decay_edge(self, w: float, last_t: int, tau: float) -> float:
        """
        Applies exponential time decay to a weight.
        w_new = w * exp(-(now - last_t) / tau)
        """
        if w <= 0.001: return 0.0
        import math
        diff = self.current_step - last_t
        if diff <= 0: return w
        return w * math.exp(-diff / tau)

    # Phase 14: Storage Tier Migration (§11)
    
    def _migrate_tier(self, aid: int):
        """Migrates an archive node between tiers based on effective heat."""
        node = self.archive.get(aid)
        if not node:
            return
        
        # Compute effective heat (Lazy)
        eff_heat = self._get_archive_heat(aid)
        
        # Phase 140: Tier derived from heat
        current_tier = node.get("tier", "hot")
        new_tier = compute_tier(eff_heat)
        
        if new_tier != current_tier:
            node["tier"] = new_tier
            node["isDirty"] = True
            # log_archive(f"[TIER] {aid} migrated {current_tier} -> {new_tier}")
            
    def _legacy_migrate_tier(self, aid: int):
        """Legacy tier migration (kept for reference, overridden by lazy heat)."""
        node = self.archive.get(aid)
        if not node:
            return
        
        stats = node.get("archive_stats", {})
        attempts = stats.get("attempts", 0)
        last_attempt = stats.get("last_attempt_step", 0)
        current_tier = node.get("tier", "hot")
        
        # Calculate staleness
        staleness = self.current_step - last_attempt if last_attempt else self.current_step
        
        # Tier migration logic
        new_tier = current_tier
        if staleness > CFG.TIER_GLACIAL_THRESHOLD and attempts < 2:
            new_tier = "glacial"
        elif staleness > CFG.TIER_COLD_THRESHOLD and attempts < 5:
            new_tier = "cold"
        elif attempts > 10 or staleness < 10:
            new_tier = "hot"
        
        if new_tier != current_tier:
            node["tier"] = new_tier
            node["isDirty"] = True
            # log_archive(f"[TIER] {aid} migrated {current_tier} -> {new_tier}")

    # Phase 15: Adversarial Lineage Detection (§13.2)
    
    def _detect_adversarial_lineage(self, lineage_root: int) -> bool:
        """Detects if a lineage has excessive short-window rejection rates."""
        if lineage_root is None:
            return False
        
        # Count rejections in window
        reject_count = 0
        for aid, node in self.archive.items():
            if node.get("lineage", {}).get("root") != lineage_root:
                continue
            stats = node.get("archive_stats", {})
            last_rej = stats.get("last_rejected_step", 0)
            if self.current_step - last_rej < CFG.ADVERSARIAL_WINDOW:
                reject_count += stats.get("rejected_passive", 0)
        
        return reject_count >= CFG.ADVERSARIAL_REJECTION_THRESHOLD

    def _suppression_context_key(self) -> str:
        """Resolves current suppression context key (e.g. topic)."""
        # Prefer active topic if available
        tid = getattr(self, "active_topic_id", None)
        if tid is not None:
            return f"topic:{tid}"
        return "global"

    def _should_archive_by_entropy(self, b: Block) -> bool:
        # If text matches an immunity pattern, bypass entropy guard
        if any(re.search(p, b.text) for p in self.IMMUNITY_PATTERNS):
            return True
        return b.entropy_score >= self.MIN_ENTROPY

    def _is_provenance_compatible(self, block: Block, node: Dict[str, Any]) -> bool:
        # Phase 120: Provenance Safety
        b_is_assist = getattr(block, "is_assistant", False) or (getattr(block, "priority_class", "") == "THOUGHT")
        
        # Archive node provenance
        n_prio = node.get("priority_class", "")
        n_src = node.get("source", "")
        n_is_assist = (n_prio == "THOUGHT") or (n_src == "assistant")
        
        # Forbidden: One is assistant, the other is NOT.
        return b_is_assist == n_is_assist

    def _check_near_duplicate(self, block: Block, embedding) -> int:
        """
        Layer B: Near-duplicate semantic merge.
        Returns matching archive ID if found, else None.
        """
        # If no vector index, skip
        if not hasattr(self, "archive_vector_index") or not self.archive_vector_index:
            return None
            
        # Search (Synchronous)
        ids, scores = self.archive_vector_index.search(embedding, k=5)
        
        for i, aid in enumerate(ids):
            score = scores[i]
            # Spec Level 2: Semantic Identity
            # Threshold: >= 0.985
            if score < 0.985:
                continue
                
            node = self.archive.get(aid)
            if not node: continue

            # Provenance Check
            if not self._is_provenance_compatible(block, node):
                continue

            # Spec Level 2b: Lineage Root Match
            # "blocks with different lineage roots ... should NOT be collapsed"
            # We enforce that IF lineage is present, it must match.
            b_root = getattr(block, "lineage_root", None)
            n_root = node.get("lineage", {}).get("root")
            
            if b_root is not None and n_root is not None:
                if b_root != n_root:
                    continue
            
            # Fingerprint Overlap Check (Optimization for Speed)
            fp_block = getattr(block, "fingerprint", 0)
            fp_node = node.get("fingerprint", 0)
            if fp_block is None or fp_node is None:
                continue
                
            overlap_bits = (fp_block & fp_node).bit_count()
            if overlap_bits < CFG.NEAR_DUP_FP_BITS:
                continue
                
            # Match Found!
            return aid
            
        return None

    def _ensure_archive_entry(self, text: str, fingerprint: int) -> int:
        """
        Phase 2: Correct Lineage Assignment.
        Creates or retrieves an archive ID for a text fragment BEFORE Block creation.
        Ensures strict lineage root assignment.

        Dedup is gated by CFG.DEDUP_ENABLED (default: False). When
        disabled, every ingest gets a brand-new archive id — this
        preserves a per-invocation timeline for tool calls and other
        repeated-text events. A later NLP-driven classifier will
        decide recallability instead of relying on exact-text dedup.
        """
        if CFG.DEDUP_ENABLED:
            # 1. Canonical Dedup
            if hasattr(self, "canonical_hash_index"):
                 key = self._calculate_canonical_hash(text)
                 dup_aid = self.canonical_hash_index.get(key)
                 if dup_aid is not None:
                     return dup_aid

            # 2. Legacy SHA-1 Fallback
            elif hasattr(self, "_archive_dedup") and hasattr(self, "_dedup_key"):
                 key = self._dedup_key(text)
                 dup_aid = self._archive_dedup.get(key)
                 if dup_aid is not None:
                     return dup_aid

        # 3. Create New Node
        # DB-assigned ID (LanceDB) or fallback to in-memory counter
        try:
            aid = self.archive.next_id("archive")
        except (NotImplementedError, AttributeError):
            aid = max(1, self.next_archive_id)
            self.next_archive_id = aid + 1

        # Safety assertion
        assert aid >= 1, f"Invalid archive ID allocated: {aid}"

        # Create minimal Block placeholder for unified node creation
        from .blocks import Block
        # We assume new lineage root = self (aid)
        placeholder = Block(
            id=aid, # Use aid as temp id
            text=text,
            token_count=len(text.split()),
            fingerprint=fingerprint,
            created_step=self.current_step,
            last_used_step=self.current_step,
            heat=1.0,
            origin_archive_id=aid,
            source="atomic_write",
            last_heat_source="atomic_write",
            priority_class="USER",
            volatility=1.0,
            protected_until_step=self.current_step + self.PROTECTION_WINDOW,
            last_heat_update_step=self.current_step,
            entropy_score=1.0,
            lineage_root=aid, # CRITICAL: Self-reference
            is_assistant=False
        )
        
        entry = self._create_unified_node(aid, placeholder)
        
        # Phase 4: Ensure archive stores lineage root explicitly
        entry["lineage"]["root"] = aid
        entry["lineage_root"] = aid # Redundant key for flat access if needed
        
        entry["timestamp"] = time.time()
        self.archive[aid] = entry
        
        # Update Dedup Maps
        if hasattr(self, "canonical_hash_index"):
            key = self._calculate_canonical_hash(text)
            self.canonical_hash_index[key] = aid
            
        if hasattr(self, "_archive_dedup") and hasattr(self, "_dedup_key"):
            key = self._dedup_key(text)
            self._archive_dedup[key] = aid
            
        # [FIX-2] Fingerprint Reverse Index
        if hasattr(self, "fingerprint_index") and fingerprint:
             self.fingerprint_index[fingerprint] = aid
            
        # Indexing (Immediate Sparse) — skip during bulk ingest, batch-built after
        _bulk = os.environ.get("TCMM_BULK_INGEST") == "1"
        if not _bulk and hasattr(self, "archive_sparse_index") and self.archive_sparse_index:
             self.archive_sparse_index.add([text], [aid])

        # [FIX-C] Immediate Dense Index Insert + Embedding Persistence
        # Skip during bulk ingest — batch embedding pass runs after ingest
        emb = None
        if not _bulk and hasattr(self, "archive_vector_index") and self.archive_vector_index and hasattr(self, "embedder"):
            try:
                emb = self.embedder.embed(text)
                if emb is not None:
                    self.archive_vector_index.add(aid, emb)
                    entry["embedding"] = emb  # Persist in archive entry
            except Exception:
                pass

        # [FIX-A] Temporal Wiring (prev_aid / next_aid + embedding-based weight)
        #
        # Changed 23 Apr 2026: was walking ``reversed(self.live_blocks)``
        # to find prev_aid. Broken when short user-messages get evicted
        # from live before the NEXT user-message gets its backlink.
        # Observed 17/138 broken links (12%) in bb400c87 where multiple
        # consecutive USER aids all backlinked to the same ASSISTANT aid
        # (the walker kept finding the assistant because the intermediate
        # user blocks had already been evicted from the live tier).
        # New approach: track ``_last_ingested_aid`` on self — strictly
        # per-instance (per-conversation), doesn't depend on live_blocks
        # membership, doesn't degrade under eviction.
        prev_aid = getattr(self, "_last_ingested_aid", None)
        if prev_aid == aid:
            prev_aid = None  # shouldn't happen but defensive
        entry["temporal"]["prev_aid"] = prev_aid

        # Compute temporal weight from embedding cosine similarity
        temporal_weight = 1.0
        if prev_aid is not None and emb is not None:
            prev_emb = None
            if hasattr(self, "archive_embeddings"):
                prev_emb = self.archive_embeddings.get(prev_aid)
            if prev_emb is None:
                prev_node_tmp = self.archive.get(prev_aid)
                if prev_node_tmp:
                    prev_emb = prev_node_tmp.get("embedding")
            if prev_emb is not None:
                try:
                    import numpy as _np
                    a = _np.array(emb, dtype=_np.float32)
                    b = _np.array(prev_emb, dtype=_np.float32)
                    cos = float(_np.dot(a, b) / ((_np.linalg.norm(a) * _np.linalg.norm(b)) + 1e-9))
                    temporal_weight = max(0.1, min(1.0, cos))
                except Exception:
                    temporal_weight = 1.0
        entry["temporal"]["prev_weight"] = temporal_weight

        # Link previous node forward. Only overwrite if the existing
        # next_aid is None or points to an aid OLDER than ours — this
        # prevents a later insert from clobbering a valid forward link
        # that was already set for an intermediate block. Observed 23
        # Apr: multiple consecutive USER inserts each overwrote the
        # previous assistant's next_aid, losing the middle links.
        if prev_aid is not None and prev_aid in self.archive:
            try:
                prev_node = self.archive[prev_aid]
                self._normalize_archive_node(prev_node)
                existing_next = prev_node.get("temporal", {}).get("next_aid")
                if existing_next is None or (
                    isinstance(existing_next, (int, float)) and int(existing_next) < int(aid)
                ):
                    prev_node["temporal"]["next_aid"] = aid
                    prev_node["temporal"]["next_weight"] = temporal_weight
                    prev_node["isDirty"] = True
                    self.archive[prev_aid] = prev_node
            except KeyError:
                pass

        # Remember this aid for the next insert's prev_aid. Per-instance
        # (per-conversation) — different TCMM instances don't share state.
        self._last_ingested_aid = aid

        # Re-persist the NEW entry — DB providers (LanceDB) serialise
        # on __setitem__, so our earlier write (before temporal wiring)
        # captured temporal.prev_aid=None. The mutation above only
        # lives in the local `entry` dict until we write it back here.
        try:
            self.archive[aid] = entry
        except Exception:
            pass

        # [FIX-B] Enqueue to Worker Queues (semantic + embedding)
        _bulk = os.environ.get("TCMM_BULK_INGEST") == "1"

        # Skip both queues during bulk ingest — catch-up passes handle everything after
        if not _bulk:
            if hasattr(self, "semantic_queue"):
                max_q = getattr(self, "MAX_SEMANTIC_QUEUE", 128)
                qsize = self.semantic_queue.qsize()
                # OLD behaviour: silently drop the enqueue when the queue
                # hit the cap. That left blocks forever unclassified —
                # observed 23 Apr 2026 with 72 blocks stuck as
                # block_class=None, the oldest 4.6 hours old, while the
                # queue itself sat at size=0 (they'd fallen through the
                # gate during earlier bursts). Fix: always enqueue so
                # we don't lose classification work. If the queue truly
                # overflows (>2048) we log it so we can raise the cap
                # instead of silently corrupting the archive.
                if qsize >= max_q:
                    _lg = _logger if "_logger" in globals() else None
                    # Warn once per ~50 over-cap enqueues to avoid log spam
                    if qsize % 50 == 0:
                        logger.warning(
                            f"[archive] semantic_queue depth={qsize} "
                            f"above cap={max_q} — continuing to enqueue "
                            f"(OLD behaviour silently dropped, losing classification)"
                        )
                self.semantic_queue.put(aid)
                self._metrics["semantic.enqueue"] += 1

        # Skip embedding queue during bulk ingest — batch embedding pass runs after
        if not _bulk:
            if hasattr(self, "embedding_queue"):
                # Same fix as semantic_queue above — never silently drop.
                # Embedding is more expensive per item so we're stricter;
                # log at every 50 over-cap enqueues.
                max_q = getattr(self, "MAX_COLD_QUEUE", 128)
                qsize = self.embedding_queue.qsize()
                if qsize >= max_q and qsize % 50 == 0:
                    logger.warning(
                        f"[archive] embedding_queue depth={qsize} "
                        f"above cap={max_q} — continuing to enqueue"
                    )
                self.embedding_queue.put(aid)

        # Checkpoint: persist new node + temporal link partner
        try:
            persist_aids = [aid]
            prev_aid = entry.get("temporal", {}).get("prev_aid")
            if prev_aid is not None and prev_aid != aid:
                persist_aids.append(prev_aid)
            self.archive.persist(persist_aids)
        except Exception:
            pass

        return aid

    # Final Architecture: Immediate Archival
    def ensure_archive_node_for_block(self, block: Block) -> int:
        """
        Guarantees that a block has a corresponding Archive Node.
        Returns the Archive ID.
        """
        if hasattr(block, "archivable") and not block.archivable:
            return None

        # HARD RULE: never archive pure assistant THOUGHT blocks
        # BUG-ARCHIVE-THOUGHT-BLOCKING
        # Allow THOUGHT blocks to be archived, but mark recallability later
        # Pure assistant thought blocks may still be non-recallable,
        # but must exist in the archive for linking and provenance.

        archive_text = block.text
        
        # Phase 73: Revert stripping. 
        # We trust entropy guards and user priority.
        # Mixed blocks are archived verbatim to prevent data loss.
            
        # 1. Idempotency (already archived?)
        if getattr(block, "origin_archive_id", None) is not None:
            return block.origin_archive_id
            
        # Phase 107: Dream Mode Guard (No new Memories)
        if getattr(self, "dream_mode", False):
            return None
            
        # 2. Check Dedup (Level 1: Canonical Identity)
        # Spec v1.0: Canonical Text Identity via SHA-256.
        # Gated by CFG.DEDUP_ENABLED (default False) — "archive every turn"
        # design relies on the episodic classifier to mark recallability
        # instead of collapsing exact-text matches at archive time.
        if CFG.DEDUP_ENABLED and hasattr(self, "canonical_hash_index"):
             # SHA-256 Dedup Check
             key = self._calculate_canonical_hash(block.text)
             dup_aid = self.canonical_hash_index.get(key)

             if dup_aid is not None:
                 # 2a. Map ID
                 block.origin_archive_id = dup_aid
                 
                 # 2b. Flag as Non-Persistent
                 block.non_persistent = True
                 
                 # 2c. Correct Temporal Linking (Forward Linking)
                 # Even if duplicated, we must link the PREVIOUS block to this OLD archive node.
                 if hasattr(self, "live_blocks") and self.live_blocks:
                     # Identify previous block (handling current block presence in list)
                     idx = -1
                     prev_block = self.live_blocks[idx]
                     
                     if prev_block.id == block.id:
                         if len(self.live_blocks) > 1:
                             prev_block = self.live_blocks[-2]
                         else:
                             prev_block = None
                     
                     if prev_block:
                         prev_aid = getattr(prev_block, "origin_archive_id", None)
                         
                         # Avoid self-ref
                         if prev_aid == dup_aid:
                             prev_aid = None
                         
                         # Update Archive Graph (Dirty State)
                         if prev_aid is not None and prev_aid in self.archive:
                              try:
                                  prev_node = self.archive[prev_aid]
                                  self._normalize_archive_node(prev_node)
                                  
                                  # Update next pointer to the EXISTING duplicate node
                                  prev_node["temporal"]["next_aid"] = dup_aid
                                  prev_node["isDirty"] = True
                                  # log_archive(f"[DEDUP] Linked prev_aid={prev_aid} to duplicate aid={dup_aid}")
                              except KeyError:
                                  pass

                 # 2d. Bypass Creation
                 return dup_aid

        # Legacy SHA-1 Fallback (Deprecated but kept for safety if Level 1 missing)
        # Same CFG.DEDUP_ENABLED gate as Level 1.
        elif CFG.DEDUP_ENABLED and hasattr(self, "_archive_dedup") and hasattr(self, "_dedup_key"):

             # SHA-1 Dedup Check
             key = self._dedup_key(block.text)
             dup_aid = self._archive_dedup.get(key)

             if dup_aid is not None:
                 # 2a. Map ID
                 block.origin_archive_id = dup_aid
                 
                 # 2b. Flag as Non-Persistent
                 block.non_persistent = True
                 
                 # 2c. Correct Temporal Linking (Forward Linking)
                 # Even if duplicated, we must link the PREVIOUS block to this OLD archive node.
                 if hasattr(self, "live_blocks") and self.live_blocks:
                     # Identify previous block (handling current block presence in list)
                     idx = -1
                     prev_block = self.live_blocks[idx]
                     
                     if prev_block.id == block.id:
                         if len(self.live_blocks) > 1:
                             prev_block = self.live_blocks[-2]
                         else:
                             prev_block = None
                     
                     if prev_block:
                         prev_aid = getattr(prev_block, "origin_archive_id", None)
                         
                         # Avoid self-ref
                         if prev_aid == dup_aid:
                             prev_aid = None
                         
                         # Update Archive Graph (Dirty State)
                         if prev_aid is not None and prev_aid in self.archive:
                              try:
                                  prev_node = self.archive[prev_aid]
                                  self._normalize_archive_node(prev_node)
                                  
                                  # Update next pointer to the EXISTING duplicate node
                                  prev_node["temporal"]["next_aid"] = dup_aid
                                  prev_node["isDirty"] = True
                                  # log_archive(f"[DEDUP] Linked prev_aid={prev_aid} to duplicate aid={dup_aid}")
                              except KeyError:
                                  pass

                 # 2d. Bypass Creation
                 return dup_aid

        # Phase 120: Layer B - Near-Duplicate Semantic Merge
        emb = None
        if hasattr(self, "live_embedding_cache") and block.id in self.live_embedding_cache:
            emb = self.live_embedding_cache[block.id]
        elif hasattr(self, "embedder") and self.embedder:
            # Use embedder for archive_text (likely a description/summary) -> passage
            try:
                emb = self.embedder.embed(archive_text)
                if emb is not None:
                    self.live_embedding_cache[block.id] = emb
            except Exception as e:
                _log_warn(f"Embedding failed during archival: {e}")
        
        # Phase 120 Layer B merge is also gated by CFG.DEDUP_ENABLED so
        # every turn gets its own archive row when dedup is off.
        if CFG.DEDUP_ENABLED and emb is not None:
            merge_aid = self._check_near_duplicate(block, emb)
            if merge_aid is not None:
                # Merge!
                node = self.archive[merge_aid]
                
                # Update provenance
                srcs = node.get("source_block_ids", [])
                if block.original_id not in srcs:
                    srcs.append(block.original_id)
                
                # Update stats
                if "archive_stats" not in node:
                    node["archive_stats"] = {"attempts": 0, "used": 0}
                node["archive_stats"]["merged"] = node["archive_stats"].get("merged", 0) + 1
                
                block.origin_archive_id = merge_aid
                log_archive(f"[MERGE] Block {block.id} merged into {merge_aid}")
                return merge_aid

        # 3. Create New Node
        # DB-assigned ID (LanceDB) or fallback to in-memory counter
        try:
            aid = self.archive.next_id("archive")
        except (NotImplementedError, AttributeError):
            aid = max(1, self.next_archive_id)
            self.next_archive_id = aid + 1

        # Safety assertion
        assert aid >= 1, f"Invalid archive ID allocated: {aid}"

        # 4. Construct Entry
        
        # 4. Construct Entry (Unified Schema)
        entry = self._create_unified_node(aid, block)
        # Fix text source (block.text might be implicit, ensuring it's set)
        entry["text"] = self.resolve_text(doc.get("id")) or ""
        
        # 5. Populate Metadata
        entry["timestamp"] = time.time()
        # Optimization: Pass pre-computed embedding to avoid re-work in worker
        entry["embedding"] = emb if emb is not None else None

        # --- semantic overlay placeholders ---
        # Schema handles init

        self.archive[aid] = entry
        block.origin_archive_id = aid
        
        # BUG-SEMANTIC-WORKER-THOUGHT-BLOCKED
        # Enqueue immediately. Workers expect 'aid' (int), not tuple.
        if hasattr(self, "semantic_queue"):
             # Phase 112: Surgical Fix - Queue Cap
             max_q = getattr(self, "MAX_SEMANTIC_QUEUE", 128)
             if self.semantic_queue.qsize() < max_q:
                 self.semantic_queue.put(aid)
                 self._metrics["semantic.enqueue"] += 1
             else:
                 _log_warn(f"Semantic Queue Full (size={self.semantic_queue.qsize()}), dropping aid={aid}")

        if hasattr(self, "embedding_queue"):
             # Phase 112: Surgical Fix - Queue Cap
             max_q = getattr(self, "MAX_COLD_QUEUE", 128)
             if self.embedding_queue.qsize() < max_q:
                 self.embedding_queue.put(aid)
             else:
                 _log_warn(f"Cold Queue Full (size={self.embedding_queue.qsize()}), dropping aid={aid}")
        
        # 5. Update Dedup
        if hasattr(self, "canonical_hash_index"):
            key = self._calculate_canonical_hash(archive_text)
            self.canonical_hash_index[key] = aid
            
        if hasattr(self, "_archive_dedup"):
            key = self._dedup_key(archive_text)
            self._archive_dedup[key] = aid

        # 6. Temporal Wiring (Connect to previous *archived* block from live stream)
        # We look at the block immediately preceding this one in the *write stream*.
        if hasattr(self, "live_blocks") and self.live_blocks:
             # Fix: Handle case where current block is already appended to live_blocks
             # If live_blocks[-1] IS this block, we must look at [-2]
             idx = -1
             prev_block = self.live_blocks[idx]

             if prev_block.id == block.id:
                 if len(self.live_blocks) > 1:
                     prev_block = self.live_blocks[-2]
                 else:
                     prev_block = None

             if prev_block:
                 prev_aid = getattr(prev_block, "origin_archive_id", None)

                 # Avoid self-ref if something went wrong
                 if prev_aid == aid:
                     prev_aid = None

                 entry["temporal"]["prev_aid"] = prev_aid

                 # Compute temporal weight from embedding cosine similarity
                 temporal_weight = 1.0
                 if prev_aid is not None and emb is not None:
                     prev_emb = None
                     if hasattr(self, "archive_embeddings"):
                         prev_emb = self.archive_embeddings.get(prev_aid)
                     if prev_emb is None:
                         prev_node_tmp = self.archive.get(prev_aid)
                         if prev_node_tmp:
                             prev_emb = prev_node_tmp.get("embedding")
                     if prev_emb is not None:
                         try:
                             import numpy as _np
                             a = _np.array(emb, dtype=_np.float32)
                             b = _np.array(prev_emb, dtype=_np.float32)
                             cos = float(_np.dot(a, b) / ((_np.linalg.norm(a) * _np.linalg.norm(b)) + 1e-9))
                             temporal_weight = max(0.1, min(1.0, cos))
                         except Exception:
                             temporal_weight = 1.0
                 entry["temporal"]["prev_weight"] = temporal_weight

                 # Link graph
                 if prev_aid is not None and prev_aid in self.archive:
                      try:
                          prev_node = self.archive[prev_aid]
                          self._normalize_archive_node(prev_node)
                          prev_node["temporal"]["next_aid"] = aid
                          prev_node["temporal"]["next_weight"] = temporal_weight
                          prev_node["isDirty"] = True
                          self.archive[prev_aid] = prev_node  # Write back for DB providers
                      except KeyError:
                          pass

        # Indexing (Immediate Sparse)
        if hasattr(self, "archive_sparse_index") and self.archive_sparse_index:
             self.archive_sparse_index.add([archive_text], [aid])

        # [FIX-RECALL-2] Immediate Dense Index Insert
        # Guarantees vector index has entry before next recall
        if emb is not None and hasattr(self, "archive_vector_index") and self.archive_vector_index:
            try:
                self.archive_vector_index.add(aid, emb)
            except Exception:
                pass

        # Checkpoint: persist new node + temporal link partner
        try:
            persist_aids = [aid]
            prev_aid = entry.get("temporal", {}).get("prev_aid")
            if prev_aid is not None and prev_aid != aid:
                persist_aids.append(prev_aid)
            self.archive.persist(persist_aids)
        except Exception:
            pass

        return aid

    def attach_semantic_overlay(self, aid: int, overlay: dict):
        """
        overlay must contain:
          - topics
          - entities
          - claims
        """
    
        entry = self.archive.get(aid)
        if not entry:
            return
    
        def _sanitize(raw_list):
            clean = []
            for item in raw_list:
                if isinstance(item, str):
                    clean.append(item)
                elif isinstance(item, dict):
                    # Attempt to extract common keys or stringify
                    val = item.get("name") or item.get("text") or item.get("entity") or str(item)
                    if val:
                        clean.append(val)
                else:
                    clean.append(str(item))
            return clean

        entry["topics"] = _sanitize(overlay.get("topics", []) or [])
        entry["entities"] = _sanitize(overlay.get("entities", []) or [])
        entry["claims"] = _sanitize(overlay.get("claims", []) or [])

        # Persist typed entity dicts (name, type, score) for dream engine
        raw_ed = overlay.get("entity_dicts", [])
        if raw_ed:
            entry["entity_dicts"] = [
                ed for ed in raw_ed
                if isinstance(ed, dict) and ed.get("name")
            ]

        # Persist typed topic dicts (name, type, score) for recall scoring
        raw_td = overlay.get("topic_dicts", [])
        if raw_td:
            entry["topic_dicts"] = [
                td for td in raw_td
                if isinstance(td, dict) and td.get("name")
            ]

        # Persist claim quality scores (parallel to claims list)
        raw_cs = overlay.get("claim_scores", [])
        if raw_cs:
            entry["claim_scores"] = [float(s) for s in raw_cs if isinstance(s, (int, float))]

        # Phase 300: Update dedicated claims index (per-claim granularity)
        if entry["claims"]:
            _valid_claims = [c for c in entry["claims"] if isinstance(c, str) and c.strip()]
            if _valid_claims and hasattr(self, "archive_claims_index") and self.archive_claims_index:
                self.archive_claims_index.add(_valid_claims, [aid] * len(_valid_claims))

        # Phase 301: Update dedicated entity index
        if entry["entities"]:
            ent_text = " ".join(e for e in entry["entities"] if isinstance(e, str))
            if ent_text.strip() and hasattr(self, "archive_entity_index") and self.archive_entity_index:
                self.archive_entity_index.add([ent_text], [aid])

        # Phase 19: Update dedicated topic index
        if entry.get("topics"):
            topic_text = " ".join(t for t in entry["topics"] if isinstance(t, str))
            if topic_text.strip() and hasattr(self, "archive_topic_index") and self.archive_topic_index:
                self.archive_topic_index.add([topic_text], [aid])

        # block_class is now written by the NLP path in workers.py
        # (episodic classifier: FACT/DECISION/.../CHATTER/ACK). The old
        # structural KNOWLEDGE/ECHO classifier is retained only for the
        # longmemeval benchmark's direct call sites and is slated for
        # deletion — no longer invoked here.

        # Linker is now handled by workers.py to satisfy SEM requirements
        # self._semantic_linker(aid)
        
        # Phase 6b: Contextual Bootstrapping & Drift
        sem = overlay.get("semantic") or {}
        
        # Surgical Fix: Persist Semantic Tail
        tail = sem.get("text_tail")
        if tail:
            entry["semantic_text_tail"] = tail
            
        flow = sem.get("flow_from_prev")

        _do_bootstrap_prev = None
        if flow:
            entry["semantic_flow_count"] = entry.get("semantic_flow_count", 0) + 1
            if flow.get("break"):
                entry["semantic_break_count"] = entry.get("semantic_break_count", 0) + 1

            # Update frequency ratio
            total = max(1, entry["semantic_flow_count"])
            entry["break_frequency"] = entry["semantic_break_count"] / total

            # Capture the prev_aid we want to bootstrap against AFTER the
            # outer entry has been persisted. If we bootstrap now (with
            # source=aid), the inner write of aid gets clobbered by the
            # outer final write at the end of this function because
            # both target the same row. Defer.
            if not flow.get("break"):
                prev_ctx = self._get_prev_semantic_context(aid)
                if prev_ctx:
                    _do_bootstrap_prev = prev_ctx.get("aid")

        entry["isDirty"] = True
        # Write back to provider (critical for DB-backed providers like LanceDB
        # where get() returns a deserialized copy, not a live reference).
        # This MUST happen BEFORE the bidirectional contextual-link bootstrap
        # below, otherwise the final write overwrites the bootstrap's update.
        self.archive[aid] = entry

        # Deferred bootstrap: now that aid's entry (topics, entities,
        # claims, flow counters) is on disk, wire up contextual links.
        # _bootstrap_contextual_link internally fetches + mutates +
        # writes back per (source), so ordering is symmetric and safe.
        if _do_bootstrap_prev is not None:
            # Link A -> B (updates prev_aid's row)
            self._bootstrap_contextual_link(_do_bootstrap_prev, aid)
            # Link B -> A (updates aid's row — no longer clobbered)
            self._bootstrap_contextual_link(aid, _do_bootstrap_prev)

    def _get_prev_semantic_context(self, aid: int):
        """Return the previous archive block's context via temporal link."""
        entry = self.archive.get(aid)
        if not entry:
            return None
        prev_aid = (entry.get("temporal") or {}).get("prev_aid")
        if prev_aid is None or prev_aid not in self.archive:
            return None
        return {"aid": prev_aid}

    def _bootstrap_contextual_link(self, source: int, target: int):
        """Seeds a weak contextual link derived from semantic continuity.

        LanceDB provider note: archive.get(aid) returns a DESERIALIZED
        COPY, not a live reference. Mutating it in memory does nothing
        unless we write it back via archive[aid] = node. Earlier
        versions of this function mutated the copy and skipped the
        write-back, so every contextual link bootstrapped was silently
        discarded — LanceDB dumps showed contextual_links: {} on every
        archive entry even though the semantic worker was firing flow
        detection correctly. Same class of bug as the temporal-wiring
        fix in _ensure_archive_entry.
        """
        # Linker Summary: Contextual Link
        summary.increment_counter("linker", "contextual_links_added")

        node = self.archive.get(source)
        if not node:
            return
        self._normalize_archive_node(node)

        links = node["contextual_links"]

        # Schema: links[target] = {weight, last_step}
        if target in links:
            current = links[target]
        else:
            current = None

        if current is None:
            current = {"weight": 0.0, "last_step": 0}

        if self._should_apply_dream_update():
            current["weight"] += CFG.CONTEXTUAL_SEED_DELTA
            current["last_step"] = self.current_step

        # Link
        links[target] = current
        node["isDirty"] = True

        # Persist back to the provider. Without this the in-memory
        # update is lost for DB-backed providers.
        try:
            self.archive[source] = node
        except Exception:
            pass

        # DEBUG-SEM-03 : contextual link bootstrap snapshot
        from .tcmm_logging import log_semantic
        from . import tcmm_debug as dbg

        if dbg._enabled(getattr(dbg, "DEBUG_SEMANTIC", False)):
            node = self.archive.get(source)
            if node:
                val = node.get("contextual_links", {}).get(target)
                log_semantic(
                    f"contextual_link source={source} target={target} value={val}"
                )


    # REMOVED: Legacy _semantic_linker (entity-only overlap)
    # Single authority is now the inline claims-anchored linker in semantic_worker_loop

    def _compress_and_archive(self, blocks: List[Block]):
        """
        Legacy / Cleanup-triggered method.
        Final Architecture: This should NO LONGER create nodes.
        It strictly cleans up live blocks. 
        But wait, `_cool_blocks` calls this?
        No, `_cool_blocks` calls `archive_mixin._compress_and_archive`.
        We must effectively disable the *creation* part here.
        """
        # Since we moved creation to `add_new_block`, 
        # blocks reaching here SHOULD already have `origin_archive_id`.
        # If they don't (e.g. from before the patch?), we create them now.
        for b in blocks:
             self.ensure_archive_node_for_block(b)
             
        return # Done. We don't batch-compress anymore.

    def _prune_graph_references(self, evicted_aids: Set[int]):
        """
        Removes all graph links pointing to the evicted archive IDs.
        Crucial for maintaining graph integrity.
        Handles both int and string keys defensively.
        """
        if not evicted_aids:
            return

        # Iterate over all remaining nodes to clean up their outgoing links
        # For ShardedArchive, this iterates all shards
        for aid, node in self.archive.items():
            # Skip if this node is also being evicted
            if aid in evicted_aids:
                continue
                
            is_dirty = False
            
            # Helper to check if target is evicted (handling str/int)
            def _is_evicted(t):
                if t in evicted_aids: return True
                try:
                    return int(t) in evicted_aids
                except (ValueError, TypeError):
                    return False

            # 1. Prune Contextual Links
            if "contextual_links" in node:
                links = node["contextual_links"]
                for target in list(links.keys()):
                    if _is_evicted(target):
                        del links[target]
                        is_dirty = True
            
            # 2. Prune Behavioural Links (Hot/Cold/Ghost)
            if "behavioural" in node:
                beh = node["behavioural"]
                for tier in ["hot", "cold", "ghost"]:
                    if tier in beh:
                        tier_map = beh[tier]
                        for target in list(tier_map.keys()):
                            if _is_evicted(target):
                                del tier_map[target]
                                is_dirty = True
                                
            # 3. Prune Suppression Links
            if "suppresses" in node:
                supp = node["suppresses"]
                for target in list(supp.keys()):
                    if _is_evicted(target):
                        del supp[target]
                        is_dirty = True
            
            # 4. Prune Temporal Links
            if "temporal" in node:
                temp = node["temporal"]
                if temp.get("prev_aid") in evicted_aids:
                    temp["prev_aid"] = None
                    is_dirty = True
                if temp.get("next_aid") in evicted_aids:
                    temp["next_aid"] = None
                    is_dirty = True

            if is_dirty:
                node["isDirty"] = True

    def _enforce_archive_limits(self):
        # 1. Count items
        if len(self.archive) <= self.max_archive_items:
            return

        # 2. Evict lowest usefulness (tie breaker: oldest)
        candidates = []
        for aid, entry in self.archive.items():
            u = self._archive_usefulness(aid)
            # Sort Key: (usefulness, step, aid) -> Ascending order puts low use, old step first.
            candidates.append((u, entry.get("step", 0), aid))
            
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        
        to_remove_count = len(self.archive) - self.max_archive_items
        evicted_aids = set()
        
        for i in range(to_remove_count):
            _, _, aid = candidates[i]
            evicted_aids.add(aid)
            del self.archive[aid]
            if aid in self.archive_embeddings:
                del self.archive_embeddings[aid]
            if aid in self.archive_stats:
                del self.archive_stats[aid]
            
            # Phase 7: Sync Indexes
            if self.archive_vector_index:
                self.archive_vector_index.remove(aid)
            if self.archive_sparse_index:
                self.archive_sparse_index.remove(aid)
        
        # 3. Prune dangling graph references
        if evicted_aids:
            self._prune_graph_references(evicted_aids)
                
        log_archive(f"[INFO] Limit enforced: Evicted {len(evicted_aids)} archive items and pruned graph links.")

    def resolve_text(self, aid: int) -> str:
        """
        Resolves the text content of an archive ID.

        File-origin blocks used to read back from disk here; that plane
        is gone — files now flow through the proxy's tool-invocation
        pipeline as tool_result blocks, so their text lives on
        entry["text"] like any other block.
        """
        entry = self.get_archive_entry(aid)

        if not entry:
            return ""

        # Dream arc_state_sheets: build richer text from structured data
        # so the LLM sees temporal context + entities inline
        if entry.get("type") == "arc_state_sheet" and entry.get("arc_state_data"):
            return self._resolve_arc_state_text(entry)

        # Prefer raw conversation text — canonical_text is NLP metadata that duplicates block content
        return entry.get("text") or entry.get("canonical_text") or entry.get("semantic_text") or ""

    def _resolve_arc_state_text(self, entry: dict) -> str:
        """Build LLM-friendly text from arc_state_sheet structured data.

        Produces richer text than raw canonical_text by:
        - Including entity names inline with types
        - Surfacing date range from source blocks
        - Presenting events/preferences/goals as natural sentences
        - Appending key topics for BM25-matching term visibility
        """
        data = entry.get("arc_state_data", {})
        parts = []

        # Date range from canonical_text header (already embedded there)
        ct = entry.get("canonical_text", "")
        # Extract the [STATE ...] or [SINGLE STATE ...] header line
        if ct.startswith("["):
            header_end = ct.find("\n")
            if header_end > 0:
                parts.append(ct[:header_end])

        # Current state
        if data.get("current_state"):
            parts.append(f"Current state: {data['current_state']}")

        # Events with descriptions
        for ev in data.get("events", []):
            desc = ev.get("description", "")
            if desc:
                parts.append(f"- {desc}")

        # Preferences
        for p in data.get("preferences", []):
            pref = p.get("preference", "")
            if pref:
                parts.append(f"- Preference: {pref}")

        # Goals
        for g in data.get("goals", []):
            goal = g.get("goal", "")
            if goal:
                parts.append(f"- Goal: {goal}")

        # Facts
        for f in data.get("facts", []):
            if isinstance(f, str) and f:
                parts.append(f"- Fact: {f}")

        # Named entities — cap at 30, preserve original order (LLM extraction order
        # is already relevance-ranked). Full list stays on node for graph traversal.
        ent_names = entry.get("entities", [])
        if not ent_names:
            ent_names = [e.get("name", "") for e in data.get("entities", []) if e.get("name")]
        if ent_names:
            parts.append(f"Mentions: {', '.join(str(e) for e in ent_names[:30])}")

        # Key topics (for BM25 term visibility in context window)
        topics = entry.get("topics", [])
        if topics:
            parts.append(f"Topics: {', '.join(topics[:10])}")

        text = "\n".join(p for p in parts if p)
        return text if text else ct

    def _archive_get_text(self, archive_id: int) -> str:
        return self.resolve_text(archive_id)

    def _get_archive_heat(self, aid: int) -> float:
        """
        Computes effective heat for an archive node lazily.
        Does not mutate the node.
        """
        node = self.archive.get(aid)
        if not node:
            # Phase 13: Dream Archive Fallback
            if getattr(self, "dream_mode", False) and hasattr(self, "dream_archive"):
                 node = self.dream_archive.get(aid)
            
            if not node:
                return 0.0
        
        # Lazy initialization fallback
        h = node.get("heat", 1.0)
        last = node.get("last_decay_step", getattr(self, "current_step", 0))
        
        return compute_effective_heat(
            h, 
            last, 
            getattr(self, "current_step", 0), 
            getattr(CFG, "ARCHIVE_DECAY_RATE", 0.995)
        )

    def _archive_usefulness(self, origin_id: int) -> float:
        # Phase 4: Usefulness Score
        if origin_id not in self.archive_stats:
            return 0.5 # Neutral start
            
        s = self.archive_stats[origin_id]
        # Bayesian smoothing: (used + 1) / (attempts + 2)
        raw = (s["used"] + 1) / (s["attempts"] + 2)
        
        # Recency-Biased Usefulness
        # ((u+1)/(a+2)) * exp( -(current_step - last_used_step) / TAU )
        if s["used"] == 0:
             # If never used, just raw likelihood
             pass
        else:
             import math
             last_used = s.get("last_used", 0) # Default to 0 if missing
             diff = self.current_step - last_used
             decay = math.exp(-diff / self.usefulness_decay_tau)
             raw *= decay
             
        return raw

    def add_suppression(self, aid_source: int, aid_target: int, weight: float):
        """Adds a suppression edge conditioned on current topic."""
        # Linker Summary: Suppression Link
        summary.increment_counter("linker", "suppression_links_added")

        # Phase 107: Dream Mode Scale
        if getattr(self, "dream_mode", False):
             weight *= self.DREAM_LEARNING_SCALE

        node_a = self.archive.get(aid_source)
        if not node_a: return
        self._normalize_archive_node(node_a)
        
        # Schema: node["suppresses"][target][ctx] = {weight, last_step}
        supp_map = node_a["suppresses"]
        
        target_map = supp_map.setdefault(aid_target, {})
        
        ctx = self._suppression_context_key()
        entry = target_map.setdefault(ctx, {"weight": 0.0, "last_step": 0})
        
        # Apply decay to existing weight before adding delta
        # entry["weight"] = self._decay_edge(entry["weight"], entry["last_step"], CFG.SUPPRESSION_TAU)
        
        if self._should_apply_dream_update():
            entry["weight"] += weight
            entry["last_step"] = self.current_step
        
        node_a["isDirty"] = True
        
        # Only cleanup if cold? We rely on _cool_archive_edges 
        # No "Hot/Cold" tiering for suppression listed strictly in schema 
        # other than "suppresses" container.
        # cooling happens in _cool_archive_edges.


    def add_new_block(self, text: str, priority_class: str = "USER", source: str = "live", origin_archive_id: int = None, initial_heat: float = 0.8, is_assistant: bool = False):
        with perf_scope("tcmm.add_new_block"):
            # Phase 21: Assistant Filler Guard Check
            # Drop procedural filler ("Okay, I will...")
            lower_txt = text.lower().strip()
            if priority_class == "THOUGHT" and (
                lower_txt.startswith("okay, i will") or 
                lower_txt.startswith("i will store") or
                lower_txt.startswith("storing the")
            ):
                 return

            # Phase 67: Shadow Staging for Reinjection
            if source == "reinjected":
                from core.blocks import Block
                
                # Level 3: Lineage Identity (Reinjection Dedup)
                # "Reinjected blocks must not duplicate live blocks with same origin"
                if origin_archive_id is not None:
                    # Check if already in live_blocks
                    if hasattr(self, "live_blocks"):
                        for lb in self.live_blocks:
                            # Level 0 Check: Structural Identity
                            if lb.origin_archive_id == origin_archive_id:
                                return # REJECT
                            # Check text identity just in case (e.g. dirty read)
                            if lb.text == text:
                                return # REJECT
                
                # Create discrete block bypassing buffer
                
                # EG-REDUNDANT-STATIC-RECOMPUTE: Ensure static entropy is computed
                # If we have access to helper... or just import?
                # ArchiveMixin is in core/archive.py.
                # Lazy import to avoid circular dependency
                from .entropy_guard import entropy_guard_static
                
                diag = entropy_guard_static(text)
                
                b = Block(
                    id=self._alloc_block_id(),
                    text=text,
                    token_count=self._count_tokens(text),
                    fingerprint=self._get_semantic_fingerprint(text),
                    created_step=self.current_step,
                    last_used_step=self.current_step,
                    heat=initial_heat,
                    origin_archive_id=origin_archive_id,
                    source=source,
                    last_heat_source="recall",
                    priority_class=priority_class,
                    volatility=self.CLASS_VOLATILITY.get(priority_class, 1.0),
                    protected_until_step=self.current_step + self.PROTECTION_WINDOW,
                    last_heat_update_step=self.current_step,
                    entropy_score=diag.score, 
                    is_assistant=is_assistant
                )
                b.entropy_static = diag
                b.entropy_diag = diag
                self._add_shadow_block(b)
                return

            # Phase 67: Unified Write Path (BUG-LIVE-WRITE-02)
            b = None
            if is_assistant:
                self._ensure_current_assistant_block()
                self._append_to_current_assistant_block(text)
                b = self._current_assistant_block
            else:
                self._ensure_current_user_block()
                self._append_to_current_user_block(text)
                b = self._current_user_block

            return

    def apply_shadow_rejection(self, aids: Set[int], rejection_type="passive", context_emb=None):
        """
        Phase 1: Record shadow rejection feedback.
        tracks rejected_passive, last_rejected_step, last_rejected_context_embedding.
        """
        for aid in aids:
            node = self.archive.get(aid)
            if not node:
                continue

            if "archive_stats" not in node:
                node["archive_stats"] = {"attempts": 0, "used": 0}
            
            stats = node["archive_stats"]
            
            # Increment specific counter based on type
            key = f"rejected_{rejection_type}" # e.g. rejected_passive
            stats[key] = stats.get(key, 0) + 1
            
            stats["last_rejected_step"] = self.current_step
            if context_emb:
                stats["last_rejected_context_embedding"] = context_emb

    def load_dream_archive(self):
        """Loads dream archive nodes from isolated persistence layer.

        Routes by provider type:
          - DB providers (Lance, SQLite): skip TinyDB entirely. Nodes and
            search indices are already durable. Just wire transient
            dream_links backlinks onto archive nodes and sync the dream
            ID counter.
          - Local / fallback: load from dream_archive.json +
            dream_archive_embeddings.json (existing behavior).
        """
        import os

        # ── DB-provider fast path ──
        _provider_name = type(self.dream_archive).__name__
        if _provider_name in ("SQLiteStorageProvider", "LanceStorageProvider"):
            loaded = 0
            max_id = self._next_dream_id

            # Buffer archive writebacks: LanceDB's get() returns a fresh
            # deserialized dict per call, so in-place edits are lost unless
            # we write the dict back. Buffer to dedupe (each archive node
            # may be referenced by multiple dream nodes).
            archive_edits: Dict[int, Dict[str, Any]] = {}
            # Track which edits actually changed something — prior versions
            # wrote back every touched archive node on every load, which
            # hammered Lance with 100s of delete+add per init even when the
            # dream_links were already durable from a previous run. Now we
            # only write back nodes whose backlinks genuinely diverge.
            dirty_aids: set = set()

            # Iterate once: the list returned by items() is walked then GC'd.
            # No persistent in-memory cache of dream nodes.
            for aid, doc in self.dream_archive.items():
                if doc is None:
                    continue
                loaded += 1
                if aid > max_id:
                    max_id = aid

                node_type = doc.get("type", "")
                tier_weight = {
                    "concept_node": 0.9,
                    "cluster_anchor": 0.7,
                    "identity_kernel": 0.85,
                    "arc_node": 0.8,
                    "semantic_node": 0.6,
                    "fact_sheet": 0.7,
                }.get(node_type, 0.5)
                src_raw = doc.get("source_block_ids") or {}
                if isinstance(src_raw, dict):
                    src_pairs = src_raw.items()
                elif isinstance(src_raw, list):
                    src_pairs = [(s, 0.75) for s in src_raw]
                else:
                    src_pairs = []
                for src_aid, src_w in src_pairs:
                    try:
                        src_aid_i = int(src_aid)
                        src_w_f = float(src_w)
                    except (ValueError, TypeError):
                        continue
                    link_w = max(tier_weight, src_w_f)
                    # Pull once per archive node (cached in archive_edits)
                    if src_aid_i not in archive_edits:
                        src_block = self.archive.get(src_aid_i)
                        if src_block is None:
                            continue
                        archive_edits[src_aid_i] = src_block
                    existing_links = archive_edits[src_aid_i].setdefault("dream_links", {})
                    # dream_links keys may round-trip as strings on some
                    # providers — normalize to compare meaningfully.
                    existing = existing_links.get(aid)
                    if existing is None:
                        existing = existing_links.get(str(aid))
                    if existing is None or abs(float(existing) - float(link_w)) > 1e-9:
                        existing_links[aid] = link_w
                        dirty_aids.add(src_aid_i)

            # Only the subset of archive nodes whose backlinks actually
            # changed get persisted. On a cached warm load that's typically
            # 0 writes — load_dream_archive becomes a read-only op.
            #
            # Use put_batch so Lance does ONE merge_insert instead of
            # N individual delete+add round-trips. Previously 300 dirty
            # aids took ~30s on spinning disk; batched it's ~50ms.
            if dirty_aids:
                try:
                    if hasattr(self.archive, "put_batch"):
                        items_list = [(aid, archive_edits[aid]) for aid in dirty_aids]
                        self.archive.put_batch(items_list)
                    else:
                        for src_aid_i in dirty_aids:
                            self.archive[src_aid_i] = archive_edits[src_aid_i]
                except Exception:
                    # Fall back to per-aid writes if batching fails.
                    for src_aid_i in dirty_aids:
                        try:
                            self.archive[src_aid_i] = archive_edits[src_aid_i]
                        except Exception:
                            pass

            # Sync dream ID counter (DB preferred, _next_dream_id fallback)
            self._next_dream_id = max_id + 1
            try:
                self.dream_archive.set_id("dream", max_id + 1)
            except (NotImplementedError, AttributeError):
                pass

            # Note: _dream_ent_to_nodes / _dream_topic_to_nodes are NOT
            # built here. recall_graph.py:697-708 rebuilds them lazily
            # on first use if absent.

            _uid = getattr(self, "_user_id", None) or "-"
            _ns = getattr(self, "_clean_ns", getattr(self, "_ns_key", "default"))
            print(f"[DREAM] Loaded {loaded} dream nodes from {_provider_name} "
                  f"(tenant user_id={_uid!r} namespace={_ns!r}, "
                  f"backlinks touched={len(archive_edits)} written={len(dirty_aids)})")
            return

        # ── TinyDB fallback (Local provider) ──
        try:
            from tinydb import TinyDB
        except ImportError:
            print("[DREAM-LOAD] TinyDB not available — skipping dream archive load")
            return

        db_dir = self._data_dir
        db_path = os.path.join(db_dir, "dream_archive.json")
        emb_db_path = os.path.join(db_dir, "dream_archive_embeddings.json")
        # DreamEngine saves to TCMM/data/ (relative to module), fallback there
        if not os.path.exists(db_path):
            alt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
            alt_db = os.path.join(alt_dir, "dream_archive.json")
            if os.path.exists(alt_db):
                db_dir = alt_dir
                db_path = alt_db
                emb_db_path = os.path.join(alt_dir, "dream_archive_embeddings.json")
            else:
                return

        try:
            db = TinyDB(db_path)
            emb_db = None
            if os.path.exists(emb_db_path):
                emb_db = TinyDB(emb_db_path)
                if hasattr(self, "dream_archive_embeddings"):
                    for emb_doc in emb_db.all():
                        if "embedding" in emb_doc:
                            self.dream_archive_embeddings[emb_doc.doc_id] = emb_doc["embedding"]
            
            loaded = 0
            max_id = self._next_dream_id
            
            for doc in db.all():
                raw_id = doc.get("id")
                if raw_id is None: continue
                
                # PART 6 — Prevent ID collision
                # Enforce integer keys and apply offset (only if not already offset)
                try:
                    aid = int(raw_id)
                    if aid < 10_000_000:
                        aid += 10_000_000
                    doc["id"] = aid
                except ValueError:
                    continue
                
                if emb_db is not None:
                    # TinyDB doc_id is 1-based, but our aid is offset.
                    # We need to query emb_db using the original raw_id if it was stored that way,
                    # or by the new aid if emb_db was also offset.
                    # Assuming emb_db stores embeddings by the original raw_id for now.
                    # If emb_db stores by the *offset* aid, then use `aid` here.
                    # For now, let's assume emb_db stores by its own internal doc_id or original raw_id.
                    # The most robust way is to store the original ID in the embedding doc itself.
                    # For simplicity, let's assume emb_db uses the same ID as the main archive *before* offset.
                    emb_doc = emb_db.get(doc_id=int(raw_id)) # Assuming emb_db uses original raw_id as doc_id
                    if emb_doc and "embedding" in emb_doc:
                        doc["embedding"] = emb_doc["embedding"]

                # Add to local engine representation
                doc["isDirty"] = False
                self.dream_archive[aid] = doc

                # PATCH 5 — Hydrate dream archive too
                if hasattr(self, "dream_archive_embeddings"):
                    emb = self.dream_archive_embeddings.get(aid)
                    if emb is not None:
                        doc["embedding"] = emb
                
                loaded += 1

                # action_event nodes are graph-traversable only, not directly
                # searchable.  Force recallable=False so they stay out of all
                # retrieval indices while remaining reachable via topology edges.
                if doc.get("type") == "action_event":
                    doc["recallable"] = False

                # recallable gate — non-recallable nodes (e.g. action_event)
                # are kept in dream_archive for graph traversal but excluded
                # from every search index (BM25, vector, entity, topic).
                _is_recallable = doc.get("recallable", True)

                # PART 2 — Add dedicated dream sparse index
                # NOTE: Use doc text directly — resolve_text() requires dream_mode=True
                # which may not be set during __init__ loading.
                if _is_recallable and hasattr(self, "dream_sparse_index"):
                    text = doc.get("canonical_text") or doc.get("text") or ""
                    overlay = doc.get("semantic_summary") or ""
                    if text:
                        self.dream_sparse_index.add([text], [aid], [overlay])

                # Dream entity + topic BM25 indices (Phase 302)
                if _is_recallable and hasattr(self, "dream_entity_index"):
                    _ents = doc.get("entities", [])
                    if _ents:
                        _et = " ".join(str(e) for e in _ents if e)
                        if _et.strip():
                            self.dream_entity_index.add([_et], [aid])
                if _is_recallable and hasattr(self, "dream_topic_index"):
                    _topics = doc.get("topics", [])
                    if _topics:
                        _tt = " ".join(str(t) for t in _topics if t)
                        if _tt.strip():
                            self.dream_topic_index.add([_tt], [aid])
                # Dream entity VECTOR index (Phase 302b) — per-entity embeddings
                if _is_recallable and hasattr(self, "dream_entity_vector_index") and hasattr(self, "embedding_adapter"):
                    _ents = doc.get("entities", [])
                    _topics = doc.get("topics", [])
                    _all_terms = list(_ents or []) + list(_topics or [])
                    _unique_ents = list(set(str(e).strip() for e in _all_terms if e and len(str(e).strip()) > 1))
                    if _unique_ents:
                        try:
                            _embs = self.embedding_adapter.embed_batch(_unique_ents)
                            if _embs is not None and len(_embs) == len(_unique_ents):
                                for _eemb in _embs:
                                    _vid = self._next_entity_vid
                                    self._next_entity_vid += 1
                                    self.dream_entity_vector_index.add(_vid, _eemb)
                                    self._dream_entity_vid_to_nid[_vid] = aid
                        except Exception:
                            pass

                # Populate Vector Index (Upgrade 3)
                if _is_recallable and doc.get("embedding"):
                    self.dream_vector_index.add(aid, doc["embedding"])
                    
                # Wire bidirectional dream links between dream nodes and their
                # source archive blocks.  These are separate from semantic_links
                # (which is the NLP overlay between archive blocks).
                source_ids_raw = doc.get("source_block_ids", {})
                node_type = doc.get("type", "")
                # Node-tier base weight for archive→dream backlinks
                tier_weight = {"concept_node": 0.9, "cluster_anchor": 0.7,
                               "arc_state_sheet": 0.85, "fact_sheet_node": 0.8,
                               "action_event": 0.85}.get(node_type, 0.5)

                # Normalise: accept both dict (weighted) and list (legacy flat)
                if isinstance(source_ids_raw, dict):
                    source_pairs = source_ids_raw.items()
                elif isinstance(source_ids_raw, list):
                    source_pairs = [(s, 0.75) for s in source_ids_raw]
                else:
                    source_pairs = []

                for src_aid, src_weight in source_pairs:
                    try:
                        src_aid = int(src_aid)
                        src_weight = float(src_weight)
                    except (ValueError, TypeError):
                        continue

                    # Archive block → dream backlink (so traversal can reach dream nodes)
                    # Backlink weight = max(node tier, source weight) — strongest signal wins
                    link_weight = max(tier_weight, src_weight)
                    src_block = self.archive.get(src_aid)
                    if src_block is not None:
                        src_block.setdefault("dream_links", {})[aid] = link_weight

                # Keep allocator ahead
                if aid >= max_id:
                    max_id = aid + 1

            self._next_dream_id = max_id
            # Sync dream sequence counter to DB provider if supported
            try:
                self.dream_archive.set_id("dream", max_id)
            except (NotImplementedError, AttributeError):
                pass

            # ── Build pre-computed inverted index for recall traversal ──
            from collections import defaultdict
            _ent_idx = defaultdict(set)
            _topic_idx = defaultdict(set)
            for nid, node in self.dream_archive.items():
                if not node.get("recallable", True):
                    continue
                for e in node.get("entities", []):
                    if isinstance(e, str) and len(e) > 1:
                        _ent_idx[e.lower()].add(nid)
                for t in node.get("topics", []):
                    if isinstance(t, str) and len(t) > 2:
                        _topic_idx[t.lower()].add(nid)
            self._dream_ent_to_nodes = dict(_ent_idx)
            self._dream_topic_to_nodes = dict(_topic_idx)

            _emb_count = sum(1 for v in self.dream_archive.values() if v.get("embedding"))
            _vi_pending = len(self.dream_vector_index._pending) if hasattr(self, "dream_vector_index") else -1
            _si_ids = len(self.dream_sparse_index.ids) if hasattr(self, "dream_sparse_index") else -1
            print(f"[DREAM] Loaded {loaded} dream nodes (emb={_emb_count} vec={_vi_pending} sparse={_si_ids} ent_idx={len(_ent_idx)} topic_idx={len(_topic_idx)})")
            log_archive(f"[DREAM] Loaded {loaded} dream blocks (wired {sum(len(d.get('source_block_ids', [])) for d in self.dream_archive.values())} links).")

            # Close TinyDB handles to free memory
            try:
                db.close()
            except Exception:
                pass
            if emb_db is not None:
                try:
                    emb_db.close()
                except Exception:
                    pass

        except Exception as e:
            import traceback
            print(f"[DREAM-LOAD] FAILED: {e}")
            traceback.print_exc()
            log_exception("Failed to load dream archive", e)

    def save_indices(self):
        """Persist all indexes to disk under data/indices/."""
        import os
        idx_dir = os.path.join(self._data_dir, "indices")
        try:
            if hasattr(self, "archive_vector_index") and self.archive_vector_index:
                self.archive_vector_index.save(os.path.join(idx_dir, "vector"))
            if hasattr(self, "archive_sparse_index") and self.archive_sparse_index:
                self.archive_sparse_index.save(os.path.join(idx_dir, "sparse"))
            if hasattr(self, "archive_claims_index") and self.archive_claims_index:
                self.archive_claims_index.save(os.path.join(idx_dir, "claims"))
            if hasattr(self, "archive_entity_index") and self.archive_entity_index:
                self.archive_entity_index.save(os.path.join(idx_dir, "entity"))
            if hasattr(self, "archive_topic_index") and self.archive_topic_index:
                self.archive_topic_index.save(os.path.join(idx_dir, "topic"))
            log_archive("[PERSISTENCE] Saved all indexes to disk")
        except Exception as e:
            log_exception("Failed to save indexes", e)

    def _build_indices_from_archive(self):
        """
        Build vector and sparse indices immediately after archive load.
        Required for benchmark / recall-only mode.
        """

        if not hasattr(self, "archive_vector_index"):
            return

        if not hasattr(self, "archive_sparse_index"):
            return

        # LanceDB / any provider-backed sparse index: the corpus already
        # lives in the provider. Re-adding everything on every startup
        # would (a) duplicate rows (b) trigger O(N) per-row table.update()
        # calls for the vector column. If the sparse table is already
        # populated, the converter or a prior session built it — skip.
        try:
            if type(self.archive_sparse_index).__name__ == "LanceSparseProvider":
                existing = len(self.archive_sparse_index.ids)
                if existing > 0:
                    print(f"[TCMM] Skipping archive-index rebuild "
                          f"(LanceDB already has {existing} sparse rows)")
                    return
        except Exception:
            pass

        # Reset indices
        self.archive_vector_index._pending.clear()
        self.archive_vector_index._dirty = True

        self.archive_sparse_index.corpus.clear()
        self.archive_sparse_index.ids.clear()
        self.archive_sparse_index._dirty = True

        texts = []
        ids = []
        overlays = []

        for aid, entry in self.archive.items():

            try:
                aid_int = int(aid)
            except:
                continue

            # VECTOR INDEX
            emb = self.archive_embeddings.get(aid_int)
            if emb is not None:
                self.archive_vector_index.add(aid_int, emb)

            # SPARSE INDEX
            text = self.resolve_text(aid) or ""
            overlay = entry.get("semantic_summary") or ""

            if text:
                texts.append(text)
                ids.append(aid_int)
                overlays.append(overlay)

        # bulk add sparse
        if texts:
            self.archive_sparse_index.add(texts, ids, overlays)

        # Phase 300: Build dedicated claims index
        if hasattr(self, "archive_claims_index") and self.archive_claims_index is not None:
            self.archive_claims_index.corpus.clear() if hasattr(self.archive_claims_index, 'corpus') else None
            self.archive_claims_index.ids.clear() if hasattr(self.archive_claims_index, 'ids') else None
            claims_texts, claims_ids = [], []
            for aid, entry in self.archive.items():
                _cls = entry.get("claims", [])
                if _cls:
                    for c in _cls:
                        if isinstance(c, str) and c.strip():
                            claims_texts.append(c)
                            claims_ids.append(int(aid))
            if claims_texts:
                self.archive_claims_index.add(claims_texts, claims_ids)
            print(f"[TCMM] Claims index built: {len(claims_texts)} claims from {len(set(claims_ids))} nodes")

        # Phase 301: Build dedicated entity index
        # Clear before rebuild to prevent accumulation from per-entry additions
        if hasattr(self, "archive_entity_index") and self.archive_entity_index is not None:
            self.archive_entity_index.corpus.clear() if hasattr(self.archive_entity_index, 'corpus') else None
            self.archive_entity_index.ids.clear() if hasattr(self.archive_entity_index, 'ids') else None
            ent_texts, ent_ids = [], []
            for aid, entry in self.archive.items():
                _ents = entry.get("entities", [])
                if _ents:
                    _et = " ".join(e for e in _ents if isinstance(e, str))
                    if _et.strip():
                        ent_texts.append(_et)
                        ent_ids.append(int(aid))
            if ent_texts:
                self.archive_entity_index.add(ent_texts, ent_ids)
            print(f"[TCMM] Entity index built: {len(ent_texts)} nodes with entities")

        # Phase 19: Build dedicated topic index
        if hasattr(self, "archive_topic_index") and self.archive_topic_index is not None:
            self.archive_topic_index.corpus.clear() if hasattr(self.archive_topic_index, 'corpus') else None
            self.archive_topic_index.ids.clear() if hasattr(self.archive_topic_index, 'ids') else None
            topic_texts, topic_ids = [], []
            for aid, entry in self.archive.items():
                _topics = entry.get("topics", [])
                if _topics:
                    _tt = " ".join(t for t in _topics if isinstance(t, str))
                    if _tt.strip():
                        topic_texts.append(_tt)
                        topic_ids.append(int(aid))
            if topic_texts:
                self.archive_topic_index.add(topic_texts, topic_ids)
            print(f"[TCMM] Topic index built: {len(topic_texts)} nodes with topics")

        # force build FAISS now
        if hasattr(self.archive_vector_index, "_ensure_index"):
            self.archive_vector_index._ensure_index()

        print(f"[TCMM] Index build complete: "
              f"vector={len(self.archive_vector_index._pending)} "
              f"sparse={len(self.archive_sparse_index.ids)}")

        # Save indexes to disk for future fast loads
        self.save_indices()

    def load_from_persistence(self):
        """Loads archive nodes from persistence layer.
        DB-backed providers (SQLite) load directly -- no TinyDB needed.
        Local providers fall back to TinyDB JSON files.
        """
        import os

        # ── DB providers: data already in the provider ──
        _provider_name = type(self.archive).__name__
        if _provider_name in ("SQLiteStorageProvider", "LanceStorageProvider"):
            # DB providers: nodes read on demand via __getitem__.
            # Walk with .items() (ONE query) instead of keys()+__getitem__
            # (N+1 queries — was ~650 round-trips for a 650-node bench).
            loaded = 0
            max_id = 0
            max_step = 0
            for aid, node in self.archive.items():
                if aid > max_id:
                    max_id = aid
                step = node.get("created_step", 0)
                if step > max_step:
                    max_step = step
                text = node.get("text", "")
                if text:
                    ch = self._calculate_canonical_hash(text)
                    self.canonical_hash_index[ch] = aid
                    fp = node.get("fingerprint")
                    if fp:
                        self.fingerprint_index[fp] = aid
                loaded += 1

            # Embeddings live in the provider and are read live via
            # EmbeddingProxy.get(). The previous code did
            #     archive_embs = provider.get_all_embeddings(...)
            #     self.archive_embeddings.update(archive_embs)
            # which for LanceDB does a read-then-write-back round-trip per
            # row (N UPDATE calls) — a pure O(N) waste spiking CPU on init.
            # Skip; the proxy already points at the source of truth.

            if loaded > 0:
                try:
                    self.archive.set_id("archive", max_id + 1)
                except (NotImplementedError, AttributeError):
                    pass
                self.next_archive_id = max(self.next_archive_id, max_id + 1)
                self.current_step = max(self.current_step, max_step)
                print(f"[TCMM] Loaded {loaded} nodes from {_provider_name} (max_id={max_id})")
            return

        # ── Local provider: fall back to TinyDB ──
        try:
            from tinydb import TinyDB
        except ImportError:
            _log_warn("TinyDB not available, skipping load.")
            return

        # Fix: User specified data/archive.json at root
        db_dir = self._data_dir
        # Ensure directory exists for future saves if we load nothing now
        if not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
            except OSError:
                pass

        db_path = os.path.join(db_dir, "archive.json")
        emb_db_path = os.path.join(db_dir, "archive_embeddings.json")
        idx_dir = os.path.join(db_dir, "indices")

        if not os.path.exists(db_path):
            return

        # Check upfront if cached indexes exist — if so, skip incremental adds during node loading
        _have_idx_cache = (
            os.path.exists(os.path.join(idx_dir, "vector", "vec_meta.json")) and
            os.path.exists(os.path.join(idx_dir, "vector", "faiss.idx")) and
            os.path.exists(os.path.join(idx_dir, "sparse", "corpus.json")) and
            os.path.exists(os.path.join(idx_dir, "claims", "corpus.json")) and
            os.path.exists(os.path.join(idx_dir, "entity", "corpus.json"))
        )

        try:
            db = TinyDB(db_path)

            # Load embedding sidecar if it exists
            emb_db = None
            if os.path.exists(emb_db_path):
                emb_db = TinyDB(emb_db_path)
                if hasattr(self, "archive_embeddings"):
                    for emb_doc in emb_db.all():
                        if "embedding" in emb_doc:
                            self.archive_embeddings[emb_doc.doc_id] = emb_doc["embedding"]

            # Load topic embedding sidecar if it exists
            topic_emb_db_path = os.path.join(db_dir, "topic_embeddings.json")
            if os.path.exists(topic_emb_db_path):
                topic_emb_db = TinyDB(topic_emb_db_path)
                if hasattr(self, "topic_embeddings"):
                    for temb_doc in topic_emb_db.all():
                        if "topic_embedding" in temb_doc:
                            self.topic_embeddings[temb_doc.doc_id] = temb_doc["topic_embedding"]

            docs = db.all()
            
            loaded_count = 0
            max_id = 0
            max_step = 0
            
            # Lazy import for reconstruction
            from .entropy_guard import EntropyDiagnostics
            
            for doc in docs:
                aid = doc.get("id")
                if aid is None: continue
                
                # Ensure integer ID
                try:
                    aid_int = int(aid)
                    # If stored as string, convert in memory for consistency
                    if aid != aid_int:
                        doc["id"] = aid_int
                    aid = aid_int
                except (ValueError, TypeError):
                    continue

                # TCMM FIX: reject invalid archive IDs
                if not isinstance(aid, int):
                    raise TypeError(f"Invalid archive ID type: {type(aid)}")

                if aid < 1:
                    raise ValueError(f"Invalid archive ID {aid}: IDs must start at 1")

                # Step Persistence (Cooling System)
                created_s = doc.get("created_step", 0)
                used_s = doc.get("last_used_step", 0)
                max_step = max(max_step, created_s, used_s)

                # Fix: Sanitize Graph Keys (String -> Int)
                # 1. Contextual Links
                if "contextual_links" in doc and doc["contextual_links"]:
                    sanitized = {}
                    for k, v in doc["contextual_links"].items():
                        try:
                            sanitized[int(k)] = v
                        except (ValueError, TypeError):
                            sanitized[k] = v # Keep as is if not int
                    doc["contextual_links"] = sanitized

                # 2. Behavioural Links
                if "behavioural" in doc:
                    beh = doc["behavioural"]
                    for tier in ["hot", "cold", "ghost"]:
                        if tier in beh and beh[tier]:
                            sanitized_tier = {}
                            for k, v in beh[tier].items():
                                try:
                                    sanitized_tier[int(k)] = v
                                except (ValueError, TypeError):
                                    sanitized_tier[k] = v
                            beh[tier] = sanitized_tier

                # 3. Suppresses
                if "suppresses" in doc and doc["suppresses"]:
                    sanitized = {}
                    for k, v in doc["suppresses"].items():
                        try:
                            sanitized[int(k)] = v
                        except (ValueError, TypeError):
                            sanitized[k] = v
                    doc["suppresses"] = sanitized

                # FIX: normalize internal references (semantic_links, temporal, lineage)
                
                # 4. Semantic Links
                if "semantic_links" in doc and doc["semantic_links"]:
                    doc["semantic_links"] = {
                        int(k): float(v)
                        for k, v in doc["semantic_links"].items()
                        if k is not None
                    }

                # 5. Temporal
                temporal = doc.get("temporal")
                if temporal:
                    if "prev_aid" in temporal and temporal["prev_aid"] is not None:
                        try:
                            temporal["prev_aid"] = int(temporal["prev_aid"])
                        except (ValueError, TypeError): pass
                    if "next_aid" in temporal and temporal["next_aid"] is not None:
                        try:
                            temporal["next_aid"] = int(temporal["next_aid"])
                        except (ValueError, TypeError): pass

                # 6. Lineage
                if "lineage_root" in doc and doc["lineage_root"] is not None:
                    try:
                        doc["lineage_root"] = int(doc["lineage_root"])
                    except (ValueError, TypeError): pass
                
                lineage = doc.get("lineage", {})
                if isinstance(lineage, dict):
                    if lineage.get("root") is not None:
                        try:
                            lineage["root"] = int(lineage["root"])
                        except (ValueError, TypeError): pass
                    if lineage.get("parents"):
                        try:
                            lineage["parents"] = [int(x) for x in lineage["parents"]]
                        except (ValueError, TypeError): pass

                # Reconstruct EntropyDiagnostics if present (as dict)
                if "entropy_static" in doc and isinstance(doc["entropy_static"], dict):
                    try:
                        doc["entropy_static"] = EntropyDiagnostics(**doc["entropy_static"])
                    except Exception as e:
                        _log_warn(f"Failed to reconstruct EntropyDiagnostics for aid={aid}: {e}")
                        doc["entropy_static"] = None

                # Inject into archive
                self.archive[aid] = doc

                # Rebuild content_hash_cache for canonical dedup on restore.
                if hasattr(self, "content_hash_cache") and doc.get("content_hash"):
                    self.content_hash_cache[doc["content_hash"]] = aid

                # PATCH 1 — Hydrate embeddings on archive load
                if hasattr(self, "archive_embeddings"):
                    emb = self.archive_embeddings.get(aid)
                    if emb is not None:
                        doc["embedding"] = emb

                # Hydrate topic embeddings on archive load
                if hasattr(self, "topic_embeddings"):
                    temb = self.topic_embeddings.get(aid)
                    if temb is not None:
                        doc["topic_embedding"] = temb

                if aid > max_id:
                    max_id = aid
                
                loaded_count += 1
                
                # Populate Dedup Map (Level 1: Canonical)
                if hasattr(self, "canonical_hash_index") and "text" in doc:
                    key = self._calculate_canonical_hash(self.resolve_text(doc.get("id")))
                    self.canonical_hash_index[key] = aid

                # Populate Dedup Map (Legacy SHA-1)
                if hasattr(self, "_archive_dedup") and hasattr(self, "_dedup_key"):
                    if "text" in doc:
                        key = self._dedup_key(self.resolve_text(doc.get("id")))
                        self._archive_dedup[key] = aid
                
                # Re-index (skip if we'll load from cached indexes — avoids dirtying them)
                if not _have_idx_cache:
                    if hasattr(self, "archive_vector_index") and self.archive_vector_index and doc.get("embedding"):
                        try:
                            self.archive_vector_index.add(aid, doc["embedding"])
                            if hasattr(self, "archive_embeddings"):
                                self.archive_embeddings[aid] = doc["embedding"]
                        except Exception: pass

                    if hasattr(self, "archive_sparse_index") and self.archive_sparse_index:
                        try:
                            text_to_index = self.resolve_text(doc.get("id"))
                            if text_to_index:
                                self.archive_sparse_index.add([text_to_index], [aid])
                        except Exception: pass
                     
            # Restore counter
            # TCMM FIX: enforce minimum ID of 1
            # Sync sequence counter to DB provider if supported
            try:
                self.archive.set_id("archive", max(1, max_id + 1))
            except (NotImplementedError, AttributeError):
                pass
            if hasattr(self, "next_archive_id"):
                self.next_archive_id = max(1, max_id + 1)
            
            # Restore Step (Cooling Fix)
            if hasattr(self, "current_step"):
                if max_step > self.current_step:
                    self.current_step = max_step + 1
            
            log_archive(f"[PERSISTENCE] Loaded {loaded_count} archive nodes from disk. Restored step={self.current_step}.")

            # Close TinyDB handles to free memory and file descriptors
            try:
                db.close()
            except Exception:
                pass
            if emb_db is not None:
                try:
                    emb_db.close()
                except Exception:
                    pass

        except Exception as e:
            log_exception("Failed to load persistence DB", e)

        # Try to load persisted indexes instead of rebuilding
        indexes_loaded = False
        if os.path.exists(idx_dir):
            if hasattr(self, "archive_vector_index"):
                vec_ok = self.archive_vector_index.load(
                    os.path.join(idx_dir, "vector"),
                    archive_embeddings=getattr(self, "archive_embeddings", None)
                )
            else:
                vec_ok = False
            if hasattr(self, "archive_sparse_index"):
                sparse_ok = self.archive_sparse_index.load(os.path.join(idx_dir, "sparse"))
            else:
                sparse_ok = False
            if hasattr(self, "archive_claims_index"):
                claims_ok = self.archive_claims_index.load(os.path.join(idx_dir, "claims"))
            else:
                claims_ok = False
            if hasattr(self, "archive_entity_index"):
                entity_ok = self.archive_entity_index.load(os.path.join(idx_dir, "entity"))
            else:
                entity_ok = False
            if hasattr(self, "archive_topic_index"):
                topic_ok = self.archive_topic_index.load(os.path.join(idx_dir, "topic"))
            else:
                topic_ok = False

            if vec_ok and sparse_ok and claims_ok and entity_ok and topic_ok:
                log_archive("[PERSISTENCE] Loaded all indexes from disk (skipped rebuild)")
                indexes_loaded = True

        # If index load failed, rebuild from archive
        if not indexes_loaded:
            self._build_indices_from_archive()

# Phase 10: Archive Scaling
class ShardedArchive(dict):
    """
    Transparently sharded dictionary for Archive Nodes.
    Buckets nodes by ID range to avoid monolithic memory pressure.
    behaving like a dict so existing code doesn't break.
    """
    SHARD_SIZE = 5000

    def __init__(self, *args, **kwargs):
        self.shards = {} # { shard_id: dict }
        self._count = 0
        self.update(*args, **kwargs)

    def _get_shard_key(self, key):
        if isinstance(key, int):
            return key // self.SHARD_SIZE
        # Handle string keys if they exist (though generic Archive IDs are ints)
        # Verify if we have non-int keys. 
        # tcmm_core: self.archive[aid] = entry. aid is int.
        # But we might have other keys? 
        # "active_topic_id" is unrelated.
        # Assuming int keys for archive nodes.
        try:
            return int(key) // self.SHARD_SIZE
        except (ValueError, TypeError):
            return "misc"

    def _get_shard(self, key, create=False):
        s_key = self._get_shard_key(key)
        if s_key not in self.shards:
            if create:
                self.shards[s_key] = {}
            else:
                return None
        return self.shards[s_key]

    def __setitem__(self, key, value):
        shard = self._get_shard(key, create=True)
        if key not in shard:
            self._count += 1
        shard[key] = value

    def __getitem__(self, key):
        shard = self._get_shard(key)
        if shard is None or key not in shard:
            raise KeyError(key)
        return shard[key]

    def __delitem__(self, key):
        shard = self._get_shard(key)
        if shard is None or key not in shard:
            raise KeyError(key)
        del shard[key]
        self._count -= 1
        # Cleanup empty shard?
        # if not shard: del self.shards[self._get_shard_key(key)]

    def __contains__(self, key):
        shard = self._get_shard(key)
        return shard is not None and key in shard

    def __len__(self):
        return self._count

    def __iter__(self):
        for shard in self.shards.values():
            yield from shard

    def keys(self):
        for shard in self.shards.values():
            yield from shard.keys()

    def values(self):
        for shard in self.shards.values():
            yield from shard.values()

    def items(self):
        for shard in self.shards.values():
            yield from shard.items()

    def get(self, key, default=None):
        shard = self._get_shard(key)
        if shard is None:
            return default
        return shard.get(key, default)
    
    def clear(self):
        self.shards.clear()
        self._count = 0
    
    def pop(self, key, default=object()):
        shard = self._get_shard(key)
        if shard is None or key not in shard:
            if default is object():
                raise KeyError(key)
            return default
        self._count -= 1
        return shard.pop(key)

    def setdefault(self, key, default=None):
        shard = self._get_shard(key, create=True)
        if key not in shard:
             self._count += 1
        return shard.setdefault(key, default)
    
    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError("update expected at most 1 arguments, "
                                "got %d" % len(args))
            other = args[0]
            if isinstance(other, dict):
                for k, v in other.items():
                    self[k] = v
            elif hasattr(other, 'keys'):
                for k in other.keys():
                    self[k] = other[k]
            else:
                for k, v in other:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def __repr__(self):
        return f"<ShardedArchive count={self._count} shards={len(self.shards)}>"
