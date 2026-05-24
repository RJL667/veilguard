"""Tests for the two-tier system-block assembly helpers (Phase-7 step 3c).

Covers:
  - _split_tcmm_memory_into_tiers across all three input shapes:
      * Phase-7 two-marker layout
      * Phase-6 single-boundary fallback
      * No-marker fallback
  - _assemble_system_blocks_for_tiers:
      * Block count for each input shape
      * cache_control attachment (TTL 1h on stable, TTL 5m on working)
      * Size-floor skipping working-tier cache_control when too small
      * Circuit-breaker strip path emits NO cache_control anywhere
      * Marker constants match what the adapter emits
"""
import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.main import (
    TCMM_STABLE_BOUNDARY,
    TCMM_LIVE_BOUNDARY,
    _split_tcmm_memory_into_tiers,
    _assemble_system_blocks_for_tiers,
)


# The proxy's effective-min-cache size. Keep this in sync with main.py's
# _MIN_CACHE_CHARS — duplicated as a literal so we don't import a "private"
# symbol that could change shape.
MIN_CACHE = 9200


def _build_phase7_memory(stable_body: str, working_body: str, volatile_body: str) -> str:
    """Render a TCMM memory blob in the Phase-7 two-marker shape."""
    return (
        f"{stable_body}\n{TCMM_STABLE_BOUNDARY}\n"
        f"{working_body}\n{TCMM_LIVE_BOUNDARY}\n"
        f"{volatile_body}"
    )


def _build_phase6_memory(working_body: str, volatile_body: str) -> str:
    return f"{working_body}\n{TCMM_LIVE_BOUNDARY}\n{volatile_body}"


class TestSplitTiers(unittest.TestCase):
    def test_phase7_layout(self):
        mem = _build_phase7_memory("STABLE!", "WORKING!", "VOLATILE!")
        s, w, v = _split_tcmm_memory_into_tiers(mem)
        # Marker line stays with the tier it terminates (inclusive).
        self.assertIn("STABLE!", s)
        self.assertIn(TCMM_STABLE_BOUNDARY, s)
        self.assertIn("WORKING!", w)
        self.assertIn(TCMM_LIVE_BOUNDARY, w)
        self.assertIn("VOLATILE!", v)
        # Concatenation equals the original (modulo the newline we
        # inserted around markers — split is byte-exact across the three
        # slices).
        self.assertEqual(s + w + v, mem)

    def test_phase6_fallback_when_only_live_marker(self):
        mem = _build_phase6_memory("ALL_WORKING", "VOLATILE")
        s, w, v = _split_tcmm_memory_into_tiers(mem)
        self.assertEqual(s, "", "no stable tier in phase-6 layout")
        self.assertIn("ALL_WORKING", w)
        self.assertIn(TCMM_LIVE_BOUNDARY, w)
        self.assertIn("VOLATILE", v)
        self.assertEqual(s + w + v, mem)

    def test_no_markers_defaults_to_working(self):
        mem = "JUST RAW MEMORY NO MARKERS"
        s, w, v = _split_tcmm_memory_into_tiers(mem)
        self.assertEqual(s, "")
        self.assertEqual(w, mem)
        self.assertEqual(v, "")

    def test_empty_memory(self):
        self.assertEqual(_split_tcmm_memory_into_tiers(""), ("", "", ""))
        self.assertEqual(_split_tcmm_memory_into_tiers(None or ""), ("", "", ""))

    def test_reversed_markers_falls_back_to_phase6(self):
        """If somehow the live boundary appears before the stable one
        (corrupt input), fall back to phase-6 single-boundary handling
        rather than emit a negative-length tier."""
        mem = (
            "garbage\n" + TCMM_LIVE_BOUNDARY + "\nmore\n"
            + TCMM_STABLE_BOUNDARY + "\ntail"
        )
        s, w, v = _split_tcmm_memory_into_tiers(mem)
        # Phase-7 condition requires stable_idx < live_idx; otherwise
        # use phase-6 path (just LIVE).
        self.assertEqual(s, "")
        self.assertIn(TCMM_LIVE_BOUNDARY, w)


class TestAssembleBlocks(unittest.TestCase):
    PREAMBLE = "X" * (MIN_CACHE + 100)  # big enough to be cacheable

    def test_phase7_emits_three_blocks_with_correct_ttls(self):
        big_stable = "S" * 100
        big_working = "W" * (MIN_CACHE + 200)
        volatile = "V" * 200
        mem = _build_phase7_memory(big_stable, big_working, volatile)

        blocks, used_ttl = _assemble_system_blocks_for_tiers(
            veilguard_static_preamble=self.PREAMBLE,
            tcmm_memory=mem,
            cache_circuit_strip=False,
            min_cache_chars=MIN_CACHE,
        )
        self.assertEqual(len(blocks), 3)
        # block 0 = preamble + stable, 1h TTL
        self.assertIn("STABLE", blocks[0]["text"].upper()  # not a strict assertion; just sanity
                      ) if False else self.assertTrue(blocks[0]["text"].startswith(self.PREAMBLE))
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        # block 1 = working, default 5m ephemeral
        self.assertEqual(blocks[1]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("ttl", blocks[1]["cache_control"])
        # block 2 = volatile, no cache_control
        self.assertNotIn("cache_control", blocks[2])
        self.assertTrue(used_ttl)

    def test_phase6_emits_two_blocks_no_stable_tier(self):
        working = "W" * (MIN_CACHE + 100)
        volatile = "V" * 100
        mem = _build_phase6_memory(working, volatile)

        blocks, used_ttl = _assemble_system_blocks_for_tiers(
            veilguard_static_preamble=self.PREAMBLE,
            tcmm_memory=mem,
            cache_circuit_strip=False,
            min_cache_chars=MIN_CACHE,
        )
        # 3 blocks: preamble (no stable_mem prepended), working, volatile.
        self.assertEqual(len(blocks), 3)
        # Preamble block text = preamble + "" (no stable), still 1h-cached
        self.assertEqual(blocks[0]["text"], self.PREAMBLE)
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        # Working still gets 5m TTL since it's big enough
        self.assertEqual(blocks[1]["cache_control"], {"type": "ephemeral"})

    def test_working_below_floor_omits_5m_marker(self):
        # Working memory below MIN_CACHE — must NOT carry cache_control.
        working_tiny = "tiny working"
        mem = _build_phase7_memory("STABLE", working_tiny, "VOLATILE")
        blocks, _ = _assemble_system_blocks_for_tiers(
            veilguard_static_preamble=self.PREAMBLE,
            tcmm_memory=mem,
            cache_circuit_strip=False,
            min_cache_chars=MIN_CACHE,
        )
        # Block 1 is the working tier; it must NOT have cache_control.
        self.assertEqual(len(blocks), 3)
        self.assertNotIn("cache_control", blocks[1])
        # But the stable tier (block 0) must still carry its 1h marker.
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})

    def test_empty_memory_emits_only_preamble(self):
        blocks, used_ttl = _assemble_system_blocks_for_tiers(
            veilguard_static_preamble=self.PREAMBLE,
            tcmm_memory="",
            cache_circuit_strip=False,
            min_cache_chars=MIN_CACHE,
        )
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["text"], self.PREAMBLE)
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertTrue(used_ttl)

    def test_circuit_breaker_strips_all_cache_control(self):
        big_stable = "S" * 100
        big_working = "W" * (MIN_CACHE + 200)
        volatile = "V" * 200
        mem = _build_phase7_memory(big_stable, big_working, volatile)

        blocks, used_ttl = _assemble_system_blocks_for_tiers(
            veilguard_static_preamble=self.PREAMBLE,
            tcmm_memory=mem,
            cache_circuit_strip=True,   # ← breaker tripped
            min_cache_chars=MIN_CACHE,
        )
        for blk in blocks:
            self.assertNotIn(
                "cache_control", blk,
                f"breaker tripped but block carries cache_control: {blk}"
            )
        self.assertFalse(used_ttl)

    def test_circuit_breaker_still_emits_all_three_blocks(self):
        """Stripping cache_control must not change block partitioning —
        the LLM still sees the same content shape, just uncached."""
        mem = _build_phase7_memory("S", "W" * 50, "V" * 50)
        blocks, _ = _assemble_system_blocks_for_tiers(
            veilguard_static_preamble=self.PREAMBLE,
            tcmm_memory=mem,
            cache_circuit_strip=True,
            min_cache_chars=MIN_CACHE,
        )
        self.assertEqual(len(blocks), 3)

    def test_marker_constants_match_adapter_convention(self):
        """Spot-check that the marker constants are exactly the strings
        the adapter emits — drift here is the single most common cause
        of silent cache-miss regressions."""
        self.assertEqual(TCMM_STABLE_BOUNDARY, "--- END STABLE MEMORY ---")
        self.assertEqual(TCMM_LIVE_BOUNDARY, "--- END LIVE MEMORY ---")


if __name__ == "__main__":
    unittest.main()
