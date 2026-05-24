"""Phase 7 step 2 + 6 cache metrics tests.

Covers:
  - record_request / record_from_usage roll up correctly per tenant.
  - get_tenant_stats math (hit_rate, write_rate, write_amplification,
    effective_token_multiplier).
  - is_cache_thrashing decision logic across edge cases:
      * insufficient samples -> no trip
      * good hit rate -> no trip
      * low hit rate + high amp -> trips
      * cache_read=0 + any writes -> infinite amp -> trips (when n large)
      * windowed query honours window_seconds.
"""
import time
import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app import cache_metrics


class _BaseTest(unittest.TestCase):
    def setUp(self):
        cache_metrics.reset()  # clear all tenants


class TestRecordAndAggregate(_BaseTest):
    def test_record_appends_sample(self):
        cache_metrics.record_request(
            tenant_id="alice",
            conv_id="c1",
            cache_create=100,
            cache_read=900,
            input_tokens=50,
            output_tokens=200,
        )
        s = cache_metrics.get_tenant_stats("alice")
        self.assertEqual(s["n_samples"], 1)
        self.assertEqual(s["cache_create_total"], 100)
        self.assertEqual(s["cache_read_total"], 900)

    def test_aggregation_across_samples(self):
        for _ in range(3):
            cache_metrics.record_request(
                tenant_id="bob",
                conv_id="c2",
                cache_create=100,
                cache_read=900,
                input_tokens=50,
                output_tokens=200,
            )
        s = cache_metrics.get_tenant_stats("bob")
        self.assertEqual(s["n_samples"], 3)
        self.assertEqual(s["cache_create_total"], 300)
        self.assertEqual(s["cache_read_total"], 2700)
        # hit_rate = 2700 / (300+2700+150) = 2700/3150 ≈ 0.857
        self.assertAlmostEqual(s["hit_rate"], 2700 / 3150, places=3)

    def test_write_amplification(self):
        cache_metrics.record_request(
            tenant_id="carol",
            conv_id="c3",
            cache_create=200,
            cache_read=100,
            input_tokens=0,
            output_tokens=0,
        )
        s = cache_metrics.get_tenant_stats("carol")
        self.assertAlmostEqual(s["write_amplification"], 2.0, places=3)

    def test_zero_reads_returns_none_amp(self):
        cache_metrics.record_request(
            tenant_id="dan",
            conv_id="c4",
            cache_create=100,
            cache_read=0,
            input_tokens=50,
        )
        s = cache_metrics.get_tenant_stats("dan")
        self.assertIsNone(s["write_amplification"])

    def test_empty_tenant_returns_zero_stats(self):
        s = cache_metrics.get_tenant_stats("nobody")
        self.assertEqual(s["n_samples"], 0)
        self.assertIsNone(s["hit_rate"])

    def test_record_from_usage_extracts_correctly(self):
        usage = {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_creation_input_tokens": 500,
            "cache_read_input_tokens": 1500,
        }
        cache_metrics.record_from_usage(usage, tenant_id="erin", conv_id="c5")
        s = cache_metrics.get_tenant_stats("erin")
        self.assertEqual(s["n_samples"], 1)
        self.assertEqual(s["cache_create_total"], 500)
        self.assertEqual(s["cache_read_total"], 1500)
        self.assertEqual(s["input_tokens_total"], 10)

    def test_record_from_usage_handles_none(self):
        cache_metrics.record_from_usage(None, tenant_id="frank", conv_id="c6")
        s = cache_metrics.get_tenant_stats("frank")
        self.assertEqual(s["n_samples"], 0)

    def test_window_filter(self):
        # Insert old sample
        cache_metrics.record_request(tenant_id="gwen", conv_id="x",
                                     cache_create=100, cache_read=0)
        # Mutate timestamp to be 2 hours ago
        with cache_metrics._state_lock:
            cache_metrics._state["gwen"][0]["ts"] = time.time() - 7200
        # Recent sample
        cache_metrics.record_request(tenant_id="gwen", conv_id="x",
                                     cache_create=10, cache_read=90)

        # Full window: both samples
        full = cache_metrics.get_tenant_stats("gwen")
        self.assertEqual(full["n_samples"], 2)
        # 1-hour window: only recent
        recent = cache_metrics.get_tenant_stats("gwen", window_seconds=3600)
        self.assertEqual(recent["n_samples"], 1)
        self.assertEqual(recent["cache_create_total"], 10)


class TestCircuitBreaker(_BaseTest):
    def _flood(self, tenant_id, n, *, cache_create=0, cache_read=0, input_tokens=100):
        for _ in range(n):
            cache_metrics.record_request(
                tenant_id=tenant_id,
                conv_id="c",
                cache_create=cache_create,
                cache_read=cache_read,
                input_tokens=input_tokens,
            )

    def test_insufficient_samples_no_trip(self):
        self._flood("a", 5, cache_create=1000, cache_read=0)
        trip, why = cache_metrics.is_cache_thrashing("a")
        self.assertFalse(trip)
        self.assertEqual(why["reason"], "insufficient_samples")

    def test_healthy_cache_no_trip(self):
        # 50 samples with strong hit rate.
        self._flood("b", 50, cache_create=100, cache_read=9000, input_tokens=10)
        trip, why = cache_metrics.is_cache_thrashing("b")
        self.assertFalse(trip)
        self.assertEqual(why["reason"], "ok")
        # Hit rate ≈ 9000 / 9110 ≈ 0.988
        self.assertGreater(why["hit_rate"], 0.9)

    def test_thrashing_pattern_trips(self):
        # 50 samples with many writes, few reads, lots of full-priced input.
        # hit_rate ≈ 50 / (1000+50+5000) = 50/6050 ≈ 0.008 (well below floor)
        # write_amp = 1000 / 50 = 20.0 (well above ceiling)
        self._flood("c", 50, cache_create=1000, cache_read=50, input_tokens=5000)
        trip, why = cache_metrics.is_cache_thrashing("c")
        self.assertTrue(trip)
        self.assertEqual(why["reason"], "cache_thrash")

    def test_zero_reads_with_writes_trips_when_n_sufficient(self):
        self._flood("d", 50, cache_create=1000, cache_read=0, input_tokens=100)
        trip, why = cache_metrics.is_cache_thrashing("d")
        self.assertTrue(trip)
        self.assertEqual(why["write_amplification"], "inf")

    def test_high_hit_rate_does_not_trip_even_with_some_writes(self):
        # 50 samples, write_amp = 0.5 (1 write per 2 reads), hit ≈ 0.66.
        # Both signals required to trip — this should NOT trip.
        self._flood("e", 50, cache_create=500, cache_read=1000, input_tokens=0)
        trip, why = cache_metrics.is_cache_thrashing("e")
        self.assertFalse(trip)

    def test_low_hit_rate_alone_does_not_trip_if_amp_ok(self):
        # Engineered: hit_rate is low, but write_amp is also low — means
        # there's just no caching activity (low cache_create), not active
        # thrash. Most cost is uncached input.
        # cache_create=10, cache_read=100, input=10000:
        # hit_rate = 100/10110 ≈ 0.0099 (below floor)
        # write_amp = 10/100 = 0.1 (well below ceiling)
        # Should NOT trip — AND gate.
        self._flood("f", 50, cache_create=10, cache_read=100, input_tokens=10000)
        trip, why = cache_metrics.is_cache_thrashing("f")
        self.assertFalse(trip)

    def test_breaker_recovers_when_pattern_improves(self):
        # Flood with bad samples first
        self._flood("g", 50, cache_create=1000, cache_read=0, input_tokens=100)
        trip, _ = cache_metrics.is_cache_thrashing("g")
        self.assertTrue(trip)
        # Now add a wave of good samples — the rolling window is 1000 by
        # default, so the bad samples stay in the calculation. We need
        # MANY more good samples to dominate.
        self._flood("g", 1000, cache_create=10, cache_read=1000, input_tokens=10)
        trip, why = cache_metrics.is_cache_thrashing("g")
        self.assertFalse(trip, f"Should recover with 1000 good samples; got {why}")


class TestListTenants(_BaseTest):
    def test_list_tenants(self):
        cache_metrics.record_request(tenant_id="t1", conv_id="x")
        cache_metrics.record_request(tenant_id="t2", conv_id="x")
        tenants = cache_metrics.list_tenants()
        self.assertIn("t1", tenants)
        self.assertIn("t2", tenants)

    def test_get_all_stats(self):
        cache_metrics.record_request(tenant_id="t1", conv_id="x",
                                     cache_create=100, cache_read=900)
        cache_metrics.record_request(tenant_id="t2", conv_id="x",
                                     cache_create=200, cache_read=800)
        all_stats = cache_metrics.get_all_stats()
        ids = {s["tenant_id"] for s in all_stats}
        self.assertEqual(ids, {"t1", "t2"})


if __name__ == "__main__":
    unittest.main()
