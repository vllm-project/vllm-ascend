"""Unit tests for batch_job_aware_scheduler.py.

Covers all public and internal components:

    - ``_cdiv``
    - ``_JobStat``
    - ``JobNameParser``
    - ``JobDecodeEstimator``
    - ``RequestBucket``
    - ``RunningBlockReserver``
    - ``BatchJobAwareRequestQueue``
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, PropertyMock, patch

from vllm.v1.request import Request

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import BatchJobSchedConfig
from vllm_ascend.core.batch_job_aware_scheduler import (
    _JobStat,
    _cdiv,
    BatchJobAwareRequestQueue,
    JobDecodeEstimator,
    JobNameParser,
    RequestBucket,
    RunningBlockReserver,
)


# ===================================================================
# _cdiv
# ===================================================================


class TestCdiv(TestBase):
    """Ceiling division helper tests."""

    def test_exact_division(self):
        self.assertEqual(_cdiv(10, 5), 2)

    def test_rounds_up(self):
        self.assertEqual(_cdiv(11, 5), 3)

    def test_single_block(self):
        self.assertEqual(_cdiv(1, 16), 1)

    def test_zero_numerator(self):
        self.assertEqual(_cdiv(0, 16), 0)

    def test_numerator_equals_denominator(self):
        self.assertEqual(_cdiv(16, 16), 1)

    def test_large_values(self):
        self.assertEqual(_cdiv(1000000, 16), 62500)


# ===================================================================
# _JobStat
# ===================================================================


class TestJobStat(TestBase):
    """Per‑job running statistics container tests."""

    def test_initial_values(self):
        stat = _JobStat()
        self.assertEqual(stat.sample_count, 0)
        self.assertEqual(stat.cumulative_sum, 0)
        self.assertEqual(stat.ewma_average, 0.0)

    def test_after_observations(self):
        stat = _JobStat()
        stat.sample_count = 3
        stat.cumulative_sum = 350
        stat.ewma_average = 116.5
        self.assertEqual(stat.sample_count, 3)
        self.assertEqual(stat.cumulative_sum, 350)
        self.assertEqual(stat.ewma_average, 116.5)


# ===================================================================
# JobNameParser
# ===================================================================


class TestJobNameParser(TestBase):
    """Job name extraction with caching."""

    def setUp(self):
        super().setUp()
        self.parser = JobNameParser()

    def test_parse_with_job_name(self):
        result = self.parser.parse("req1#job_name[my_job]#")
        self.assertEqual(result, "my_job")

    def test_parse_without_job_name_returns_default(self):
        result = self.parser.parse("req_no_job")
        self.assertEqual(result, "__default__")

    def test_parse_with_empty_job_name(self):
        result = self.parser.parse("req1#job_name[]#")
        self.assertEqual(result, "__default__")

    def test_parse_caches_result(self):
        first = self.parser.parse("req1#job_name[job1]#")
        second = self.parser.parse("req1#job_name[job1]#")
        self.assertEqual(first, second)
        self.assertIn("req1#job_name[job1]#", self.parser._cache)

    def test_remove_clears_cache(self):
        self.parser.parse("req1#job_name[job1]#")
        self.parser.remove("req1#job_name[job1]#")
        self.assertNotIn("req1#job_name[job1]#", self.parser._cache)

    def test_remove_non_existent_does_not_raise(self):
        # Should not raise KeyError or other exceptions
        self.parser.remove("non_existent_req")

    def test_clear_empties_cache(self):
        self.parser.parse("req1#job_name[job1]#")
        self.parser.parse("req2#job_name[job2]#")
        self.parser.clear()
        self.assertEqual(len(self.parser._cache), 0)

    def test_multiple_jobs_cached_correctly(self):
        j1 = self.parser.parse("r1#job_name[alpha]#")
        j2 = self.parser.parse("r2#job_name[beta]#")
        self.assertEqual(j1, "alpha")
        self.assertEqual(j2, "beta")


# ===================================================================
# JobDecodeEstimator
# ===================================================================


class TestJobDecodeEstimator(TestBase):
    """EWMA + Bayesian cold‑start decode length estimator tests."""

    def setUp(self):
        super().setUp()
        self.config = BatchJobSchedConfig()
        self.estimator = JobDecodeEstimator(self.config)

    def test_predict_no_samples_returns_default(self):
        result = self.estimator.predict("unknown_job")
        self.assertEqual(result, self.config.cold_start_default_decode)

    def test_observe_creates_job_stat_on_first_call(self):
        self.estimator.observe("job1", 100)
        self.assertIn("job1", self.estimator._job_stats)
        self.assertEqual(self.estimator._job_stats["job1"].sample_count, 1)
        self.assertEqual(self.estimator._job_stats["job1"].cumulative_sum, 100)
        self.assertEqual(self.estimator._job_stats["job1"].ewma_average, 100.0)

    def test_cold_start_predict_uses_prior_shrinkage(self):
        """With samples < cold_start_min_samples, Bayesian shrinkage is used."""
        cfg = BatchJobSchedConfig({
            "cold_start_default_decode": 128,
            "cold_start_prior_weight": 2.0,
            "cold_start_min_samples": 3,
        })
        estimator = JobDecodeEstimator(cfg)
        estimator.observe("job1", 100)
        estimator.observe("job1", 150)
        # L_pred = (2.0 * 128 + 250) / (2.0 + 2) = 506 / 4 = 126.5 ≈ 126
        result = estimator.predict("job1")
        self.assertEqual(result, 126)

    def test_stable_phase_predict_uses_ewma(self):
        """When samples >= cold_start_min_samples, pure EWMA is used."""
        cfg = BatchJobSchedConfig({
            "cold_start_min_samples": 2,
            "ewma_alpha": 0.3,
        })
        estimator = JobDecodeEstimator(cfg)
        # n=1: EWMA = 100
        estimator.observe("job1", 100)
        # n=2: EWMA = 0.3 * 150 + 0.7 * 100 = 115
        estimator.observe("job1", 150)
        result = estimator.predict("job1")
        self.assertEqual(result, 115)

    def test_ewma_convergence(self):
        """EWMA with alpha=1.0 should track the latest observation exactly."""
        cfg = BatchJobSchedConfig({
            "cold_start_min_samples": 1,
            "ewma_alpha": 1.0,
        })
        estimator = JobDecodeEstimator(cfg)
        estimator.observe("job1", 100)
        estimator.observe("job1", 200)
        result = estimator.predict("job1")
        self.assertEqual(result, 200)

    def test_observe_max_jobs_exceeded_raises(self):
        cfg = BatchJobSchedConfig({"max_jobs": 2})
        estimator = JobDecodeEstimator(cfg)
        estimator.observe("job1", 100)
        estimator.observe("job2", 100)
        with self.assertRaises(RuntimeError) as ctx:
            estimator.observe("job3", 100)
        self.assertIn("Maximum number of jobs", str(ctx.exception))

    def test_observe_max_jobs_zero_allows_unlimited(self):
        cfg = BatchJobSchedConfig({"max_jobs": 51})
        estimator = JobDecodeEstimator(cfg)
        for i in range(50):
            estimator.observe(f"job{i}", 100)
        self.assertEqual(len(estimator._job_stats), 50)

    def test_observe_max_samples_per_job_stops_updating(self):
        cfg = BatchJobSchedConfig({
            "max_samples_per_job": 2,
            "cold_start_min_samples": 2,
        })
        estimator = JobDecodeEstimator(cfg)
        estimator.observe("job1", 100)
        estimator.observe("job1", 200)
        # Third call should be skipped
        estimator.observe("job1", 300)
        stat = estimator._job_stats["job1"]
        self.assertEqual(stat.sample_count, 2)
        self.assertEqual(stat.cumulative_sum, 300)  # sum of first two
        self.assertEqual(stat.ewma_average, 130.0)  # EWMA of second obs

    def test_observe_zero_max_samples_disables_all_updates(self):
        cfg = BatchJobSchedConfig({
            "max_samples_per_job": 0,
            "cold_start_min_samples": 0,
        })
        estimator = JobDecodeEstimator(cfg)
        estimator.observe("job1", 100)
        stat = estimator._job_stats["job1"]
        self.assertEqual(stat.sample_count, 1)  # 0 means unlimited, not disabled
        self.assertEqual(stat.cumulative_sum, 100)
        self.assertEqual(stat.ewma_average, 100.0)

    def test_get_stats_returns_human_readable_dict(self):
        self.estimator.observe("job1", 100)
        self.estimator.observe("job1", 150)
        stats = self.estimator.get_stats()
        self.assertIn("job1", stats)
        self.assertEqual(stats["job1"]["sample_count"], 2)
        self.assertEqual(stats["job1"]["cumulative_sum"], 250)
        self.assertIsInstance(stats["job1"]["ewma_average"], float)

    def test_get_stats_empty(self):
        stats = self.estimator.get_stats()
        self.assertEqual(stats, {})

    def test_multiple_jobs_independent_estimates(self):
        self.estimator.observe("long_job", 500)
        self.estimator.observe("long_job", 600)
        self.estimator.observe("short_job", 10)
        long_pred = self.estimator.predict("long_job")
        short_pred = self.estimator.predict("short_job")
        self.assertGreater(long_pred, short_pred)

    def test_observe_updates_cumulative_sum_correctly(self):
        self.estimator.observe("job1", 10)
        self.estimator.observe("job1", 20)
        self.estimator.observe("job1", 30)
        self.assertEqual(self.estimator._job_stats["job1"].cumulative_sum, 60)


# ===================================================================
# RequestBucket
# ===================================================================


class _MockRequest:
    """Minimal request-like object for RequestBucket tests.

    Avoids importing the real ``Request`` which requires heavy vLLM
    initialisation.  Only exposes the attributes that RequestBucket uses.
    """
    def __init__(self, request_id: str, num_prompt_tokens: int):
        self.request_id = request_id
        self.num_prompt_tokens = num_prompt_tokens
        self.num_computed_tokens = 0
        self.num_output_tokens = 0


class TestRequestBucket(TestBase):
    """O(log n) best-fit request bucket tests."""

    def setUp(self):
        super().setUp()
        self.bucket = RequestBucket()

    def test_put_increases_length(self):
        req = _MockRequest("req1", 100)
        self.bucket.put(req)
        self.assertEqual(len(self.bucket), 1)

    def test_put_none_is_noop(self):
        self.bucket.put(None)
        self.assertTrue(self.bucket.is_empty())

    def test_put_requests_with_same_length_grouped(self):
        r1 = _MockRequest("req1", 100)
        r2 = _MockRequest("req2", 100)
        self.bucket.put(r1)
        self.bucket.put(r2)
        self.assertEqual(len(self.bucket), 2)

    def test_is_empty_on_creation(self):
        self.assertTrue(self.bucket.is_empty())

    def test_is_empty_after_put_and_remove(self):
        req = _MockRequest("req1", 100)
        self.bucket.put(req)
        self.bucket.remove(req)
        self.assertTrue(self.bucket.is_empty())

    def test_peek_best_fit_returns_none_when_empty(self):
        self.assertIsNone(self.bucket.peek_best_fit_request(100))

    def test_peek_best_fit_exact_match(self):
        req = _MockRequest("req1", 100)
        self.bucket.put(req)
        # search_prefill_length=200: bisect_right on [100] returns idx=1,
        # idx-1=0 yields best_length=100
        result = self.bucket.peek_best_fit_request(200)
        self.assertIs(result, req)

    def test_peek_best_fit_returns_nearest_lower(self):
        r1 = _MockRequest("req1", 50)
        r2 = _MockRequest("req2", 200)
        self.bucket.put(r1)
        self.bucket.put(r2)
        # search=150: bisect_right on [50,200] returns idx=1,
        # idx-1=0 yields best_length=50
        result = self.bucket.peek_best_fit_request(150)
        self.assertIs(result, r1)

    def test_peek_best_fit_when_search_less_than_all(self):
        r1 = _MockRequest("req1", 100)
        r2 = _MockRequest("req2", 200)
        self.bucket.put(r1)
        self.bucket.put(r2)
        # search=50: bisect_right on [50,100,200] -> idx=0 (no lower),
        # idx=0 < len -> returns first bucket entry
        result = self.bucket.peek_best_fit_request(50)
        self.assertIs(result, r1)

    def test_remove_existing_request_returns_true(self):
        req = _MockRequest("req1", 100)
        self.bucket.put(req)
        self.assertTrue(self.bucket.remove(req))

    def test_remove_none_returns_false(self):
        self.assertFalse(self.bucket.remove(None))

    def test_remove_non_existent_request_returns_false(self):
        req = _MockRequest("req1", 100)
        self.assertFalse(self.bucket.remove(req))

    def test_remove_request_wrong_length_returns_false(self):
        self.bucket.put(_MockRequest("req1", 100))
        other = _MockRequest("req2", 200)
        self.assertFalse(self.bucket.remove(other))

    def test_remove_cleans_up_empty_bucket_entry(self):
        req = _MockRequest("req1", 100)
        self.bucket.put(req)
        self.bucket.remove(req)
        self.assertEqual(len(self.bucket._bucket), 0)
        self.assertEqual(len(self.bucket._sorted_lengths), 0)

    def test_iteration_yields_all_requests(self):
        reqs = [_MockRequest(f"req{i}", i * 100) for i in range(3)]
        for r in reqs:
            self.bucket.put(r)
        iterated = list(self.bucket)
        self.assertEqual(len(iterated), 3)
        for r in reqs:
            self.assertIn(r, iterated)

    def test_bool_empty_is_false(self):
        self.assertFalse(self.bucket)

    def test_bool_non_empty_is_true(self):
        self.bucket.put(_MockRequest("req1", 100))
        self.assertTrue(self.bucket)

    def test_peek_best_fit_no_sorted_lengths_returns_none(self):
        # Direct manipulation to test edge case
        self.bucket._bucket[100] = deque()
        self.bucket._sorted_lengths = []
        self.assertIsNone(self.bucket.peek_best_fit_request(50))


# ===================================================================
# RunningBlockReserver
# ===================================================================


class TestRunningBlockReserver(TestBase):
    """KV‑block demand prediction with incremental cache tests."""

    def setUp(self):
        super().setUp()
        self.config = BatchJobSchedConfig({
            "reserve_margin_blocks": 2,
            "reserve_max_blocks": 8,
        })

        # Build a minimal mock scheduler
        self.mock_scheduler = MagicMock()
        self.mock_scheduler.block_size = 16
        self.mock_scheduler.max_num_scheduled_tokens = 32
        self.mock_scheduler.num_lookahead_tokens = 0
        self.mock_scheduler.running = []

        self.reserver = RunningBlockReserver(self.mock_scheduler, self.config)

    @staticmethod
    def _make_request(num_prompt_tokens=100, num_computed_tokens=0):
        req = MagicMock(spec=Request)
        req.num_prompt_tokens = num_prompt_tokens
        req.num_computed_tokens = num_computed_tokens
        req.request_id = f"req_{id(req)}"
        return req

    def test_invalidate_cache_sets_cache_stale(self):
        self.reserver._cached_running_queue_len = 5
        self.reserver.invalidate_cache()
        self.assertEqual(self.reserver._cached_running_queue_len, -1)

    def test_predict_empty_running_returns_margin(self):
        # _finalize_reserve(0) = min(0 + 2, 8) = 2
        result = self.reserver.predict()
        self.assertEqual(result, 2)

    def test_predict_cache_hit_returns_cached(self):
        req = self._make_request()
        self.mock_scheduler.running = [req]
        # First call populates cache
        first = self.reserver.predict()
        # Second call should hit cache
        second = self.reserver.predict()
        self.assertEqual(first, second)

    def test_predict_incremental_update_single_request(self):
        req1 = self._make_request(num_prompt_tokens=100, num_computed_tokens=0)
        self.mock_scheduler.running = [req1]
        first = self.reserver.predict()

        req2 = self._make_request(num_prompt_tokens=50, num_computed_tokens=0)
        self.mock_scheduler.running = [req1, req2]
        # Running grew by 1 -> incremental update
        second = self.reserver.predict()

        # Should be >= first since new request was added
        self.assertGreaterEqual(second, first)

    def test_predict_full_recompute_on_complex_change(self):
        req1 = self._make_request()
        self.mock_scheduler.running = [req1]
        self.reserver.predict()

        # Change running length by more than 1 (simulating preemption)
        self.mock_scheduler.running = []
        third = self.reserver.predict()
        self.assertEqual(third, 2)  # just margin

    def test_compute_one_prefill_phase(self):
        """Request in prefill phase: num_computed < num_prompt."""
        req = self._make_request(num_prompt_tokens=100, num_computed_tokens=0)
        # _compute_one simulation:
        # remaining_prompt = 100 - 0 = 100
        # this_round_tokens = min(100, 32) = 32
        # num_computed_after = 0 + 32 = 32
        # Still in prefill: remaining_prompt_after = 100 - 32 = 68
        # next_round_tokens = min(68, 32) = 32
        # pos_in_block = 32 % 16 = 0
        # remaining = 0
        # next_round_tokens(32) > remaining(0) -> _cdiv(32 - 0, 16) = 2
        blocks = self.reserver._compute_one(req)
        self.assertEqual(blocks, 2)

    def test_compute_one_decode_phase(self):
        """Request that has completed prefill."""
        req = self._make_request(num_prompt_tokens=100, num_computed_tokens=100)
        # this_round_tokens = 1 + 0 = 1
        # num_computed_after = 100 + 1 = 101
        # num_computed_after(101) >= num_prompt(100) -> decode
        # next_round_tokens = 1 + 0 = 1
        # pos_in_block = 101 % 16 = 5
        # remaining = 16 - 5 = 11
        # next_round_tokens(1) <= remaining(11) -> 0
        blocks = self.reserver._compute_one(req)
        self.assertEqual(blocks, 0)

    def test_compute_one_decode_at_block_boundary(self):
        """Request at block boundary needs a new block."""
        req = self._make_request(num_prompt_tokens=100, num_computed_tokens=99)
        # this_round_tokens = 1 + 0 = 1
        # num_computed_after = 99 + 1 = 100
        # decode: next_round_tokens = 1
        # pos_in_block = 100 % 16 = 4
        # remaining = 16 - 4 = 12
        # next_round_tokens(1) <= remaining(12) -> 0
        blocks = self.reserver._compute_one(req)
        self.assertEqual(blocks, 0)

    def test_finalize_reserve_caps_at_max_blocks(self):
        self.reserver._config.reserve_max_blocks = 5
        self.reserver._config.reserve_margin_blocks = 10
        result = self.reserver._finalize_reserve(100)
        self.assertEqual(result, 5)


# ===================================================================
# BatchJobAwareRequestQueue
# ===================================================================


class TestBatchJobAwareRequestQueue(TestBase):
    """Job‑grouped request queue with reserve‑aware admission tests."""

    def setUp(self):
        super().setUp()
        self.config = BatchJobSchedConfig({
            "cold_start_min_samples": 2,
            "short_decode_token_threshold": 32,
            "low_available_tokens_threshold": 4096,
            "max_jobs": 20,
        })

        # Mock scheduler with minimal attributes
        self.mock_scheduler = MagicMock()
        self.mock_scheduler.block_size = 16
        self.mock_scheduler.max_num_scheduled_tokens = 32
        self.mock_scheduler.num_lookahead_tokens = 0
        # Mock kv_cache_manager.block_pool.get_num_free_blocks
        type(self.mock_scheduler).kv_cache_manager = PropertyMock(
            return_value=MagicMock(
                block_pool=MagicMock(
                    get_num_free_blocks=MagicMock(return_value=1000)
                )
            )
        )

        self.job_decode_estimator = JobDecodeEstimator(self.config)
        self.block_reserver = RunningBlockReserver(self.mock_scheduler, self.config)
        self.job_name_parser = JobNameParser()

        with patch(
            "vllm.v1.core.sched.request_queue.RequestQueue.__init__",
            MagicMock(return_value=None),
        ):
            self.queue = BatchJobAwareRequestQueue(
                self.mock_scheduler,
                self.job_decode_estimator,
                self.block_reserver,
                self.job_name_parser,
                self.config,
            )

    @staticmethod
    def _make_request(request_id: str, num_prompt_tokens: int = 100):
        req = MagicMock(spec=Request)
        req.request_id = request_id
        req.num_prompt_tokens = num_prompt_tokens
        req.num_computed_tokens = 0
        req.num_output_tokens = 0
        return req

    def test_add_request_increases_length(self):
        req = self._make_request("req1")
        self.queue.add_request(req)
        self.assertEqual(len(self.queue), 1)

    def test_add_request_none_is_noop(self):
        self.queue.add_request(None)
        self.assertEqual(len(self.queue), 0)

    def test_add_request_routes_to_correct_job_bucket(self):
        req = self._make_request("r1#job_name[job1]#")
        self.queue.add_request(req)
        self.assertIn("job1", self.queue._job_buckets)
        self.assertEqual(len(self.queue._job_buckets["job1"]), 1)

    def test_add_request_tracks_cold_start(self):
        req = self._make_request("r1#job_name[job1]#")
        self.queue.add_request(req)
        self.assertIn("job1", self.queue._cold_start_reqs)
        self.assertIn(req.request_id, self.queue._cold_start_reqs["job1"])

    def test_add_request_beyond_cold_start_requests_not_tracked(self):
        cfg = BatchJobSchedConfig({"cold_start_min_samples": 1})
        with patch(
            "vllm.v1.core.sched.request_queue.RequestQueue.__init__",
            MagicMock(return_value=None),
        ):
            queue = BatchJobAwareRequestQueue(
                self.mock_scheduler,
                JobDecodeEstimator(cfg),
                self.block_reserver,
                self.job_name_parser,
                cfg,
            )
        # First request -> cold start tracked
        r1 = self._make_request("r1#job_name[job1]#")
        queue.add_request(r1)
        self.assertIn("job1", queue._cold_start_reqs)
        self.assertIn(r1.request_id, queue._cold_start_reqs["job1"])
        # Second request -> exceeds min_samples, not tracked
        r2 = self._make_request("r2#job_name[job1]#")
        queue.add_request(r2)
        # With cold_start_min_samples=1, only the first request ID is tracked
        self.assertIn("job1", queue._cold_start_reqs)  # job key remains
        self.assertNotIn(r2.request_id, queue._cold_start_reqs["job1"])

    def test_remove_request_decreases_length(self):
        req = self._make_request("r1#job_name[job1]#")
        self.queue.add_request(req)
        self.queue.remove_request(req)
        self.assertEqual(len(self.queue), 0)

    def test_remove_request_none_is_noop(self):
        self.queue.remove_request(None)  # should not raise

    def test_remove_request_unknown_job_is_noop(self):
        req = self._make_request("r1")
        # Not added yet
        self.queue.remove_request(req)
        self.assertEqual(len(self.queue), 0)

    def test_remove_request_cleans_up_empty_job_bucket(self):
        req = self._make_request("r1#job_name[job1]#")
        self.queue.add_request(req)
        self.queue.remove_request(req)
        self.assertNotIn("job1", self.queue._job_buckets)

    def test_remove_request_cleans_up_cold_start_tracking(self):
        req = self._make_request("r1#job_name[job1]#")
        self.queue.add_request(req)
        self.queue.remove_request(req)
        self.assertNotIn("job1", self.queue._cold_start_reqs)

    def test_remove_requests_removes_multiple(self):
        reqs = [
            self._make_request(f"r{i}#job_name[job1]#")
            for i in range(3)
        ]
        for r in reqs:
            self.queue.add_request(r)
        self.queue.remove_requests(reqs)
        self.assertEqual(len(self.queue), 0)

    def test_prepend_requests_adds_all(self):
        reqs = [
            self._make_request(f"r{i}#job_name[job1]#")
            for i in range(3)
        ]
        mock_queue = MagicMock()
        mock_queue.__iter__.return_value = iter(reqs)
        self.queue.prepend_requests(mock_queue)
        self.assertEqual(len(self.queue), 3)

    def test_peek_request_returns_admittable(self):
        req = self._make_request("r1#job_name[job1]#", num_prompt_tokens=50)
        self.queue.add_request(req)
        result = self.queue.peek_request()
        self.assertIsNotNone(result)

    def test_peek_request_raises_on_empty(self):
        with self.assertRaises(IndexError):
            self.queue.peek_request()

    def test_pop_request_returns_and_removes(self):
        req = self._make_request("r1#job_name[job1]#")
        self.queue.add_request(req)
        popped = self.queue.pop_request()
        self.assertIs(popped, req)
        self.assertEqual(len(self.queue), 0)

    def test_pop_request_raises_on_empty(self):
        with self.assertRaises(IndexError):
            self.queue.pop_request()

    def test_get_cold_start_request_returns_prioritized(self):
        r1 = self._make_request("r1#job_name[job1]#")
        r2 = self._make_request("r2#job_name[job2]#")
        self.queue.add_request(r1)
        self.queue.add_request(r2)
        # Both are cold-start; should return one of them
        result = self.queue._get_cold_start_request()
        self.assertIsNotNone(result)
        self.assertIn(result, [r1, r2])

    def test_collect_job_decode_info_separates_long_short(self):
        # Observe enough samples for short_job to enter stable phase,
        # otherwise cold-start shrinkage (prior_weight=2.0) pulls the
        # prediction toward default_decode=128, misclassifying it as long.
        self.job_decode_estimator.observe("long_job", 100)
        self.job_decode_estimator.observe("short_job", 10)
        self.job_decode_estimator.observe("short_job", 10)
        req_long = self._make_request("r1#job_name[long_job]#")
        req_short = self._make_request("r2#job_name[short_job]#")
        self.queue.add_request(req_long)
        self.queue.add_request(req_short)

        long_jobs, short_jobs = self.queue._collect_job_decode_info()
        long_names = [name for name, _ in long_jobs]
        short_names = [name for name, _ in short_jobs]
        self.assertIn("long_job", long_names)
        self.assertIn("short_job", short_names)

    def test_sort_jobs_prioritizes_long_when_many_tokens(self):
        """When admission_budget > low_available_tokens_threshold, long jobs first."""
        # admission_budget=5000 > threshold=4096 → prioritize_long=True
        long_jobs = [("job_a", 100), ("job_b", 50)]
        short_jobs = [("job_c", 10)]
        ordered = self.queue._sort_jobs(5000, long_jobs, short_jobs)
        # Both categories sorted descending; long jobs concatenated first
        self.assertEqual(ordered, [("job_a", 100), ("job_b", 50), ("job_c", 10)])

    def test_sort_jobs_prioritizes_short_when_few_tokens(self):
        """When admission_budget <= low_available_tokens_threshold, short jobs first."""
        # admission_budget=3000 <= threshold=4096 → prioritize_long=False
        long_jobs = [("job_a", 100)]
        short_jobs = [("job_b", 10)]
        ordered = self.queue._sort_jobs(3000, long_jobs, short_jobs)
        # Both categories sorted descending; short jobs concatenated first
        self.assertEqual(ordered, [("job_b", 10), ("job_a", 100)])

    def test_sort_jobs_only_long_high_budget_descending(self):
        """With only long jobs and high budget, sort by decode length descending."""
        long_jobs = [("job_b", 50), ("job_a", 100)]
        ordered = self.queue._sort_jobs(5000, long_jobs, [])
        self.assertEqual(ordered, [("job_a", 100), ("job_b", 50)])

    def test_sort_jobs_only_long_low_budget_ascending(self):
        """With only long jobs and low budget, sort by decode length ascending."""
        long_jobs = [("job_a", 100), ("job_b", 50)]
        ordered = self.queue._sort_jobs(3000, long_jobs, [])
        self.assertEqual(ordered, [("job_b", 50), ("job_a", 100)])

    def test_sort_jobs_only_short_high_budget_descending(self):
        """With only short jobs and high budget, sort by decode length descending."""
        short_jobs = [("job_b", 5), ("job_a", 10)]
        ordered = self.queue._sort_jobs(5000, [], short_jobs)
        self.assertEqual(ordered, [("job_a", 10), ("job_b", 5)])

    def test_sort_jobs_only_short_low_budget_ascending(self):
        """With only short jobs and low budget, sort by decode length ascending."""
        short_jobs = [("job_a", 10), ("job_b", 5)]
        ordered = self.queue._sort_jobs(3000, [], short_jobs)
        self.assertEqual(ordered, [("job_b", 5), ("job_a", 10)])

    def test_admission_budget_zero_when_no_free_blocks(self):
        type(self.mock_scheduler).kv_cache_manager = PropertyMock(
            return_value=MagicMock(
                block_pool=MagicMock(
                    get_num_free_blocks=MagicMock(return_value=0)
                )
            )
        )
        budget = self.queue._admission_budget()
        self.assertEqual(budget, 0)

    def test_find_admittable_returns_none_with_zero_budget(self):
        type(self.mock_scheduler).kv_cache_manager = PropertyMock(
            return_value=MagicMock(
                block_pool=MagicMock(
                    get_num_free_blocks=MagicMock(return_value=0)
                )
            )
        )
        result = self.queue._find_admittable()
        self.assertIsNone(result)

    def test_queue_bool_checks_admittable(self):
        # Empty queue should be falsy
        self.assertFalse(self.queue)

    def test_queue_bool_with_admittable_request(self):
        req = self._make_request("r1#job_name[job1]#", num_prompt_tokens=50)
        self.queue.add_request(req)
        self.assertTrue(self.queue)

    def test_max_jobs_exceeded_raises_in_get_or_create(self):
        cfg = BatchJobSchedConfig({"max_jobs": 1})
        with patch(
            "vllm.v1.core.sched.request_queue.RequestQueue.__init__",
            MagicMock(return_value=None),
        ):
            queue = BatchJobAwareRequestQueue(
                self.mock_scheduler,
                JobDecodeEstimator(cfg),
                self.block_reserver,
                self.job_name_parser,
                cfg,
            )
        queue.add_request(self._make_request("r1#job_name[job1]#"))
        with self.assertRaises(RuntimeError) as ctx:
            queue.add_request(self._make_request("r2#job_name[job2]#"))
        self.assertIn("Maximum number of jobs", str(ctx.exception))

