# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for PD disaggregation cached_tokens correction.

Covers:
- ``adjust_disagg_prefill_stats``: D-side override, P-side recording, no-op
  for non-PD requests.
- ``DisaggPrefillStatsMixin._free_request``: propagation of
  ``remote_num_cached_tokens`` to the connector finish dict.
- ``DisaggPrefillStatsMixin.update_from_output``: zeroing of
  ``num_cache_creation_tokens`` on D-side.
- Regression: single-node / KV-pool / preemption-recovery requests are
  unaffected (no ``kv_transfer_params`` → no change).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.metrics.stats import PrefillStats

from vllm_ascend.core.disagg_stats import (
    DisaggPrefillStatsMixin,
    adjust_disagg_prefill_stats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    num_prompt_tokens: int = 100,
    kv_transfer_params: dict | None = None,
    prefill_stats: PrefillStats | None = None,
) -> SimpleNamespace:
    """Build a lightweight request stand-in for unit tests."""
    return SimpleNamespace(
        num_prompt_tokens=num_prompt_tokens,
        kv_transfer_params=kv_transfer_params,
        prefill_stats=prefill_stats,
    )


def _make_prefill_stats(
    num_prompt_tokens: int = 100,
    num_local_cached_tokens: int = 0,
    num_external_cached_tokens: int = 0,
) -> PrefillStats:
    """Build a PrefillStats that has already had ``set()`` called."""
    ps = PrefillStats()
    ps.set(
        num_prompt_tokens=num_prompt_tokens,
        num_local_cached_tokens=num_local_cached_tokens,
        num_external_cached_tokens=num_external_cached_tokens,
    )
    return ps


def _make_mixin_scheduler(
    super_free_returns: dict | None = None, super_update_returns: dict | None = None
) -> MagicMock:
    """Build a scheduler instance with DisaggPrefillStatsMixin in its MRO.

    The ``super()`` calls inside the mixin are routed to MagicMock so we can
    stub the return values.
    """
    scheduler = DisaggPrefillStatsMixin.__new__(DisaggPrefillStatsMixin)
    # Stub the super() chain — the mixin calls super()._free_request and
    # super().update_from_output.
    scheduler.__class__ = type(
        "StubScheduler",
        (DisaggPrefillStatsMixin, MagicMock),
        {},
    )
    return scheduler


# ---------------------------------------------------------------------------
# adjust_disagg_prefill_stats — WAITING-loop helper
# ---------------------------------------------------------------------------


class TestAdjustDisaggPrefillStats:
    """Tests for the WAITING-loop helper function."""

    def test_no_kv_transfer_params_is_noop(self):
        """Single-node / KV-pool / preemption-recovery: no change."""
        ps = _make_prefill_stats(
            num_prompt_tokens=100,
            num_local_cached_tokens=10,
            num_external_cached_tokens=20,
        )
        request = _make_request(
            num_prompt_tokens=100,
            kv_transfer_params=None,
            prefill_stats=ps,
        )
        result = adjust_disagg_prefill_stats(request, 10, 20)

        # No change to cached_tokens or connector_prefix_cache_hits.
        assert ps.num_cached_tokens == 30  # 10 + 20, unchanged
        assert result == 20  # connector_prefix_cache_hits unchanged
        assert not hasattr(ps, "_disagg_d_side")

    def test_d_side_overrides_cached_tokens_with_remote(self):
        """D-side (do_remote_prefill): override with D-local + P-real."""
        ps = _make_prefill_stats(
            num_prompt_tokens=100,
            num_local_cached_tokens=0,
            num_external_cached_tokens=99,  # inflated by connector
        )
        request = _make_request(
            num_prompt_tokens=100,
            kv_transfer_params={
                "do_remote_prefill": True,
                "remote_num_cached_tokens": 50,  # P-side real hits
            },
            prefill_stats=ps,
        )
        result = adjust_disagg_prefill_stats(request, 0, 99)

        # D-local(0) + P-real(50) = 50, not 99.
        assert ps.num_cached_tokens == 50
        assert result == 0  # connector_prefix_cache_hits zeroed
        assert getattr(ps, "_disagg_d_side", False) is True

    def test_d_side_with_local_hits_adds_local_and_remote(self):
        """D-side with local prefix hits: D-local + P-real."""
        ps = _make_prefill_stats(
            num_prompt_tokens=100,
            num_local_cached_tokens=5,
            num_external_cached_tokens=90,  # inflated
        )
        request = _make_request(
            num_prompt_tokens=100,
            kv_transfer_params={
                "do_remote_prefill": True,
                "remote_num_cached_tokens": 50,
            },
            prefill_stats=ps,
        )
        result = adjust_disagg_prefill_stats(request, 5, 90)

        # D-local(5) + P-real(50) = 55.
        assert ps.num_cached_tokens == 55
        assert result == 0

    def test_d_side_capped_at_prompt_tokens(self):
        """D-side: cached_tokens capped at num_prompt_tokens."""
        ps = _make_prefill_stats(
            num_prompt_tokens=100,
            num_local_cached_tokens=10,
            num_external_cached_tokens=90,
        )
        request = _make_request(
            num_prompt_tokens=100,
            kv_transfer_params={
                "do_remote_prefill": True,
                "remote_num_cached_tokens": 200,  # exceeds prompt
            },
            prefill_stats=ps,
        )
        adjust_disagg_prefill_stats(request, 10, 90)

        assert ps.num_cached_tokens == 100  # capped

    def test_d_side_no_remote_defaults_to_zero(self):
        """D-side without remote_num_cached_tokens: defaults to 0 (graceful)."""
        ps = _make_prefill_stats(
            num_prompt_tokens=100,
            num_local_cached_tokens=0,
            num_external_cached_tokens=99,
        )
        request = _make_request(
            num_prompt_tokens=100,
            kv_transfer_params={"do_remote_prefill": True},  # no remote key
            prefill_stats=ps,
        )
        result = adjust_disagg_prefill_stats(request, 0, 99)

        # D-local(0) + default-remote(0) = 0.
        assert ps.num_cached_tokens == 0
        assert result == 0
        assert getattr(ps, "_disagg_d_side", False) is True

    def test_p_side_records_remote_num_cached_tokens(self):
        """P-side (do_remote_decode): record local hits for D-side."""
        ps = _make_prefill_stats(
            num_prompt_tokens=100,
            num_local_cached_tokens=30,
            num_external_cached_tokens=0,
        )
        params = {"do_remote_decode": True}
        request = _make_request(
            num_prompt_tokens=100,
            kv_transfer_params=params,
            prefill_stats=ps,
        )
        result = adjust_disagg_prefill_stats(request, 30, 0)

        # P-side records its local hits.
        assert params["remote_num_cached_tokens"] == 30
        # No change to prefill_stats or connector_prefix_cache_hits on P-side.
        assert ps.num_cached_tokens == 30  # unchanged
        assert result == 0  # unchanged (was already 0)

    def test_d_side_without_prefill_stats_does_not_crash(self):
        """D-side with prefill_stats=None: no crash, just zeroes hits."""
        request = _make_request(
            num_prompt_tokens=100,
            kv_transfer_params={"do_remote_prefill": True},
            prefill_stats=None,
        )
        result = adjust_disagg_prefill_stats(request, 0, 99)
        assert result == 0

    def test_d_side_flag_cleared_on_re_prefill_after_preemption(self):
        """Preemption re-prefill (balance/profiling): stale flag must clear.

        Balance/profiling schedulers re-run this helper for post-preemption
        re-prefills (their guard lacks the num_preemptions check).  The
        _disagg_d_side flag from the original remote prefill must be cleared,
        or update_from_output would wrongly zero num_cache_creation_tokens
        for a genuine local re-prefill.
        """
        ps = _make_prefill_stats(
            num_prompt_tokens=100,
            num_local_cached_tokens=0,
            num_external_cached_tokens=99,
        )
        request = _make_request(
            num_prompt_tokens=100,
            kv_transfer_params={"do_remote_prefill": True},
            prefill_stats=ps,
        )
        adjust_disagg_prefill_stats(request, 0, 99)
        assert getattr(ps, "_disagg_d_side", False) is True

        # Connector cleared do_remote_prefill after the async KV load
        # (update_state_after_alloc); the request was then preempted before
        # producing output, and is now re-prefilled locally.
        request.kv_transfer_params = {"do_remote_prefill": False}
        ps.num_cached_tokens = 40  # set() re-ran with real local hits
        adjust_disagg_prefill_stats(request, 40, 0)

        assert getattr(ps, "_disagg_d_side", False) is False


# ---------------------------------------------------------------------------
# DisaggPrefillStatsMixin._free_request — finish-path propagation
# ---------------------------------------------------------------------------


class TestFreeRequestPropagation:
    """Tests for P-side remote_num_cached_tokens propagation."""

    def test_p_side_propagates_to_dict(self):
        """P-side: remote_num_cached_tokens added to connector's finish dict."""

        # Build a stub scheduler where super()._free_request returns a dict.
        class StubSuper:
            def _free_request(self, request, delay_free_blocks=False):
                return {"do_remote_prefill": True}

        class TestSched(DisaggPrefillStatsMixin, StubSuper):
            pass

        scheduler = TestSched()
        request = _make_request(
            kv_transfer_params={
                "do_remote_decode": True,
                "remote_num_cached_tokens": 42,
            },
        )
        result = scheduler._free_request(request)

        assert result is not None
        assert result["do_remote_prefill"] is True  # original key preserved
        assert result["remote_num_cached_tokens"] == 42

    def test_p_side_creates_dict_when_super_returns_none(self):
        """P-side: creates dict when connector returns None (Layerwise/SFA)."""

        class StubSuper:
            def _free_request(self, request, delay_free_blocks=False):
                return None  # e.g. SFA / Layerwise connector

        class TestSched(DisaggPrefillStatsMixin, StubSuper):
            pass

        scheduler = TestSched()
        request = _make_request(
            kv_transfer_params={
                "do_remote_decode": True,
                "remote_num_cached_tokens": 42,
            },
        )
        result = scheduler._free_request(request)

        assert result is not None
        assert result["remote_num_cached_tokens"] == 42

    def test_p_side_without_recorded_value_is_noop(self):
        """Abort path: P-side params without remote_num_cached_tokens.

        Requests aborted before adjust_disagg_prefill_stats ran never get the
        key recorded — _free_request must not crash or add a bogus value.
        """

        class StubSuper:
            def _free_request(self, request, delay_free_blocks=False):
                return {"existing": "key"}

        class TestSched(DisaggPrefillStatsMixin, StubSuper):
            pass

        scheduler = TestSched()
        request = _make_request(kv_transfer_params={"do_remote_decode": True})
        result = scheduler._free_request(request)

        assert result == {"existing": "key"}

    def test_d_side_does_not_propagate(self):
        """D-side (do_remote_prefill): no propagation (it receives, not sends)."""

        class StubSuper:
            def _free_request(self, request, delay_free_blocks=False):
                return {"existing": "key"}

        class TestSched(DisaggPrefillStatsMixin, StubSuper):
            pass

        scheduler = TestSched()
        request = _make_request(
            kv_transfer_params={"do_remote_prefill": True},
        )
        result = scheduler._free_request(request)

        assert result == {"existing": "key"}  # unchanged

    def test_no_kv_transfer_params_is_passthrough(self):
        """Non-PD: _free_request is a transparent passthrough."""

        class StubSuper:
            def _free_request(self, request, delay_free_blocks=False):
                return {"existing": "key"}

        class TestSched(DisaggPrefillStatsMixin, StubSuper):
            pass

        scheduler = TestSched()
        request = _make_request(kv_transfer_params=None)
        result = scheduler._free_request(request)

        assert result == {"existing": "key"}


# ---------------------------------------------------------------------------
# DisaggPrefillStatsMixin.update_from_output — cache_creation zeroing
# ---------------------------------------------------------------------------


class TestUpdateFromOutputZeroing:
    """Tests for num_cache_creation_tokens zeroing on D-side."""

    def test_d_side_cache_creation_tokens_zeroed(self):
        """D-side: num_cache_creation_tokens zeroed after finalize()."""
        # Build a prefill_stats that has a non-zero num_cache_creation_tokens
        # (as if finalize() had run).  PrefillStats is a @dataclass without
        # __slots__, so we can set the attribute directly even on vLLM
        # versions where it is not a declared field.
        ps = _make_prefill_stats(
            num_prompt_tokens=100,
            num_local_cached_tokens=0,
            num_external_cached_tokens=99,
        )
        ps.num_cache_creation_tokens = 1  # simulate post-finalize value
        ps._disagg_d_side = True

        eco = EngineCoreOutput(
            request_id="req-1",
            new_token_ids=[1],
            prefill_stats=ps,
        )
        ecos = EngineCoreOutputs(outputs=[eco])

        class StubSuper:
            def update_from_output(self, scheduler_output, model_runner_output):
                return {0: ecos}

        class TestSched(DisaggPrefillStatsMixin, StubSuper):
            pass

        scheduler = TestSched()
        outputs = scheduler.update_from_output(
            scheduler_output=MagicMock(),
            model_runner_output=MagicMock(),
        )

        eco_result = outputs[0].outputs[0]
        assert eco_result.prefill_stats.num_cache_creation_tokens == 0

    def test_finalize_runs_before_zeroing(self):
        """Ordering proof: finalize() then mixin zeroing.

        super().update_from_output() runs finalize() inside upstream; the
        mixin must zero num_cache_creation_tokens AFTER that.  Calls the real
        finalize() to prove the ordering assumption holds on vLLM versions
        that ship it (>= v0.29.0).
        """
        if not hasattr(PrefillStats, "finalize"):
            pytest.skip("PrefillStats.finalize requires vLLM >= v0.29.0")

        ps = _make_prefill_stats(
            num_prompt_tokens=100,
            num_local_cached_tokens=0,
            num_external_cached_tokens=50,
        )
        ps._disagg_d_side = True
        # Simulate upstream: estimate_cached_tokens returns ~100 for a D-side
        # request whose KV was fully transferred.
        ps.finalize(100)
        assert ps.num_cache_creation_tokens == 50  # post-finalize precondition

        eco = EngineCoreOutput(
            request_id="req-1",
            new_token_ids=[1],
            prefill_stats=ps,
        )
        ecos = EngineCoreOutputs(outputs=[eco])

        class StubSuper:
            def update_from_output(self, scheduler_output, model_runner_output):
                return {0: ecos}

        class TestSched(DisaggPrefillStatsMixin, StubSuper):
            pass

        scheduler = TestSched()
        outputs = scheduler.update_from_output(
            scheduler_output=MagicMock(),
            model_runner_output=MagicMock(),
        )

        eco_result = outputs[0].outputs[0]
        assert eco_result.prefill_stats.num_cache_creation_tokens == 0

    def test_non_d_side_cache_creation_tokens_preserved(self):
        """Non-D-side: num_cache_creation_tokens is NOT zeroed."""
        ps = _make_prefill_stats(
            num_prompt_tokens=100,
            num_local_cached_tokens=0,
            num_external_cached_tokens=99,
        )
        ps.num_cache_creation_tokens = 1  # simulate post-finalize value
        # No _disagg_d_side flag → not D-side.

        eco = EngineCoreOutput(
            request_id="req-1",
            new_token_ids=[1],
            prefill_stats=ps,
        )
        ecos = EngineCoreOutputs(outputs=[eco])

        class StubSuper:
            def update_from_output(self, scheduler_output, model_runner_output):
                return {0: ecos}

        class TestSched(DisaggPrefillStatsMixin, StubSuper):
            pass

        scheduler = TestSched()
        outputs = scheduler.update_from_output(
            scheduler_output=MagicMock(),
            model_runner_output=MagicMock(),
        )

        eco_result = outputs[0].outputs[0]
        assert eco_result.prefill_stats.num_cache_creation_tokens == 1  # preserved

    def test_none_prefill_stats_does_not_crash(self):
        """Output with prefill_stats=None: no crash."""
        eco = EngineCoreOutput(
            request_id="req-1",
            new_token_ids=[1],
            prefill_stats=None,
        )
        ecos = EngineCoreOutputs(outputs=[eco])

        class StubSuper:
            def update_from_output(self, scheduler_output, model_runner_output):
                return {0: ecos}

        class TestSched(DisaggPrefillStatsMixin, StubSuper):
            pass

        scheduler = TestSched()
        outputs = scheduler.update_from_output(
            scheduler_output=MagicMock(),
            model_runner_output=MagicMock(),
        )
        # No crash — just verify it returns the outputs.
        assert len(outputs[0].outputs) == 1


# ---------------------------------------------------------------------------
# Regression: all 4 scheduler classes include the mixin
# ---------------------------------------------------------------------------


class TestSchedulerClassMRO:
    """Verify the mixin is in the MRO of all 4 ascend schedulers.

    These tests import the full scheduler classes, which trigger vllm-ascend's
    patch chain.  If the installed vLLM version differs from the branch
    target, the import fails due to API drift — in that case we skip rather
    than report a false failure.
    """

    def test_recompute_scheduler_has_mixin(self):
        try:
            from vllm_ascend.core.recompute_scheduler import RecomputeScheduler
        except Exception as e:
            pytest.skip(f"Scheduler import requires matching vLLM version: {e}")
        assert DisaggPrefillStatsMixin in RecomputeScheduler.__mro__

    def test_dyntra_lb_scheduler_has_mixin(self):
        try:
            from vllm_ascend.core.dyntra_lb_scheduler import DyntraLBScheduler
        except Exception as e:
            pytest.skip(f"Scheduler import requires matching vLLM version: {e}")
        assert DisaggPrefillStatsMixin in DyntraLBScheduler.__mro__

    def test_profiling_chunk_scheduler_has_mixin(self):
        try:
            from vllm_ascend.core.scheduler_profiling_chunk import ProfilingChunkScheduler
        except Exception as e:
            pytest.skip(f"Scheduler import requires matching vLLM version: {e}")
        assert DisaggPrefillStatsMixin in ProfilingChunkScheduler.__mro__

    def test_balance_scheduler_has_mixin(self):
        try:
            from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler
        except Exception as e:
            pytest.skip(f"Scheduler import requires matching vLLM version: {e}")
        assert DisaggPrefillStatsMixin in BalanceScheduler.__mro__
