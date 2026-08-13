# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy

from vllm_ascend.patch.platform.patch_balance_schedule import (
    _ORIGINAL_SCHEDULER,
    BalanceScheduler,
    _disable_preemption_on_prefill_node,
)


@pytest.mark.parametrize(
    ("kv_role", "expected"),
    [
        ("kv_producer", True),
        ("kv_consumer", False),
        ("kv_both", False),
        (None, False),
    ],
)
def test_disable_preemption_only_on_v023_prefill_nodes(kv_role, expected):
    kv_transfer_config = None if kv_role is None else SimpleNamespace(kv_role=kv_role)
    vllm_config = SimpleNamespace(kv_transfer_config=kv_transfer_config)

    with patch(
        "vllm_ascend.patch.platform.patch_balance_schedule.vllm_version_is",
        return_value=True,
    ):
        assert _disable_preemption_on_prefill_node(vllm_config) is expected


def test_disable_preemption_is_limited_to_v023():
    vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_role="kv_producer"))

    with patch(
        "vllm_ascend.patch.platform.patch_balance_schedule.vllm_version_is",
        return_value=False,
    ):
        assert not _disable_preemption_on_prefill_node(vllm_config)


@pytest.mark.parametrize("scheduler_name", ["default", "profiling_chunk"])
def test_prefill_node_keeps_running_request_when_allocation_fails(
    scheduler_name,
):
    request = SimpleNamespace(
        request_id="prefill-request",
        num_output_placeholders=0,
        num_tokens_with_spec=2,
        num_computed_tokens=1,
        num_prompt_tokens=2,
        has_encoder_inputs=False,
        spec_token_ids=[],
    )
    if scheduler_name == "default":
        scheduler = BalanceScheduler.__new__(BalanceScheduler)
        scheduler._balance_enabled = False
    else:
        from vllm_ascend.core.scheduler_profiling_chunk import (
            ProfilingChunkScheduler,
        )

        scheduler = ProfilingChunkScheduler.__new__(ProfilingChunkScheduler)
        scheduler.profiling_chunk_manager = SimpleNamespace(
            predictor=SimpleNamespace(target_latency=None),
            is_ready=False,
        )
        scheduler.needs_kv_cache_zeroing = False
    scheduler._disable_preemption = True
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.running = [request]
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.max_num_scheduled_tokens = 1
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.max_num_running_reqs = 1
    scheduler.max_model_len = 16
    scheduler.num_lookahead_tokens = 0
    scheduler.need_mamba_block_aligned_split = False
    scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.allocate_slots.return_value = None
    scheduler.kv_cache_manager.get_num_common_prefix_blocks.return_value = [0]
    scheduler.encoder_cache_manager = MagicMock()
    scheduler.encoder_cache_manager.get_freed_mm_hashes.return_value = []
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.connector_prefix_cache_stats = None
    scheduler.lora_config = None
    scheduler.is_encoder_decoder = False
    scheduler.log_stats = False
    scheduler.use_eagle = False
    scheduler.use_v2_model_runner = False
    scheduler.finished_req_ids = set()
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler._preempt_request = MagicMock()
    scheduler._make_cached_request_data = MagicMock()
    scheduler._update_after_schedule = MagicMock()

    output = scheduler.schedule()

    scheduler.kv_cache_manager.allocate_slots.assert_called_once_with(
        request,
        1,
        num_lookahead_tokens=0,
    )
    scheduler._preempt_request.assert_not_called()
    assert scheduler.running == [request]
    assert output.total_num_scheduled_tokens == 0
    assert output.preempted_req_ids == set()


def test_non_prefill_node_uses_upstream_scheduler():
    scheduler = BalanceScheduler.__new__(BalanceScheduler)
    scheduler._balance_enabled = False
    scheduler._disable_preemption = False
    expected = object()

    with (
        patch(
            "vllm_ascend.patch.platform.patch_balance_schedule.vllm_version_is",
            return_value=True,
        ),
        patch.object(_ORIGINAL_SCHEDULER, "schedule", return_value=expected) as schedule,
    ):
        assert scheduler.schedule() is expected

    schedule.assert_called_once_with()


def test_async_scheduler_inherits_prefill_preemption_guard():
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    assert BalanceScheduler in AsyncScheduler.__mro__


def test_profiling_chunk_scheduler_inherits_prefill_preemption_guard():
    from vllm_ascend.core.scheduler_profiling_chunk import (
        ProfilingChunkScheduler,
    )

    assert BalanceScheduler in ProfilingChunkScheduler.__mro__


def test_prefill_node_rejects_forced_prefix_cache_reset_while_running():
    scheduler = BalanceScheduler.__new__(BalanceScheduler)
    scheduler._disable_preemption = True
    scheduler.running = [object()]
    scheduler._preempt_request = MagicMock()

    with pytest.raises(RuntimeError, match="drain or abort"):
        scheduler.reset_prefix_cache(reset_running_requests=True)

    scheduler._preempt_request.assert_not_called()


@pytest.mark.parametrize(
    ("disable_preemption", "running"),
    [
        (True, []),
        (False, [object()]),
    ],
)
def test_forced_prefix_cache_reset_delegates_when_safe(
    disable_preemption,
    running,
):
    scheduler = BalanceScheduler.__new__(BalanceScheduler)
    scheduler._disable_preemption = disable_preemption
    scheduler.running = running

    with patch.object(
        _ORIGINAL_SCHEDULER,
        "reset_prefix_cache",
        return_value=True,
    ) as reset_prefix_cache:
        assert scheduler.reset_prefix_cache(True, True)

    reset_prefix_cache.assert_called_once_with(True, True)


def test_non_forced_prefix_cache_reset_keeps_upstream_behavior():
    scheduler = BalanceScheduler.__new__(BalanceScheduler)
    scheduler._disable_preemption = True
    scheduler.running = [object()]

    with patch.object(
        _ORIGINAL_SCHEDULER,
        "reset_prefix_cache",
        return_value=False,
    ) as reset_prefix_cache:
        assert not scheduler.reset_prefix_cache()

    reset_prefix_cache.assert_called_once_with(False, False)
