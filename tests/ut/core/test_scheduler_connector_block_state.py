# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.core.test_dyntra_lb_scheduler import create_dyntra_lb_scheduler, make_dyntra_test_config
from tests.ut.kv_offload.utils import create_request
from vllm_ascend.core.dyntra_lb_scheduler import AsyncDyntraLBScheduler, DyntraLBScheduler
from vllm_ascend.core.recompute_scheduler import (
    AsyncDyntraLBRecomputeScheduler,
    AsyncRecomputeScheduler,
    DyntraLBRecomputeScheduler,
    RecomputeScheduler,
)
from vllm_ascend.core.scheduler_profiling_chunk import ProfilingChunkAsyncScheduler, ProfilingChunkScheduler
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler


@pytest.mark.parametrize(
    "scheduler_cls",
    [DyntraLBScheduler, AsyncDyntraLBScheduler, BalanceScheduler, ProfilingChunkScheduler],
)
@pytest.mark.parametrize("with_connector", [False, True])
def test_boundary_state_is_drained_consumed_and_not_dispatched(monkeypatch, scheduler_cls, with_connector):
    """Hand off main snapshots locally; preserve release connector metadata."""
    scheduler = create_dyntra_lb_scheduler(make_dyntra_test_config(), scheduler_cls=scheduler_cls)
    if isinstance(scheduler, BalanceScheduler):
        # Exercise the local schedule implementation, not its super fallback.
        scheduler._balance_enabled = True
        scheduler.balance_queue = [torch.tensor([0], dtype=torch.int)]

    scheduler.connector = object() if with_connector else None
    scheduler.ec_connector = None
    scheduler.requests = {"cached": object(), "boundary": object()}
    cached_data = SimpleNamespace(req_ids=["cached", "unchanged"], new_block_ids=[([9],), None])
    monkeypatch.setattr(scheduler, "_make_cached_request_data", lambda *args: cached_data)
    offers = {"boundary": [(0, 42, 128)], "finished": [(0, 43, 128)]}
    drain = Mock(side_effect=[offers, {}])
    get_blocks = Mock(side_effect=lambda req_id: {"cached": ([1, 9],), "boundary": ([42],)}[req_id])
    monkeypatch.setattr(scheduler.kv_cache_manager, "take_boundary_state_offloads", drain)
    monkeypatch.setattr(scheduler.kv_cache_manager, "get_block_ids", get_blocks)
    seen_states: list[Any] = []
    metadata = object()

    def build_metadata(connector, output):
        assert connector is scheduler.connector
        seen_states.append(output.kv_connector_block_state)
        return metadata

    def update_after_schedule(output):
        assert output.kv_connector_block_state is None

    monkeypatch.setattr(scheduler, "_build_kv_connector_meta", build_metadata)
    monkeypatch.setattr(scheduler, "_update_after_schedule", update_after_schedule)

    first_output = scheduler.schedule()
    second_output = scheduler.schedule()
    assert drain.call_count == 2
    assert first_output.kv_connector_block_state is None
    assert second_output.kv_connector_block_state is None
    if with_connector:
        assert first_output.kv_connector_metadata is metadata
        assert second_output.kv_connector_metadata is metadata
        get_blocks.assert_not_called()
        assert seen_states[0].req_ids == {"boundary"}
        assert seen_states[0].get_block_ids("boundary") == ([42],)
        assert seen_states[0].get_block_ids("cached") is None
        assert seen_states[0].get_block_ids("finished") is None
        assert seen_states[1].req_ids == set()
        get_blocks.assert_called_once_with("boundary")
        assert seen_states[0].boundary_state_offloads is offers
        assert seen_states[1].boundary_state_offloads == {}
    else:
        assert seen_states == []
        get_blocks.assert_not_called()


@pytest.mark.parametrize(
    "scheduler_cls",
    [
        DyntraLBScheduler,
        AsyncDyntraLBScheduler,
        RecomputeScheduler,
        AsyncRecomputeScheduler,
        DyntraLBRecomputeScheduler,
        AsyncDyntraLBRecomputeScheduler,
        ProfilingChunkScheduler,
        ProfilingChunkAsyncScheduler,
    ],
)
@pytest.mark.parametrize("loaded_groups", [(), (0,), (0, 1)])
@pytest.mark.parametrize("async_load", [False, True])
def test_kv_zeroing_preserves_only_async_loaded_groups(scheduler_cls, loaded_groups, async_load):
    """Keep transferred pages intact without skipping other groups or local allocations."""
    block_size = 16
    num_blocks = 100
    config = make_dyntra_test_config(block_size=block_size)
    config.kv_transfer_config = None
    config.scheduler_config.async_scheduling = issubclass(scheduler_cls, AsyncScheduler)
    config.cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [f"layer-{index}"],
                FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=1, dtype=dtype),
            )
            for index, dtype in enumerate([torch.float16, torch.float32])
        ],
    )
    scheduler = scheduler_cls(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        structured_output_manager=StructuredOutputManager(config),
    )
    assert scheduler.needs_kv_cache_zeroing
    connector = MagicMock(spec=KVConnectorBase_V1)
    connector.get_num_new_matched_tokens.return_value = (2 * block_size, async_load)
    connector.get_loaded_kv_cache_group_ids.return_value = loaded_groups
    scheduler.connector = connector
    request = create_request(1, num_tokens=4 * block_size, block_size=block_size)
    scheduler.add_request(request)

    output = scheduler.schedule()

    blocks = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    assert all(blocks)
    expected_zeroed = {
        block_id
        for index, group_blocks in enumerate(blocks)
        for block_id in group_blocks
        if not async_load or index not in loaded_groups
    }
    assert set(output.new_block_ids_to_zero or []) == expected_zeroed
    if async_load:
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert request in scheduler.kv_holding_waiting
        assert request.request_id not in output.num_scheduled_tokens
        connector.get_loaded_kv_cache_group_ids.assert_called_once_with(request)
    else:
        assert request.request_id in output.num_scheduled_tokens
        connector.get_loaded_kv_cache_group_ids.assert_not_called()

    # A later local allocation must still be zeroed, even for a transferred group.
    connector.get_num_new_matched_tokens.return_value = (0, False)
    local_request = create_request(2, num_tokens=block_size, block_size=block_size)
    scheduler.add_request(local_request)
    local_output = scheduler.schedule()
    local_blocks = scheduler.kv_cache_manager.get_block_ids(local_request.request_id)
    assert local_request.request_id in local_output.num_scheduled_tokens
    local_block_ids = {block_id for group_blocks in local_blocks for block_id in group_blocks}
    local_zeroed = set(local_output.new_block_ids_to_zero or [])
    assert local_block_ids <= local_zeroed
    assert local_zeroed.isdisjoint(block_id for group_blocks in blocks for block_id in group_blocks)
