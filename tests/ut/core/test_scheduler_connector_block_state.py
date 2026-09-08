# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.ut.core.test_dyntra_lb_scheduler import create_dyntra_lb_scheduler, make_dyntra_test_config
from vllm_ascend.core.dyntra_lb_scheduler import AsyncDyntraLBScheduler, DyntraLBScheduler
from vllm_ascend.core.scheduler_profiling_chunk import ProfilingChunkScheduler
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler


@pytest.mark.parametrize(
    "scheduler_cls",
    [DyntraLBScheduler, AsyncDyntraLBScheduler, BalanceScheduler, ProfilingChunkScheduler],
)
@pytest.mark.parametrize("with_connector", [False, True])
def test_boundary_state_is_drained_consumed_and_not_dispatched(monkeypatch, scheduler_cls, with_connector):
    """vLLM #51358: hand off exact snapshots only during metadata building."""
    scheduler = create_dyntra_lb_scheduler(make_dyntra_test_config(), scheduler_cls=scheduler_cls)
    if isinstance(scheduler, BalanceScheduler):
        # Exercise the local schedule implementation, not its super fallback.
        scheduler._balance_enabled = True

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
    seen_states = []
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
        assert seen_states[0].block_ids == {"cached": ([1, 9],), "boundary": ([42],)}
        assert seen_states[0].boundary_state_offloads is offers
        assert seen_states[1].block_ids == {"cached": ([1, 9],)}
        assert seen_states[1].boundary_state_offloads == {}
        assert get_blocks.call_count == 3
    else:
        assert seen_states == []
        get_blocks.assert_not_called()
