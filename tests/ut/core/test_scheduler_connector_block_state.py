# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tests.ut.core.test_dyntra_lb_scheduler import create_dyntra_lb_scheduler, make_dyntra_test_config
from vllm_ascend.core.dyntra_lb_scheduler import AsyncDyntraLBScheduler, DyntraLBScheduler
from vllm_ascend.core.scheduler_profiling_chunk import ProfilingChunkScheduler
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler
from vllm_ascend.utils import vllm_version_is


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

    scheduler.connector = object() if with_connector else None
    scheduler.ec_connector = None
    scheduler.requests = {"cached": object(), "boundary": object()}
    cached_data = SimpleNamespace(req_ids=["cached", "unchanged"], new_block_ids=[([9],), None])
    monkeypatch.setattr(scheduler, "_make_cached_request_data", lambda *args: cached_data)
    offers = {"boundary": [(0, 42, 128)], "finished": [(0, 43, 128)]}
    drain = Mock(side_effect=[offers, {}])
    get_blocks = Mock(side_effect=lambda req_id: {"cached": ([1, 9],), "boundary": ([42],)}[req_id])
    if not vllm_version_is("0.28.0"):
        monkeypatch.setattr(scheduler.kv_cache_manager, "take_boundary_state_offloads", drain)
    monkeypatch.setattr(scheduler.kv_cache_manager, "get_block_ids", get_blocks)
    seen_states: list[Any] = []
    metadata = object()

    def build_metadata(connector, output):
        assert connector is scheduler.connector
        if vllm_version_is("0.28.0"):
            assert "kv_connector_block_state" not in vars(output)
            seen_states.append(None)
        else:
            seen_states.append(output.kv_connector_block_state)
        return metadata

    def update_after_schedule(output):
        if vllm_version_is("0.28.0"):
            assert "kv_connector_block_state" not in vars(output)
        else:
            assert output.kv_connector_block_state is None

    monkeypatch.setattr(scheduler, "_build_kv_connector_meta", build_metadata)
    monkeypatch.setattr(scheduler, "_update_after_schedule", update_after_schedule)

    first_output = scheduler.schedule()
    second_output = scheduler.schedule()

    if vllm_version_is("0.28.0"):
        # No boundary-state interface exists in this release. Neither invent
        # one on the manager nor attach a main-only field to worker output.
        drain.assert_not_called()
        get_blocks.assert_not_called()
        assert "kv_connector_block_state" not in vars(first_output)
        assert "kv_connector_block_state" not in vars(second_output)
    else:
        assert drain.call_count == 2
        assert first_output.kv_connector_block_state is None
        assert second_output.kv_connector_block_state is None
    if with_connector:
        assert first_output.kv_connector_metadata is metadata
        assert second_output.kv_connector_metadata is metadata
        if vllm_version_is("0.28.0"):
            assert seen_states == [None, None]
        else:
            assert seen_states[0].block_ids == {"cached": ([1, 9],), "boundary": ([42],)}
            assert seen_states[0].boundary_state_offloads is offers
            assert seen_states[1].block_ids == {"cached": ([1, 9],)}
            assert seen_states[1].boundary_state_offloads == {}
            assert get_blocks.call_count == 3
    else:
        assert seen_states == []
        get_blocks.assert_not_called()
