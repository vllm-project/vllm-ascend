# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from tests.ut.core.test_recompute_scheduler import _create_live_recompute_scheduler
from tests.ut.kv_offload.utils import create_request
from vllm_ascend.utils import vllm_version_is


def test_connector_receives_current_block_tables_and_exact_boundary_offers(monkeypatch):
    """Exercise #51358's handoff through schedule(), not a private helper."""
    config, scheduler = _create_live_recompute_scheduler()
    request = create_request(request_id=1, block_size=config.cache_config.block_size)
    scheduler.add_request(request)
    scheduler.requests.update({req_id: Mock() for req_id in ("cached", "boundary", "unchanged")})
    offers = {"boundary": [(1, 42, 128)], "finished": [(1, 99, 128)]}
    block_tables = {
        request.request_id: ([10], [11]),
        "cached": ([20, 21], [22]),
        "boundary": ([40], [42]),
    }
    drain = Mock(return_value=offers)
    if not vllm_version_is("0.28.0"):
        monkeypatch.setattr(scheduler.kv_cache_manager, "take_boundary_state_offloads", drain)
    get_blocks = Mock(side_effect=block_tables.__getitem__)
    connector = Mock()

    def make_cached_request_data(*args):
        # Let real request allocation finish before isolating connector handoff.
        scheduler.connector = connector
        monkeypatch.setattr(scheduler.kv_cache_manager, "get_block_ids", get_blocks)
        return SimpleNamespace(req_ids=["cached", "unchanged"], new_block_ids=[([21], []), None])

    monkeypatch.setattr(scheduler, "_make_cached_request_data", make_cached_request_data)
    seen_states: list[Any] = []
    metadata = object()

    def build_metadata(actual_connector, output):
        assert actual_connector is connector
        if vllm_version_is("0.28.0"):
            assert "kv_connector_block_state" not in vars(output)
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
    output = scheduler.schedule()

    assert output.kv_connector_metadata is metadata
    assert [item.req_id for item in output.scheduled_new_reqs] == [request.request_id]
    if vllm_version_is("0.28.0"):
        drain.assert_not_called()
        get_blocks.assert_not_called()
        assert seen_states == []
    else:
        assert len(seen_states) == 1
        state = seen_states[0]
        assert state.block_ids == block_tables
        assert state.boundary_state_offloads is offers
        assert "finished" not in state.block_ids
        drain.assert_called_once_with()
        assert get_blocks.call_count == len(block_tables)


def test_boundary_offers_are_drained_without_a_connector(monkeypatch):
    _, scheduler = _create_live_recompute_scheduler()
    scheduler.connector = None
    drain = Mock(return_value={"stale": [(0, 1, 128)]})
    if not vllm_version_is("0.28.0"):
        monkeypatch.setattr(scheduler.kv_cache_manager, "take_boundary_state_offloads", drain)
    get_blocks = Mock()
    monkeypatch.setattr(scheduler.kv_cache_manager, "get_block_ids", get_blocks)

    output = scheduler.schedule()

    if vllm_version_is("0.28.0"):
        drain.assert_not_called()
        assert "kv_connector_block_state" not in vars(output)
    else:
        drain.assert_called_once_with()
        assert output.kv_connector_block_state is None
    get_blocks.assert_not_called()
