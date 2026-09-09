# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import vllm_ascend.core.recompute_scheduler as scheduler_module
from vllm_ascend.core.recompute_scheduler import RecomputeScheduler
from vllm_ascend.utils import vllm_version_is


def test_connector_receives_current_block_tables_and_exact_boundary_offers(monkeypatch):
    """#51358 requires snapshots, not reconstruction from appended blocks."""
    # Exercise the pure snapshot algorithm on release as well; its production
    # call site is main-only, and release handoff is covered by live schedule tests.
    if vllm_version_is("0.28.0"):
        monkeypatch.setattr(scheduler_module, "KVConnectorBlockState", SimpleNamespace, raising=False)
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.connector = Mock()
    scheduler.requests = {req_id: Mock() for req_id in ("new", "cached", "boundary", "unchanged")}
    offers = {"boundary": [(1, 42, 128)], "finished": [(1, 99, 128)]}
    block_tables = {"new": ([10], [11]), "cached": ([20, 21], [22]), "boundary": ([40], [42])}
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_boundary_state_offloads.return_value = offers
    scheduler.kv_cache_manager.get_block_ids.side_effect = block_tables.__getitem__
    cached_reqs = SimpleNamespace(req_ids=["cached", "unchanged"], new_block_ids=[([21], []), None])

    state = scheduler._take_kv_connector_block_state([SimpleNamespace(req_id="new")], cached_reqs)

    assert state is not None
    assert state.block_ids == block_tables
    assert state.boundary_state_offloads is offers
    assert "finished" not in state.block_ids
    scheduler.kv_cache_manager.take_boundary_state_offloads.assert_called_once_with()


def test_boundary_offers_are_drained_without_a_connector(monkeypatch):
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.connector = None
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_boundary_state_offloads.return_value = {"stale": [(0, 1, 128)]}

    state = scheduler._take_kv_connector_block_state([], SimpleNamespace(req_ids=[], new_block_ids=[]))

    assert state is None
    scheduler.kv_cache_manager.take_boundary_state_offloads.assert_called_once_with()
    scheduler.kv_cache_manager.get_block_ids.assert_not_called()
