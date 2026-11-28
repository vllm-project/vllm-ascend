from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiKVConnectorMetadata

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    update_sparse_kv_offload_metadata,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import SfaPDConsumerMetadata
from vllm_ascend.worker.v2.model_states.default import AscendModelState, DefaultModelState
from vllm_ascend.worker.v2.model_runner import GPUModelRunner, NPUModelRunner


class Buffer:
    def __init__(self, size):
        self.np = np.full(size, -1, dtype=np.int64)
        self.copied = []

    def copy_to_gpu(self, size):
        self.copied.append(self.np[:size].copy())


def test_request_instances_follow_scheduler_lifecycle_on_each_rank():
    states = []
    for _ in range(2):
        state = AscendModelState.__new__(AscendModelState)
        state._selective_native_experiment = True
        state._selective_request_ids = {}
        state._selective_cache_ids = {}
        state._selective_data_versions = {}
        state._selective_write_frontiers = {}
        state._selective_next_request_id = 1 << 32
        state._offload_request_states = None
        state._selective_tails = None
        states.append(state)
    with patch.object(DefaultModelState, "add_request"), patch.object(DefaultModelState, "remove_request"):
        for state in states:
            state.add_request(0, SimpleNamespace(req_id="a", num_computed_tokens=0))
            state.add_request(1, SimpleNamespace(req_id="b", num_computed_tokens=0))
            assert state._selective_request_ids == {"a": 1 << 32, "b": (1 << 32) + 1}
            state.remove_request("a")
            state.add_request(0, SimpleNamespace(req_id="a", num_computed_tokens=0))
            assert state._selective_request_ids == {"a": (1 << 32) + 2, "b": (1 << 32) + 1}


def test_old_path_does_not_allocate_request_id_or_version():
    state = AscendModelState.__new__(AscendModelState)
    state._selective_native_experiment = False
    state._selective_request_ids = {}
    state._selective_cache_ids = {}
    state._selective_data_versions = {}
    state._selective_write_frontiers = {}
    state._selective_next_request_id = 1 << 32
    state._offload_request_states = None
    state._selective_tails = None
    with patch.object(DefaultModelState, "add_request"), patch.object(DefaultModelState, "remove_request"):
        state.add_request(0, SimpleNamespace(req_id="a"))
        state.remove_request("a")
    assert state._selective_request_ids == state._selective_cache_ids == state._selective_data_versions == {}
    assert state._selective_next_request_id == 1 << 32


def test_metadata_copies_instances_and_clears_padding():
    ids, mapping = Buffer(4), Buffer(4)
    update_sparse_kv_offload_metadata(
        2, 2, 4, 4, ["b", "a"], np.array([0, 1, 2]), ids, mapping, request_instance_ids=[2, 3]
    )
    assert ids.copied[-1].tolist() == [2, 3, 0, 0]
    assert mapping.copied[-1].tolist() == [0, 1, 0, 0]
    update_sparse_kv_offload_metadata(
        1, 1, 4, 4, ["a"], np.array([0, 1]), ids, mapping, request_instance_ids=[3]
    )
    assert ids.copied[-1].tolist() == [3, 0, 0, 0]


def test_dummy_and_real_instance_ids_are_disjoint():
    ids, mapping = Buffer(1), Buffer(1)
    update_sparse_kv_offload_metadata(1, 1, 1, 1, ["dummy"], np.array([0, 1]), ids, mapping)
    dummy_id = ids.copied[-1][0]
    real_id = 1 << 32
    update_sparse_kv_offload_metadata(
        1, 1, 1, 1, ["request"], np.array([0, 1]), ids, mapping, request_instance_ids=[real_id]
    )
    assert 0 <= dummy_id < real_id
    assert ids.copied[-1].tolist() == [real_id]


def test_exhausted_instance_ids_fail_before_parent_mutates_state():
    state = AscendModelState.__new__(AscendModelState)
    state._selective_native_experiment = True
    state._selective_request_ids = {}
    state._selective_next_request_id = 1 << 63
    with patch.object(DefaultModelState, "add_request") as parent_add:
        with pytest.raises(OverflowError):
            state.add_request(0, SimpleNamespace(req_id="a"))
    parent_add.assert_not_called()
    assert state._selective_request_ids == {}


def test_rollback_invalidates_only_the_affected_request_before_update():
    state = AscendModelState.__new__(AscendModelState)
    state._selective_native_experiment = True
    state._selective_request_ids = {"a": 1 << 32, "b": (1 << 32) + 1}
    state._selective_cache_ids = dict(state._selective_request_ids)
    state._selective_data_versions = {"a": 0, "b": 0}
    state._selective_write_frontiers = {"a": 110, "b": 100}
    state._selective_next_request_id = (1 << 32) + 2
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_state = state
    runner.req_states = SimpleNamespace(
        req_id_to_index={"a": 0, "b": 1}, num_computed_tokens_np=np.array([100, 100])
    )
    output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["a", "b"], num_computed_tokens=[105, 100], new_block_ids=[None, ([7],)]
        ),
        num_scheduled_tokens={"a": 1, "b": 1},
    )
    def update_parent(_output):
        runner.req_states.num_computed_tokens_np[:] = [105, 100]

    with patch.object(GPUModelRunner, "update_requests", side_effect=update_parent) as parent_update:
        runner.update_requests(output)
    parent_update.assert_called_once_with(output)
    assert state._selective_request_ids == {"a": 1 << 32, "b": (1 << 32) + 1}
    assert state._selective_cache_ids == {"a": (1 << 32) + 2, "b": (1 << 32) + 1}
    assert state._selective_data_versions == {"a": 1, "b": 0}
    assert state._selective_write_frontiers == {"a": 106, "b": 101}


def test_one_token_rollback_is_detected_when_scheduler_start_is_unchanged():
    state = AscendModelState.__new__(AscendModelState)
    state._selective_native_experiment = True
    state._selective_request_ids = {"a": 1 << 32}
    state._selective_cache_ids = dict(state._selective_request_ids)
    state._selective_data_versions = {"a": 0}
    state._selective_write_frontiers = {"a": 101}
    state._selective_next_request_id = (1 << 32) + 1
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_state = state
    runner.req_states = SimpleNamespace(req_id_to_index={"a": 0}, num_computed_tokens_np=np.array([100]))
    output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(req_ids=["a"], num_computed_tokens=[100]),
        num_scheduled_tokens={"a": 1},
    )
    with patch.object(GPUModelRunner, "update_requests"):
        runner.update_requests(output)
    assert state._selective_data_versions == {"a": 1}
    assert state._selective_cache_ids["a"] != state._selective_request_ids["a"]


def test_ordinary_page_append_does_not_invalidate_existing_cache():
    state = AscendModelState.__new__(AscendModelState)
    state._selective_native_experiment = True
    state._selective_request_ids = {"a": 1 << 32}
    state._selective_cache_ids = dict(state._selective_request_ids)
    state._selective_data_versions = {"a": 0}
    state._selective_write_frontiers = {"a": 100}
    state._selective_next_request_id = (1 << 32) + 1
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_state = state
    runner.req_states = SimpleNamespace(req_id_to_index={"a": 0}, num_computed_tokens_np=np.array([100]))
    output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(req_ids=["a"], num_computed_tokens=[100], new_block_ids=[([7],)]),
        num_scheduled_tokens={"a": 1},
    )
    with patch.object(GPUModelRunner, "update_requests"):
        runner.update_requests(output)
    assert state._selective_data_versions == {"a": 0}
    assert state._selective_cache_ids == state._selective_request_ids
    assert state._selective_write_frontiers == {"a": 101}


def test_version_events_deduplicate_and_assign_same_ids_in_different_order():
    results = []
    for req_ids in (["b", "a", "a"], ["a", "b"]):
        state = AscendModelState.__new__(AscendModelState)
        state._selective_cache_ids = {"a": 1 << 32, "b": (1 << 32) + 1}
        state._selective_data_versions = {"a": 0, "b": 0}
        state._selective_next_request_id = (1 << 32) + 2
        state.invalidate_selective_requests(req_ids)
        results.append((state._selective_cache_ids, state._selective_data_versions))
    assert results[0] == results[1]
    assert results[0][1] == {"a": 1, "b": 1}


def test_modified_resident_page_invalidates_only_its_request():
    state = AscendModelState.__new__(AscendModelState)
    state._selective_native_experiment = True
    state._selective_request_ids = {"a": 1 << 32, "b": (1 << 32) + 1}
    state._selective_cache_ids = dict(state._selective_request_ids)
    state._selective_data_versions = {"a": 0, "b": 0}
    state._selective_write_frontiers = {"a": 100, "b": 100}
    state._selective_next_request_id = (1 << 32) + 2
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_state = state
    runner.req_states = SimpleNamespace(req_id_to_index={"a": 0, "b": 1}, num_computed_tokens_np=np.array([100, 100]))
    native = SimpleNamespace(
        _selective_native_started=True,
        tp_size=1,
        tp_rank=0,
        selective_native_cache_ids_for_modified_pages=lambda blocks, active: {1 << 32} if blocks == {7, 9} else set(),
    )
    runner.sparse_kv_offload_manager = native
    output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(req_ids=["a", "b"], num_computed_tokens=[100, 100]),
        num_scheduled_tokens={"a": 1, "b": 1},
        new_block_ids_to_zero=[7], kv_cache_block_copies=[SimpleNamespace(dst_block_id=9)],
    )
    with patch.object(GPUModelRunner, "update_requests"):
        runner.update_requests(output)
    assert state._selective_data_versions == {"a": 1, "b": 0}
    assert state._selective_cache_ids["b"] == state._selective_request_ids["b"]


def test_tp_follower_uses_owner_page_invalidation_result():
    state = AscendModelState.__new__(AscendModelState)
    state._selective_native_experiment = True
    state._selective_cache_ids = {"a": 1 << 32, "b": (1 << 32) + 1}
    state._selective_data_versions = {"a": 0, "b": 0}
    state._selective_write_frontiers = {"a": 100, "b": 100}
    state._selective_next_request_id = (1 << 32) + 2
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_state = state
    runner.req_states = SimpleNamespace(req_id_to_index={"a": 0, "b": 1}, num_computed_tokens_np=np.array([100, 100]))
    scan = MagicMock(side_effect=AssertionError("follower must not scan planner state"))
    broadcast = MagicMock(return_value=(None, {1 << 32}))
    runner.sparse_kv_offload_manager = SimpleNamespace(
        _selective_native_started=True, tp_size=2, tp_rank=1,
        tp_group=SimpleNamespace(broadcast_object=broadcast, cpu_group=object()),
        selective_native_cache_ids_for_modified_pages=scan,
    )
    output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(req_ids=["a", "b"], num_computed_tokens=[100, 100]),
        num_scheduled_tokens={"a": 1, "b": 1}, new_block_ids_to_zero=[7], kv_cache_block_copies=[],
    )
    def gather(output, events, group):
        output[:] = [events, events]

    with patch.object(GPUModelRunner, "update_requests"), patch.object(torch.distributed, "all_gather_object", gather):
        runner.update_requests(output)
    scan.assert_not_called()
    broadcast.assert_called_once_with(None, src=0)
    assert state._selective_data_versions == {"a": 1, "b": 0}


def test_tp_page_event_mismatch_fails_before_parent_update():
    state = SimpleNamespace(_selective_native_experiment=True, _selective_write_frontiers={},
                            _selective_cache_ids={})
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_state = state
    native = SimpleNamespace(_selective_native_started=True, tp_size=2, tp_rank=1,
                             tp_group=SimpleNamespace(cpu_group=object()))
    runner.sparse_kv_offload_manager = native
    output = SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(req_ids=[], num_computed_tokens=[]),
                             new_block_ids_to_zero=[], kv_cache_block_copies=[])

    def gather(results, events, group):
        results[:] = [(None, ((), ())), (None, ((7,), ()))]

    with (patch.object(GPUModelRunner, "update_requests") as parent,
          patch.object(torch.distributed, "all_gather_object", gather)):
        with pytest.raises(RuntimeError, match="events differ"):
            runner.update_requests(output)
    parent.assert_not_called()
    assert native._selective_native_failed


def test_tp_owner_page_scan_error_is_broadcast_before_parent_update():
    state = SimpleNamespace(_selective_native_experiment=True, _selective_write_frontiers={},
                            _selective_cache_ids={})
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_state = state
    group = SimpleNamespace(cpu_group=object(), broadcast_object=MagicMock(side_effect=lambda value, src: value))
    native = SimpleNamespace(_selective_native_started=True, tp_size=2, tp_rank=0, tp_group=group,
                             selective_native_cache_ids_for_modified_pages=MagicMock(side_effect=RuntimeError("scan")))
    runner.sparse_kv_offload_manager = native
    output = SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(req_ids=[], num_computed_tokens=[]),
                             new_block_ids_to_zero=[7], kv_cache_block_copies=[])

    def gather(results, value, group):
        results[:] = [value, value]

    with (patch.object(GPUModelRunner, "update_requests") as parent,
          patch.object(torch.distributed, "all_gather_object", gather)):
        with pytest.raises(RuntimeError, match="page scan failed: scan"):
            runner.update_requests(output)
    group.broadcast_object.assert_called_once_with(("scan", None), src=0)
    parent.assert_not_called()
    assert native._selective_native_failed


@pytest.mark.parametrize("nested", [False, True])
def test_pd_load_invalidates_only_matching_live_request(nested):
    state = AscendModelState.__new__(AscendModelState)
    state._selective_native_experiment = True
    state._selective_request_ids = {"a": 1 << 32, "b": (1 << 32) + 1}
    state._selective_cache_ids = dict(state._selective_request_ids)
    state._selective_data_versions = {"a": 0, "b": 0}
    state._selective_next_request_id = (1 << 32) + 2
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_state = state
    metadata = SfaPDConsumerMetadata()
    metadata.add_request("a", [1], [2])
    metadata.add_request("new", [3], [4])
    if nested:
        other = SimpleNamespace(requests=[SimpleNamespace(req_id="b")])
        metadata = MultiKVConnectorMetadata(metadata=(other, metadata))
    runner._invalidate_selective_pd_loads(metadata)
    assert state._selective_data_versions == {"a": 1, "b": 0}
    assert state._selective_cache_ids == {"a": (1 << 32) + 2, "b": (1 << 32) + 1}


def test_old_path_does_not_change_cache_ids_on_pd_or_rollback():
    state = AscendModelState.__new__(AscendModelState)
    state._selective_native_experiment = False
    state._selective_cache_ids = {"a": 1 << 32}
    state._selective_data_versions = {"a": 0}
    state._selective_write_frontiers = {"a": 110}
    state._selective_next_request_id = (1 << 32) + 1
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_state = state
    metadata = SfaPDConsumerMetadata()
    metadata.add_request("a", [1], [2])
    runner._invalidate_selective_pd_loads(metadata)
    output = SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(req_ids=["a"], num_computed_tokens=[100]),
                             num_scheduled_tokens={"a": 1})
    with patch.object(GPUModelRunner, "update_requests"):
        runner.update_requests(output)
    assert state._selective_cache_ids == {"a": 1 << 32}
    assert state._selective_data_versions == {"a": 0}


@pytest.mark.parametrize("values", [[1], [0, 2], [-1, 2]])
def test_metadata_rejects_invalid_instance_ids(values):
    with pytest.raises(ValueError):
        update_sparse_kv_offload_metadata(
            2, 2, 2, 2, ["a", "b"], np.array([0, 1, 2]), Buffer(2), Buffer(2), request_instance_ids=values
        )


def test_tp_missing_active_ids_reports_before_page_broadcast():
    state = SimpleNamespace(_selective_native_experiment=True, _selective_write_frontiers={},
                            _selective_cache_ids=None)
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_state = state
    group = SimpleNamespace(cpu_group=object(), broadcast_object=MagicMock())
    native = SimpleNamespace(_selective_native_started=True, tp_size=2, tp_rank=1, tp_group=group)
    runner.sparse_kv_offload_manager = native
    output = SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(req_ids=[], num_computed_tokens=[]),
                             new_block_ids_to_zero=[7], kv_cache_block_copies=[])
    def gather(results, status, group):
        results[:] = [status, status]
    with (patch.object(GPUModelRunner, "update_requests") as parent,
          patch.object(torch.distributed, "all_gather_object", gather)):
        with pytest.raises(RuntimeError, match="event preparation failed"):
            runner.update_requests(output)
    group.broadcast_object.assert_not_called()
    parent.assert_not_called()
    assert native._selective_native_failed


def test_tp_event_collective_failure_quarantines_native():
    state = SimpleNamespace(_selective_native_experiment=True, _selective_write_frontiers={},
                            _selective_cache_ids={})
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_state = state
    native = SimpleNamespace(_selective_native_started=True, tp_size=2, tp_rank=1,
                             tp_group=SimpleNamespace(cpu_group=object()))
    runner.sparse_kv_offload_manager = native
    output = SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(req_ids=[], num_computed_tokens=[]),
                             new_block_ids_to_zero=[], kv_cache_block_copies=[])
    with (patch.object(GPUModelRunner, "update_requests") as parent,
          patch.object(torch.distributed, "all_gather_object", side_effect=RuntimeError("collective failed"))):
        with pytest.raises(RuntimeError, match="collective failed"):
            runner.update_requests(output)
    parent.assert_not_called()
    assert native._selective_native_failed


def test_tp_pd_metadata_peer_failure_is_shared_before_invalidation():
    state = SimpleNamespace(_selective_native_experiment=True, invalidate_selective_requests=MagicMock())
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_state = state
    native = SimpleNamespace(_selective_native_started=True, tp_size=2, tp_group=SimpleNamespace(cpu_group=object()))
    runner.sparse_kv_offload_manager = native
    metadata = SfaPDConsumerMetadata()
    metadata.add_request("a", [1], [2])
    def gather(results, status, group):
        results[:] = [status, ("invalid peer metadata", ())]
    with patch.object(torch.distributed, "all_gather_object", gather):
        with pytest.raises(RuntimeError, match="P/D metadata failed or differs"):
            runner._invalidate_selective_pd_loads(metadata)
    state.invalidate_selective_requests.assert_not_called()
    assert native._selective_native_failed


def test_tp_pd_invalidation_failure_is_shared():
    state = SimpleNamespace(_selective_native_experiment=True,
                            invalidate_selective_requests=MagicMock(side_effect=RuntimeError("bad version")))
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_state = state
    native = SimpleNamespace(_selective_native_started=True, tp_size=2, tp_group=SimpleNamespace(cpu_group=object()))
    runner.sparse_kv_offload_manager = native
    metadata = SfaPDConsumerMetadata()
    metadata.add_request("a", [1], [2])
    def gather(results, status, group):
        results[:] = [status, status]
    with patch.object(torch.distributed, "all_gather_object", gather):
        with pytest.raises(RuntimeError, match="P/D invalidation failed"):
            runner._invalidate_selective_pd_loads(metadata)
    assert native._selective_native_failed


def test_tp_page_result_local_iteration_failure_is_shared():
    class BadIds(dict):
        def items(self):
            raise RuntimeError("bad request map")
    state = SimpleNamespace(_selective_native_experiment=True, _selective_write_frontiers={},
                            _selective_cache_ids=BadIds({"a": 1}), invalidate_selective_requests=MagicMock())
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_state = state
    group = SimpleNamespace(cpu_group=object(), broadcast_object=MagicMock(return_value=(None, {1})))
    native = SimpleNamespace(_selective_native_started=True, tp_size=2, tp_rank=1, tp_group=group)
    runner.sparse_kv_offload_manager = native
    output = SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(req_ids=[], num_computed_tokens=[]),
                             new_block_ids_to_zero=[7], kv_cache_block_copies=[])
    def gather(results, value, group):
        results[:] = [value, value]
    with (patch.object(GPUModelRunner, "update_requests") as parent,
          patch.object(torch.distributed, "all_gather_object", gather)):
        with pytest.raises(RuntimeError, match="invalidation failed or differs"):
            runner.update_requests(output)
    state.invalidate_selective_requests.assert_not_called()
    parent.assert_not_called()
    assert native._selective_native_failed


def test_tp_connector_preemption_failure_is_reported_before_pd_work():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    failure = RuntimeError("connector failed")
    native = SimpleNamespace(_selective_native_started=True,
                             check_selective_native_phase_status=MagicMock(side_effect=failure))
    runner.sparse_kv_offload_manager = native
    connector = SimpleNamespace(handle_preemptions=MagicMock(side_effect=failure))
    with patch("vllm_ascend.worker.v2.model_runner.get_kv_transfer_group", return_value=connector):
        with pytest.raises(RuntimeError, match="connector failed"):
            runner._handle_selective_preemptions(object())
    native.check_selective_native_phase_status.assert_called_once_with("connector preemption", failure)


def test_tp_missing_connector_metadata_is_reported():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    phase = MagicMock(side_effect=RuntimeError("peer metadata missing"))
    runner.sparse_kv_offload_manager = SimpleNamespace(_selective_native_started=True,
                                                        check_selective_native_phase_status=phase)
    with pytest.raises(RuntimeError, match="peer metadata missing"):
        runner._handle_selective_preemptions(None)
    assert isinstance(phase.call_args.args[1], AssertionError)
