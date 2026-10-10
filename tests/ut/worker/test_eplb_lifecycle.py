# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.eplb import eplb_state as upstream_eplb_state
from vllm.model_executor.models.interfaces import MixtureOfExperts

from vllm_ascend.distributed.eplb import eplb_state as state_module
from vllm_ascend.distributed.eplb.eplb_communicator import AscendHixlEplbCommunicator
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbLayerState, AscendEplbState
from vllm_ascend.worker import worker as worker_module


def _worker(monkeypatch):
    state = MagicMock(spec=AscendEplbState)
    state.model_states = {
        "model": SimpleNamespace(
            communicator=SimpleNamespace(_registered_regions=[(0, 4096)]),
            model=SimpleNamespace(set_eplb_state=MagicMock()),
            expert_load_pass_buffer=object(),
            logical_to_physical_map=object(),
            logical_replica_count=object(),
        )
    }
    worker = worker_module.NPUWorker.__new__(worker_module.NPUWorker)
    worker.model_runner = SimpleNamespace(eplb_state=state)
    allocator = MagicMock()
    allocator.pointer_to_data = {
        0: SimpleNamespace(tag="weights", handle=(0, 1024, 0, 0)),
        1024: SimpleNamespace(tag="kv_cache", handle=(0, 1024, 1024, 0)),
        2048: SimpleNamespace(tag="persistent", handle=(0, 1024, 2048, 0)),
    }
    monkeypatch.setattr(worker_module.CaMemAllocator, "get_instance", lambda: allocator)
    monkeypatch.setattr(worker_module.CaMemAllocator, "sleep_persistent_tag", "persistent")
    monkeypatch.setattr(torch.npu, "mem_get_info", lambda: (0, 4096))
    monkeypatch.setattr(
        worker_module,
        "get_ascend_config",
        lambda: SimpleNamespace(
            weight_nz_mode=0, rl_config=SimpleNamespace(enabled=False, sleep_mode_extra_cleanup=False)
        ),
    )
    return worker, state, allocator


def test_sleep_closes_before_unmap_and_partial_wake_waits_for_all_registered_tags(monkeypatch):
    worker, state, allocator = _worker(monkeypatch)
    allocator.sleep.side_effect = lambda **_kwargs: state.close.assert_called_once_with()
    worker.sleep()
    assert worker._eplb_pending_wake_tags == {"weights", "kv_cache"}
    worker.wake_up(tags=["weights"])
    state.resume.assert_not_called()
    worker.wake_up(tags=["kv_cache"])
    state.resume.assert_called_once_with()
    assert not hasattr(worker, "_eplb_pending_wake_tags")


def test_failed_close_blocks_sleep_unmap(monkeypatch):
    worker, state, allocator = _worker(monkeypatch)
    state.close.side_effect = RuntimeError("still bound")
    with pytest.raises(RuntimeError, match="still bound"):
        worker.sleep()
    allocator.sleep.assert_not_called()


def test_failed_close_blocks_wake_before_remap(monkeypatch):
    worker, state, allocator = _worker(monkeypatch)
    state.raise_if_close_failed.side_effect = RuntimeError("worker must terminate")
    with pytest.raises(RuntimeError, match="worker must terminate"):
        worker.wake_up()
    allocator.wake_up.assert_not_called()


def test_reload_weights_closes_old_registration_before_rebinding(monkeypatch):
    worker, state, _ = _worker(monkeypatch)
    worker.model_runner.reload_weights = MagicMock(side_effect=lambda **_kwargs: state.close.assert_called_once_with())
    worker.reload_weights(weights_path="replacement")
    worker.model_runner.reload_weights.assert_called_once_with(
        weights_iterator=None, weights_path="replacement", is_checkpoint_format=True
    )
    state.finish_weight_reload.assert_called_once_with(True)
    state.resume.assert_called_once_with()


def test_failed_reload_does_not_resume_migration(monkeypatch):
    worker, state, _ = _worker(monkeypatch)
    worker.model_runner.reload_weights = MagicMock(side_effect=RuntimeError("load failed"))
    with pytest.raises(RuntimeError, match="load failed"):
        worker.reload_weights()
    state.close.assert_called_once_with()
    state.resume.assert_not_called()


class _ExpertLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[20.0], [10.0]]), requires_grad=False)
        self.eplb_state = AscendEplbLayerState()

    def get_expert_weights(self):
        return [self.weight]

    def set_eplb_state(self, **kwargs):
        self.eplb_state.set_layer_state(**kwargs)


class _ExpertModel(torch.nn.Module):
    set_eplb_state = MixtureOfExperts.set_eplb_state
    num_moe_layers = 1
    num_routed_experts = 2
    num_redundant_experts = 0
    num_physical_experts = 2

    def __init__(self):
        super().__init__()
        self.moe_layers = torch.nn.ModuleList([_ExpertLayer()])

    def logical_weight(self, logical_id):
        physical_id = self.moe_layers[0].eplb_state.expert_replica_routing_table[0, logical_id]
        return self.moe_layers[0].weight[physical_id].item()


def _live_worker(monkeypatch):
    worker, _, allocator = _worker(monkeypatch)
    monkeypatch.setattr(state_module, "get_ep_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0))
    monkeypatch.setattr(state_module, "get_eplb_group", lambda: object())
    monkeypatch.setattr(upstream_eplb_state, "PIN_MEMORY", False)
    monkeypatch.setattr(torch.npu, "synchronize", lambda: None)
    model = _ExpertModel()
    state = AscendEplbState.__new__(AscendEplbState)
    state._suspended = False
    state._close_error = None
    state._sleep_saved_mappings = None
    state._pending_checkpoint_reload = None
    state._stop_async = threading.Event()
    state.async_worker = None
    state.policy = object()
    state.should_record_tensor = torch.tensor(True)
    state.start_async_loop = MagicMock()
    physical_buffer = torch.tensor([[1, 0, -1]])
    ms = SimpleNamespace(
        model=model,
        physical_to_logical_map_buffer=physical_buffer,
        physical_to_logical_map=physical_buffer[:, :2],
        logical_to_physical_map=torch.tensor([[[1, -1, -1, -1], [0, -1, -1, -1]]]),
        logical_replica_count=torch.ones((1, 2), dtype=torch.long),
        expert_load_pass_buffer=torch.ones((1, 3), dtype=torch.int32),
        expert_load_window=torch.ones((2, 1, 2), dtype=torch.int32),
        num_unpadded_tokens_tensors=[torch.tensor(3)],
        communicator=MagicMock(spec=AscendHixlEplbCommunicator),
        expert_buffer=[],
    )
    state.model_states = {"model": ms}
    ms.communicator._registered_regions = [(model.moe_layers[0].weight.data_ptr(), model.moe_layers[0].weight.nbytes)]
    model.set_eplb_state(ms.expert_load_pass_buffer, ms.logical_to_physical_map, ms.logical_replica_count)
    factory = MagicMock(return_value=MagicMock(spec=AscendHixlEplbCommunicator))
    monkeypatch.setattr(upstream_eplb_state, "create_eplb_communicator", factory)
    worker.model_runner = SimpleNamespace(eplb_state=state, model=model)
    # Metadata can live outside the registered weights pool. Partial wake must
    # wait for this pool too, before writing through graph-captured references.
    tensors = [model.moe_layers[0].weight, *state.lifecycle_tensors()]
    allocator.pointer_to_data = {
        tensor.data_ptr(): SimpleNamespace(
            tag="weights" if tensor is tensors[0] else "metadata",
            handle=(0, tensor.nbytes, tensor.data_ptr(), 0),
        )
        for tensor in tensors
    }
    return worker, state, ms, allocator, factory


@pytest.mark.parametrize("is_checkpoint_format", [True, False])
def test_reload_after_migration_keeps_weight_identity_and_graph_addresses(monkeypatch, is_checkpoint_format):
    worker, state, ms, _, factory = _live_worker(monkeypatch)
    assert ms.model.logical_weight(0) == 10
    mapping_ptr = ms.logical_to_physical_map.data_ptr()
    table_ptr = ms.model.moe_layers[0].eplb_state.expert_replica_routing_table.data_ptr()

    def reload(**kwargs):
        assert state._suspended
        ms.communicator.close.assert_called_once_with()
        if kwargs["is_checkpoint_format"]:
            ms.model.moe_layers[0].weight.copy_(torch.tensor([[10.0], [20.0]]))
        else:
            # Kernel format describes the existing physical slots.
            ms.model.moe_layers[0].weight.copy_(torch.tensor([[40.0], [30.0]]))

    worker.model_runner.reload_weights = reload
    worker.reload_weights(is_checkpoint_format=is_checkpoint_format)
    assert ms.model.logical_weight(0) == (10 if is_checkpoint_format else 30)
    assert ms.logical_to_physical_map.data_ptr() == mapping_ptr
    assert ms.model.moe_layers[0].eplb_state.expert_replica_routing_table.data_ptr() == table_ptr
    assert ms.physical_to_logical_map_buffer[0, 2] == -1
    assert not ms.expert_load_pass_buffer.any()
    assert not ms.expert_load_window.any()
    assert not state.should_record_tensor
    factory.assert_called_once()


@pytest.mark.parametrize("reload_during_partial_wake", [False, True])
def test_level_two_sleep_restores_unregistered_metadata_before_resume(monkeypatch, reload_during_partial_wake):
    worker, state, ms, allocator, factory = _live_worker(monkeypatch)
    assert list(ms.model.named_buffers()) == []
    tensors = list(state.lifecycle_tensors())
    pointers = [tensor.data_ptr() for tensor in tensors]
    table = ms.model.moe_layers[0].eplb_state.expert_replica_routing_table

    def discard(**kwargs):
        assert kwargs["offload_tags"] == ()
        assert state._suspended
        for tensor in tensors:
            tensor.zero_()

    allocator.sleep.side_effect = discard
    worker.sleep(level=2)
    assert worker._eplb_pending_wake_tags == {"weights", "metadata"}
    worker.wake_up(tags=["weights"])
    factory.assert_not_called()
    assert not table.any()
    if reload_during_partial_wake:
        worker.model_runner.reload_weights = lambda **_kwargs: ms.model.moe_layers[0].weight.copy_(
            torch.tensor([[10.0], [20.0]])
        )
        worker.reload_weights()
        factory.assert_not_called()
        assert not table.any()
    worker.wake_up(tags=["metadata"])
    assert ms.model.logical_weight(0) == 10
    assert ms.physical_to_logical_map.tolist() == ([[0, 1]] if reload_during_partial_wake else [[1, 0]])
    assert ms.logical_replica_count.tolist() == [[1, 1]]
    assert [tensor.data_ptr() for tensor in tensors] == pointers
    assert ms.model.moe_layers[0].eplb_state.expert_replica_routing_table is table
    assert not ms.expert_load_window.any()
    assert not hasattr(worker, "_eplb_pending_wake_tags")
    factory.assert_called_once()
