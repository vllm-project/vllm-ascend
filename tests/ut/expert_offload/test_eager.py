# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.expert_offload.config import ExpertOffloadConfig
from vllm_ascend.expert_offload.expert_offload_manager import (
    ExpertOffloadManager,
    LayerOffloadBinding,
    OffloadState,
)
from vllm_ascend.expert_offload.quantization import MXFP4ExpertAdapter
from vllm_ascend.expert_offload.residency import ExpertResidency
from vllm_ascend.expert_offload.transfer import TorchExpertTransfer


class LayoutScheme:
    """CPU contract double, not a substitute for an MXFP NPU kernel test."""

    group_size = 32

    def process_weights_after_loading(self, layer):
        for name in MXFP4ExpertAdapter.tensor_names:
            getattr(layer, name).data = getattr(layer, name).data.transpose(-1, -2)


class ExpertLayer(torch.nn.Module):
    def __init__(self, name="model.layers.0.mlp.experts"):
        super().__init__()
        self.layer_name = name
        self.local_num_experts = 6
        self.top_k = 2
        self.activation = "silu"
        self.moe_config = SimpleNamespace(device="cpu", is_sequence_parallel=False, has_bias=False)
        shapes = {
            "w13_weight": (6, 128, 32),
            "w2_weight": (6, 64, 32),
            "w13_weight_scale": (6, 128, 2),
            "w2_weight_scale": (6, 64, 2),
        }
        for tensor_name, shape in shapes.items():
            self.register_parameter(
                tensor_name, torch.nn.Parameter(torch.zeros(shape, dtype=torch.uint8), requires_grad=False)
            )
        self._expert_offload_binding = LayerOffloadBinding(
            self, LayoutScheme(), ExpertOffloadConfig(enabled=True, num_device_experts=2)
        )

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
        if weight_name == "ignored":
            return False if return_success else None
        expert_data = param.data if loaded_weight.ndim == 3 else param.data[expert_id]
        if shard_id in ("w1", "w3"):
            half = expert_data.shape[-2] // 2
            start = 0 if shard_id == "w1" else half
            expert_data[..., start : start + half, :].copy_(loaded_weight)
        else:
            expert_data.copy_(loaded_weight)
        return True if return_success else None

    def load(self):
        for eid in range(self.local_num_experts):
            for name in MXFP4ExpertAdapter.tensor_names:
                shards = ("w1", "w3") if name.startswith("w13") else ("w2",)
                for shard in shards:
                    shape = list(getattr(self, name).shape[1:])
                    if name.startswith("w13"):
                        shape[0] //= 2
                    tensor = torch.full(shape, eid + (2 if shard == "w3" else 1), dtype=torch.uint8)
                    getattr(self, name).weight_loader(getattr(self, name), tensor, name, shard, eid)


def ready_model(*layers):
    for layer in layers:
        layer.load()
        layer._expert_offload_binding.process_weights_after_loading(layer)
    model = torch.nn.ModuleList(layers)
    manager = ExpertOffloadManager()
    manager.finalize_loading(model)
    return manager


def assert_selected_bundles(binding, view, physical_ids, logical_ids):
    expected = binding.adapter.materialize(binding.host_store, logical_ids.reshape(-1).tolist(), "cpu")
    for name in MXFP4ExpertAdapter.tensor_names:
        actual = getattr(view, name).index_select(0, physical_ids.reshape(-1).long())
        torch.testing.assert_close(actual, expected[name])
        assert getattr(view, name).stride()[-2:] == expected[name].stride()[-2:]


def test_complete_host_bundle_and_owned_lifetime():
    layer = ExpertLayer()
    layer.load()
    binding = layer._expert_offload_binding
    original = layer.w13_weight.data
    binding.process_weights_after_loading(layer)
    original.zero_()
    assert binding.host_store.tensors["w13_weight"].sum() > 0
    assert layer.w13_weight.shape[0] == 2
    assert set(binding.resident) == set(MXFP4ExpertAdapter.tensor_names)


def test_incomplete_loading_is_rejected_before_slot_allocation():
    layer = ExpertLayer()
    binding = layer._expert_offload_binding
    with pytest.raises(RuntimeError, match="incomplete"):
        binding.process_weights_after_loading(layer)
    assert not binding.resident


def test_rejected_loader_input_does_not_mark_completion():
    layer = ExpertLayer()
    param = layer.w13_weight
    assert param.weight_loader(param, torch.ones(2, 3), "ignored", "w1", 0, return_success=True) is False
    assert not layer._expert_offload_binding.host_store.loaded["w13_weight"]


def test_weight_reload_is_rejected_even_after_close():
    layer = ExpertLayer()
    manager = ready_model(layer)
    manager.close()
    with pytest.raises(RuntimeError, match="reload"):
        layer.w2_weight.weight_loader(layer.w2_weight, torch.ones(4, 3), "w2_weight", "w2", 0)


def test_decode_eviction_prefill_and_overflow_preserve_parameters():
    layer = ExpertLayer()
    manager = ready_model(layer)
    binding = layer._expert_offload_binding
    addresses = {name: getattr(layer, name).data_ptr() for name in binding.resident}
    manager.begin_execution()
    manager.has_prefill = False
    with patch.object(manager, "get_comm", return_value=object()):
        for ids in (torch.tensor([[5, 4]]), torch.tensor([[4, 2]]), torch.tensor([[2, 2]])):
            view, physical, _ = binding.prepare(layer, ids)
            assert_selected_bundles(binding, view, physical, ids)
        resident_before = {name: value.clone() for name, value in binding.resident.items()}
        map_before = binding.residency.logical_to_physical()
        # Oversized decode falls back without changing decode residency.
        ids = torch.tensor([[0, 1], [2, 3]])
        view, physical, _ = binding.prepare(layer, ids)
        assert torch.equal(physical, ids)
        assert_selected_bundles(binding, view, physical, ids)
        manager.has_prefill = True
        view, physical, _ = binding.prepare(layer, torch.tensor([[5, 1]]))
        assert_selected_bundles(binding, view, physical, torch.tensor([[5, 1]]))
        assert binding.residency.logical_to_physical() == map_before
        for name, tensor in resident_before.items():
            torch.testing.assert_close(binding.resident[name], tensor)
        manager.has_prefill = False
        ids = torch.tensor([[2, 4]])
        view, physical, _ = binding.prepare(layer, ids)
        assert_selected_bundles(binding, view, physical, ids)
    assert addresses == {name: getattr(layer, name).data_ptr() for name in binding.resident}
    manager.finish_execution(failed=False)
    assert manager.state is OffloadState.READY


def test_pool_is_reused_between_layers_and_restores_correct_contents():
    first, second = ExpertLayer(), ExpertLayer("model.layers.1.mlp.experts")
    manager = ready_model(first, second)
    second_binding = second._expert_offload_binding
    for tensor in second_binding.host_store.tensors.values():
        tensor.add_(10)
    addresses = {name: tensor.data_ptr() for name, tensor in manager.prefill_pool.tensors.items()}
    manager.begin_execution()
    with patch.object(manager, "get_comm", return_value=None):
        for layer in (first, second, first):
            ids = torch.tensor([[1, 4]])
            view, physical, _ = layer._expert_offload_binding.prepare(layer, ids)
            assert_selected_bundles(layer._expert_offload_binding, view, physical, ids)
    assert addresses == {name: tensor.data_ptr() for name, tensor in manager.prefill_pool.tensors.items()}
    manager.finish_execution(failed=False)


def test_transfer_failure_never_publishes_plan_and_invalidates_worker():
    layer = ExpertLayer()
    manager = ready_model(layer)
    binding = layer._expert_offload_binding
    manager.begin_execution()
    manager.has_prefill = False
    before = binding.residency.logical_to_physical()
    with (
        patch.object(TorchExpertTransfer, "copy_experts", side_effect=RuntimeError("copy failed")),
        pytest.raises(RuntimeError, match="copy failed"),
    ):
        binding.prepare(layer, torch.tensor([[4, 5]]))
    assert binding.residency.logical_to_physical() == before
    assert manager.state is OffloadState.FAILED
    manager.finish_execution(failed=True)
    with pytest.raises(RuntimeError, match="ready"):
        manager.begin_execution()


def test_lifecycle_reentrancy_profile_and_close():
    layer = ExpertLayer()
    manager = ready_model(layer)
    manager.begin_execution(is_profile=True)
    with pytest.raises(RuntimeError, match="serialized"):
        manager.begin_execution()
    manager.has_prefill = False
    with patch.object(manager, "get_comm", return_value=None):
        ids = torch.tensor([[0, 1]])
        view, physical, _ = layer._expert_offload_binding.prepare(layer, ids)
        assert view.w13_weight.shape[0] == 6
        assert torch.equal(physical, ids)
    manager.finish_execution(failed=False)
    manager.close()
    manager.close()
    assert manager.state is OffloadState.CLOSED
    assert not manager.bindings and not manager.prefill_pool.tensors


def test_missing_and_duplicate_layers_are_rejected():
    with pytest.raises(RuntimeError, match="no supported"):
        ExpertOffloadManager().finalize_loading(torch.nn.Module())
    first, second = ExpertLayer(), ExpertLayer()
    with pytest.raises(RuntimeError, match="duplicate"):
        ready_model(first, second)


def test_incompatible_pool_layout_is_rejected_at_initialization():
    first, second = ExpertLayer(), ExpertLayer("model.layers.1.mlp.experts")
    second._expert_offload_binding.adapter.activation = "situ"
    with pytest.raises(ValueError, match="homogeneous"):
        ready_model(first, second)


@pytest.mark.parametrize("fail_preparation", [False, True])
def test_native_loading_prepares_host_storage_before_publishing_residency(monkeypatch, fail_preparation):
    events = []
    binding = object.__new__(LayerOffloadBinding)
    binding.resident = {}
    binding.num_slots = 2
    binding.device = SimpleNamespace(type="npu")
    binding.graph_staging = {}
    binding.residency = ExpertResidency(6, 2)
    tensors = {name: object() for name in MXFP4ExpertAdapter.tensor_names}
    binding.adapter = SimpleNamespace(materialize=lambda *args: tensors)

    def prepare(*args):
        assert binding.residency.logical_to_physical() == (-1,) * 6
        events.append("prepare")
        if fail_preparation:
            raise RuntimeError("native preparation failed")

    binding.host_store = SimpleNamespace(
        finalize=lambda *args, **kwargs: events.append("finalize"),
        prepare_native=prepare,
        close=lambda: events.append("close"),
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_stream=lambda device: SimpleNamespace(synchronize=lambda: events.append("sync"))),
        raising=False,
    )
    layer = SimpleNamespace(**{name: SimpleNamespace(data=None) for name in tensors})
    if fail_preparation:
        with pytest.raises(RuntimeError, match="native preparation failed"):
            binding.process_weights_after_loading(layer)
        assert binding.residency.logical_to_physical() == (-1,) * 6
        assert not binding.resident
        assert events == ["finalize", "sync", "prepare", "close"]
    else:
        binding.process_weights_after_loading(layer)
        assert binding.residency.logical_to_physical() == (0, 1, -1, -1, -1, -1)
        assert events == ["finalize", "sync", "prepare"]
        assert all(getattr(layer, name).data is value for name, value in tensors.items())
