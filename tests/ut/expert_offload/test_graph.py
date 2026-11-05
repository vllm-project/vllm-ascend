# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager


def test_graph_capacity_includes_speculative_and_padding_rows():
    manager = ExpertOffloadManager()
    manager.bindings = {"layer": SimpleNamespace(top_k=8, num_slots=68)}
    for tokens, uniform, expected in ((4, 4, True), (8, 4, True), (16, 4, False), (4, None, False)):
        desc = SimpleNamespace(num_tokens=tokens, uniform_token_count=uniform)
        assert manager._supports_graph_descriptor(desc) is expected
    descs = [SimpleNamespace(num_tokens=n, uniform_token_count=4) for n in (4, 8, 16)]
    graph = SimpleNamespace(_capture_descs={"full": descs}, _candidates={(3, 0): [descs[-1]]})
    manager.prepare_graph_capture(graph, lambda: False)
    assert graph._capture_descs["full"] == descs[:2]
    assert graph._candidates[(3, 0)] == []


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_graph_replay_refreshes_experts_and_reuses_storage(expert_factory, manager_factory, native_runtime, dtype):
    layer, _ = expert_factory()
    manager = manager_factory(layer)
    binding = layer._expert_offload_binding
    reference = {name: tensor.clone() for name, tensor in manager.full_pool.tensors.items()}
    binding.host_store.convert_to_native_layout(binding.adapter, "cpu")
    addresses = (binding.ids_cpu.data_ptr(), binding.mapping.host_map.data_ptr(), binding.mapping.device_map.data_ptr())
    desc = SimpleNamespace(num_tokens=1, uniform_token_count=1)
    capture = Mock(side_effect=lambda *a, **k: not manager.has_prefill)
    graph = SimpleNamespace(device=binding.device, _capture_descs={"full": [desc]}, _candidates={}, capture=capture)
    manager.prepare_graph_capture(graph, lambda: True)
    assert graph.capture("model", progress_bar_desc="test") is True
    capture.assert_called_once_with("model", progress_bar_desc="test")
    assert graph._capture_stream(desc) is native_runtime.stream
    assert graph._capture_stream(desc) is native_runtime.stream
    for graph_mode, logical in (
        (True, [[4, 5]]),
        (True, [[1]]),
        (False, [[2, 3]]),
        (True, [[4]]),
        (True, [[0, 1]]),
        (True, [[2], [2]]),
    ):
        ids = torch.tensor(logical, dtype=dtype)
        expected_slots, _ = binding.plan_expert_loads(set(ids.reshape(-1).tolist()))
        if graph_mode:
            physical = binding.prepare_graph(ids)
            assert native_runtime.npu._launch_host_func.call_args.args[-1] == ids.numel()
        else:
            _, physical, _ = binding.prepare(layer, ids)
        assert binding.slot_to_expert == expected_slots
        for name in binding.adapter.output_names:
            torch.testing.assert_close(binding.resident[name][physical.long()], reference[name][ids.long()])
        slots = binding.slot_to_expert
        assert binding.mapping.host_map.tolist() == [
            slots.index(eid) if eid in slots else -1 for eid in range(binding.num_experts)
        ]
        assert addresses == (
            binding.ids_cpu.data_ptr(),
            binding.mapping.host_map.data_ptr(),
            binding.mapping.device_map.data_ptr(),
        )
    manager.unsubscribe_graph_streams()
    native_runtime.npu._subscribe_report.assert_called_once_with(native_runtime.stream)
    native_runtime.npu._unsubscribe_report.assert_called_once_with(native_runtime.stream)
