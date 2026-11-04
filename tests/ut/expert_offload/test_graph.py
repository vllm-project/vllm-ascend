# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager, GraphExpertStaging


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


def test_graph_replay_refreshes_experts_and_reuses_storage(expert_factory, manager_factory, native_runtime):
    layer, _ = expert_factory()
    manager = manager_factory(layer)
    binding = layer._expert_offload_binding
    reference = {name: tensor.clone() for name, tensor in manager.prefill_pool.tensors.items()}
    binding.host_store.prepare_native(binding.adapter, binding.resident, "cpu")
    staging = GraphExpertStaging(binding, (1, 2), torch.int32)
    addresses = (staging.ids.data_ptr(), staging.mapping.host_map.data_ptr(), staging.mapping.device_map.data_ptr())
    manager.subscribe_graph_stream(native_runtime.stream)
    manager.subscribe_graph_stream(native_runtime.stream)
    with manager.graph_capture():
        assert manager.capturing_graphs and not manager.has_prefill
        for logical in ([[4, 5]], [[1, 4]], [[2, 2]]):
            ids = torch.tensor(logical, dtype=torch.int32)
            physical = staging.prepare(ids)
            manager.check_graph_errors()
            for name in binding.adapter.output_names:
                torch.testing.assert_close(binding.resident[name][physical.long()], reference[name][ids.long()])
            assert addresses == (
                staging.ids.data_ptr(),
                staging.mapping.host_map.data_ptr(),
                staging.mapping.device_map.data_ptr(),
            )
    assert not manager.capturing_graphs
    manager.close()
    native_runtime.npu._subscribe_report.assert_called_once_with(native_runtime.stream)
    native_runtime.npu._unsubscribe_report.assert_called_once_with(native_runtime.stream)
