# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
import textwrap
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
    addresses = (
        binding.topk_ids_cpu.data_ptr(),
        binding.mapping.host_map.data_ptr(),
        binding.mapping.device_map.data_ptr(),
    )
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
            binding.topk_ids_cpu.data_ptr(),
            binding.mapping.host_map.data_ptr(),
            binding.mapping.device_map.data_ptr(),
        )
    manager.unsubscribe_graph_streams()
    native_runtime.npu._subscribe_report.assert_called_once_with(native_runtime.stream)
    native_runtime.npu._unsubscribe_report.assert_called_once_with(native_runtime.stream)


@pytest.mark.parametrize("graph_callback", [False, True])
def test_partial_transfer_failure_cannot_publish_residency(
    expert_factory, manager_factory, native_runtime, monkeypatch, graph_callback
):
    import acl  # type: ignore[import-not-found]

    layer, _ = expert_factory()
    manager_factory(layer)
    binding = layer._expert_offload_binding
    binding.host_store.convert_to_native_layout(binding.adapter, "cpu")
    before = binding.slot_to_expert
    host_map = binding.mapping.host_map.clone()
    copy = acl.rt.memcpy_async
    copies = []

    def fail_second_expert(*args):
        copies.append(args)
        return 1 if len(copies) == len(binding.adapter.output_names) + 1 else copy(*args)

    monkeypatch.setattr(acl.rt, "memcpy_async", fail_second_expert)
    exit_worker = Mock(side_effect=SystemExit(1))
    monkeypatch.setattr(os, "_exit", exit_worker)
    if graph_callback:
        binding.topk_ids_cpu[:2].copy_(torch.tensor([4, 5], dtype=torch.int32))
        with pytest.raises(SystemExit):
            binding.update_graph(2)
        exit_worker.assert_called_once_with(1)
    else:
        slots, loads = binding.plan_expert_loads({4, 5})
        with pytest.raises(RuntimeError, match="asynchronous expert host copy failed"):
            binding.load_resident(slots, loads)
        exit_worker.assert_not_called()
    assert len(copies) == len(binding.adapter.output_names) + 1
    assert native_runtime.calls[-1] == "sync"
    assert binding.slot_to_expert == before
    torch.testing.assert_close(binding.mapping.host_map, host_map)
    assert binding.resident["w13_weight"][0].eq(5).all()  # The first expert really was overwritten.


def test_graph_callback_failure_exits_even_if_callback_exceptions_are_swallowed():
    program = textwrap.dedent("""
        import sys
        from contextlib import nullcontext
        from types import SimpleNamespace
        import torch
        sys.modules["vllm_ascend.ascend_config"] = SimpleNamespace(get_ascend_config=lambda: None)
        sys.modules["vllm_ascend.ops.fused_moe.routed_experts"] = SimpleNamespace(AscendRoutedExperts=torch.nn.Module)
        from vllm_ascend.expert_offload.expert_offload_manager import LayerOffloadBinding

        binding = LayerOffloadBinding.__new__(LayerOffloadBinding)
        binding.layer_name = "injected_failure"
        binding.device = "cpu"
        binding.manager = SimpleNamespace(graph_load_stream=None)
        binding.topk_ids_cpu = torch.tensor([4])
        binding.slot_to_expert = (0, 1)
        torch.npu = SimpleNamespace(set_device=lambda _: None, stream=lambda _: nullcontext())
        def fail(*args):
            raise RuntimeError("injected transfer failure")
        binding.load_resident = fail
        try:
            binding.update_graph(1)
        except BaseException:
            pass  # Model torch_npu clearing callback exceptions before continuing the graph.
        raise SystemExit(99)
    """)
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "terminating worker" in result.stdout + result.stderr
