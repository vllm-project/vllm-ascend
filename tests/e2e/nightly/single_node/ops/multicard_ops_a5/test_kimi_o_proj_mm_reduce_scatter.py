# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the fused Kimi O projection with GEMM + ReduceScatter on A5.

Run on reserved devices with the same CANN environment as inference:
torchrun --standalone --nproc-per-node=8 -m pytest --noconftest -s <this_file>
"""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops import linear_op


@pytest.fixture(scope="module")
def tp_group():
    if "RANK" not in os.environ:
        pytest.skip("Launch with torchrun --nproc-per-node=8 to test TP communication.")
    if not get_current_hardware_profile().supports(HardwareCapability.MM_REDUCE_SCATTER_AI_CPU_INFERENCE):
        pytest.skip("This test requires Ascend A5.")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    yield SimpleNamespace(
        device_group=dist.group.WORLD,
        rank_in_group=dist.get_rank(),
        world_size=dist.get_world_size(),
    )
    dist.destroy_process_group()


def _latency_ms(fn, iterations=20):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


@pytest.mark.parametrize("num_tokens", [1, 7, 8, 17, 128, 8192])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_kimi_o_proj_mm_reduce_scatter(tp_group, monkeypatch, num_tokens, use_graph):
    rank = tp_group.rank_in_group
    world_size = tp_group.world_size
    # Rank-distinct operands catch missing reductions and incorrect TP groups.
    torch.manual_seed(2026 + rank)
    local_k = 12288 // world_size
    x = torch.randn(num_tokens, local_k, dtype=torch.bfloat16, device="npu") * 0.1
    weight = torch.randn(7168, local_k, dtype=torch.bfloat16, device="npu") * 0.1
    layer = SimpleNamespace(
        weight=weight,
        bias=None,
        custom_op=None,
        quant_method=UnquantizedLinearMethod(),
        input_is_parallel=True,
        input_size_per_partition=local_k,
        reduce_results=False,
        return_bias=True,
        skip_bias_add=False,
        prefix="model.layers.0.self_attn.o_proj",
    )
    # Use the real HCCL process group without loading a model or KV caches.
    monkeypatch.setattr(linear_op, "get_tp_group", lambda: tp_group)
    fused_op = linear_op.KimiOProjMMReduceScatterOp(layer)

    def baseline():
        partial = torch.nn.functional.linear(x, weight)
        padding = (-num_tokens) % world_size
        if padding:
            partial = torch.nn.functional.pad(partial, (0, 0, 0, padding))
        output = torch.empty((partial.shape[0] // world_size, 7168), dtype=x.dtype, device=x.device)
        dist.reduce_scatter_tensor(output, partial, group=tp_group.device_group)
        return output

    def fused():
        return fused_op.apply(x)[0]

    expected = baseline()
    actual = fused()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    if use_graph:
        baseline_graph, fused_graph = torch.npu.NPUGraph(), torch.npu.NPUGraph()
        torch.npu.synchronize()
        with torch.npu.graph(baseline_graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            expected = baseline()
        with torch.npu.graph(fused_graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            actual = fused()
        baseline_graph.replay()
        fused_graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        baseline_fn, fused_fn = baseline_graph.replay, fused_graph.replay
    else:
        baseline_fn, fused_fn = baseline, fused

    baseline_ms = _latency_ms(baseline_fn)
    fused_ms = _latency_ms(fused_fn)
    if rank == 0:
        print(
            f"TP={world_size} M={num_tokens} graph={use_graph} rank=0 "
            f"baseline_ms={baseline_ms:.4f} fused_ms={fused_ms:.4f}"
        )
