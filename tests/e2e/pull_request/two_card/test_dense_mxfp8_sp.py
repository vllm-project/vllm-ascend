# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""950DT TP2 operator accuracy, ACL replay and timing (no model checkpoint).

Run: torchrun --standalone --nproc-per-node=2 -m pytest --noconftest -s \
    tests/e2e/pull_request/two_card/test_dense_mxfp8_sp.py

Both timing paths use ND weights; full-model benchmarking must also compare
against the original NZ-capable path with fuse_gemm_comms disabled.
"""

import math
import os
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

torch_npu = pytest.importorskip("torch_npu")

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed.parallel_state import (  # noqa: E402
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)

from vllm_ascend.models.common.ops.sequence_parallel import sp_all_gather, sp_reduce_scatter  # noqa: E402
from vllm_ascend.ops import dense_mxfp8_sp  # noqa: E402

TP_SIZE = 2
HIDDEN_SIZE = 1024
LOCAL_INTERMEDIATE_SIZE = 1024
WARMUP_ITERATIONS = 3
TIMING_ITERATIONS = 20
pytestmark = pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != TP_SIZE, reason="requires torchrun TP2")


@pytest.fixture(scope="module")
def context():
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    name = torch.npu.get_device_name(local_rank).lower()
    if not any(tag in name for tag in ("950", "910_95")):
        pytest.skip("MXFP8 communication fusion requires 950DT")
    if not dense_mxfp8_sp.dense_mxfp8_sp_available():
        pytest.skip("torch_npu does not provide the quantized collective APIs")
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=TP_SIZE))
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=TP_SIZE,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method="env://",
            backend="hccl",
            timeout=timedelta(seconds=300),
        )
        initialize_model_parallel(tensor_parallel_size=TP_SIZE, backend="hccl")
        try:
            yield rank, torch.device("npu", local_rank)
        finally:
            torch.npu.synchronize()
            destroy_model_parallel()
            destroy_distributed_environment()


def _quantize_weight(weight, scale_alg):
    quantized, scale = torch_npu.npu_dynamic_mx_quant(weight, dst_type=torch.float8_e4m3fn, scale_alg=scale_alg)
    return quantized.t().contiguous(), scale.reshape(weight.shape[0], -1, 2).transpose(0, 1).contiguous()


def _case(context, tokens, scale_alg):
    rank, device = context
    generator = torch.Generator().manual_seed(1234)
    full_x = torch.randn(tokens, HIDDEN_SIZE, generator=generator, dtype=torch.bfloat16)
    padded = math.ceil(tokens / TP_SIZE) * TP_SIZE
    full_x = torch.nn.functional.pad(full_x, (0, 0, 0, padded - tokens))
    local_x = full_x.chunk(TP_SIZE)[rank].contiguous().to(device)
    # Different rank weights make missing or duplicate reduction observable.
    generator.manual_seed(5678 + rank)
    gate_weight = torch.randn(2 * LOCAL_INTERMEDIATE_SIZE, HIDDEN_SIZE, generator=generator) / math.sqrt(HIDDEN_SIZE)
    down_weight = torch.randn(HIDDEN_SIZE, LOCAL_INTERMEDIATE_SIZE, generator=generator) / math.sqrt(
        LOCAL_INTERMEDIATE_SIZE * TP_SIZE
    )
    gate_weight, gate_scale = _quantize_weight(gate_weight.to(device=device, dtype=torch.bfloat16), scale_alg)
    down_weight, down_scale = _quantize_weight(down_weight.to(device=device, dtype=torch.bfloat16), scale_alg)

    def qmm(x, weight, scale):
        quantized, activation_scale = torch_npu.npu_dynamic_mx_quant(
            x, dst_type=torch.float8_e4m3fn, scale_alg=scale_alg
        )
        return torch_npu.npu_quant_matmul(
            quantized,
            weight,
            scale,
            pertoken_scale=activation_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            output_dtype=x.dtype,
            group_sizes=[1, 1, 32],
        )

    def baseline():
        gate_up = qmm(sp_all_gather(local_x), gate_weight, gate_scale)
        activated = torch_npu.npu_swiglu(gate_up)
        return sp_reduce_scatter(qmm(activated, down_weight, down_scale))

    def fused():
        gate_up = torch.ops.vllm.dense_mxfp8_all_gather_mm(local_x, gate_weight, gate_scale, scale_alg, TP_SIZE)
        activated = torch_npu.npu_swiglu(gate_up)
        return torch.ops.vllm.dense_mxfp8_mm_reduce_scatter(activated, down_weight, down_scale, scale_alg, TP_SIZE)

    return local_x, baseline, fused


def _assert_close_on_all_ranks(actual, expected):
    torch.npu.synchronize()
    error = None
    try:
        torch.testing.assert_close(actual.float().cpu(), expected.float().cpu(), atol=0.02, rtol=0.05)
    except AssertionError as exc:
        error = str(exc)
    failed = torch.tensor(int(error is not None), device=actual.device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    assert not failed.item(), error or "Numerical validation failed on another TP rank"


@pytest.mark.parametrize("tokens", [2, 255, 256, 257, 512])
@pytest.mark.parametrize("scale_alg", [0, 1])
@torch.inference_mode()
def test_dense_mxfp8_sp_accuracy(context, tokens, scale_alg):
    local_x, baseline, fused = _case(context, tokens, scale_alg)
    actual, expected = fused(), baseline()
    assert actual.shape == local_x.shape
    _assert_close_on_all_ranks(actual, expected)


@torch.inference_mode()
def test_dense_mxfp8_sp_aclgraph_replay(context):
    local_x, baseline, fused = _case(context, 512, 1)
    for _ in range(WARMUP_ITERATIONS):
        fused()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fused()
    for multiplier in (1.0, 1.5):
        local_x.mul_(multiplier)
        expected = baseline()
        graph.replay()
        _assert_close_on_all_ranks(output, expected)


def _time_ms(run, device):
    for _ in range(WARMUP_ITERATIONS):
        run()
    torch.npu.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(TIMING_ITERATIONS):
        run()
    torch.npu.synchronize()
    elapsed = torch.tensor((time.perf_counter() - start) * 1000 / TIMING_ITERATIONS, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.item()


@torch.inference_mode()
def test_dense_mxfp8_sp_operator_timing(context):
    local_x, baseline, fused = _case(context, 2048, 1)
    _assert_close_on_all_ranks(fused(), baseline())
    baseline_ms = _time_ms(baseline, local_x.device)
    fused_ms = _time_ms(fused, local_x.device)
    if context[0] == 0:
        print(f"ND operator timing: baseline_ms={baseline_ms:.4f}, fused_ms={fused_ms:.4f}")
