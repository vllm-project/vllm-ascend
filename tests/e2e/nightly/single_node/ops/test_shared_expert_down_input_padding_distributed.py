# SPDX-License-Identifier: Apache-2.0
"""Compare quantized Down plus TP16 reduce-scatter on 16 idle NPUs."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import tensor_model_parallel_reduce_scatter
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)

from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts, SharedExpertParallelMode

TP = 16


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != TP, reason="requires 16 NPUs")
def test_quantized_down_before_tp16_dcp16_reduce_scatter():
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=TP, decode_context_parallel_size=TP))
    with set_current_vllm_config(config):
        init_distributed_environment(world_size=TP, rank=rank, local_rank=local_rank, backend="hccl")
        try:
            initialize_model_parallel(
                tensor_model_parallel_size=TP, decode_context_model_parallel_size=TP, backend="hccl"
            )
            _check(rank)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def _check(rank):
    shared = AscendSharedExperts.__new__(AscendSharedExperts)
    shared.layer = SimpleNamespace(expert_gate=None)
    shared.moe_config = SimpleNamespace(tp_group=get_tp_group())
    mode = SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
    torch.manual_seed(2026 + rank)
    weight = torch.randint(-127, 128, (384, 7168), dtype=torch.int8, device="npu")
    weight = torch_npu.npu_format_cast(weight, 29)
    weight_scale = torch.rand(7168, dtype=torch.bfloat16, device="npu") / 128

    def down(x, scale):
        return torch_npu.npu_quant_matmul(
            x, weight, weight_scale, pertoken_scale=scale, bias=None, output_dtype=torch.bfloat16
        )

    def candidate(x, scale):
        padded_x, padded_scale = shared._pad_a8_down_input(x, scale, mode)
        return tensor_model_parallel_reduce_scatter(down(padded_x, padded_scale), dim=0)

    def reference(x, scale):
        result = down(x, scale)
        pad = (-x.shape[0]) % TP
        return tensor_model_parallel_reduce_scatter(F.pad(result, (0, 0, 0, pad)) if pad else result, dim=0)

    for rows in (1, 8, 15, 16, 17, 31, 32, 64):
        x = torch.randint(-127, 128, (rows, 384), dtype=torch.int8, device="npu")
        scale = torch.rand(rows, dtype=torch.float32, device="npu") / 128
        expected, actual = reference(x, scale), candidate(x, scale)
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)

    x = torch.ones((8, 384), dtype=torch.int8, device="npu")
    scale = torch.ones(8, dtype=torch.float32, device="npu") / 128
    candidate(x, scale)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = candidate(x, scale)
    for iteration in range(3):
        x.fill_(iteration + 1)
        scale.fill_(0.01 * (iteration + 1))
        torch.npu.synchronize()
        graph.replay()
        expected = reference(x, scale)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
    dist.barrier()
