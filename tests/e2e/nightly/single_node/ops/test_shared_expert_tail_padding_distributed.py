# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check shared-expert padding against the original path on 16 idle NPUs.

Run on one node with the CANN environment and all 16 devices available:

    torchrun --standalone --nproc_per_node=16 -m pytest -q \
        tests/e2e/nightly/single_node/ops/test_shared_expert_tail_padding_distributed.py

This compares the real vLLM TP reduce-scatter after the old and candidate
padding paths. It is not an end-to-end model accuracy or performance test.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import tensor_model_parallel_reduce_scatter
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)

from vllm_ascend.ops.fused_moe import moe_utils
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts

WORLD_SIZE = 16
HIDDEN_SIZE = 7168
TOKEN_COUNTS = (1, 8, 15, 16, 17, 31, 32, 64)


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE, reason="requires torchrun on 16 NPUs")
def test_shared_expert_tail_padding_tp16_dcp16():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    config = VllmConfig(
        parallel_config=ParallelConfig(tensor_parallel_size=WORLD_SIZE, decode_context_parallel_size=WORLD_SIZE)
    )
    with set_current_vllm_config(config):
        init_distributed_environment(world_size=WORLD_SIZE, rank=rank, local_rank=local_rank, backend="hccl")
        try:
            _check_padding(rank)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def _check_padding(rank: int) -> None:
    initialize_model_parallel(
        tensor_model_parallel_size=WORLD_SIZE,
        decode_context_model_parallel_size=WORLD_SIZE,
        backend="hccl",
    )
    shared = SimpleNamespace(moe_config=SimpleNamespace(tp_group=get_tp_group()))
    # Graph warmup can run without a current-config context. The shared-expert
    # TP group is authoritative and must still select the cached cat path.
    with patch.object(moe_utils, "get_current_vllm_config_or_none", return_value=None):
        moe_utils._PAD_ZERO_BLOCKS.clear()
        for dtype in (torch.bfloat16, torch.float32):
            for rows in TOKEN_COUNTS:
                torch.manual_seed(2026 + rank * 100 + rows)
                value = torch.randn((rows, HIDDEN_SIZE), dtype=dtype, device="npu")
                pad = (-rows) % WORLD_SIZE
                reference = F.pad(value, (0, 0, 0, pad)) if pad else value
                expected = tensor_model_parallel_reduce_scatter(reference, dim=0)
                actual = AscendSharedExperts._pad_and_reduce_scatter(shared, value)
                torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
                if pad:
                    assert any(key[0] == WORLD_SIZE for key in moe_utils._PAD_ZERO_BLOCKS)

        # Exercise collective capture and changing input on a warm zero block.
        value = torch.randn((8, HIDDEN_SIZE), dtype=torch.bfloat16, device="npu")
        expected_graph, actual_graph = torch.npu.NPUGraph(), torch.npu.NPUGraph()
        with torch.npu.graph(expected_graph):
            expected = tensor_model_parallel_reduce_scatter(F.pad(value, (0, 0, 0, 8)), dim=0)
        with torch.npu.graph(actual_graph):
            actual = AscendSharedExperts._pad_and_reduce_scatter(shared, value)
        for step in range(3):
            value.fill_(rank + step)
            torch.npu.synchronize()
            actual_graph.replay()
            expected_graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    dist.barrier()
