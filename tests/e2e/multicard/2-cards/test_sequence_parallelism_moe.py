#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import torch
import torch.nn as nn
import vllm.config
from vllm.compilation.passes.fx_utils import OpOverload
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    init_distributed_environment,
)
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    initialize_model_parallel,
)
from vllm.utils.system_utils import update_environment_variables

import vllm_ascend.ops.register_custom_ops  # noqa
from tests.e2e.singlecard.compile.backend import TestBackend as CompileTestBackend
from vllm_ascend.compilation.passes.sequence_parallelism_moe import (
    SequenceParallelismMoePass,
)
from vllm_ascend.utils import enable_custom_op

MASTER_PORT = 29500


class AllGatherRMSNormModel(nn.Module):
    def __init__(
        self,
        tp_size: int,
        group_name: str,
        hidden_size: int,
        dtype: torch.dtype,
        eps: float = 1e-6,
        device: str = "npu",
    ):
        super().__init__()
        self.tp_size = tp_size
        self.group_name = group_name
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm_w = torch.randn(hidden_size, dtype=dtype, device=device)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        num_tokens_helper: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = num_tokens_helper.shape[0]
        z = torch.relu(x)
        gathered = torch.ops.vllm.all_gather(z, 0, self.tp_size, self.group_name)
        sliced = gathered[:num_tokens]
        rms_out = torch.ops._C_ascend.npu_add_rms_norm_bias(sliced, residual, self.norm_w, None, self.eps)
        return rms_out[0]

    def ops_in_model_before(self) -> list[OpOverload]:
        return [
            torch.ops.vllm.all_gather.default,
            torch.ops._C_ascend.npu_add_rms_norm_bias.default,
        ]

    def ops_in_model_after(self) -> list[OpOverload]:
        return [
            torch.ops.vllm.all_gather.default,
            torch.ops._C_ascend.npu_add_rms_norm_bias.default,
            torch.ops.vllm.maybe_chunk_residual.default,
        ]


def _run_sequence_parallelism_moe_test(
    local_rank: int,
    world_size: int,
    master_port: int,
    batch_size: int = 8,
    seq_len: int = 16,
    hidden_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    eps: float = 1e-5,
):
    torch.npu.set_device(local_rank)
    torch.set_default_device(f"npu:{local_rank}")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
        }
    )

    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        local_rank=local_rank,
        backend="hccl",
    )

    model_config = ModelConfig(
        model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        dtype=dtype,
    )
    vllm_config = VllmConfig(model_config=model_config)

    try:
        with vllm.config.set_current_vllm_config(vllm_config):
            initialize_model_parallel(tensor_model_parallel_size=world_size)

            if not enable_custom_op():
                raise RuntimeError("vllm_ascend custom ops are not available")

            tp_group = get_tp_group()
            tp_size = get_tensor_model_parallel_world_size()
            group_name = tp_group.unique_name

            # Force-create the TP group's HCCL communicator before
            # SequenceParallelismMoePass.__init__ traces patterns with
            # make_fx(tracing_mode="real"), which executes real all_gather
            # ops and would otherwise trigger lazy communicator creation
            # that can fail with Bind_Failed on Ascend.
            dummy = torch.zeros(1, dtype=dtype)
            torch.distributed.all_reduce(dummy, group=tp_group.device_group)

            sp_moe_pass = SequenceParallelismMoePass(vllm_config)
            backend = CompileTestBackend(custom_passes=[sp_moe_pass])
            model = AllGatherRMSNormModel(
                tp_size=tp_size,
                group_name=group_name,
                hidden_size=hidden_size,
                dtype=dtype,
                eps=eps,
                device=f"npu:{local_rank}",
            )

            local_tokens = batch_size * seq_len
            num_tokens = local_tokens * tp_size
            x = torch.randn(local_tokens, hidden_size, dtype=dtype)
            residual = torch.zeros(num_tokens, hidden_size, dtype=dtype)
            num_tokens_helper = torch.empty(num_tokens, device=x.device, dtype=dtype)
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(num_tokens_helper, 0)

            unfused = model(x, residual, num_tokens_helper)
            compiled = torch.compile(model, backend=backend)
            fused = compiled(x, residual, num_tokens_helper)
            assert unfused.shape == fused.shape

            assert sp_moe_pass.matched_count == 1
            for op in model.ops_in_model_before():
                assert backend.op_count(op, before=True) == 1
            for op in model.ops_in_model_after():
                assert backend.op_count(op) == 1
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_sequence_parallelism_moe_pass():
    """Test SequenceParallelismMoePass fuses all_gather+slice+npu_add_rms_norm_bias."""
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required")
    if torch.npu.device_count() < 2:
        pytest.skip("Two NPUs are required")

    torch.multiprocessing.spawn(
        _run_sequence_parallelism_moe_test,
        args=(2, MASTER_PORT),
        nprocs=2,
        join=True,
    )
