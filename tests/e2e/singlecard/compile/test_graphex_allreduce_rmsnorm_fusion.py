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
import numpy as np
import gc

from vllm.config import VllmConfig
from vllm_ascend.compilation.npugraph_ex_passes.graphex_allreduce_rmsnorm_fusion_pass.py import \
    (GraphEXMiddleLayerMatmulAllReduceAddRMSNormPattern, GraphEXLastLayerMatmulAllReduceAddRMSNormPattern)
from vllm.distributed import get_tensor_model_parallel_world_size


BATCH_SIZES = [1, 2]
SEQ_LENS = [1, 4, 32, 128]
HIDDEN_SIZES = [4096]
EPS = [1e-6]
DTYPES = [torch.bfloat16, torch.float16]
SEEDS = [0]
DEVICES = ["npu:0"]
DEFAULT_ATOL = 5e-2
DEFAULT_RTOL = 5e-3


def manual_matmul_allreduce_add_rmsnorm(x, weight, residual, rms_norm_weight, eps):
    mm = torch.mm(x, weight.t())
    all_reduced = mm
    added = all_reduced + residual

    added_f32 = added.to(torch.float32)
    variance = torch.mean(added_f32 ** 2, dim = -1, keepdim=True)
    normalized = added_f32 * torch.rsqrt(variance + eps)
    output = normalized * rms_norm_weight.to(torch.float32)
    output = output.to(x.dtype)

    return output, added


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_graphex_middle_layer_fusion(batch_size, seq_len, hidden_size, eps, dtype, seed, device):
    if get_tensor_model_parallel_world_size() != 1:
        pytest.skip("This test assumes TP = 1 for simplicity.")

    torch.manual_seed(seed)
    torch.set_default_device(device)
    vllm_config = VllmConfig()
    pattern_inst = GraphEXMiddleLayerMatmulAllReduceAddRMSNormPattern(vllm_config, eps)

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    rms_norm_weight = torch.randn(hidden_size, dtype=dtype, device=device)

    out0_fused, out1_fused = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm(
        x,
        weight,
        residual,
        rms_norm_weight,
        pattern_inst.tp_group_name,
        pattern_inst.tp_size,
        pattern_inst.local_rank,
        eps,
        True,
        False,
    )

    out0_matmul, out1_matmul = manual_matmul_allreduce_add_rmsnorm(
        x, weight, residual, rms_norm_weight, eps
    )

    torch.testing.assert_close(
        out0_fused.to(torch.float32).cpu(),
        out0_matmul.cpu(),
        atol=DEFAULT_ATOL,
        rtol=DEFAULT_RTOL,
    )
    torch.testing.assert_close(
        out1_fused.to(torch.float32).cpu(),
        out1_matmul.cpu(),
        atol=DEFAULT_ATOL,
        rtol=DEFAULT_RTOL,
    )

    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_graphex_last_layer_fusion(batch_size, seq_len, hidden_size, eps, dtype, seed, device):
    if get_tensor_model_parallel_world_size() != 1:
        pytest.skip("This test assumes TP=1 for simplicity.")

    torch.manual_seed(seed)
    torch.set_default_device(device)

    vllm_config = VllmConfig()
    pattern_inst = GraphEXLastLayerMatmulAllReduceAddRMSNormPattern(vllm_config, eps=eps)

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    rms_norm_weight = torch.randn(hidden_size, dtype=dtype, device=device)

    out0_fused = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm(
        x,
        weight,
        residual,
        rms_norm_weight,
        pattern_inst.tp_group_name,
        pattern_inst.tp_size,
        pattern_inst.local_rank,
        eps,
        True,
        False,
    )[0]

    out0_manual, _ = manual_matmul_allreduce_add_rmsnorm(
        x, weight, residual, rms_norm_weight, eps
    )

    torch.testing.assert_close(
        out0_fused.to(torch.float32).cpu(),
        out0_manual.cpu(),
        atol=DEFAULT_ATOL,
        rtol=DEFAULT_RTOL,
    )

    gc.collect()
    torch.npu.empty_cache()
