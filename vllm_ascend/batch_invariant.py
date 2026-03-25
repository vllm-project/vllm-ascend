# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/batch_invariant.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
import os

import torch
import torch_npu
import vllm.envs as envs
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

# in case recursive call in reduce_sum.
torch_sum = torch.sum


def vllm_is_batch_invariant() -> bool:
    """Check if batch-invariant mode is enabled.

    This is a compatibility wrapper for the vllm function that was removed
    in recent upstream vLLM refactoring.
    """
    # Try to access from envs module, fall back to environment variable
    if hasattr(envs, 'VLLM_BATCH_INVARIANT'):
        return bool(envs.VLLM_BATCH_INVARIANT)
    else:
        # Fallback to environment variable for older vLLM versions
        return bool(int(os.getenv("VLLM_BATCH_INVARIANT", "0")))


if HAS_TRITON:
    from vllm_ascend.ops.triton.batch_invariant.matmul import (
        addmm_batch_invariant,
        bmm_batch_invariant,
        linear_batch_invariant,
        matmul_batch_invariant,
        mm_batch_invariant,
    )
    from vllm_ascend.ops.triton.batch_invariant.softmax import softmax_batch_invariant


try:
    import batch_invariant_ops  # type: ignore[import-not-found] # noqa

    HAS_ASCENDC_BATCH_INVARIANT = True
except ImportError:
    HAS_ASCENDC_BATCH_INVARIANT = False


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
):
    """AclnnAddRmsNorm can't ensure batch invariant,
    so we need to split it into add and rms_norm.
    """
    x_ = x + residual
    residual_ = x_
    x_, _ = torch_npu.npu_rms_norm(x_, weight, eps)
    return x_, None, residual_


def reduce_sum(x: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    """npu_reduce_sum_batch_invariant requires dim to be specified, but torch.sum
    doesn't require it, so we set dim to -1 by default if dim is None and x.dim()==1.
    """
    dim = -1 if dim is None and x.dim() == 1 else dim
    if x.device.type == "npu" and dim is not None:
        return torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant(x, dim, keepdim)
    # cpu tensor can't use npu_reduce_sum_batch_invariant, so we use torch.sum instead.
    return torch_sum(x, dim, keepdim)


def override_envs_for_invariance():
    # enabling NZ mode introduces NZ format input to the triton operator,
    # resulting in accuracy anomalies.
    os.environ["VLLM_ASCEND_ENABLE_NZ"] = "0"
    # fused operator can't ensure batch invariant, so we disable it.
    os.environ["VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE"] = "0"

    # communication determinism settings
    os.environ["HCCL_DETERMINISTIC"] = "strict"
    os.environ["LCCL_DETERMINISTIC"] = "1"


_batch_invariant_LIB = None


def enable_batch_invariant_mode():
    global _batch_invariant_LIB
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")

    # Register operators only implemented in triton.
    if HAS_TRITON:
        _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::softmax", softmax_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::_softmax", softmax_batch_invariant, "NPU")

    # Register operators implemented in Ascend batch-invariant ops in priority.
    if HAS_ASCENDC_BATCH_INVARIANT:
        _batch_invariant_LIB.impl("aten::mm", torch.ops.batch_invariant_ops.npu_mm_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::matmul", torch.ops.batch_invariant_ops.npu_matmul_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::sum", torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant, "NPU")
        # torch_npu.npu_fused_infer_attention_score is a function of torch_npu, not a torch.ops.Operator,
        # so we need to patch it directly.
        torch_npu.npu_fused_infer_attention_score = (
            torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant
        )
        # patch npu_add_rms_norm to ensure batch invariant.
        torch_npu.npu_add_rms_norm = add_rms_norm
        # torch.sum can't be replaced by dispatch logic, so we patch it directly.
        torch.sum = reduce_sum

    # register triton implementations if ascendc is not available.
    elif HAS_TRITON:
        _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, "NPU")

        # linear call matmul internally, so register linear only when ascendc
        # is not available. it will get better performance with ascendc.
        _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, "NPU")


def init_batch_invariance():
    """
    Initialize batch-invariant mode for vLLM on Ascend NPU.

    This function:
    1. Sets environment variables for deterministic computation
    2. Registers batch-invariant implementations for torch operators
    3. Enables batch-invariant flash attention

    Call this function early in your application, or set VLLM_BATCH_INVARIANT=1
    environment variable to enable automatically.
    """
    if vllm_is_batch_invariant():
        if HAS_TRITON or HAS_ASCENDC_BATCH_INVARIANT:
            logger.info(
                "Enabling batch-invariant mode for vLLM on Ascend NPU.",
            )
            override_envs_for_invariance()
            enable_batch_invariant_mode()
        else:
            logger.warning(
                "Batch-invariant mode requested but Triton or AscendC batch-invariant "
                "ops is not available.skipping batch-invariant initialization."
            )
