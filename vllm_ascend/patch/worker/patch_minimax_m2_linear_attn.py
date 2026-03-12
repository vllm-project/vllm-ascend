#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
# MiniMax-M2 linear attention: MiniMaxText01RMSNormTP weight sharding and NPU q/k norm path.
#

from functools import partial

import torch
import torch.nn as nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.mamba.linear_attn import (
    CustomOp,
    MiniMaxText01RMSNormTP,
)
from vllm.platforms import current_platform

_ORIG_QK_METHOD_NAME: str | None = None
_original_qk_method = None
_qk_is_staticmethod = False

if hasattr(MiniMaxText01RMSNormTP, "forward_qk"):
    _ORIG_QK_METHOD_NAME = "forward_qk"
    _original_qk_method = getattr(MiniMaxText01RMSNormTP, _ORIG_QK_METHOD_NAME)
elif hasattr(MiniMaxText01RMSNormTP, "_normalize_qk"):
    # Older vLLM versions
    _ORIG_QK_METHOD_NAME = "_normalize_qk"
    _original_qk_method = getattr(MiniMaxText01RMSNormTP, _ORIG_QK_METHOD_NAME)

if _ORIG_QK_METHOD_NAME is not None:
    # Detect whether upstream defined it as a staticmethod (some versions do).
    _orig_desc = MiniMaxText01RMSNormTP.__dict__.get(_ORIG_QK_METHOD_NAME)
    _qk_is_staticmethod = isinstance(_orig_desc, staticmethod)


def _patched_qk(
    q_norm: "MiniMaxText01RMSNormTP",
    k_norm: "MiniMaxText01RMSNormTP",
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # NPU fast path: kernelized local RMSNorm for q/k, then TP-global rstd correction.
    if current_platform.device_name == "npu":
        q, q_inv_rms = torch.ops.npu.npu_rms_norm(q, q_norm.weight, q_norm.variance_epsilon)
        k, k_inv_rms = torch.ops.npu.npu_rms_norm(k, k_norm.weight, k_norm.variance_epsilon)

        if q_norm.tp_world > 1:
            q_local_inv_rms = q_inv_rms.to(torch.float32)
            if q_local_inv_rms.shape[-1] != 1:
                q_local_inv_rms = q_local_inv_rms.mean(dim=-1, keepdim=True)
            q_local_var = (q_local_inv_rms.reciprocal().pow(2) - q_norm.variance_epsilon).clamp_min_(0.0)

            k_local_inv_rms = k_inv_rms.to(torch.float32)
            if k_local_inv_rms.shape[-1] != 1:
                k_local_inv_rms = k_local_inv_rms.mean(dim=-1, keepdim=True)
            k_local_var = (k_local_inv_rms.reciprocal().pow(2) - k_norm.variance_epsilon).clamp_min_(0.0)

            qk_var = torch.cat([q_local_var, k_local_var], dim=-1)
            qk_var = tensor_model_parallel_all_reduce(qk_var) / q_norm.tp_world
            q_global_var, k_global_var = qk_var.chunk(2, dim=-1)

            q_local_rstd = torch.rsqrt(q_local_var + q_norm.variance_epsilon)
            k_local_rstd = torch.rsqrt(k_local_var + k_norm.variance_epsilon)
            q_global_rstd = torch.rsqrt(q_global_var + q_norm.variance_epsilon)
            k_global_rstd = torch.rsqrt(k_global_var + k_norm.variance_epsilon)

            q = q * (q_global_rstd / q_local_rstd).to(q.dtype)
            k = k * (k_global_rstd / k_local_rstd).to(k.dtype)

        return q, k

    assert _original_qk_method is not None
    # We install the patch as a staticmethod below, so prefer the static calling
    # convention for the original as well.
    return _original_qk_method(q_norm, k_norm, q, k)


def _patched_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_world_size: int | None = None,
    shard_rank: int | None = None,
) -> None:
    if shard_world_size is None:
        shard_world_size = get_tensor_model_parallel_world_size()
    if shard_rank is None:
        shard_rank = get_tensor_model_parallel_rank()
    shard_size = loaded_weight.shape[0] // shard_world_size
    shard = slice(shard_rank * shard_size, (shard_rank + 1) * shard_size)
    param.data.copy_(loaded_weight[shard])


def _patched_init(
    self: "MiniMaxText01RMSNormTP",
    hidden_size: int,
    eps: float = 1e-6,
    *,
    weight_shard_world_size: int | None = None,
    weight_shard_rank: int | None = None,
) -> None:
    CustomOp.__init__(self)
    self.tp_world = get_tensor_model_parallel_world_size()
    self.tp_rank = get_tensor_model_parallel_rank()
    self.weight_shard_world = weight_shard_world_size or self.tp_world
    self.weight_shard_rank = self.tp_rank if weight_shard_rank is None else weight_shard_rank

    if hidden_size % self.weight_shard_world != 0:
        raise ValueError(
            "MiniMaxText01RMSNormTP hidden_size must be divisible by "
            f"weight_shard_world_size, got hidden_size={hidden_size}, "
            f"weight_shard_world_size={self.weight_shard_world}"
        )

    self.weight = nn.Parameter(torch.ones(int(hidden_size / self.weight_shard_world)))
    self.weight.weight_loader = partial(
        _patched_weight_loader,
        shard_world_size=self.weight_shard_world,
        shard_rank=self.weight_shard_rank,
    )
    self.variance_epsilon = eps


MiniMaxText01RMSNormTP.__init__ = _patched_init
MiniMaxText01RMSNormTP.weight_loader = staticmethod(_patched_weight_loader)

if _ORIG_QK_METHOD_NAME is not None:
    # Force staticmethod style, as requested.
    setattr(MiniMaxText01RMSNormTP, _ORIG_QK_METHOD_NAME, staticmethod(_patched_qk))
