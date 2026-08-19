#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""VIT Sequence Parallelism (TP+SP hybrid) strategy interface.

Defines ``VisionSPStrategy`` — an abstract interface with 2 methods that the
Forward patches call. Phase 1 implements ``NaiveVisionSPStrategy`` (sequential
comm + matmul using existing primitives). Phase 2 will implement
``FusedVisionSPStrategy`` (fused comm+matmul ops for 通算掩盖). The Forward
code is identical for both phases — only the strategy implementation changes.

``matmul_and_reducescatter`` is a standalone function (not part of the
strategy interface) because no fused version is planned for ReduceScatter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from vllm.distributed import (
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)


class VisionSPStrategy(ABC):
    """Abstract interface for VIT SP communication strategy.

    Forward patches call these 2 methods for the attention path (AllGather+
    matmul and AllToAll+matmul). The MLP path uses
    ``vision_matmul_and_reducescatter`` directly (no strategy, no fused plan).
    """

    @abstractmethod
    def allgather_and_matmul(self, input_: torch.Tensor, layer) -> torch.Tensor:
        """AllGather(seq dim=0) + column-parallel matmul.

        Used for qkv_proj and FFN UP (linear_fc1).
        Args:
            input_: [local_seq, ..., in_features]  (SP sharded on dim=0)
            layer:  ColumnParallelLinear / QKVParallelLinear
        Returns:
            [full_seq, ..., out_features_per_partition]

        Note: assumes ``layer.quant_method.apply`` returns a single tensor
        (not a tuple). This holds when ``return_bias=False`` and
        ``skip_bias_add=False``, which is the case for all current Qwen3-VL
        vision layers. If a future layer uses ``skip_bias_add=True`` with a
        real bias, this method would need to unpack the tuple and add bias.
        """

    @abstractmethod
    def alltoall_matmul(self, input_: torch.Tensor, layer) -> torch.Tensor:
        """AllToAll(seq->head) + full matmul with ReplicatedLinear weight.

        Used for o_proj (proj). After AllToAll each rank holds all heads
        for its local_seq. The o_proj is a ReplicatedLinear (full weight
        [H, all_h*head_dim] on every rank), so each rank can compute the
        full output without AllReduce.

        Args:
            input_: [full_seq, ..., local_h*head_dim]  (FA output)
            layer:  ReplicatedLinear (full weight on every rank)
        Returns:
            [local_seq, ..., hidden]
        """


def vision_matmul_and_reducescatter(input_: torch.Tensor, layer) -> torch.Tensor:
    """Row-parallel matmul + ReduceScatter(seq dim=0).

    Standalone function — not part of VisionSPStrategy because no fused
    version is planned for ReduceScatter. Uses existing primitives directly.

    Used for FFN Down (linear_fc2).
    Args:
        input_: [full_seq, ..., local_ffn_dim]
        layer:  RowParallelLinear
    Returns:
        [local_seq, ..., hidden]
    """
    tp_rank = get_tp_group().rank_in_group
    # Row-parallel matmul: bias only on rank 0 to avoid double-add after
    # ReduceScatter (ReduceScatter sums across ranks).
    bias_ = None if (tp_rank > 0 or getattr(layer, "skip_bias_add", False)) else layer.bias
    output_parallel = layer.quant_method.apply(layer, input_, bias_)
    # ReduceScatter on seq dim: [full_seq, ...] -> [local_seq, ...]
    output = tensor_model_parallel_reduce_scatter(output_parallel, dim=0)
    return output


class NaiveVisionSPStrategy(VisionSPStrategy):
    """Phase 1: sequential comm + matmul using existing primitives.

    No fused ops. All communication is done first, then matmul (or vice
    versa). This is the simplest correct implementation to validate the
    SP data flow before Phase 2 fusion.
    """

    def allgather_and_matmul(self, input_, layer):
        # 1. AllGather on seq dim: [local_seq, ...] -> [full_seq, ...]
        full_input = tensor_model_parallel_all_gather(input_, dim=0)
        # 2. Column-parallel matmul (bias added on all ranks since each
        #    rank holds a different output shard)
        bias = layer.bias if not getattr(layer, "skip_bias_add", False) else None
        output = layer.quant_method.apply(layer, full_input, bias)
        return output

    def alltoall_matmul(self, input_, layer):
        tp_group = get_tp_group()
        # 1. AllToAll: scatter seq (dim=0), gather head (dim=-1)
        #    [full_seq, ..., local_h*head_dim] -> [local_seq, ..., all_h*head_dim]
        gathered = tp_group.all_to_all(input_, scatter_dim=0, gather_dim=-1)
        # 2. Full matmul with ReplicatedLinear weight [H, all_h*head_dim].
        #    No AllReduce needed — each rank already has all heads for its
        #    local_seq after AllToAll, and the weight is replicated.
        bias = layer.bias if not getattr(layer, "skip_bias_add", False) else None
        output = layer.quant_method.apply(layer, gathered, bias)
        return output


_strategy_instance: VisionSPStrategy | None = None


def get_vision_sp_strategy() -> VisionSPStrategy:
    """Factory: returns Naive (Phase 1) or Fused (Phase 2) strategy.

    The choice is controlled by ``enable_vision_sp_fused()``. Forward code
    calls this to get the strategy instance and is identical for both phases.
    """
    global _strategy_instance
    if _strategy_instance is None:
        from vllm_ascend.utils import enable_vision_sp_fused

        if enable_vision_sp_fused():
            _strategy_instance = FusedVisionSPStrategy()
        else:
            _strategy_instance = NaiveVisionSPStrategy()
    return _strategy_instance


def clear_vision_sp_strategy():
    """Reset the strategy singleton. Called from clear_enable_sp()."""
    global _strategy_instance
    _strategy_instance = None


class FusedVisionSPStrategy(VisionSPStrategy):
    """Phase 2: fused comm+matmul (通算掩盖). Interface is identical to Naive.

    Each method replaces sequential comm+matmul with a single fused op that
    overlaps communication and computation. Requires Phase 2 fused ops to be
    available (developed by other team members).

    Reference implementations:
    - allgather_and_matmul: see SequenceColumnParallelOp (linear_op.py:288)
    - alltoall_matmul: see OProjRowParallelOp (linear_op.py:239)

    Note: matmul_and_reducescatter is NOT part of this strategy — it uses
    existing ReduceScatter directly (no fused version planned).
    """

    def allgather_and_matmul(self, input_, layer):
        raise NotImplementedError(
            "FusedVisionSPStrategy is Phase 2. Set additional_config."
            "enable_vision_sp_fused to False to use NaiveVisionSPStrategy "
            "(Phase 1)."
        )

    def alltoall_matmul(self, input_, layer):
        raise NotImplementedError(
            "FusedVisionSPStrategy is Phase 2. Set additional_config."
            "enable_vision_sp_fused to False to use NaiveVisionSPStrategy "
            "(Phase 1)."
        )
