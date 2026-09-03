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
"""
This file extends the functionality of linear operations by encapsulating custom
communication groups and forward functions into classes (linear ops).

Current class inheritance structure:
CustomLinearOp
├── CustomColumnParallelOp
│   ├── DSV4OProjColumnParallelOp
│   └── ShardedCPColumnParallelOp
├── CustomRowParallelOp
│   ├── DSV4OProjRowParallelOp
│   └── OProjRowParallelOp
└── CustomReplicatedOp
How to extend a new linear op? Taking column parallel op as an example:
1. Inherit from CustomColumnParallelOp and create a new class MyColumnParallelOp
2. [Optional] The default communication group is the TP group. If a custom communication group is needed,
   override the comm_group method
3. Override the apply method according to requirements, which will replace the original linear.forward
4. Add selection logic for MyColumnParallelOp in the get_column_parallel_op method, typically based on
   prefix and configuration judgments
Row parallel op follows a similar approach - inherit from RowColumnParallelOp and register the new class in
get_row_parallel_op.
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter
from vllm.distributed import split_tensor_along_last_dim
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger

from vllm_ascend.distributed.parallel_state import get_otp_group
from vllm_ascend.utils import (
    enable_dsa_cp,
    oproj_tp_enable,
    shared_expert_dp_enabled,
)


class CustomLinearOp:
    def __init__(self, layer):
        self.layer = layer
        self.bias = None
        self.skip_bias_add = None
        self.return_bias = None
        self.quant_method = None

    # Custom communication group, while determining weight sharding
    @property
    def comm_group(self):
        return get_tp_group()

    @property
    def tp_rank(self):
        return self.comm_group.rank_in_group

    @property
    def tp_size(self):
        return self.comm_group.world_size

    # Update the attributes required by apply(), obtaining them from the layer.
    # Call this after the layer completes its initialization, specifically at the end of layer.init().
    def update_attrs(self):
        if hasattr(self.layer, "bias"):
            self.bias = self.layer.bias
        self.skip_bias_add = self.layer.skip_bias_add
        self.return_bias = self.layer.return_bias
        self.quant_method = self.layer.quant_method
        self.prefix = self.layer.prefix

    def apply_impl(self, input_):
        raise NotImplementedError

    # Replace layer.forward to customize the layer computation process.
    def apply(self, input_):
        output, output_bias = self.apply_impl(input_)
        if not self.return_bias:
            return output
        return output, output_bias


class CustomColumnParallelOp(CustomLinearOp):
    def __init__(self, layer):
        super().__init__(layer)
        self.gather_output = None

    def update_attrs(self):
        super().update_attrs()
        self.gather_output = self.layer.gather_output


class CustomRowParallelOp(CustomLinearOp):
    def __init__(self, layer):
        super().__init__(layer)
        self.reduce_results = None
        self.input_is_parallel = None
        self.input_size_per_partition = None

    def update_attrs(self):
        super().update_attrs()
        self.input_is_parallel = self.layer.input_is_parallel
        self.reduce_results = self.layer.reduce_results
        self.input_size_per_partition = self.layer.input_size_per_partition

    def apply(self, input_):
        output, output_bias = self.apply_impl(input_)

        if not self.return_bias:
            return output
        return output, output_bias

    def get_input_parallel(self, input_: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel:
            return input_

        split_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
        return split_input[self.tp_rank].contiguous()


class CustomReplicatedOp(CustomLinearOp):
    def apply_impl(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None

        output = self.quant_method.apply(self.layer, input_, bias)
        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias


class DSV4OProjColumnParallelOp(CustomColumnParallelOp):
    @property
    def comm_group(self):
        return get_otp_group()

    def apply_impl(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self.layer, input_, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output_parallel, output_bias


class DSV4OProjRowParallelOp(CustomRowParallelOp):
    @property
    def comm_group(self):
        return get_otp_group()

    def apply_impl(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        input_parallel = self.get_input_parallel(input_)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self.layer, input_parallel, bias=bias_)
        output_bias = self.bias if self.skip_bias_add else None
        return output_parallel, output_bias


class OProjRowParallelOp(CustomRowParallelOp):
    def __init__(self, layer):
        super().__init__(layer)

    @property
    def comm_group(self):
        return get_otp_group()

    def apply_impl(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        input_parallel = self.get_input_parallel(input_)

        # Prepare tensors for all-to-all communication
        local_batch_size = input_parallel.size(0)
        chunk_size = self.input_size_per_partition
        total_batch_size = local_batch_size * self.tp_size

        # Reshape tensor for efficient cross-device transfer:
        # [batch, dim] -> [tp_size, batch, chunk] -> flattened
        send_buf = input_parallel.reshape(-1, self.tp_size, chunk_size).transpose(0, 1).contiguous().view(-1)

        # Create receive buffer
        recv_buf = torch.empty(total_batch_size * chunk_size, dtype=input_parallel.dtype, device=input_parallel.device)

        # Perform all-to-all communication
        dist.all_to_all_single(recv_buf, send_buf, group=self.comm_group.device_group)
        input_parallel = recv_buf.view(total_batch_size, chunk_size)

        # Only fuse bias add for rank 0 to avoid duplicate bias addition in TP>1
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self.layer, input_parallel, bias=bias_)

        # otp-specific: Combine partial results across devices
        output = self.comm_group.reduce_scatter(output_parallel, dim=0)
        output = output.view(input_.shape[0], self.layer.output_size)

        # Handle bias return based on configuration
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def update_attrs(self):
        super().update_attrs()
        self.input_is_parallel = self.layer.input_is_parallel
        self.input_size_per_partition = self.layer.input_size_per_partition


class ShardedCPColumnParallelOp(CustomColumnParallelOp):
    @property
    def comm_group(self):
        # fake comm group to bypass tp logic
        return SimpleNamespace(world_size=1, rank_in_group=0, device_group=None)

    def apply_impl(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None
        output = self.quant_method.apply(self.layer, input_, bias)
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias


def _is_shared_expert_layer(prefix: str) -> bool:
    return "shared_experts" in prefix or "shared_expert" in prefix or "share_expert" in prefix


def _get_column_parallel_op(prefix, layer) -> DSV4OProjColumnParallelOp | ShardedCPColumnParallelOp | None:
    if enable_dsa_cp() and ("q_b_proj" in prefix or "kv_b_proj" in prefix):
        return ShardedCPColumnParallelOp(layer)
    if "wo_a" in prefix and oproj_tp_enable():
        return DSV4OProjColumnParallelOp(layer)
    return None


def _get_row_parallel_op(prefix, layer) -> OProjRowParallelOp | DSV4OProjRowParallelOp | None:
    if "wo_b" in prefix and oproj_tp_enable():
        return DSV4OProjRowParallelOp(layer)
    if "o_proj" in prefix and oproj_tp_enable():
        return OProjRowParallelOp(layer)
    return None


def get_parallel_op(disable_tp, prefix, layer, direct):
    if _is_shared_expert_layer(prefix):
        # Shared-expert weight layout is decoupled from sequence parallelism:
        # only the shared-expert DP switch replicates weights. Models still
        # pass disable_tp=is_sequence_parallel for shared experts, so SP alone
        # must not force replication here.
        if shared_expert_dp_enabled():
            return None, 0, 1
    elif disable_tp:
        return None, 0, 1
    custom_op: (
        DSV4OProjColumnParallelOp | OProjRowParallelOp | DSV4OProjRowParallelOp | ShardedCPColumnParallelOp | None
    ) = None
    if direct == "row":
        custom_op = _get_row_parallel_op(prefix, layer)

    if direct == "column":
        custom_op = _get_column_parallel_op(prefix, layer)

    if custom_op is not None:
        logger.debug(
            "get_parallel_op: prefix=%s, direct=%s -> %s (tp_rank=%d, tp_size=%d)",
            prefix,
            direct,
            type(custom_op).__name__,
            custom_op.tp_rank,
            custom_op.tp_size,
        )
        return custom_op, custom_op.tp_rank, custom_op.tp_size

    return None, get_tp_group().rank_in_group, get_tp_group().world_size


def get_replicated_op(disable_tp, prefix, layer) -> tuple[CustomReplicatedOp | None, int | None, int | None]:
    if disable_tp:
        return None, None, None

    custom_op = CustomReplicatedOp(layer)
    return custom_op, custom_op.tp_rank, custom_op.tp_size
