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
│   ├── MLPColumnParallelOp
│   ├── SequenceColumnParallelOp
└── CustomRowParallelOp
│   ├── MLPRowParallelOp
│   ├── OProjRowParallelOp
│   └── SequenceRowParallelOp
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

from functools import lru_cache
from types import SimpleNamespace

import regex as re
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from vllm.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger
from vllm.model_executor.models.utils import extract_layer_index

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.parallel_state import (
    get_mlp_tp_group,
    get_otp_group,
)
from vllm_ascend.utils import (
    enable_dsa_cp,
    enable_sp,
    is_vl_model,
    mlp_tp_enable,
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


class MLPColumnParallelOp(CustomColumnParallelOp):
    def __init__(self, layer):
        super().__init__(layer)

    @property
    def comm_group(self):
        return get_mlp_tp_group()

    def apply_impl(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None
        # Matrix multiply.
        assert self.quant_method is not None
        input_parallel = self.comm_group.all_gather(input_, 0)
        output = self.quant_method.apply(self.layer, input_parallel, bias)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class MLPRowParallelOp(CustomRowParallelOp):
    def __init__(self, layer):
        super().__init__(layer)

    @property
    def comm_group(self):
        return get_mlp_tp_group()

    def apply_impl(self, input_: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        input_parallel = self.get_input_parallel(input_)

        assert self.quant_method is not None
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.layer.bias
        output_parallel = self.quant_method.apply(self.layer, input_parallel, bias=bias_)
        output = self.comm_group.reduce_scatter(output_parallel, 0)

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


class SequenceColumnParallelOp(CustomColumnParallelOp):
    def apply_impl(self, input_: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        """Linear layer with column parallelism.

        Implemented multiple optimization projects for dense models, such as FlashComm and
        communication-computation fusion.
        """

        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        need_all_gather = not (extract_layer_index(self.layer.prefix) == 0 and is_vl_model() and "attn" in self.prefix)
        input_ = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(input_, label=need_all_gather)
        output_parallel = self.quant_method.apply(self.layer, input_, bias)

        if self.gather_output:
            # All-gather across the partitions.
            output = self.comm_group.all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class SequenceRowParallelOp(CustomRowParallelOp):
    def __init__(self, layer):
        super().__init__(layer)
        self.unique_prefix = None

    def apply_impl(self, input_: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        """Linear layer with column parallelism.

        Implemented multiple optimization projects for dense models, such as FlashComm and
        communication-computation fusion.
        """
        input_parallel = self.get_input_parallel(input_)

        assert self.quant_method is not None
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias

        if self.tp_size == 1 or not self.reduce_results:
            output = self.quant_method.apply(self.layer, input_parallel, bias=bias_)
        else:
            output = torch.ops.vllm.matmul_and_reduce(input_parallel, self.unique_prefix)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def matmul_and_reduce(self, input_parallel: torch.Tensor, bias_: Parameter | None) -> torch.Tensor:
        assert self.quant_method is not None
        try:
            flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled
            mmrs_fusion = _EXTRA_CTX.mmrs_fusion
        except AssertionError:
            flash_comm_v1_enabled = False
            mmrs_fusion = False
            logger.debug(
                "matmul_and_reduce: _EXTRA_CTX access failed (profile_run?), "
                "using defaults: flash_comm_v1=False, mmrs_fusion=False",
            )

        x = input_parallel

        if not flash_comm_v1_enabled:
            output_parallel = self.layer.quant_method.apply(self.layer, x, bias=bias_)
            return tensor_model_parallel_all_reduce(output_parallel)

        pad_size = _EXTRA_CTX.pad_size
        dsa_cp_attn_out = enable_dsa_cp() and ("o_proj" in self.layer.prefix or "wo_b" in self.layer.prefix)
        if pad_size > 0 and not dsa_cp_attn_out:
            x = F.pad(x, (0, 0, 0, pad_size))

        world_size = self.layer.tp_size
        hcom_name = get_tp_group().device_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.layer.tp_rank)

        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        from vllm_ascend.quantization.method_adapters import AscendLinearMethod
        from vllm_ascend.quantization.methods import AscendW8A8LinearMethod

        # For unquant
        if mmrs_fusion and isinstance(self.layer.quant_method, UnquantizedLinearMethod):
            output = DeviceOperator.npu_mm_reduce_scatter_base(
                x,
                self.layer.weight.t(),
                hcom_name,
                world_size,
                reduce_op="sum",
                bias=None,
                comm_turn=0,
            )
            if bias_ is not None:
                output.add_(bias_)
        # For w8a8 quant
        elif mmrs_fusion and (
            isinstance(self.layer.quant_method, AscendLinearMethod)
            and isinstance(self.layer.quant_method.quant_method, AscendW8A8LinearMethod)
        ):
            if x.dtype != torch.int8:
                x_quant = torch.ops.vllm.quantize(
                    x,
                    self.layer.aclnn_input_scale,
                    self.layer.aclnn_input_scale_reciprocal,
                    self.layer.aclnn_input_offset,
                )
            else:
                x_quant = x
            quant_bias = self.layer.quant_bias
            deq_scale = self.layer.deq_scale
            output_dtype = torch.bfloat16
            output = DeviceOperator.npu_mm_reduce_scatter_base(
                x_quant,
                self.layer.weight,
                hcom_name,
                world_size,
                reduce_op="sum",
                bias=None,
                comm_turn=0,
                x2_scale=deq_scale,
                output_dtype=output_dtype,
            )
            output = torch.add(output, torch.mul(quant_bias, deq_scale).to(self.layer.params_dtype))
        else:
            output_parallel = self.layer.quant_method.apply(self.layer, x, bias=bias_)
            output = tensor_model_parallel_reduce_scatter(output_parallel, 0)

        return output

    def update_attrs(self):
        super().update_attrs()
        self.input_is_parallel = self.layer.input_is_parallel
        self.reduce_results = self.layer.reduce_results
        self.unique_prefix = self.layer.unique_prefix


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


def _get_column_parallel_op(
    prefix, layer
) -> MLPColumnParallelOp | DSV4OProjColumnParallelOp | SequenceColumnParallelOp | ShardedCPColumnParallelOp | None:
    if enable_dsa_cp() and ("q_b_proj" in prefix or "kv_b_proj" in prefix):
        return ShardedCPColumnParallelOp(layer)
    if "wo_a" in prefix and oproj_tp_enable():
        return DSV4OProjColumnParallelOp(layer)
    if "gate_up_proj" in prefix and mlp_tp_enable() and not is_moe_layer(prefix):
        return MLPColumnParallelOp(layer)
    if enable_sp():
        # "share_expert" added for Step3p5
        if "shared_expert" in prefix or "share_expert" in prefix:
            return None
        sp_column_prefix = [
            "gate_up_proj",  # first MLP of most LLMs
            "in_proj",  # gated deltanet of Qwen3 Next
            "qkv_proj",  # qkv linear of most LLMs
            "conv1d",  # gated deltanet of Qwen3 Next
            "query_key_value",  # qkv linear of Bailing
            "g_proj",  # attention gate projection of Step3p5
        ]
        for a_prefix in sp_column_prefix:
            if a_prefix in prefix and "vision_model" not in prefix:
                return SequenceColumnParallelOp(layer)

    return None


def _get_row_parallel_op(
    prefix, layer
) -> MLPRowParallelOp | OProjRowParallelOp | DSV4OProjRowParallelOp | SequenceRowParallelOp | None:
    if "wo_b" in prefix and oproj_tp_enable():
        return DSV4OProjRowParallelOp(layer)
    if "down_proj" in prefix and mlp_tp_enable() and not is_moe_layer(prefix):
        return MLPRowParallelOp(layer)
    if "o_proj" in prefix and oproj_tp_enable():
        return OProjRowParallelOp(layer)
    if enable_sp():
        # "share_expert" added for Step3p5
        if "shared_expert" in prefix or "share_expert" in prefix:
            return None
        sp_row_prefixes = [
            "o_proj",  # attn output linear of most LLMs
            "out_proj",  # attn output linear of Qwen3 Next
            "down_proj",  # second MLP of most LLMs
            "attention.dense",  # attn output linear of Bailing
            "wo_b",  # attn output linear of v4
        ]
        for a_prefix in sp_row_prefixes:
            if a_prefix in prefix and "vision_model" not in prefix:
                return SequenceRowParallelOp(layer)

    return None


def get_parallel_op(disable_tp, prefix, layer, direct):
    if (
        disable_tp
        or ("shared_experts" in prefix and shared_expert_dp_enabled())
        or ("shared_expert" in prefix and shared_expert_dp_enabled())
        or ("share_expert" in prefix and shared_expert_dp_enabled())  # "share_expert" added for Step3p5
    ):
        return None, 0, 1
    custom_op: (
        MLPColumnParallelOp
        | DSV4OProjColumnParallelOp
        | SequenceColumnParallelOp
        | MLPRowParallelOp
        | OProjRowParallelOp
        | DSV4OProjRowParallelOp
        | SequenceRowParallelOp
        | ShardedCPColumnParallelOp
        | None
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


def is_moe_layer(prefix: str) -> bool:
    @lru_cache(maxsize=1)
    def get_moe_params():
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        config = vllm_config.model_config.hf_text_config
        n_routed_experts = getattr(config, "n_routed_experts", 0)
        first_k_dense_replace = getattr(config, "first_k_dense_replace", float("inf"))
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        return n_routed_experts, first_k_dense_replace, moe_layer_freq

    match = re.search(r"layers\.(\d+)\.", prefix)
    if match is None:
        return False
    layer_idx = int(match.group(1))

    n_routed_experts, first_k_dense_replace, moe_layer_freq = get_moe_params()

    return n_routed_experts is not None and layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0
