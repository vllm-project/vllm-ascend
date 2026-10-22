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
└── CustomRowParallelOp
│   ├── MLPRowParallelOp
│   ├── OProjRowParallelOp
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
import torch_npu
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter
from vllm.distributed import split_tensor_along_last_dim
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.common.ops.sequence_parallel import sp_reduce_scatter

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import (
    get_mlp_tp_group,
    get_otp_group,
)
from vllm_ascend.utils import (
    enable_dsa_cp,
    get_weight_prefetch_method,
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


class MatmulAllreduceRowParallelOp(CustomRowParallelOp):
    def __init__(self, layer):
        super().__init__(layer)
        self.matmul_comm_mode = None
        group = self.comm_group
        if get_ascend_device_type() == AscendDeviceType.A5 and self.tp_size > 1:
            group = get_matmul_allreduce_group()
            assert group.ranks == self.comm_group.ranks, "MC2 and TP rank ordering must match"
            self.matmul_comm_mode = "ai_cpu"
        self.hcomm_info = self.get_hcomm_info(group.device_group)

    def apply_impl(self, input_: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        input_parallel = self.get_input_parallel(input_)
        """Calculate the output tensor of forward by considering
        fusing communication and computation."""
        # Dispatch through the custom op so the row-count decision in
        # matmul_and_reduce() below stays opaque to torch.compile. Calling
        # npu_mm_all_reduce_base() inline instead bakes whichever branch trace time
        # happened to take -- profile_run traces at max_num_batched_tokens, so decode
        # would replay a captured graph containing the fused op regardless of its own
        # row count, and the op's internal rtMemcpy aborts a capturing stream.
        output = torch.ops.vllm.matmul_and_reduce(input_parallel, self.unique_prefix)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def matmul_and_reduce(self, input_parallel: torch.Tensor, bias_: Parameter | None) -> torch.Tensor:
        """Fuse matmul with its AllReduce once M is large enough to be worth it.

        aclnnMatmulAllReduce pipelines communication against compute along M, so it
        needs enough rows to tile. Measured on A5, tp2, bf16, [*,3072]x[3072,5120]:
        0.32x at M=4, 0.35x at 8, 0.66x at 64, 0.92x at 256, 1.18x at 512, 1.35x at
        1024, 1.47x at 1536, 1.20-1.29x from 2048 up to 16384. Fusing unconditionally
        therefore cost ~3x on decode-sized batches, so reuse the same 512 threshold the
        AddRMSNorm fusion pass uses. Staying unfused below it also keeps the fused op
        out of decode aclgraph capture, which would otherwise abort on its rtMemcpy
        (acl_graph.py captures with capture_error_mode="global").

        Needs HCCL_BUFFSIZE >= 2048: the 200MB default aborts with 50701 past M~1536.
        """
        from vllm_ascend.compilation.passes.allreduce_rmsnorm_fusion_pass import (
            ALLREDUCE_NORM_FUSE_THRESHOLD,
        )

        if self.reduce_results and self.tp_size > 1 and input_parallel.shape[0] >= ALLREDUCE_NORM_FUSE_THRESHOLD:
            return DeviceOperator.npu_mm_all_reduce_base(
                input_parallel,
                self.layer.weight.t(),
                self.hcomm_info,
                bias=bias_,
                comm_mode=self.matmul_comm_mode,
            )

        assert self.quant_method is not None
        output = self.quant_method.apply(self.layer, input_parallel, bias=bias_)
        if self.reduce_results and self.tp_size > 1:
            # Fork the gate_up weight prefetch here, between this MatMul and the
            # collective, so it overlaps the AllReduce instead of only the
            # AddRmsNorm that prefetch_gemma_rms_norm can reach. That fallback
            # stands down whenever _attention_gate_up_target() is eligible, so
            # without this call gate_up is never prefetched at all.
            # Pass self.prefix, not self.unique_prefix: the latter gains a
            # .unique_prefixN suffix on collision and would fail the target lookup.
            from vllm_ascend.ops.mlp_weight_prefetch import attention_all_reduce_with_prefetch

            output = attention_all_reduce_with_prefetch(output, self.prefix)
        return output

    def update_attrs(self):
        # torch.ops.vllm.matmul_and_reduce resolves the layer by this prefix.
        super().update_attrs()
        self.unique_prefix = self.layer.unique_prefix

    @staticmethod
    def get_hcomm_info(group: ProcessGroup) -> str:
        # A class-wide cache can return a handle from a different domain or
        # a destroyed process group. The backend already caches its handle.
        rank = torch.distributed.get_rank(group)
        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            return group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
        return group.get_hccl_comm_name(rank)


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
        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mlp_weight_preprocess(
            weight_prefetch_method.MLP_GATE_UP, output, self.prefix
        )

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


class KimiOProjMMReduceScatterOp(CustomRowParallelOp):
    """Return the token shard of Kimi's BF16 TP output projection."""

    SUPPORTED_TP_SIZES = (2, 4, 8, 16, 32, 64)
    MIN_K = 256
    MAX_K = 65535
    MAX_COMM_BYTES = 16 * 256 * 1024 * 1024

    @staticmethod
    def unsupported_reason(layer) -> str | None:
        if layer.custom_op is not None:
            return "Kimi O-projection MM ReduceScatter requires the original TP group."
        if not isinstance(layer.quant_method, UnquantizedLinearMethod) or layer.weight.dtype != torch.bfloat16:
            return "Kimi O-projection MM ReduceScatter requires unquantized BF16 O-projection weights."
        if layer.bias is not None:
            return "Kimi O-projection MM ReduceScatter requires bias-free O projections."
        if not callable(getattr(torch_npu, "npu_quant_mm_reduce_scatter", None)):
            return "Kimi O-projection MM ReduceScatter requires the torch_npu V2 interface."
        if get_tp_group().world_size not in KimiOProjMMReduceScatterOp.SUPPORTED_TP_SIZES:
            return "Kimi O-projection MM ReduceScatter requires TP size 2, 4, 8, 16, 32, or 64."
        weight = layer.weight
        if weight.ndim != 2 or weight.shape[0] == 0:
            return "Kimi O-projection MM ReduceScatter requires nonempty 2D weights."
        if not KimiOProjMMReduceScatterOp.MIN_K <= weight.shape[1] < KimiOProjMMReduceScatterOp.MAX_K:
            return "Kimi O-projection MM ReduceScatter requires local K in [256, 65535)."
        return None

    def __init__(self, layer):
        super().__init__(layer)
        if reason := self.unsupported_reason(layer):
            raise ValueError(reason)
        self.update_attrs()
        device_group = self.comm_group.device_group
        backend = device_group._get_backend(torch.device("npu"))
        self.hcom = backend.get_hccl_comm_name(self.tp_rank)
        self.world_size = self.tp_size

    def apply_impl(self, input_: torch.Tensor) -> tuple[torch.Tensor, None]:
        assert self.quant_method is not None
        input_parallel = self.get_input_parallel(input_)
        weight = self.layer.weight
        # Branch only on TP-consistent tensor metadata, before any collective.
        # The decoder skips its RS and MLA already owns a sharded output buffer,
        # so the unfused path must return the same token shard as fusion.
        if not self._can_fuse(input_parallel, weight):
            output = self.quant_method.apply(self.layer, input_parallel, None)
            return sp_reduce_scatter(output), None
        input_parallel = input_parallel.contiguous()
        # sp_reduce_scatter pads the GEMM result. With bias-free O projections,
        # padding the input before the fused GEMM gives the same zero rows.
        sp_pad = (-input_parallel.shape[0]) % self.world_size
        if sp_pad:
            input_parallel = torch.nn.functional.pad(input_parallel, (0, 0, 0, sp_pad))
        # The V2 interface supports BF16 inference without quantization scales.
        # Keep TP communication on AI CPU, independently of the MoE EP engine.
        output, _ = torch_npu.npu_quant_mm_reduce_scatter(
            input_parallel,
            self.layer.weight.t(),
            self.hcom,
            self.world_size,
            reduce_op="sum",
            comm_mode="ai_cpu",
        )
        return output, None

    def apply_into(self, input_: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Write the ordinary decode projection directly into its TP shard."""
        assert self.quant_method is not None
        input_parallel = self.get_input_parallel(input_)
        parallel_output = self.quant_method.apply(self.layer, input_parallel, None)
        dist.reduce_scatter_tensor(output, parallel_output, group=self.comm_group.device_group)
        return output

    def _can_fuse(self, input_parallel: torch.Tensor, weight: torch.Tensor) -> bool:
        # Keep decode and mixed batches on the existing MM + RS path.
        if is_forward_context_available():
            attn_metadata = get_forward_context().attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = next(
                    (
                        meta
                        for meta in attn_metadata.values()
                        if hasattr(meta, "num_prefills") and hasattr(meta, "num_decodes")
                    ),
                    None,
                )
            if (
                attn_metadata is None
                or getattr(attn_metadata, "num_prefills", 0) == 0
                or getattr(attn_metadata, "num_decodes", 0) != 0
            ):
                return False
        if input_parallel.ndim != 2 or weight.ndim != 2:
            return False
        if input_parallel.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            return False
        if not self.MIN_K <= input_parallel.shape[1] < self.MAX_K or input_parallel.shape[1] != weight.shape[1]:
            return False
        if not (weight.is_contiguous() or weight.t().is_contiguous()):
            return False
        num_tokens, _ = input_parallel.shape
        if num_tokens == 0 or weight.shape[0] == 0:
            return False
        padded_tokens = num_tokens + (-num_tokens) % self.world_size
        return padded_tokens * weight.shape[0] * weight.element_size() < self.MAX_COMM_BYTES


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


def _get_column_parallel_op(
    prefix, layer
) -> MLPColumnParallelOp | DSV4OProjColumnParallelOp | ShardedCPColumnParallelOp | None:
    if enable_dsa_cp() and ("q_b_proj" in prefix or "kv_b_proj" in prefix):
        return ShardedCPColumnParallelOp(layer)
    if "wo_a" in prefix and oproj_tp_enable():
        return DSV4OProjColumnParallelOp(layer)
    if "gate_up_proj" in prefix and mlp_tp_enable() and not is_moe_layer(prefix):
        return MLPColumnParallelOp(layer)
    return None


def _get_row_parallel_op(prefix, layer) -> MLPRowParallelOp | OProjRowParallelOp | DSV4OProjRowParallelOp | None:
    if "wo_b" in prefix and oproj_tp_enable():
        return DSV4OProjRowParallelOp(layer)
    if "down_proj" in prefix and mlp_tp_enable() and not is_moe_layer(prefix):
        return MLPRowParallelOp(layer)
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
        MLPColumnParallelOp
        | DSV4OProjColumnParallelOp
        | MLPRowParallelOp
        | OProjRowParallelOp
        | DSV4OProjRowParallelOp
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
