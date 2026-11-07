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

import regex as re
import torch
import torch.distributed as dist
import torch_npu
from torch.nn.parameter import Parameter
from vllm.distributed import split_tensor_along_last_dim
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
)
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.parallel_state import (
    get_mlp_tp_group,
    get_otp_group,
)
from vllm_ascend.utils import (
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


def _is_shared_expert_layer(prefix: str) -> bool:
    return "shared_experts" in prefix or "shared_expert" in prefix or "share_expert" in prefix


def _get_column_parallel_op(prefix, layer) -> MLPColumnParallelOp | DSV4OProjColumnParallelOp | None:
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


def _qwen4_fused_quant_mm_rs_impl(
    input_parallel: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    hcom: str,
    world_size: int,
) -> torch.Tensor:
    sp_pad = (-input_parallel.shape[0]) % world_size
    if sp_pad:
        input_parallel = torch.nn.functional.pad(input_parallel, (0, 0, 0, sp_pad))
    quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(input_parallel)
    if quantized_x.dim() != 2:
        quantized_x = quantized_x.reshape(-1, input_parallel.shape[1])
    if pertoken_scale.dim() == 2:
        pertoken_scale = pertoken_scale.reshape(-1, 1)
    else:
        pertoken_scale = pertoken_scale.unsqueeze(dim=1)
    return DeviceOperator.npu_mm_reduce_scatter_base(
        quantized_x,
        weight,
        hcom,
        world_size,
        x1_scale=pertoken_scale,
        x2_scale=weight_scale.view(1, -1),
        output_dtype=torch.bfloat16,
        comm_mode="aiv",
    )


def _qwen4_fused_quant_mm_rs_fake(
    input_parallel: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    hcom: str,
    world_size: int,
) -> torch.Tensor:
    rows = (input_parallel.shape[0] + world_size - 1) // world_size
    return torch.empty([rows, weight.shape[-1]], dtype=torch.bfloat16, device=input_parallel.device)


direct_register_custom_op(
    op_name="qwen4_fused_quant_mm_rs",
    op_func=_qwen4_fused_quant_mm_rs_impl,
    mutates_args=[],
    fake_impl=_qwen4_fused_quant_mm_rs_fake,
    dispatch_key="PrivateUse1",
)


def _qwen4_fused_bf16_mm_rs_impl(
    input_parallel: torch.Tensor,
    weight: torch.Tensor,
    hcom: str,
    world_size: int,
) -> torch.Tensor:
    sp_pad = (-input_parallel.shape[0]) % world_size
    if sp_pad:
        input_parallel = torch.nn.functional.pad(input_parallel, (0, 0, 0, sp_pad))
    # CANN aclnnMatmulReduceScatterV2 only accepts commMode="aiv"; "ai_cpu"
    # fails tiling (EZ0026). Same call as the upstream fusion pass's bf16
    # replacement: npu_mm_reduce_scatter_base, aiv default, no scales.
    return DeviceOperator.npu_mm_reduce_scatter_base(
        input_parallel,
        weight.t(),
        hcom,
        world_size,
        reduce_op="sum",
    )


def _qwen4_fused_bf16_mm_rs_fake(
    input_parallel: torch.Tensor,
    weight: torch.Tensor,
    hcom: str,
    world_size: int,
) -> torch.Tensor:
    rows = (input_parallel.shape[0] + world_size - 1) // world_size
    return torch.empty([rows, weight.shape[0]], dtype=torch.bfloat16, device=input_parallel.device)


direct_register_custom_op(
    op_name="qwen4_fused_bf16_mm_rs",
    op_func=_qwen4_fused_bf16_mm_rs_impl,
    mutates_args=[],
    fake_impl=_qwen4_fused_bf16_mm_rs_fake,
    dispatch_key="PrivateUse1",
)

# Per-layer registry for the opaque O-proj dispatch op, keyed by the weight
# parameter. The dispatcher calls _impl at runtime; the registry recovers the
# layer-bound op instance (quant method, comm handle, branch flags), which
# cannot travel through the traced graph.
_QWEN4_MMRS_OPS: dict = {}
_QWEN4_MMRS_STATS = {"fused": 0, "unfused": 0}
_MMRS_DEBUG = __import__("os").environ.get("VLLM_MMRS_DEBUG", "0") == "1"


def _qwen4_o_proj_mm_rs_impl(
    input_parallel: torch.Tensor,
    weight: torch.Tensor,
    weight_n: int,
    world_size: int,
) -> torch.Tensor:
    op = _QWEN4_MMRS_OPS.get(weight)
    if op is None:
        # Heal a stale key: process_weights_after_loading may rebind
        # layer.weight to a new tensor after registration.
        for candidate in _QWEN4_MMRS_OPS.values():
            if candidate.layer.weight is weight:
                op = candidate
                _QWEN4_MMRS_OPS[weight] = op
                break
    if op is None:
        raise RuntimeError("qwen4_o_proj_mm_rs: no op registered for this weight")
    return op.run_mm_rs(input_parallel)


def _qwen4_o_proj_mm_rs_fake(
    input_parallel: torch.Tensor,
    weight: torch.Tensor,
    weight_n: int,
    world_size: int,
) -> torch.Tensor:
    rows = (input_parallel.shape[0] + world_size - 1) // world_size
    return torch.empty([rows, weight_n], dtype=torch.bfloat16, device=input_parallel.device)


direct_register_custom_op(
    op_name="qwen4_o_proj_mm_rs",
    op_func=_qwen4_o_proj_mm_rs_impl,
    mutates_args=[],
    fake_impl=_qwen4_o_proj_mm_rs_fake,
    dispatch_key="PrivateUse1",
)


# ---------------------------------------------------------------------------
# GDN input-projection AllGather+Matmul fusion (Qwen4Exp linear_attention
# layers). Mirrors Qwen4OProjMMReduceScatterOp: an opaque dispatch op records
# one graph node; the prefill-vs-fallback decision runs at runtime inside the
# op body with the real forward context. In the fused branch the sequence-
# parallel AllGather feeding in_proj_qkvz is fused into the GEMM via
# npu_all_gather_base_mm, whose second output (the gathered tokens) also
# feeds in_proj_ba, so the standalone AllGather disappears entirely.
# ---------------------------------------------------------------------------
_QWEN4_AGMM_OPS: dict = {}
_QWEN4_AGMM_STATS = {"fused": 0, "unfused": 0}


def _qwen4_gdn_in_proj_ag_mm_impl(
    x_shard: torch.Tensor,
    w_qkvz: torch.Tensor,
    n_out: int,
    world_size: int,
) -> torch.Tensor:
    op = _QWEN4_AGMM_OPS.get(w_qkvz)
    if op is None:
        for candidate in _QWEN4_AGMM_OPS.values():
            if candidate.layer.in_proj_qkvz.weight is w_qkvz:
                op = candidate
                _QWEN4_AGMM_OPS[w_qkvz] = op
                break
    if op is None:
        raise RuntimeError("qwen4_gdn_in_proj_ag_mm: no op registered for this weight")
    # Full gdn-attention forward: fused AG+MM input projections, core
    # attention, output projection. Runs eagerly at runtime with the real
    # forward context; never traced (opaque dispatch op).
    return op.run_layer(x_shard)


def _qwen4_gdn_in_proj_ag_mm_fake(
    x_shard: torch.Tensor,
    w_qkvz: torch.Tensor,
    n_out: int,
    world_size: int,
) -> torch.Tensor:
    # The op runs the full gdn attention including out_proj, whose custom op
    # (MM+ReduceScatter fusion) always returns the LOCAL token shard: the
    # fused prefill branch ReduceScatters inside npu_mm_reduce_scatter_base
    # and the decode branch ReduceScatters via sp_reduce_scatter. Model the
    # output as one shard of rows so the traced graph matches runtime.
    rows = x_shard.shape[0]
    return torch.empty([rows, n_out], dtype=torch.bfloat16, device=x_shard.device)


direct_register_custom_op(
    op_name="qwen4_gdn_in_proj_ag_mm",
    op_func=_qwen4_gdn_in_proj_ag_mm_impl,
    mutates_args=[],
    fake_impl=_qwen4_gdn_in_proj_ag_mm_fake,
    dispatch_key="PrivateUse1",
)


class Qwen4GdnInProjAGMMOp:
    """Runtime body for the gdn in_proj AG+MM dispatch op (never traced)."""

    SUPPORTED_TP_SIZES = (2, 4, 8)
    MIN_K = 256
    MAX_K = 65535
    MAX_COMM_BYTES = 16 * 256 * 1024 * 1024

    def __init__(self, gdn_attn):
        self.layer = gdn_attn
        self.qkvz = gdn_attn.in_proj_qkvz
        self.ba = gdn_attn.in_proj_ba
        self.scheme = getattr(self.qkvz.quant_method, "quant_method", self.qkvz.quant_method)
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        self.is_bf16 = isinstance(self.scheme, UnquantizedLinearMethod)
        self.n_qkvz = sum(s for s in self.qkvz.output_sizes)
        self.n_ba = sum(s for s in self.ba.output_sizes)
        self.weight_k = self.qkvz.input_size_per_partition
        _QWEN4_AGMM_OPS[self.qkvz.weight] = self
        device_group = get_tp_group().device_group
        backend = device_group._get_backend(torch.device("npu"))
        self.hcom = backend.get_hccl_comm_name(get_tp_group().rank_in_group)
        self.world_size = get_tp_group().world_size
        self.n_out = self.layer.out_proj.output_size

    @staticmethod
    def unsupported_reason(gdn_attn) -> str | None:
        if get_tp_group().world_size not in Qwen4GdnInProjAGMMOp.SUPPORTED_TP_SIZES:
            return "GDN in_proj AG MM requires TP size 2, 4, or 8."
        if not callable(getattr(torch_npu, "npu_all_gather_base_mm", None)):
            return "GDN in_proj AG MM requires the torch_npu V2 interface."
        qkvz = gdn_attn.in_proj_qkvz
        ba = gdn_attn.in_proj_ba
        if qkvz.bias is not None or ba.bias is not None:
            return "GDN in_proj AG MM requires bias-free projections."
        scheme = getattr(qkvz.quant_method, "quant_method", qkvz.quant_method)
        if not isinstance(scheme, UnquantizedLinearMethod):
            return "GDN in_proj AG MM requires unquantized (BF16) projections."
        if gdn_attn.disable_tp_for_ba_proj:
            return "GDN in_proj AG MM requires TP-sharded in_proj_ba."
        # Output partition checks: [N_local] must divide evenly across the
        # TP group and the full output must round-trip through the op.
        if qkvz.output_size % get_tp_group().world_size != 0:
            return "GDN in_proj AG MM requires divisible qkvz output size."
        if ba.output_size % get_tp_group().world_size != 0:
            return "GDN in_proj AG MM requires divisible ba output size."
        weight = qkvz.weight
        if weight.ndim != 2 or weight.shape[1] != qkvz.input_size_per_partition:
            return "GDN in_proj AG MM requires 2D ND qkvz weights."
        if not Qwen4GdnInProjAGMMOp.MIN_K <= qkvz.input_size_per_partition < Qwen4GdnInProjAGMMOp.MAX_K:
            return "GDN in_proj AG MM requires local K in [256, 65535)."
        return None

    def run_ag_mm(self, x_shard: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._can_fuse(x_shard):
            if _MMRS_DEBUG:
                _QWEN4_AGMM_STATS["unfused"] += 1
                _total = _QWEN4_AGMM_STATS["fused"] + _QWEN4_AGMM_STATS["unfused"]
                if _total % 20000 == 0:
                    logger.warning("AGMM stats: %s", _QWEN4_AGMM_STATS)
            gathered = sp_all_gather(x_shard)
            _r = self.qkvz.quant_method.apply(self.qkvz, gathered, None)
            if isinstance(_r, tuple):
                qkvz_out = _r[0]
            else:
                qkvz_out = _r
            _r = self.ba.quant_method.apply(self.ba, gathered, None)
            ba_out = _r[0] if isinstance(_r, tuple) else _r
            return qkvz_out, ba_out
        if _MMRS_DEBUG:
            _QWEN4_AGMM_STATS["fused"] += 1
            _total = _QWEN4_AGMM_STATS["fused"] + _QWEN4_AGMM_STATS["unfused"]
            if _total % 20000 == 0:
                logger.warning("AGMM stats: %s", _QWEN4_AGMM_STATS)
        x_shard = x_shard.contiguous()
        # Fused: AG(x_shard) is folded into the qkvz GEMM; the returned
        # gathered tensor feeds in_proj_ba. npu_all_gather_base_mm returns
        # (mm_out, gathered). Per the CANN contract, x2 must be the
        # TRANSPOSED weight ([K, N], first dim == x1's last dim): pass
        # weight.t() (a free view; [K,N] column-major is the native GEMM-B
        # layout), NOT the raw [N, K] parameter.
        qkvz_out, gathered = DeviceOperator.npu_all_gather_base_mm(
            x_shard,
            self.qkvz.weight.t(),
            self.hcom,
            self.world_size,
        )
        _r = self.ba.quant_method.apply(self.ba, gathered, None)
        ba_out = _r[0] if isinstance(_r, tuple) else _r
        return qkvz_out, ba_out

    def _can_fuse(self, x_shard: torch.Tensor) -> bool:
        # Keep decode and mixed batches on the baseline AG + MM path: decode
        # runs under ACL graph replay where the fused op must stay identical
        # to the captured baseline graph.
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
        if x_shard.ndim != 2 or x_shard.dtype != torch.bfloat16:
            return False
        if x_shard.shape[1] != self.weight_k:
            return False
        num_tokens = x_shard.shape[0]
        if num_tokens == 0:
            return False
        return num_tokens * self.weight_k * x_shard.element_size() < self.MAX_COMM_BYTES

    def run_layer(self, x_shard: torch.Tensor) -> torch.Tensor:
        """Full gdn-attention forward with the input projections replaced by
        the AG+MM dispatch op. Mirrors AscendGatedDeltaNetAttention.forward
        for the flat (gqa_interleaved_layout=False) Qwen3.5 layout only;
        every other layer falls back to sp_all_gather + the two baseline
        projections inside run_ag_mm."""
        attn = self.layer
        mixed_qkvz, ba = self.run_ag_mm(x_shard)
        num_tokens = mixed_qkvz.size(0)
        qkv_size = (attn.key_dim * 2 + attn.value_dim) // attn.tp_size
        z_size = attn.value_dim // attn.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(z.size(0), -1, attn.head_v_dim)
        b, a = attn._split_ba_for_tp(ba)
        b = b.contiguous()
        a = a.contiguous()

        core_attn_out = torch.zeros(
            (num_tokens, attn.num_v_heads // attn.tp_size, attn.head_v_dim),
            dtype=x_shard.dtype,
            device=x_shard.device,
        )
        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            attn.prefix,
            False,
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = attn.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        from einops import rearrange

        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        out, _ = attn.out_proj(core_attn_out)
        return out


class Qwen4OProjMMReduceScatterOp(CustomRowParallelOp):
    """Return the token shard of the Qwen4Exp W8A8 output projection.

    For prefill-only batches the projection GEMM and the sequence-parallel
    ReduceScatter are fused into npu_mm_reduce_scatter_base: the per-token
    quantized activation goes straight into the fused operator, whose
    ReduceScatter moves 1/tp_size of the data instead of the separate
    projection followed by a full-size ReduceScatter. Decode and mixed
    batches keep the existing MM + RS path so decode graph capture stays
    unchanged.
    """

    SUPPORTED_TP_SIZES = (2, 4, 8)
    MIN_K = 256
    MAX_K = 65535
    MAX_COMM_BYTES = 16 * 256 * 1024 * 1024

    @staticmethod
    def unsupported_reason(layer) -> str | None:
        if layer.custom_op is not None:
            return "Qwen4Exp O-projection MM ReduceScatter requires the original linear layer."
        from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
            AscendW8A8DynamicLinearMethod,
        )

        # modelslim W8A8_DYNAMIC wraps the scheme in AscendLinearMethod
        # (method_adapters.py), whose .quant_method is the actual scheme.
        scheme = getattr(layer.quant_method, "quant_method", layer.quant_method)
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        if not isinstance(scheme, (AscendW8A8DynamicLinearMethod, AscendUnquantizedLinearMethod)):
            return (
                "Qwen4Exp O-projection MM ReduceScatter requires W8A8 dynamic or unquantized bf16;"
                f" got wrapper={type(layer.quant_method).__name__} scheme={type(scheme).__name__}"
            )
        if layer.bias is not None:
            return "Qwen4Exp O-projection MM ReduceScatter requires bias-free O projections."
        if not callable(getattr(torch_npu, "npu_mm_reduce_scatter_base", None)):
            return "Qwen4Exp O-projection MM ReduceScatter requires the torch_npu V2 interface."
        if get_tp_group().world_size not in Qwen4OProjMMReduceScatterOp.SUPPORTED_TP_SIZES:
            return "Qwen4Exp O-projection MM ReduceScatter requires TP size 2, 4, or 8."
        # Checked on input_size_per_partition: the weight layout changes from
        # [N, K] to [K, N] in process_weights_after_loading.
        if not Qwen4OProjMMReduceScatterOp.MIN_K <= layer.input_size_per_partition < Qwen4OProjMMReduceScatterOp.MAX_K:
            return "Qwen4Exp O-projection MM ReduceScatter requires local K in [256, 65535)."
        return None

    def __init__(self, layer):
        super().__init__(layer)
        if reason := self.unsupported_reason(layer):
            raise ValueError(reason)
        self.update_attrs()
        # AscendLinearMethod wrapper -> .quant_method is the real scheme.
        self.scheme = getattr(self.quant_method, "quant_method", self.quant_method)
        from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
            AscendW8A8DynamicLinearMethod,
        )

        self.is_quant = isinstance(self.scheme, AscendW8A8DynamicLinearMethod)
        # 2D weight geometry captured before process_weights_after_loading may
        # convert the weight to 4D FRACTAL_NZ; used by runtime shape checks.
        self.weight_k = layer.input_size_per_partition
        self.weight_n = layer.output_size
        _QWEN4_MMRS_OPS[layer.weight] = self
        device_group = self.comm_group.device_group
        backend = device_group._get_backend(torch.device("npu"))
        self.hcom = backend.get_hccl_comm_name(self.tp_rank)
        self.world_size = self.tp_size

    def apply_impl(self, input_: torch.Tensor) -> tuple[torch.Tensor, None]:
        assert self.quant_method is not None
        input_parallel = self.get_input_parallel(input_)
        # Opaque dispatch op: dynamo records a single node. The
        # prefill-vs-decode decision runs at runtime inside the op body with
        # the real forward context; tracing sees dummy metadata and would
        # bake the wrong branch into the graph shared by prefill and decode.
        output = torch.ops.vllm.qwen4_o_proj_mm_rs(input_parallel, self.layer.weight, self.weight_n, self.world_size)
        return output, None

    def run_mm_rs(self, input_parallel: torch.Tensor) -> torch.Tensor:
        assert self.quant_method is not None
        # Runtime body of qwen4_o_proj_mm_rs. Called through the dispatcher,
        # so this code never enters the traced graph.
        if not self._can_fuse(input_parallel):
            if _MMRS_DEBUG:
                _QWEN4_MMRS_STATS["unfused"] += 1
                _total = _QWEN4_MMRS_STATS["fused"] + _QWEN4_MMRS_STATS["unfused"]
                if _total % 20000 == 0:
                    from vllm.logger import logger as _vl

                    _vl.warning("MMRS stats: %s", _QWEN4_MMRS_STATS)
            output = self.quant_method.apply(self.layer, input_parallel, None)
            return sp_reduce_scatter(output)
        if _MMRS_DEBUG:
            _QWEN4_MMRS_STATS["fused"] += 1
            _total = _QWEN4_MMRS_STATS["fused"] + _QWEN4_MMRS_STATS["unfused"]
            if _total % 20000 == 0:
                from vllm.logger import logger as _vl

                _vl.warning("MMRS stats: %s", _QWEN4_MMRS_STATS)
        input_parallel = input_parallel.contiguous()
        # The fused bodies (pad + operator) live inside custom ops: piecewise
        # compilation warmup bakes the m%world_size==0 no-pad branch away, so
        # an inline pad breaks on replay with unaligned m. Custom ops keep the
        # modulo padding invisible to dynamo.
        if self.is_quant:
            weight_scale = getattr(self.layer, "weight_scale_fp32", self.layer.weight_scale)
            output = torch.ops.vllm.qwen4_fused_quant_mm_rs(
                input_parallel,
                self.layer.weight,
                weight_scale,
                self.hcom,
                self.world_size,
            )
        else:
            output = torch.ops.vllm.qwen4_fused_bf16_mm_rs(
                input_parallel,
                self.layer.weight,
                self.hcom,
                self.world_size,
            )
        return output

    def _can_fuse(self, input_parallel: torch.Tensor) -> bool:
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
        if input_parallel.ndim != 2 or input_parallel.dtype != torch.bfloat16:
            return False
        weight = self.layer.weight
        if self.is_quant:
            # [K, N] ND or 4D FRACTAL_NZ; npu_mm_reduce_scatter_base takes both.
            if weight.ndim not in (2, 4):
                return False
        else:
            # bf16 branch projects with weight.t(), which needs the ND layout.
            if weight.ndim != 2:
                return False
        # Geometry captured pre-NZ-conversion in __init__.
        if not self.MIN_K <= input_parallel.shape[1] < self.MAX_K or input_parallel.shape[1] != self.weight_k:
            return False
        num_tokens = input_parallel.shape[0]
        if num_tokens == 0:
            return False
        padded_tokens = num_tokens + (-num_tokens) % self.world_size
        return padded_tokens * self.weight_k * input_parallel.element_size() < self.MAX_COMM_BYTES
