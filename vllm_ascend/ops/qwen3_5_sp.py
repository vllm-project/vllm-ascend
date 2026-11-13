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

"""Sequence-parallel communication/GEMM fusions for Qwen3.5."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any

import torch
import torch_npu
from einops import rearrange
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.common.ops.sequence_parallel import sp_all_gather, sp_reduce_scatter
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.linear_op import CustomRowParallelOp

_SUPPORTED_TP_SIZES = (2, 4, 8)
_MIN_K = 256
_MAX_K = 65535
_MAX_COMM_BYTES = 16 * 256 * 1024 * 1024

# Each worker owns its model and registry. Weak values avoid extending a
# layer's lifetime across model reloads; the projection custom op and decoder
# layer hold the strong references used during normal inference.
_MM_RS_OPS: weakref.WeakValueDictionary[int, Qwen35OProjMMReduceScatterOp] = weakref.WeakValueDictionary()
_AG_MM_OPS: weakref.WeakValueDictionary[int, Qwen35GdnInProjAllGatherMatmulOp] = weakref.WeakValueDictionary()


def _lookup_registered_op(
    registry: weakref.WeakValueDictionary[int, Any],
    weight: torch.Tensor,
    get_weight: Callable[[Any], torch.Tensor],
) -> Any | None:
    op = registry.get(id(weight))
    if op is not None:
        return op
    # Weight processing may replace a Parameter after layer construction.
    for candidate in tuple(registry.values()):
        if get_weight(candidate) is weight:
            registry[id(weight)] = candidate
            return candidate
    return None


def _prefill_only() -> bool:
    if not is_forward_context_available():
        return True
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = next(
            (
                metadata
                for metadata in attn_metadata.values()
                if hasattr(metadata, "num_prefills") and hasattr(metadata, "num_decodes")
            ),
            None,
        )
    return bool(
        attn_metadata is not None
        and getattr(attn_metadata, "num_prefills", 0) > 0
        and getattr(attn_metadata, "num_decodes", 0) == 0
    )


def _qwen35_fused_bf16_mm_rs_impl(
    input_parallel: torch.Tensor,
    weight: torch.Tensor,
    hcom: str,
    world_size: int,
) -> torch.Tensor:
    sp_pad = (-input_parallel.shape[0]) % world_size
    if sp_pad:
        input_parallel = torch.nn.functional.pad(input_parallel, (0, 0, 0, sp_pad))
    return DeviceOperator.npu_mm_reduce_scatter_base(
        input_parallel,
        weight.t(),
        hcom,
        world_size,
        reduce_op="sum",
    )


def _qwen35_fused_bf16_mm_rs_fake(
    input_parallel: torch.Tensor,
    weight: torch.Tensor,
    hcom: str,
    world_size: int,
) -> torch.Tensor:
    del hcom
    rows = (input_parallel.shape[0] + world_size - 1) // world_size
    return torch.empty((rows, weight.shape[0]), dtype=input_parallel.dtype, device=input_parallel.device)


direct_register_custom_op(
    op_name="qwen35_fused_bf16_mm_rs",
    op_func=_qwen35_fused_bf16_mm_rs_impl,
    mutates_args=[],
    fake_impl=_qwen35_fused_bf16_mm_rs_fake,
    dispatch_key="PrivateUse1",
)


def _qwen35_o_proj_mm_rs_impl(
    input_parallel: torch.Tensor,
    weight: torch.Tensor,
    output_size: int,
    world_size: int,
) -> torch.Tensor:
    del output_size, world_size
    op = _lookup_registered_op(_MM_RS_OPS, weight, lambda candidate: candidate.layer.weight)
    if op is None:
        raise RuntimeError("qwen35_o_proj_mm_rs: no operator registered for this weight")
    return op.run(input_parallel)


def _qwen35_o_proj_mm_rs_fake(
    input_parallel: torch.Tensor,
    weight: torch.Tensor,
    output_size: int,
    world_size: int,
) -> torch.Tensor:
    del weight
    rows = (input_parallel.shape[0] + world_size - 1) // world_size
    return torch.empty((rows, output_size), dtype=input_parallel.dtype, device=input_parallel.device)


direct_register_custom_op(
    op_name="qwen35_o_proj_mm_rs",
    op_func=_qwen35_o_proj_mm_rs_impl,
    mutates_args=[],
    fake_impl=_qwen35_o_proj_mm_rs_fake,
    dispatch_key="PrivateUse1",
)


class Qwen35OProjMMReduceScatterOp(CustomRowParallelOp):
    """Fuse a Qwen3.5 attention output projection with ReduceScatter."""

    @staticmethod
    def unsupported_reason(layer: Any) -> str | None:
        if layer.custom_op is not None:
            return "Qwen3.5 MM+ReduceScatter requires the original row-parallel operator."
        scheme = getattr(layer.quant_method, "quant_method", layer.quant_method)
        if not isinstance(scheme, (UnquantizedLinearMethod, AscendUnquantizedLinearMethod)):
            return "Qwen3.5 MM+ReduceScatter currently requires an unquantized BF16 output projection."
        if layer.weight.dtype != torch.bfloat16:
            return "Qwen3.5 MM+ReduceScatter requires BF16 output-projection weights."
        if layer.bias is not None:
            return "Qwen3.5 MM+ReduceScatter requires a bias-free output projection."
        if not callable(getattr(torch_npu, "npu_mm_reduce_scatter_base", None)):
            return "Qwen3.5 MM+ReduceScatter requires the torch_npu V2 interface."
        if get_tp_group().world_size not in _SUPPORTED_TP_SIZES:
            return "Qwen3.5 MM+ReduceScatter requires TP size 2, 4, or 8."
        if not _MIN_K <= layer.input_size_per_partition < _MAX_K:
            return "Qwen3.5 MM+ReduceScatter requires local K in [256, 65535)."
        return None

    def __init__(self, layer: Any):
        super().__init__(layer)
        if reason := self.unsupported_reason(layer):
            raise ValueError(reason)
        self.update_attrs()
        self.weight_k = layer.input_size_per_partition
        self.output_size = layer.output_size
        self.world_size = self.tp_size
        device_group = self.comm_group.device_group
        backend = device_group._get_backend(torch.device("npu"))
        self.hcom = backend.get_hccl_comm_name(self.tp_rank)
        _MM_RS_OPS[id(layer.weight)] = self

    def apply_impl(self, input_: torch.Tensor) -> tuple[torch.Tensor, None]:
        input_parallel = self.get_input_parallel(input_)
        output = torch.ops.vllm.qwen35_o_proj_mm_rs(
            input_parallel,
            self.layer.weight,
            self.output_size,
            self.world_size,
        )
        return output, None

    def run(self, input_parallel: torch.Tensor) -> torch.Tensor:
        assert self.quant_method is not None
        if not self._can_fuse(input_parallel):
            output = self.quant_method.apply(self.layer, input_parallel, None)
            return sp_reduce_scatter(output)
        return torch.ops.vllm.qwen35_fused_bf16_mm_rs(
            input_parallel.contiguous(),
            self.layer.weight,
            self.hcom,
            self.world_size,
        )

    def _can_fuse(self, input_parallel: torch.Tensor) -> bool:
        if not _prefill_only():
            return False
        weight = self.layer.weight
        if input_parallel.ndim != 2 or input_parallel.dtype != torch.bfloat16:
            return False
        if weight.ndim != 2 or weight.dtype != torch.bfloat16:
            return False
        if input_parallel.shape[1] != self.weight_k:
            return False
        num_tokens = input_parallel.shape[0]
        if num_tokens == 0:
            return False
        padded_tokens = num_tokens + (-num_tokens) % self.world_size
        return padded_tokens * self.weight_k * input_parallel.element_size() < _MAX_COMM_BYTES


def _qwen35_gdn_in_proj_ag_mm_impl(
    x_shard: torch.Tensor,
    weight: torch.Tensor,
    output_size: int,
    world_size: int,
) -> torch.Tensor:
    del output_size, world_size
    op = _lookup_registered_op(_AG_MM_OPS, weight, lambda candidate: candidate.qkvz.weight)
    if op is None:
        raise RuntimeError("qwen35_gdn_in_proj_ag_mm: no operator registered for this weight")
    return op.run_layer(x_shard)


def _qwen35_gdn_in_proj_ag_mm_fake(
    x_shard: torch.Tensor,
    weight: torch.Tensor,
    output_size: int,
    world_size: int,
) -> torch.Tensor:
    del weight, world_size
    return torch.empty((x_shard.shape[0], output_size), dtype=x_shard.dtype, device=x_shard.device)


direct_register_custom_op(
    op_name="qwen35_gdn_in_proj_ag_mm",
    op_func=_qwen35_gdn_in_proj_ag_mm_impl,
    mutates_args=[],
    fake_impl=_qwen35_gdn_in_proj_ag_mm_fake,
    dispatch_key="PrivateUse1",
)


class Qwen35GdnInProjAllGatherMatmulOp:
    """Fuse Qwen3.5 GDN's input AllGather with its qkvz projection."""

    @staticmethod
    def unsupported_reason(gdn_attn: Any) -> str | None:
        if get_tp_group().world_size not in _SUPPORTED_TP_SIZES:
            return "Qwen3.5 GDN AG+MM requires TP size 2, 4, or 8."
        if not callable(getattr(torch_npu, "npu_all_gather_base_mm", None)):
            return "Qwen3.5 GDN AG+MM requires the torch_npu V2 interface."
        qkvz = gdn_attn.in_proj_qkvz
        ba = gdn_attn.in_proj_ba
        if qkvz.bias is not None or ba.bias is not None:
            return "Qwen3.5 GDN AG+MM requires bias-free input projections."
        scheme = getattr(qkvz.quant_method, "quant_method", qkvz.quant_method)
        if not isinstance(scheme, (UnquantizedLinearMethod, AscendUnquantizedLinearMethod)):
            return "Qwen3.5 GDN AG+MM currently requires an unquantized BF16 qkvz projection."
        if qkvz.weight.dtype != torch.bfloat16:
            return "Qwen3.5 GDN AG+MM requires BF16 qkvz weights."
        if gdn_attn.disable_tp_for_ba_proj:
            return "Qwen3.5 GDN AG+MM requires a TP-sharded in_proj_ba."
        world_size = get_tp_group().world_size
        if qkvz.output_size % world_size or ba.output_size % world_size:
            return "Qwen3.5 GDN AG+MM requires divisible input-projection output sizes."
        weight = qkvz.weight
        if weight.ndim != 2 or weight.shape[1] != qkvz.input_size_per_partition:
            return "Qwen3.5 GDN AG+MM requires a 2D ND qkvz weight."
        if not _MIN_K <= qkvz.input_size_per_partition < _MAX_K:
            return "Qwen3.5 GDN AG+MM requires local K in [256, 65535)."
        return None

    def __init__(self, gdn_attn: Any):
        if reason := self.unsupported_reason(gdn_attn):
            raise ValueError(reason)
        self.layer = gdn_attn
        self.qkvz = gdn_attn.in_proj_qkvz
        self.ba = gdn_attn.in_proj_ba
        self.weight_k = self.qkvz.input_size_per_partition
        self.world_size = get_tp_group().world_size
        self.output_size = gdn_attn.out_proj.output_size
        device_group = get_tp_group().device_group
        backend = device_group._get_backend(torch.device("npu"))
        self.hcom = backend.get_hccl_comm_name(get_tp_group().rank_in_group)
        _AG_MM_OPS[id(self.qkvz.weight)] = self

    def __call__(self, x_shard: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.qwen35_gdn_in_proj_ag_mm(
            x_shard,
            self.qkvz.weight,
            self.output_size,
            self.world_size,
        )

    def run_input_projections(self, x_shard: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._can_fuse(x_shard):
            gathered = sp_all_gather(x_shard)
            qkvz_out = self.qkvz.quant_method.apply(self.qkvz, gathered, None)
            ba_out = self.ba.quant_method.apply(self.ba, gathered, None)
            return self._unwrap(qkvz_out), self._unwrap(ba_out)

        qkvz_out, gathered = torch_npu.npu_all_gather_base_mm(
            x_shard.contiguous(),
            self.qkvz.weight.t(),
            self.hcom,
            self.world_size,
        )
        ba_out = self.ba.quant_method.apply(self.ba, gathered, None)
        return qkvz_out, self._unwrap(ba_out)

    @staticmethod
    def _unwrap(output: Any) -> torch.Tensor:
        return output[0] if isinstance(output, tuple) else output

    def _can_fuse(self, x_shard: torch.Tensor) -> bool:
        if not _prefill_only():
            return False
        if x_shard.ndim != 2 or x_shard.dtype != torch.bfloat16:
            return False
        if x_shard.shape[1] != self.weight_k or x_shard.shape[0] == 0:
            return False
        return x_shard.numel() * x_shard.element_size() < _MAX_COMM_BYTES

    def run_layer(self, x_shard: torch.Tensor) -> torch.Tensor:
        attn = self.layer
        mixed_qkvz, ba = self.run_input_projections(x_shard)
        num_tokens = mixed_qkvz.size(0)
        qkv_size = (attn.key_dim * 2 + attn.value_dim) // attn.tp_size
        z_size = attn.value_dim // attn.tp_size
        mixed_qkv, z = mixed_qkvz.split((qkv_size, z_size), dim=-1)
        z = z.reshape(z.size(0), -1, attn.head_v_dim)
        b, a = attn._split_ba_for_tp(ba)

        core_attn_out = torch.zeros(
            (num_tokens, attn.num_v_heads // attn.tp_size, attn.head_v_dim),
            dtype=x_shard.dtype,
            device=x_shard.device,
        )
        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b.contiguous(),
            a.contiguous(),
            core_attn_out,
            attn.prefix,
            False,
        )

        z_shape = z.shape
        core_attn_out = attn.norm(core_attn_out.reshape(-1, core_attn_out.shape[-1]), z.reshape(-1, z.shape[-1]))
        core_attn_out = rearrange(core_attn_out.reshape(z_shape), "... h d -> ... (h d)")
        output, _ = attn.out_proj(core_attn_out)
        return output


def log_disabled_fusion(layer_idx: int, fusion: str, reason: str) -> None:
    logger.debug("Qwen3.5 layer %s: %s fusion disabled (%s)", layer_idx, fusion, reason)
