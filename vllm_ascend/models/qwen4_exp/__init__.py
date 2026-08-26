# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendored Qwen3.8 Flash-Next model with an Ascend backend.

The paired vLLM 0.26.0 release does not contain Qwen4Exp.  The complete model
therefore lives in the hardware plugin and is registered through vLLM's model
registry without changing the vLLM installation.
"""

import sys
import types
from typing import Any

import torch

from vllm_ascend.ops.triton.qwen4_exp import hc as ascend_hc_ops
from vllm_ascend.ops.triton.qwen4_exp import qsa as ascend_qsa_ops

from .common.hyperconnection import (
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionBase,
    HyperConnectionConfig,
)


def _disable_cuda_low_latency_gemm(*args: Any, **kwargs: Any) -> None:
    del args, kwargs


# Resolve the vendored NVIDIA model's relative operator imports to NPU
# implementations before importing the shared Python model definition.
_PACKAGE = __name__
sys.modules[f"{_PACKAGE}.nvidia.ops.hc"] = ascend_hc_ops
sys.modules[f"{_PACKAGE}.nvidia.ops.qsa"] = ascend_qsa_ops
low_latency_gemm = types.ModuleType(f"{_PACKAGE}.nvidia.low_latency_gemm")
low_latency_gemm.enable_qwen4_exp_low_latency_gemm = _disable_cuda_low_latency_gemm
sys.modules[low_latency_gemm.__name__] = low_latency_gemm

from .common import qsa_cache  # noqa: E402
from .nvidia import model as _model  # noqa: E402
from .nvidia import mtp as _mtp  # noqa: E402
from .nvidia import qsa as _qsa  # noqa: E402

qsa_cache.build_qsa_metadata = qsa_cache._build_qsa_metadata_torch


def _qsa_impl_init(
    self: Any,
    num_heads: int,
    head_size: int,
    scale: float,
    num_kv_heads: int,
    alibi_slopes: Any,
    sliding_window: Any,
    kv_cache_dtype: str,
    blocksparse_params: Any,
    attn_type: Any,
    kv_sharing_target_layer_name: Any,
    **kwargs: Any,
) -> None:
    del blocksparse_params, attn_type, kv_sharing_target_layer_name, kwargs
    self.num_heads = num_heads
    self.head_size = head_size
    self.scale = scale
    self.num_kv_heads = num_kv_heads
    self.alibi_slopes = alibi_slopes
    self.sliding_window = sliding_window or (-1, -1)
    self.kv_cache_dtype = kv_cache_dtype
    self.sinks = None
    self.dcp_world_size = 1
    self.supports_quant_query_input = False


def _qsa_cache_update(
    self: Any,
    layer: torch.nn.Module,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> None:
    del layer, args, kwargs
    key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
    slots = slot_mapping[: key.shape[0]].long()
    valid = slots >= 0
    slots = slots[valid]
    page_size = key_cache.shape[1]
    block_ids = torch.div(slots, page_size, rounding_mode="floor")
    block_offsets = slots.remainder(page_size)
    key_cache[block_ids, block_offsets, :, :] = key[valid]
    value_cache[block_ids, block_offsets, :, :] = value[valid]


_qsa.Qwen4ExpQSAFlashAttentionImpl.__init__ = _qsa_impl_init
_qsa.Qwen4ExpQSAFlashAttentionImpl.do_kv_cache_update = _qsa_cache_update


def _ascend_model_state_cls() -> type:
    from vllm_ascend.models.qwen4_exp_model_state import AscendQwen4ExpModelState

    return AscendQwen4ExpModelState


class AscendQwen4ExpForCausalLM(_model.Qwen4ExpForCausalLM):
    get_model_state_cls = staticmethod(_ascend_model_state_cls)


class AscendQwen4ExpForConditionalGeneration(_model.Qwen4ExpForConditionalGeneration):
    get_model_state_cls = staticmethod(_ascend_model_state_cls)


class AscendQwen4ExpMTP(_mtp.Qwen4ExpMTP):
    pass


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "AscendQwen4ExpForCausalLM",
    "AscendQwen4ExpForConditionalGeneration",
    "AscendQwen4ExpMTP",
]
