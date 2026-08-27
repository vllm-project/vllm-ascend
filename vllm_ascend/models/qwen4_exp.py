"""Ascend adaptation for the experimental Qwen3.8 Flash-Next model.

The model definition is owned by vLLM.  This module replaces only the
CUDA-specific glue used by that definition and then registers thin Ascend
subclasses, keeping checkpoint mapping and model behavior aligned upstream.
"""
# mypy: disable-error-code=import-not-found

from __future__ import annotations

import sys
import types
from typing import Any

import torch
import vllm.envs as vllm_envs
from vllm.model_executor.layers.ple_offload_layer import PleOffloadLayer

from vllm_ascend.ops.triton.qwen4_exp import hc as ascend_hc_ops
from vllm_ascend.ops.triton.qwen4_exp import qsa as ascend_qsa_ops


def _disable_cuda_low_latency_gemm(*args: Any, **kwargs: Any) -> None:
    del args, kwargs


# Install the NPU operator modules before importing the upstream NVIDIA model.
# Its relative imports then resolve to implementations with the same contract,
# without copying the model definition into the hardware plugin.
sys.modules["vllm.models.qwen4_exp.nvidia.ops.hc"] = ascend_hc_ops
sys.modules["vllm.models.qwen4_exp.nvidia.ops.qsa"] = ascend_qsa_ops
low_latency_gemm = types.ModuleType("vllm.models.qwen4_exp.nvidia.low_latency_gemm")
setattr(  # noqa: B010
    low_latency_gemm,
    "enable_qwen4_exp_low_latency_gemm",
    _disable_cuda_low_latency_gemm,
)
sys.modules[low_latency_gemm.__name__] = low_latency_gemm

from vllm.models.qwen4_exp.common import qsa_cache  # noqa: E402
from vllm.models.qwen4_exp.nvidia import model as upstream_model  # noqa: E402
from vllm.models.qwen4_exp.nvidia import mtp as upstream_mtp  # noqa: E402
from vllm.models.qwen4_exp.nvidia import qsa as upstream_qsa  # noqa: E402

# The upstream Triton metadata builder uses CUDA graph dependency control.
# The torch implementation is device-agnostic and remains graphable by the
# Ascend runner, so select it once outside the per-step path.
qsa_cache.build_qsa_metadata = qsa_cache._build_qsa_metadata_torch


def _ple_target_device(cls: type[PleOffloadLayer]) -> torch.device:
    del cls
    if vllm_envs.VLLM_PLE_CPU_OFFLOAD:
        return torch.device("cpu")
    return torch.device("npu", torch.npu.current_device())


PleOffloadLayer.get_target_device = classmethod(_ple_target_device)


def _ascend_qsa_impl_init(
    self: upstream_qsa.Qwen4ExpQSAFlashAttentionImpl,
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


def _ascend_qsa_cache_update(
    self: upstream_qsa.Qwen4ExpQSAFlashAttentionImpl,
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


upstream_qsa.Qwen4ExpQSAFlashAttentionImpl.__init__ = _ascend_qsa_impl_init
upstream_qsa.Qwen4ExpQSAFlashAttentionImpl.do_kv_cache_update = _ascend_qsa_cache_update


def _ascend_model_state_cls():
    from vllm_ascend.models.qwen4_exp_model_state import (
        AscendQwen4ExpModelState,
    )

    return AscendQwen4ExpModelState


class AscendQwen4ExpForCausalLM(upstream_model.Qwen4ExpForCausalLM):
    """Qwen3.8 Flash-Next text model using Ascend runtime state."""

    get_model_state_cls = staticmethod(_ascend_model_state_cls)


class AscendQwen4ExpForConditionalGeneration(upstream_model.Qwen4ExpForConditionalGeneration):
    """Qwen3.8 Flash-Next multimodal model using Ascend runtime state."""

    get_model_state_cls = staticmethod(_ascend_model_state_cls)


class AscendQwen4ExpMTP(upstream_mtp.Qwen4ExpMTP):
    """Qwen3.8 Flash-Next multi-token predictor for Ascend."""


__all__ = [
    "AscendQwen4ExpForCausalLM",
    "AscendQwen4ExpForConditionalGeneration",
    "AscendQwen4ExpMTP",
]
