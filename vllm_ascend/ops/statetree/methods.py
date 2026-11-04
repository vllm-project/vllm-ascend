# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM weight-method integration for lossless StateTree compression."""

from __future__ import annotations

import torch
from vllm.logger import logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.utils import replace_parameter

from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.statetree import operator

_MIN_WEIGHT_ELEMENTS = 1 << 21
_PAYLOAD_BUFFER = "_statetree_payload"
_SHAPE_ATTRIBUTE = "_statetree_shape"
_TILES_ATTRIBUTE = "_statetree_tiles"
_TIED_WEIGHT_ATTRIBUTE = "_statetree_tied_weight"


def _unsupported_reason(layer: torch.nn.Module) -> str | None:
    weight = getattr(layer, "weight", None)
    if weight is None or weight.ndim != 2:
        return "weight is not a matrix"
    if weight.dtype != torch.bfloat16 or weight.device.type != "npu":
        return "weight is not BF16 on NPU"
    n, k = map(int, weight.shape)
    # This is the performance-validated subset of the more general codec.
    # Other layouts retain the native BF16 implementation.
    if n % 128 or k % 256:
        return "weight shape is outside the validated N%128/K%256 subset"
    if weight.numel() < _MIN_WEIGHT_ELEMENTS:
        return f"weight has fewer than {_MIN_WEIGHT_ELEMENTS} elements"
    if getattr(layer, "bias", None) is not None:
        return "biased linears are not yet validated bit-for-bit"
    if getattr(layer, "custom_op", None) is not None:
        return "layer forward bypasses its quantization method"
    if getattr(layer, "skip_weight_nz_conversion", False):
        return "layer requests its original dense layout"
    if getattr(layer, "precast_fp32_weight", False):
        return "layer requires an FP32 weight copy"
    prefix = getattr(layer, "prefix", "")
    if "conv1d" in prefix or "wo_a" in prefix:
        return "layer uses a specialized weight layout"
    if getattr(layer, _TIED_WEIGHT_ATTRIBUTE, False):
        return "LM head shares its weight with the token embedding"
    return None


def _install_packed_weight(layer: torch.nn.Module) -> bool:
    reason = _unsupported_reason(layer)
    if reason is not None:
        logger.debug("StateTree keeps %s in BF16: %s", getattr(layer, "prefix", "<unnamed>"), reason)
        return False

    weight = layer.weight.detach()
    packed = operator.pack_verify_and_prepare(weight)
    dense_bytes = weight.numel() * weight.element_size()
    if packed.nbytes >= dense_bytes:
        logger.info(
            "StateTree keeps incompressible layer %s in BF16 (%d >= %d bytes)",
            getattr(layer, "prefix", "<unnamed>"),
            packed.nbytes,
            dense_bytes,
        )
        return False

    if _PAYLOAD_BUFFER in layer._buffers:
        layer._buffers[_PAYLOAD_BUFFER] = packed.payload
    else:
        layer.register_buffer(_PAYLOAD_BUFFER, packed.payload)
    setattr(layer, _SHAPE_ATTRIBUTE, packed.shape)
    setattr(
        layer,
        _TILES_ATTRIBUTE,
        (packed.n_tile, packed.k_tile, packed.escape_slots),
    )
    replace_parameter(layer, "weight", weight.new_empty((0,)))
    logger.info(
        "StateTree encoded %s %s: %d -> %d bytes; device reconstruction is bit-exact",
        getattr(layer, "prefix", "<unnamed>"),
        tuple(weight.shape),
        dense_bytes,
        packed.nbytes,
    )
    return True


def _apply_packed_weight(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    payload = getattr(layer, _PAYLOAD_BUFFER, None)
    if payload is None:
        return None
    output = operator.apply(
        x,
        payload,
        getattr(layer, _SHAPE_ATTRIBUTE),
        getattr(layer, _TILES_ATTRIBUTE),
    )
    return output if bias is None else output + bias


class StateTreeLinearMethod(AscendUnquantizedLinearMethod):
    """Ascend unquantized linear with opt-in packed BF16 storage."""

    supports_weight_switch = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _install_packed_weight(layer):
            super().process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = _apply_packed_weight(layer, x, bias)
        if output is not None:
            return output
        return super().apply(layer, x, bias)


class StateTreeEmbeddingMethod(UnquantizedEmbeddingMethod):
    """Compress an untied LM head while preserving normal token embeddings."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _install_packed_weight(layer):
            super().process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = _apply_packed_weight(layer, x, bias)
        if output is not None:
            return output
        return super().apply(layer, x, bias)

    def tie_weights(self, layer: torch.nn.Module, embed_tokens: torch.nn.Module):
        # Tied heads keep the shared BF16 parameter because embedding lookup
        # cannot consume the matrix-oriented packed representation.
        setattr(layer, _TIED_WEIGHT_ATTRIBUTE, True)
        if _PAYLOAD_BUFFER in layer._buffers:
            layer._buffers[_PAYLOAD_BUFFER] = None
        return super().tie_weights(layer, embed_tokens)
