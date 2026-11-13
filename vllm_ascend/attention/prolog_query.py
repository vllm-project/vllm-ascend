# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Query contracts shared by Prolog admission and indexer execution."""

from typing import Any

import torch

MXFP8_QUERY_GROUP_SIZE = 32


def mxfp8_query_scale_width(q_lora_rank: int) -> int:
    """Number of E8M0 entries including the final pair's padding."""
    if type(q_lora_rank) is not int or q_lora_rank <= 0:
        raise ValueError("Prolog query requires a positive integer q_lora_rank.")
    return 2 * ((q_lora_rank + 2 * MXFP8_QUERY_GROUP_SIZE - 1) // (2 * MXFP8_QUERY_GROUP_SIZE))


def _mxfp8_query_padding(projection: Any) -> tuple[int, int] | None:
    # Keep metadata-only imports independent of linear/quantization registration.
    from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear

    observed = getattr(projection, "mxfp8_tp_padding", None)
    if observed is not None and (
        not isinstance(observed, (tuple, list))
        or len(observed) != 2
        or any(type(value) is not int or value < 0 for value in observed)
    ):
        return None
    if isinstance(projection, RowParallelLinear):
        rank = getattr(projection, "tp_rank", None)
        width = getattr(projection, "input_size_per_partition", None)
        if type(rank) is not int or rank < 0 or type(width) is not int or width <= 0:
            return None
        prefix = rank * width % MXFP8_QUERY_GROUP_SIZE
        expected = (prefix, -(prefix + width) % MXFP8_QUERY_GROUP_SIZE)
    elif isinstance(projection, (ReplicatedLinear, ColumnParallelLinear)):
        expected = (0, 0)
    else:
        # An unknown layer without a processed padding contract is not proof
        # of zero padding. Do not use getattr(..., (0, 0)) here.
        return tuple(observed) if observed is not None else None
    if observed is not None and tuple(observed) != expected:
        return None
    return expected


def is_unquantized_query_projection(projection: Any) -> bool:
    # AscendUnquantizedLinearMethod inherits this class. A missing/unknown
    # quant_method is not evidence of an unquantized linear.
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    return isinstance(getattr(projection, "quant_method", None), UnquantizedLinearMethod)


def prolog_query_unsupported_reasons(
    projection: Any,
    q_lora_rank: int,
    producer_quant_type: type | None = None,
    *,
    allow_unquantized: bool = False,
) -> list[str]:
    """Check configuration before either query or MLA weights are transformed.

    Only stable layer attributes are used; weight shape changes during loading.
    Runtime scale widths and bucket coverage still need output validation.
    Unquantized consumers require the caller to dequantize Prolog's query;
    keep them disabled for callers that pass prepared quantized inputs.
    """
    # Lazy imports avoid cycles through quantization's model/operator registration.
    from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod
    from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod

    reasons = []
    scheme = getattr(getattr(projection, "quant_method", None), "quant_method", None)
    scheme_type = type(scheme)
    supported = (AscendW8A8DynamicLinearMethod, AscendW8A8MXFP8DynamicLinearMethod)
    unquantized = allow_unquantized and is_unquantized_query_projection(projection)
    if unquantized:
        weight = getattr(projection, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            reasons.append("Prolog unquantized query linear requires floating-point weights.")
    elif scheme_type not in supported:
        reasons.append("Prolog query requires a W8A8_DYNAMIC or W8A8_MXFP8 query linear.")
    if not unquantized and not hasattr(projection, "weight_scale"):
        reasons.append("Prolog query requires weight_scale on the query linear.")
    if type(q_lora_rank) is not int or q_lora_rank <= 0:
        reasons.append("Prolog query requires a positive integer q_lora_rank.")
    elif type(getattr(projection, "input_size", None)) is not int or projection.input_size != q_lora_rank:
        reasons.append("Prolog query q_lora_rank must match the query linear input_size.")
    if getattr(projection, "input_size_per_partition", q_lora_rank) != q_lora_rank:
        reasons.append("Prolog query linear must consume the complete q_c input width.")
    if getattr(projection, "input_is_parallel", True) is False:
        reasons.append("Prolog prepared query cannot be split by the query linear.")
    if not unquantized and producer_quant_type is not None and producer_quant_type is not scheme_type:
        reasons.append("Prolog query producer and query linear must use the same quantization scheme.")
    if scheme_type is AscendW8A8MXFP8DynamicLinearMethod:
        if getattr(scheme, "group_size", None) != MXFP8_QUERY_GROUP_SIZE:
            reasons.append("Prolog MXFP8 query requires group size 32.")
        if _mxfp8_query_padding(projection) != (0, 0):
            reasons.append("Prolog MXFP8 query requires known zero TP input padding.")
    return reasons


def normalize_prolog_query(
    q_c: torch.Tensor | None,
    scale: torch.Tensor | None,
    num_tokens: int,
    q_lora_rank: int,
    *,
    mxfp8: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Validate returned rows against token_x, including FULL graph padding.

    These are metadata-only checks; no device synchronization is introduced.
    A violation fails at the adapter boundary: the operator may already have
    written KV, so retrying this call through native is not safe.
    """
    if (
        q_c is None
        or q_c.ndim < 2
        or q_c.shape[0] != num_tokens
        or q_c.shape[-1] != q_lora_rank
        or q_c.numel() != num_tokens * q_lora_rank
    ):
        shape = None if q_c is None else tuple(q_c.shape)
        raise RuntimeError(
            f"Prolog q_c must cover all token_x rows (including graph padding): "
            f"expected {num_tokens} x {q_lora_rank}, got {shape}."
        )
    q_c = q_c.view(num_tokens, q_lora_rank)
    if scale is None:
        if q_c.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError("Prolog quantized q_c is missing its scale tensor.")
        return q_c
    # The shared adapter also feeds the standard RoPE indexer, which does
    # not go through KPool's prepared-input checks. Enforce the same encoding
    # contract here before either consumer dispatches a quantized matmul.
    if scale.device != q_c.device:
        raise RuntimeError("Prolog q_c and scale must be on the same device.")
    if mxfp8:
        if q_c.dtype != torch.float8_e4m3fn or scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise RuntimeError("Prolog MXFP8 query requires E4M3FN q_c and E8M0 scale storage.")
    elif q_c.dtype != torch.int8 or scale.dtype != torch.float32:
        raise RuntimeError("Prolog INT8 query requires INT8 q_c and FP32 per-token scales.")
    width = mxfp8_query_scale_width(q_lora_rank) if mxfp8 else 1
    if scale.ndim == 0 or scale.shape[0] != num_tokens or scale.numel() != num_tokens * width:
        raise RuntimeError(
            f"Prolog q_c scale must cover all token_x rows: expected {num_tokens} x {width}, got {tuple(scale.shape)}."
        )
    return q_c, scale.view(num_tokens, width) if mxfp8 else scale.view(num_tokens)
