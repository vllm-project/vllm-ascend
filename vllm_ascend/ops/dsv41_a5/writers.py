# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qualified A5 cache writers."""

from __future__ import annotations

import torch

from vllm_ascend.ops.triton.quantize_mxfp4_indexer import (
    write_mxfp4_indexer_cache,
)


def write_attention_cache(
    cache: torch.Tensor,
    flat_slots: torch.Tensor,
    values: torch.Tensor,
    *,
    kind: str,
) -> None:
    import custom_ops  # noqa: F401, PLC0415  # Registers torch.ops.custom.*.

    if kind == "cmp":
        cache_arg = cache
        group_size = 16
        quant_mode = "mxfp4_bf16"
    elif kind == "win":
        cache_arg = cache.view(torch.float8_e4m3fn)
        group_size = 32
        quant_mode = "mxfp8_bf16"
    else:
        raise ValueError(f"unsupported A5 cache kind: {kind}")
    torch.ops.custom.kv_compress_epilog_v2(
        cache_arg,
        values.contiguous(),
        flat_slots.contiguous(),
        quant_group_size=group_size,
        quant_mode=quant_mode,
        round_scale=True,
        x_scale=1.0,
    )


def write_index_cache(
    cache: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    coordinates: torch.Tensor,
    values: torch.Tensor,
) -> None:
    write_mxfp4_indexer_cache(values, coordinates, cache[0], cache[1])
