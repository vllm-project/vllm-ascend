# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 cache writers."""

from __future__ import annotations

import torch

from vllm_ascend.ops import packaged_attention
from vllm_ascend.ops.mixed_quant_sparse_attention import build_smla_metadata, qsmla
from vllm_ascend.ops.quant_lightning_indexer import run_a5_indexer
from vllm_ascend.ops.triton.a5_slot_mapping import build_a5_slot_mapping
from vllm_ascend.ops.triton.build_window_indices import build_window_indices_triton
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
    torch.ops._C_ascend.kv_compress_epilog_v2(
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


class MixedQuantPackedCacheOps:
    """Packed mixed-quant cache ABI selected by the hardware adaptor."""

    build_a5_slot_mapping = staticmethod(build_a5_slot_mapping)
    build_window_indices = staticmethod(build_window_indices_triton)
    build_smla_metadata = staticmethod(build_smla_metadata)
    qsmla = staticmethod(qsmla)
    run_a5_indexer = staticmethod(run_a5_indexer)
    write_attention_cache = staticmethod(write_attention_cache)
    write_index_cache = staticmethod(write_index_cache)
    mixed_quant_sparse_flash_mla_metadata = staticmethod(packaged_attention.mixed_quant_sparse_flash_mla_metadata)
    quant_lightning_indexer_metadata = staticmethod(packaged_attention.quant_lightning_indexer_metadata)
