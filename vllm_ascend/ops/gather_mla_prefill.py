# SPDX-License-Identifier: Apache-2.0
"""AscendC fused paged C8 history gather, dequantization and positional copy."""

import torch


def gather_dequant_mla_prefill(
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    cumulative_lengths: torch.Tensor,
    lengths: torch.Tensor,
    starts: torch.Tensor,
    scale: torch.Tensor,
    *,
    num_tokens: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather bounded history without materializing layer-interleaved caches.

    CPU-owned planning supplies the output size and launch bound. The AscendC
    kernel reads request lengths and offsets on device and preserves the cache's
    page/token strides. The calibrated scale is applied in FP32 before BF16
    rounding; positional keys are copied unchanged by the same launch.
    """
    return torch.ops._C_ascend.gather_mla_prefill(
        latent_cache,
        rope_cache,
        block_table,
        cumulative_lengths,
        lengths,
        starts,
        scale,
        num_tokens,
        max_seq_len,
    )
