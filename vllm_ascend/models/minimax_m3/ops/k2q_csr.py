# SPDX-License-Identifier: Apache-2.0
"""Python entry for the MiniMax-M3 AscendC ``npu_k2q_csr`` op."""

from __future__ import annotations

import torch

import vllm_ascend.vllm_ascend_C  # noqa: F401
from vllm_ascend.utils import enable_custom_op

__all__ = ["npu_k2q_csr", "k2q_csr_block_stats"]


def k2q_csr_block_stats(cu_block_lens: torch.Tensor) -> tuple[int, int]:
    """Derive ``(total_rows, max_kv)`` from ``cu_block_lens`` on host."""
    cu = cu_block_lens.reshape(-1)
    if cu.numel() <= 1:
        return 0, 0
    block_lens = cu[1:] - cu[:-1]
    total_rows = int(cu[-1].item())
    max_kv = int(block_lens.max().item()) if block_lens.numel() else 0
    return total_rows, max_kv


@torch.no_grad()
def npu_k2q_csr(
    q2k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_block_lens: torch.Tensor,
    order_method: int = 0,
    total_rows: int = -1,
    max_kv: int = -1,
    use_simt: int | bool = 0,
    q_global_offset: int | bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert MiniMax-M3 q2k indices to k2q CSR on NPU."""
    enable_custom_op()
    if total_rows < 0 or max_kv < 0:
        derived_total_rows, derived_max_kv = k2q_csr_block_stats(cu_block_lens)
        if total_rows < 0:
            total_rows = derived_total_rows
        if max_kv < 0:
            max_kv = derived_max_kv
    return torch.ops._C_ascend.npu_k2q_csr(
        q2k,
        cu_seqlens,
        cu_block_lens,
        int(order_method),
        int(total_rows),
        int(max_kv),
        int(use_simt),
        int(q_global_offset),
    )
