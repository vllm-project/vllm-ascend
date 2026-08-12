# SPDX-License-Identifier: Apache-2.0
"""Python entry for AscendC ``npu_k2q_csr`` (q2k -> k2q CSR).

Wraps the compiled custom op registered as ``torch.ops._C_ascend.npu_k2q_csr``.
Requires ``enable_custom_op()`` (or an import of ``vllm_ascend.vllm_ascend_C``)
before the first call.
"""

from __future__ import annotations

import torch

import vllm_ascend.vllm_ascend_C  # noqa: F401
from vllm_ascend.utils import enable_custom_op

__all__ = ["npu_k2q_csr", "k2q_csr_block_stats"]


def k2q_csr_block_stats(cu_block_lens: torch.Tensor) -> tuple[int, int]:
    """Derive ``(total_rows, max_kv)`` from ``cu_block_lens`` on Host.

    Prefer passing these explicitly into ``npu_k2q_csr`` to avoid an extra
    device→host sync inside the C++ adapter when both default to ``-1``.
    """
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
    """Convert q2k index tensor to k2q CSR on NPU.

    Args:
        q2k: int32 ``[H, T, topk]``. Local block indices in ``[-1, block_len)``.
        cu_seqlens: int32 ``[B+1]`` exclusive prefix of token lengths.
        cu_block_lens: int32 ``[B+1]`` exclusive prefix of KV block counts.
        order_method: ``0`` = Concat (batch), ``1`` = Round-robin.
        total_rows: Global CSR row count. ``<0`` → derive from ``cu_block_lens``.
        max_kv: Max blocks per sample. ``<0`` → derive from ``cu_block_lens``.
        use_simt: Non-zero enables Hist/Scatter SIMT on ascend950
            (``use_simt=0`` stays on portable MC / SIMD path for A2+A5).
        q_global_offset: ``False`` → batch-local ``q_ind`` (``qAbs - cu_q[bi]``);
            ``True`` → global Q token index (``qAbs``).

    Returns:
        ``(row_ptr, q_ind, slot)`` all int32 on the same device as ``q2k``:
        - ``row_ptr``: ``[H, total_rows + 1]``
        - ``q_ind``: ``[H, T * topk]`` (invalid = -1)
        - ``slot``: ``[H, T * topk]`` (invalid = -1)
    """
    enable_custom_op()
    if total_rows < 0 or max_kv < 0:
        tr, mk = k2q_csr_block_stats(cu_block_lens)
        if total_rows < 0:
            total_rows = tr
        if max_kv < 0:
            max_kv = mk
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
