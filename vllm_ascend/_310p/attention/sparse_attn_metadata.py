# SPDX-License-Identifier: Apache-2.0
"""Sparse-attention metadata fallback for Ascend 310P.

The upstream custom metadata operator only schedules work; it does not perform
attention arithmetic.  Its output is a fixed int32[1024] buffer containing
contiguous half-open work ranges for each AIC/AIV core.

The current reference AICPU implementation has flash-decode cross-core row
splitting disabled (``supportFd = false``).  A fully valid conservative plan is
therefore to assign the complete BN2 range to AIC core 0 and disable every
other core.  This preserves correctness while leaving performance tuning to a
later balanced scheduler implementation.
"""

from __future__ import annotations

import torch

SAS_METADATA_SIZE = 1024
AIC_CORE_COUNT = 36
AIV_CORE_COUNT = 72
FA_METADATA_SIZE = 8
FD_METADATA_SIZE = 8

FA_CORE_ENABLE = 0
FA_BN2_START = 1
FA_M_START = 2
FA_S2_START = 3
FA_BN2_END = 4
FA_M_END = 5
FA_S2_END = 6
FA_FIRST_FD_WORKSPACE = 7


def _infer_output_device(
    tensors: tuple[torch.Tensor | None, ...],
    device: str | torch.device,
) -> torch.device:
    for tensor in tensors:
        if tensor is not None:
            return tensor.device
    return torch.device(device)


def sparse_attn_sharedkv_metadata_310p(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_ori_kv: torch.Tensor | None = None,
    cu_seqlens_cmp_kv: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_kv: torch.Tensor | None = None,
    batch_size: int = 0,
    max_seqlen_q: int = 0,
    max_seqlen_kv: int = 0,
    ori_topk: int = 0,
    cmp_topk: int = 0,
    cmp_ratio: int = 4,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 128,
    ori_win_right: int = 0,
    layout_q: str = "BSND",
    layout_kv: str = "PA_ND",
    has_ori_kv: bool = True,
    has_cmp_kv: bool = True,
    device: str | torch.device = "npu",
) -> torch.Tensor:
    """Build a correct single-AIC sparse-attention schedule.

    The many scalar arguments intentionally mirror the custom operator schema.
    Most affect attention semantics inside the compute kernel rather than the
    validity of this conservative full-range schedule.
    """
    del (
        num_heads_q,
        head_dim,
        max_seqlen_q,
        max_seqlen_kv,
        ori_topk,
        cmp_topk,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_kv,
        has_ori_kv,
        has_cmp_kv,
    )

    if num_heads_kv <= 0:
        raise ValueError(f"num_heads_kv must be positive, got {num_heads_kv}")

    if seqused_q is not None:
        actual_batch_size = int(seqused_q.shape[0])
    elif layout_q == "TND" and cu_seqlens_q is not None:
        actual_batch_size = max(int(cu_seqlens_q.shape[0]) - 1, 0)
    else:
        actual_batch_size = int(batch_size)

    output_device = _infer_output_device(
        (
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
            seqused_q,
            seqused_kv,
        ),
        device,
    )

    # Build on CPU to avoid relying on unsupported 310P scalar in-place ops.
    metadata = torch.zeros(SAS_METADATA_SIZE, dtype=torch.int32)
    fa_metadata = metadata[: AIC_CORE_COUNT * FA_METADATA_SIZE].view(
        AIC_CORE_COUNT,
        FA_METADATA_SIZE,
    )

    # Core 0 starts at the implicit origin (0, 0, 0) and consumes all BN2
    # entries.  An end cursor of (batch*kv_heads, 0, 0) denotes all rows of all
    # preceding BN2 entries under the kernel's half-open cursor convention.
    fa_metadata[0, FA_CORE_ENABLE] = 1
    fa_metadata[0, FA_BN2_START] = 0
    fa_metadata[0, FA_M_START] = 0
    fa_metadata[0, FA_S2_START] = 0
    fa_metadata[0, FA_BN2_END] = actual_batch_size * int(num_heads_kv)
    fa_metadata[0, FA_M_END] = 0
    fa_metadata[0, FA_S2_END] = 0
    fa_metadata[0, FA_FIRST_FD_WORKSPACE] = 0

    # AIV flash-decode metadata remains all-zero because the reference
    # scheduler does not split a query row across AIC cores.
    fd_start = AIC_CORE_COUNT * FA_METADATA_SIZE
    fd_end = fd_start + AIV_CORE_COUNT * FD_METADATA_SIZE
    assert torch.count_nonzero(metadata[fd_start:fd_end]).item() == 0

    return metadata.to(output_device)
