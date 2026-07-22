# SPDX-License-Identifier: Apache-2.0
#
# Candidate-only Domino correction kernel. It scores a fixed base-logit top-k
# set and is selected by ``additional_config.domino_candidate_topk``.

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _fused_candidate_mlp_argmax_kernel(
    z_ptr,  # [B, M]
    s_ptr,  # [B, M]
    base_top_ptr,  # [B, K]
    candidate_ids_ptr,  # [B, K]
    fc2_ptr,  # [V, M]
    fc2_bias_ptr,  # [V], or dummy when HAS_FC2_BIAS=False
    out_token_ptr,  # [B]
    M,
    stride_z_b,
    stride_z_m,
    stride_s_b,
    stride_s_m,
    stride_base_b,
    stride_base_k,
    stride_ids_b,
    stride_ids_k,
    stride_fc2_v,
    stride_fc2_m,
    stride_out_b,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_FC2_BIAS: tl.constexpr,
):
    """Score only K base-logit candidates for one batch row.

    This Vector Core kernel has one program per batch row. Candidate vocabulary
    rows are irregular, so batching rows with tl.dot would not reuse weights;
    the small K=64 reduction is more naturally handled as vector FMA + sum.
    """
    pid_b = tl.program_id(0)
    offs_k = tl.arange(0, K)
    candidate_ids = tl.load(candidate_ids_ptr + pid_b * stride_ids_b + offs_k * stride_ids_k)
    scores = tl.zeros([K], dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        z = tl.load(
            z_ptr + pid_b * stride_z_b + offs_m * stride_z_m,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        s = tl.load(
            s_ptr + pid_b * stride_s_b + offs_m * stride_s_m,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        preact = z + s
        mid = preact * tl.sigmoid(preact)
        weight_ptrs = fc2_ptr + candidate_ids[:, None] * stride_fc2_v + offs_m[None, :] * stride_fc2_m
        weight = tl.load(
            weight_ptrs,
            mask=(offs_k[:, None] < K) & mask_m[None, :],
            other=0.0,
        ).to(tl.float32)
        scores += tl.sum(weight * mid[None, :], axis=1)

    base_top = tl.load(base_top_ptr + pid_b * stride_base_b + offs_k * stride_base_k).to(tl.float32)
    scores += base_top
    if HAS_FC2_BIAS:
        scores += tl.load(fc2_bias_ptr + candidate_ids).to(tl.float32)

    # Match full-vocabulary greedy argmax semantics: when corrected scores tie,
    # choose the smallest global token id rather than the first top-k position.
    # ``torch.topk`` orders by base score, so choosing its first tied position
    # would otherwise make candidate mode differ even when both winners are in K.
    max_score = tl.max(scores, axis=0)
    tied_candidate_ids = tl.where(
        scores == max_score,
        candidate_ids.to(tl.int32),
        2147483647,
    )
    winner = -tl.max(-tied_candidate_ids, axis=0)
    tl.store(out_token_ptr + pid_b * stride_out_b, winner)


def fused_candidate_mlp_argmax(
    z_proj: torch.Tensor,  # [B, M]
    s_proj: torch.Tensor,  # [B, M]
    base_top_values: torch.Tensor,  # [B, K]
    candidate_ids: torch.Tensor,  # [B, K]
    fc2_weight: torch.Tensor,  # [V, M]
    fc2_bias: torch.Tensor | None = None,  # [V]
    out_tokens: torch.Tensor | None = None,  # [B], optional preallocated output
) -> torch.Tensor:
    """Return Domino's greedy winner restricted to supplied base top-k ids.

    ``out_tokens`` lets the correction loop write directly into its fixed
    ``[B, k]`` result buffer. This avoids a per-position temporary allocation
    and copy, and keeps the allocation pattern stable during ACL graph capture.
    """
    batch_size, mlp_size = z_proj.shape
    if (
        s_proj.shape != z_proj.shape
        or base_top_values.shape != candidate_ids.shape
        or base_top_values.shape[0] != batch_size
        or fc2_weight.shape[1] != mlp_size
    ):
        raise ValueError("Incompatible Domino candidate correction tensor shapes")

    candidate_topk = candidate_ids.shape[1]
    if candidate_topk <= 0 or candidate_topk & (candidate_topk - 1):
        raise ValueError("candidate_ids width must be a positive power of two")
    if out_tokens is None:
        output = torch.empty(batch_size, dtype=torch.int64, device=z_proj.device)
    else:
        if (
            out_tokens.shape != (batch_size,)
            or out_tokens.dtype != torch.int64
            or out_tokens.device != z_proj.device
        ):
            raise ValueError("out_tokens must be an int64 [B] tensor on z_proj.device")
        output = out_tokens
    bias_ptr = fc2_bias if fc2_bias is not None else output
    _fused_candidate_mlp_argmax_kernel[(batch_size,)](
        z_proj,
        s_proj,
        base_top_values,
        candidate_ids,
        fc2_weight,
        bias_ptr,
        output,
        mlp_size,
        z_proj.stride(0),
        z_proj.stride(1),
        s_proj.stride(0),
        s_proj.stride(1),
        base_top_values.stride(0),
        base_top_values.stride(1),
        candidate_ids.stride(0),
        candidate_ids.stride(1),
        fc2_weight.stride(0),
        fc2_weight.stride(1),
        output.stride(0),
        K=candidate_topk,
        BLOCK_M=64,
        HAS_FC2_BIAS=fc2_bias is not None,
    )
    return output
