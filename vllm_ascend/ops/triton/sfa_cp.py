# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.distributed as dist
from vllm.triton_utils import tl, triton


@triton.jit
def _pack_sfa_dcp_output_lse_kernel(
    output_ptr,
    lse_ptr,
    send_ptr,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    lse_stride_t,
    lse_stride_h,
    send_stride_rank,
    send_stride_scatter,
    send_stride_replicated,
    send_stride_d,
    local_scatter_size,
    head_dim,
    SCATTER_TOKENS: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, BLOCK_D)

    if SCATTER_TOKENS:
        rank_idx = token_idx // local_scatter_size
        scatter_idx = token_idx % local_scatter_size
        replicated_idx = head_idx
    else:
        rank_idx = head_idx // local_scatter_size
        scatter_idx = head_idx % local_scatter_size
        replicated_idx = token_idx

    send_base = (
        rank_idx * send_stride_rank + scatter_idx * send_stride_scatter + replicated_idx * send_stride_replicated
    )
    output_offsets = token_idx * output_stride_t + head_idx * output_stride_h + d_offsets * output_stride_d
    d_mask = d_offsets < head_dim
    output = tl.load(output_ptr + output_offsets, mask=d_mask, other=0.0)
    tl.store(send_ptr + send_base + d_offsets * send_stride_d, output, mask=d_mask)

    lse = tl.load(lse_ptr + token_idx * lse_stride_t + head_idx * lse_stride_h).to(tl.float32)
    if LSE_PACK_DIM == 1:
        tl.store(
            send_ptr + send_base + head_dim * send_stride_d,
            lse.to(send_ptr.dtype.element_ty),
        )
    else:
        # Arbitrary uint16 bit patterns are not reliably preserved when they
        # are bit-cast through an Ascend activation tensor. A two-term
        # numerical expansion survives HCCL while retaining almost all FP32
        # LSE precision.
        lse_hi = lse.to(send_ptr.dtype.element_ty)
        lse_residual = (lse - lse_hi.to(tl.float32)).to(send_ptr.dtype.element_ty)
        tl.store(
            send_ptr + send_base + head_dim * send_stride_d,
            lse_hi,
        )
        tl.store(
            send_ptr + send_base + (head_dim + 1) * send_stride_d,
            lse_residual,
        )


@triton.jit
def _fused_sfa_dcp_lse_combine_kernel(
    recv_ptr,
    output_ptr,
    recv_stride_rank,
    recv_stride_scatter,
    recv_stride_replicated,
    recv_stride_d,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    head_dim,
    DCP_SIZE: tl.constexpr,
    SCATTER_TOKENS: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, BLOCK_D)

    if SCATTER_TOKENS:
        scatter_idx = token_idx
        replicated_idx = head_idx
    else:
        scatter_idx = head_idx
        replicated_idx = token_idx

    # Keep only scalar LSE state and one D-vector accumulator live. A
    # [DCP_SIZE, BLOCK_D] reduction causes excessive register/UB pressure on
    # Ascend for the GLM-5.2 D=256 path.
    lse_max = -float("inf")
    for rank_idx in tl.static_range(DCP_SIZE):
        recv_base = (
            rank_idx * recv_stride_rank + scatter_idx * recv_stride_scatter + replicated_idx * recv_stride_replicated
        )
        if LSE_PACK_DIM == 1:
            lse = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d).to(tl.float32)
        else:
            lse_hi = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d).to(tl.float32)
            lse_residual = tl.load(recv_ptr + recv_base + (head_dim + 1) * recv_stride_d).to(tl.float32)
            lse = lse_hi + lse_residual
        valid_lse = (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
        lse_max = tl.maximum(lse_max, tl.where(valid_lse, lse, -float("inf")))

    any_valid_lse = lse_max != -float("inf")
    safe_lse_max = tl.where(any_valid_lse, lse_max, 0.0)
    weight_sum = 0.0
    merged = tl.zeros([BLOCK_D], dtype=tl.float32)
    d_mask = d_offsets < head_dim
    for rank_idx in tl.static_range(DCP_SIZE):
        recv_base = (
            rank_idx * recv_stride_rank + scatter_idx * recv_stride_scatter + replicated_idx * recv_stride_replicated
        )
        if LSE_PACK_DIM == 1:
            lse = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d).to(tl.float32)
        else:
            lse_hi = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d).to(tl.float32)
            lse_residual = tl.load(recv_ptr + recv_base + (head_dim + 1) * recv_stride_d).to(tl.float32)
            lse = lse_hi + lse_residual
        valid_lse = (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
        weight = tl.where(valid_lse, tl.exp(lse - safe_lse_max), 0.0)
        partial_output = tl.load(
            recv_ptr + recv_base + d_offsets * recv_stride_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        # Select before multiplication: multiplying a zero weight by a NaN
        # from an invalid rank would otherwise contaminate the result.
        partial_output = tl.where(valid_lse, partial_output, 0.0)
        merged += partial_output * weight
        weight_sum += weight

    denominator = tl.where(weight_sum > 0.0, weight_sum, 1.0)
    merged /= denominator
    output_offsets = token_idx * output_stride_t + head_idx * output_stride_h + d_offsets * output_stride_d
    tl.store(output_ptr + output_offsets, merged, mask=d_mask)


def _lse_pack_dim(output_dtype: torch.dtype) -> int:
    if output_dtype in (torch.bfloat16, torch.float16):
        return 2
    if output_dtype == torch.float32:
        return 1
    raise TypeError(f"SFA DCP fused A2A supports bfloat16, float16, or float32 output, got {output_dtype}.")


def _validate_sfa_dcp_inputs(
    sfa_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    dcp_size: int,
    scatter_dim: int,
) -> tuple[int, int, int, int, int]:
    if sfa_output.ndim != 3:
        raise RuntimeError(
            f"SFA DCP fused A2A expects output [tokens, heads, head_dim], got {tuple(sfa_output.shape)}."
        )
    if softmax_lse.shape != (*sfa_output.shape[:2], 1):
        raise RuntimeError(
            "SFA DCP fused A2A expects LSE [tokens, heads, 1] matching output, "
            f"got {tuple(sfa_output.shape)} and {tuple(softmax_lse.shape)}."
        )
    if softmax_lse.dtype != torch.float32:
        raise TypeError(f"SFA DCP fused A2A requires float32 LSE, got {softmax_lse.dtype}.")
    if sfa_output.device != softmax_lse.device:
        raise RuntimeError(
            "SFA DCP fused A2A requires output and LSE on the same device, "
            f"got {sfa_output.device} and {softmax_lse.device}."
        )
    if sfa_output.device.type != "npu":
        raise RuntimeError(f"SFA DCP fused A2A requires an NPU tensor, got {sfa_output.device}.")
    if not isinstance(dcp_size, int) or isinstance(dcp_size, bool) or dcp_size <= 0:
        raise ValueError(f"SFA DCP fused A2A requires a positive integer dcp_size, got {dcp_size}.")
    if scatter_dim not in (0, 1):
        raise ValueError(f"SFA DCP fused A2A scatter_dim must be 0 or 1, got {scatter_dim}.")

    num_tokens, num_heads, head_dim = sfa_output.shape
    if num_tokens <= 0 or num_heads <= 0 or head_dim <= 0:
        raise RuntimeError(f"SFA DCP fused A2A requires non-empty dimensions, got {tuple(sfa_output.shape)}.")
    scatter_size = sfa_output.shape[scatter_dim]
    if scatter_size % dcp_size != 0:
        raise RuntimeError(
            "SFA DCP fused A2A requires the scatter dimension to be divisible "
            f"by dcp_size, got shape={tuple(sfa_output.shape)}, "
            f"scatter_dim={scatter_dim}, and dcp_size={dcp_size}."
        )
    local_scatter_size = scatter_size // dcp_size
    replicated_size = num_heads if scatter_dim == 0 else num_tokens
    return num_tokens, num_heads, head_dim, local_scatter_size, replicated_size


def pack_sfa_dcp_output_lse(
    sfa_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    dcp_size: int,
    scatter_dim: int,
) -> torch.Tensor:
    """Pack strided SFA output and FP32 LSE into one HCCL payload."""
    num_tokens, num_heads, head_dim, local_scatter_size, replicated_size = _validate_sfa_dcp_inputs(
        sfa_output, softmax_lse, dcp_size, scatter_dim
    )
    lse_pack_dim = _lse_pack_dim(sfa_output.dtype)
    send = torch.empty(
        (dcp_size, local_scatter_size, replicated_size, head_dim + lse_pack_dim),
        dtype=sfa_output.dtype,
        device=sfa_output.device,
    )
    _pack_sfa_dcp_output_lse_kernel[(num_tokens, num_heads)](
        sfa_output,
        softmax_lse,
        send,
        *sfa_output.stride(),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        *send.stride(),
        local_scatter_size,
        head_dim,
        SCATTER_TOKENS=scatter_dim == 0,
        LSE_PACK_DIM=lse_pack_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return send


def fused_sfa_dcp_lse_combine(
    recv: torch.Tensor,
    head_dim: int,
    scatter_dim: int,
) -> torch.Tensor:
    """Unpack one HCCL payload and merge rank outputs using their LSE."""
    if recv.ndim != 4:
        raise RuntimeError(f"SFA DCP fused combine expects a 4D receive buffer, got {tuple(recv.shape)}.")
    if not recv.is_contiguous():
        raise RuntimeError("SFA DCP fused combine requires a contiguous HCCL receive buffer.")
    if recv.device.type != "npu":
        raise RuntimeError(f"SFA DCP fused combine requires an NPU tensor, got {recv.device}.")
    if scatter_dim not in (0, 1):
        raise ValueError(f"SFA DCP fused combine scatter_dim must be 0 or 1, got {scatter_dim}.")
    if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0:
        raise ValueError(f"SFA DCP fused combine requires a positive integer head_dim, got {head_dim}.")

    dcp_size, local_scatter_size, replicated_size, packed_dim = recv.shape
    lse_pack_dim = _lse_pack_dim(recv.dtype)
    if packed_dim != head_dim + lse_pack_dim:
        raise RuntimeError(
            "SFA DCP fused combine received an invalid packed dimension: "
            f"expected {head_dim + lse_pack_dim}, got {packed_dim}."
        )
    num_tokens, num_heads = (
        (local_scatter_size, replicated_size) if scatter_dim == 0 else (replicated_size, local_scatter_size)
    )
    output = torch.empty(
        (num_tokens, num_heads, head_dim),
        dtype=recv.dtype,
        device=recv.device,
    )
    _fused_sfa_dcp_lse_combine_kernel[(num_tokens, num_heads)](
        recv,
        output,
        *recv.stride(),
        *output.stride(),
        head_dim,
        DCP_SIZE=dcp_size,
        SCATTER_TOKENS=scatter_dim == 0,
        LSE_PACK_DIM=lse_pack_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return output


def sfa_dcp_a2a_fused_combine(
    sfa_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    dcp_size: int,
    scatter_dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Run stride-aware pack, one HCCL All2All, and fused LSE combine."""
    send = pack_sfa_dcp_output_lse(
        sfa_output,
        softmax_lse,
        dcp_size,
        scatter_dim,
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return fused_sfa_dcp_lse_combine(recv, sfa_output.shape[-1], scatter_dim)
