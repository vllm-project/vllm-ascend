# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator, _groups
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.dcp.dcp_a2a_batched import (
    MAX_UNROLLED_COMBINE_ELEMENTS,
    _encode_lse_fields,
    _fused_dcp_lse_combine_batched_kernel,
    _pack_dcp_output_lse_batched_kernel,
)
from vllm_ascend.ops.triton.dcp.dcp_jit import _launch_kernel
from vllm_ascend.ops.triton.dcp.sfa_dcp_exchange import can_exchange, exchange
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit
def _pack_dcp_output_lse_kernel(
    output_ptr,
    lse_ptr,
    sum_ptr,
    send_ptr,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    lse_stride_t,
    lse_stride_h,
    sum_stride_t,
    sum_stride_h,
    send_stride_rank,
    send_stride_scatter,
    send_stride_replicated,
    send_stride_d,
    local_scatter_size,
    head_dim,
    num_heads,
    total_rows,
    SCATTER_TOKENS: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
    FROM_STATS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    program_idx = tl.program_id(0)
    num_programs = tl.num_programs(0)
    d_offsets = tl.arange(0, BLOCK_D)

    for linear_idx in range(program_idx, total_rows, num_programs):
        token_idx = (linear_idx // num_heads).to(tl.int64)
        head_idx = (linear_idx % num_heads).to(tl.int64)

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
        if FROM_STATS:
            total = tl.load(sum_ptr + token_idx * sum_stride_t + head_idx * sum_stride_h).to(tl.float32)
            lse = lse + tl.log(total)
        if LSE_PACK_DIM == 1:
            tl.store(
                send_ptr + send_base + head_dim * send_stride_d,
                lse.to(send_ptr.dtype.element_ty),
            )
        else:
            exponent_code, significand_hi, significand_mid, significand_lo = _encode_lse_fields(lse)
            tl.store(
                send_ptr + send_base + head_dim * send_stride_d,
                exponent_code.to(send_ptr.dtype.element_ty),
            )
            tl.store(
                send_ptr + send_base + (head_dim + 1) * send_stride_d,
                significand_hi.to(send_ptr.dtype.element_ty),
            )
            tl.store(
                send_ptr + send_base + (head_dim + 2) * send_stride_d,
                significand_mid.to(send_ptr.dtype.element_ty),
            )
            tl.store(
                send_ptr + send_base + (head_dim + 3) * send_stride_d,
                significand_lo.to(send_ptr.dtype.element_ty),
            )


@triton.jit
def _fused_dcp_lse_combine_kernel(
    recv_ptr,
    output_ptr,
    local_output_ptr,
    local_lse_ptr,
    local_output_stride_t,
    local_output_stride_h,
    local_output_stride_d,
    local_lse_stride_t,
    local_lse_stride_h,
    recv_stride_rank,
    recv_stride_scatter,
    recv_stride_replicated,
    recv_stride_d,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    head_dim,
    num_heads,
    total_rows,
    DCP_SIZE: tl.constexpr,
    SCATTER_TOKENS: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    RETURN_LSE: tl.constexpr = False,
    HAS_LOCAL: tl.constexpr = False,
):
    program_idx = tl.program_id(0)
    num_programs = tl.num_programs(0)
    d_offsets = tl.arange(0, BLOCK_D)

    for linear_idx in range(program_idx, total_rows, num_programs):
        token_idx = (linear_idx // num_heads).to(tl.int64)
        head_idx = (linear_idx % num_heads).to(tl.int64)

        if SCATTER_TOKENS:
            scatter_idx = token_idx
            replicated_idx = head_idx
        else:
            scatter_idx = head_idx
            replicated_idx = token_idx

        # Keep only scalar LSE state and one D-vector accumulator live. A
        # [DCP_SIZE, BLOCK_D] reduction causes excessive register/UB pressure
        # on Ascend for the GLM-5.2 D=256 path.
        lse_max = -float("inf")
        if HAS_LOCAL:
            local_lse = tl.load(local_lse_ptr + token_idx * local_lse_stride_t + head_idx * local_lse_stride_h).to(
                tl.float32
            )
            local_valid = (local_lse == local_lse) & (local_lse != float("inf")) & (local_lse != -float("inf"))
            lse_max = tl.where(local_valid, local_lse, -float("inf"))
        for rank_idx in tl.static_range(DCP_SIZE):
            recv_base = (
                rank_idx * recv_stride_rank
                + scatter_idx * recv_stride_scatter
                + replicated_idx * recv_stride_replicated
            )
            if LSE_PACK_DIM == 1:
                lse = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d).to(tl.float32)
                valid_lse = (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            else:
                exponent_code = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d).to(tl.float32)
                significand_hi = tl.load(recv_ptr + recv_base + (head_dim + 1) * recv_stride_d).to(tl.float32)
                significand_mid = tl.load(recv_ptr + recv_base + (head_dim + 2) * recv_stride_d).to(tl.float32)
                significand_lo = tl.load(recv_ptr + recv_base + (head_dim + 3) * recv_stride_d).to(tl.float32)
                packed_valid = exponent_code != 0.0
                sign = tl.where(exponent_code < 0.0, -1.0, 1.0)
                exponent_magnitude = tl.where(exponent_code < 0.0, -exponent_code, exponent_code)
                safe_exponent = exponent_magnitude - 128.0
                significand = significand_hi * 65536.0 + significand_mid * 256.0 + significand_lo
                lse = sign * significand * tl.exp2(safe_exponent - 23.0)
                valid_lse = packed_valid & (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            lse_max = tl.maximum(lse_max, tl.where(valid_lse, lse, -float("inf")))

        any_valid_lse = lse_max != -float("inf")
        safe_lse_max = tl.where(any_valid_lse, lse_max, 0.0)
        weight_sum = 0.0
        merged = tl.zeros([BLOCK_D], dtype=tl.float32)
        d_mask = d_offsets < head_dim
        for rank_idx in tl.static_range(DCP_SIZE):
            recv_base = (
                rank_idx * recv_stride_rank
                + scatter_idx * recv_stride_scatter
                + replicated_idx * recv_stride_replicated
            )
            if LSE_PACK_DIM == 1:
                lse = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d).to(tl.float32)
                valid_lse = (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            else:
                exponent_code = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d).to(tl.float32)
                significand_hi = tl.load(recv_ptr + recv_base + (head_dim + 1) * recv_stride_d).to(tl.float32)
                significand_mid = tl.load(recv_ptr + recv_base + (head_dim + 2) * recv_stride_d).to(tl.float32)
                significand_lo = tl.load(recv_ptr + recv_base + (head_dim + 3) * recv_stride_d).to(tl.float32)
                packed_valid = exponent_code != 0.0
                sign = tl.where(exponent_code < 0.0, -1.0, 1.0)
                exponent_magnitude = tl.where(exponent_code < 0.0, -exponent_code, exponent_code)
                safe_exponent = exponent_magnitude - 128.0
                significand = significand_hi * 65536.0 + significand_mid * 256.0 + significand_lo
                lse = sign * significand * tl.exp2(safe_exponent - 23.0)
                valid_lse = packed_valid & (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            weight = tl.where(valid_lse, tl.exp(lse - safe_lse_max), 0.0)
            partial_output = tl.load(
                recv_ptr + recv_base + d_offsets * recv_stride_d,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            # Select before multiplication: multiplying a zero weight by a
            # NaN from an invalid rank would otherwise contaminate the result.
            partial_output = tl.where(valid_lse, partial_output, 0.0)
            merged += partial_output * weight
            weight_sum += weight

        if HAS_LOCAL:
            local_weight = tl.where(local_valid, tl.exp(local_lse - safe_lse_max), 0.0)
            local_offsets = (
                token_idx * local_output_stride_t + head_idx * local_output_stride_h + d_offsets * local_output_stride_d
            )
            local_output = tl.load(local_output_ptr + local_offsets, mask=d_mask, other=0.0).to(tl.float32)
            merged += tl.where(local_valid, local_output, 0.0) * local_weight
            weight_sum += local_weight

        denominator = tl.where(weight_sum > 0.0, weight_sum, 1.0)
        merged /= denominator
        output_offsets = token_idx * output_stride_t + head_idx * output_stride_h + d_offsets * output_stride_d
        tl.store(output_ptr + output_offsets, merged, mask=d_mask)
        if RETURN_LSE:
            merged_lse = tl.where(any_valid_lse, safe_lse_max + tl.log(denominator), -float("inf"))
            lse_offset = token_idx * output_stride_t + head_idx * output_stride_h + head_dim * output_stride_d
            tl.store(output_ptr + lse_offset, merged_lse)


def _lse_pack_dim(output_dtype: torch.dtype) -> int:
    if output_dtype in (torch.bfloat16, torch.float16):
        return 4
    if output_dtype == torch.float32:
        return 1
    raise TypeError(f"DCP output/LSE pack supports bfloat16, float16, or float32 output, got {output_dtype}.")


def _validate_dcp_inputs(
    partial_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    dcp_size: int,
    scatter_dim: int,
) -> tuple[int, int, int, int, int]:
    if partial_output.ndim != 3:
        raise RuntimeError(
            f"DCP output/LSE pack expects output [tokens, heads, head_dim], got {tuple(partial_output.shape)}."
        )
    if softmax_lse.shape != (*partial_output.shape[:2], 1):
        raise RuntimeError(
            "DCP output/LSE pack expects LSE [tokens, heads, 1] matching output, "
            f"got {tuple(partial_output.shape)} and {tuple(softmax_lse.shape)}."
        )
    if softmax_lse.dtype != torch.float32:
        raise TypeError(f"DCP output/LSE pack requires float32 LSE, got {softmax_lse.dtype}.")
    if partial_output.device != softmax_lse.device:
        raise RuntimeError(
            "DCP output/LSE pack requires output and LSE on the same device, "
            f"got {partial_output.device} and {softmax_lse.device}."
        )
    if partial_output.device.type != "npu":
        raise RuntimeError(f"DCP output/LSE pack requires an NPU tensor, got {partial_output.device}.")
    if not isinstance(dcp_size, int) or isinstance(dcp_size, bool) or dcp_size <= 0:
        raise ValueError(f"DCP output/LSE pack requires a positive integer dcp_size, got {dcp_size}.")
    if scatter_dim not in (0, 1):
        raise ValueError(f"DCP output/LSE pack scatter_dim must be 0 or 1, got {scatter_dim}.")

    num_tokens, num_heads, head_dim = partial_output.shape
    if num_tokens <= 0 or num_heads <= 0 or head_dim <= 0:
        raise RuntimeError(f"DCP output/LSE pack requires non-empty dimensions, got {tuple(partial_output.shape)}.")
    scatter_size = partial_output.shape[scatter_dim]
    if scatter_size % dcp_size != 0:
        raise RuntimeError(
            "DCP output/LSE pack requires the scatter dimension to be divisible "
            f"by dcp_size, got shape={tuple(partial_output.shape)}, "
            f"scatter_dim={scatter_dim}, and dcp_size={dcp_size}."
        )
    local_scatter_size = scatter_size // dcp_size
    replicated_size = num_heads if scatter_dim == 0 else num_tokens
    return num_tokens, num_heads, head_dim, local_scatter_size, replicated_size


def pack_dcp_output_lse(
    partial_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    dcp_size: int,
    scatter_dim: int,
    softmax_sum: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack output and LSE, or compute LSE from max/sum during packing."""
    _, _, head_dim, local_scatter_size, replicated_size = _validate_dcp_inputs(
        partial_output, softmax_lse, dcp_size, scatter_dim
    )
    if softmax_sum is not None:
        if (
            softmax_sum.shape != softmax_lse.shape
            or softmax_sum.dtype != torch.float32
            or softmax_sum.device != partial_output.device
        ):
            raise ValueError("softmax_sum must match the FP32 softmax_max shape and device")
    lse_pack_dim = _lse_pack_dim(partial_output.dtype)
    send = torch.empty(
        (dcp_size, local_scatter_size, replicated_size, head_dim + lse_pack_dim),
        dtype=partial_output.dtype,
        device=partial_output.device,
    )
    init_device_properties_triton()
    _launch_pack_dcp_output_lse(
        partial_output, softmax_lse, softmax_sum, send, scatter_dim, get_vectorcore_num(), _launch_kernel
    )
    return send


def _launch_pack_dcp_output_lse(partial_output, softmax_lse, softmax_sum, send, scatter_dim, vector_cores, launch):
    """Shared runtime/warmup launch construction; no allocation or collective."""
    num_tokens, num_heads, head_dim = partial_output.shape
    local_scatter_size = send.shape[1]
    sum_input = softmax_lse if softmax_sum is None else softmax_sum
    lse_pack_dim = _lse_pack_dim(partial_output.dtype)
    total_rows = num_tokens * num_heads
    grid_size = total_rows if total_rows < vector_cores else vector_cores
    # Keep enough row tiles to occupy the vector cores. Feature-strided
    # loads can regress with batching, especially for wide heads.
    batched = (
        partial_output.dtype == torch.bfloat16
        and scatter_dim == 1
        and head_dim <= 2048
        and total_rows >= 8 * vector_cores
        and (partial_output.stride(-1) == 1 or (head_dim == 512 and num_tokens >= 128))
    )
    if batched:
        launch(
            _pack_dcp_output_lse_batched_kernel,
            (grid_size,),
            partial_output,
            softmax_lse,
            sum_input,
            send,
            *partial_output.stride(),
            softmax_lse.stride(0),
            softmax_lse.stride(1),
            sum_input.stride(0),
            sum_input.stride(1),
            *send.stride(),
            local_scatter_size,
            head_dim,
            num_heads,
            total_rows,
            SCATTER_TOKENS=scatter_dim == 0,
            LSE_PACK_DIM=lse_pack_dim,
            FROM_STATS=softmax_sum is not None,
            BLOCK_D=triton.next_power_of_2(head_dim),
            BLOCK_ROWS=32 if head_dim == 512 and num_tokens >= 128 and partial_output.stride(-1) == 1 else 8,
            OUTPUT_HEAD_MAJOR=head_dim == 512,
        )
        return
    launch(
        _pack_dcp_output_lse_kernel,
        (grid_size,),
        partial_output,
        softmax_lse,
        sum_input,
        send,
        *partial_output.stride(),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        sum_input.stride(0),
        sum_input.stride(1),
        *send.stride(),
        local_scatter_size,
        head_dim,
        num_heads,
        total_rows,
        SCATTER_TOKENS=scatter_dim == 0,
        LSE_PACK_DIM=lse_pack_dim,
        FROM_STATS=softmax_sum is not None,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return


def fused_dcp_lse_combine(
    recv: torch.Tensor,
    head_dim: int,
    scatter_dim: int,
    return_lse: bool = False,
    local_output: torch.Tensor | None = None,
    local_lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Merge rank outputs and an optional local FIA contribution in one pass.

    Local tensors use [tokens, heads, D/1] layout and may be strided. The local
    contribution is counted once, with FP32 weights shared by all ranks.
    """
    # Appended LSE must retain FP32 precision.
    if return_lse and recv.dtype != torch.float32:
        raise TypeError("Returning merged LSE requires an FP32 receive buffer.")
    if recv.ndim != 4:
        raise RuntimeError(f"DCP fused combine expects a 4D receive buffer, got {tuple(recv.shape)}.")
    if not recv.is_contiguous():
        raise RuntimeError("DCP fused combine requires a contiguous HCCL receive buffer.")
    if recv.device.type != "npu":
        raise RuntimeError(f"DCP fused combine requires an NPU tensor, got {recv.device}.")
    if scatter_dim not in (0, 1):
        raise ValueError(f"DCP fused combine scatter_dim must be 0 or 1, got {scatter_dim}.")
    if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0:
        raise ValueError(f"DCP fused combine requires a positive integer head_dim, got {head_dim}.")

    _, local_scatter_size, replicated_size, packed_dim = recv.shape
    lse_pack_dim = _lse_pack_dim(recv.dtype)
    if packed_dim != head_dim + lse_pack_dim:
        raise RuntimeError(
            "DCP fused combine received an invalid packed dimension: "
            f"expected {head_dim + lse_pack_dim}, got {packed_dim}."
        )
    num_tokens, num_heads = (
        (local_scatter_size, replicated_size) if scatter_dim == 0 else (replicated_size, local_scatter_size)
    )
    if (local_output is None) != (local_lse is None):
        raise ValueError("Local output and LSE must be supplied together.")
    if local_output is not None:
        _validate_dcp_inputs(local_output, local_lse, 1, scatter_dim)
        if local_output.shape != (num_tokens, num_heads, head_dim):
            raise RuntimeError("Local output must match the post-scatter token, head and feature dimensions.")
        if local_output.device != recv.device:
            raise RuntimeError("Local contribution and receive buffer must be on the same device.")
        _lse_pack_dim(local_output.dtype)
    output = torch.empty(
        (num_tokens, num_heads, head_dim + int(return_lse)),
        dtype=recv.dtype,
        device=recv.device,
    )
    init_device_properties_triton()
    _launch_fused_dcp_lse_combine(
        recv, output, head_dim, scatter_dim, return_lse, local_output, local_lse, get_vectorcore_num(), _launch_kernel
    )
    return output


def _launch_fused_dcp_lse_combine(
    recv, output, head_dim, scatter_dim, return_lse, local_output, local_lse, vector_cores, launch
):
    """Shared runtime/warmup launch construction for the packed receive layout."""
    num_tokens, num_heads = output.shape[:2]
    dcp_size = recv.shape[0]
    lse_pack_dim = _lse_pack_dim(recv.dtype)
    total_rows = num_tokens * num_heads
    grid_size = total_rows if total_rows < vector_cores else vector_cores
    # Stream ranks so the live output accumulator is independent of DCP size.
    # Four rows per core amortize the wider row tile; small workloads keep the
    # scalar path. No chip model, token/head list or DCP whitelist is required.
    batched = (
        recv.dtype == torch.bfloat16
        and scatter_dim == 1
        and head_dim <= 512
        and total_rows >= 4 * vector_cores
        # Feature-strided code exceeds 192 KiB UB on the measured A3;
        # one-row coalesced loads produce invalid CANN 9.2 VEC parameters.
        and (local_output is None or local_output.stride(-1) == 1)
        and not return_lse
    )
    if batched:
        block_rows = 8
        block_d = triton.next_power_of_2(head_dim)
        # Unrolling amortizes loop overhead when cores process multiple tiles.
        # Bound its live output tiles; larger worlds use the same streaming math.
        rank_unroll = (
            dcp_size
            if total_rows >= 2 * block_rows * vector_cores
            and dcp_size * block_rows * block_d <= MAX_UNROLLED_COMBINE_ELEMENTS
            else 1
        )
        grid_size = min(vector_cores, triton.cdiv(total_rows, block_rows))
        launch(
            _fused_dcp_lse_combine_batched_kernel,
            (grid_size,),
            recv,
            output,
            local_output if local_output is not None else recv,
            local_lse if local_lse is not None else recv,
            *(local_output.stride() if local_output is not None else (0, 0, 0)),
            *(local_lse.stride()[:2] if local_lse is not None else (0, 0)),
            *recv.stride(),
            *output.stride(),
            head_dim,
            num_heads,
            total_rows,
            DCP_SIZE=dcp_size,
            HAS_LOCAL=local_output is not None,
            RANK_UNROLL=rank_unroll,
            BLOCK_D=block_d,
            BLOCK_ROWS=block_rows,
            multibuffer=False,
        )
        return
    launch(
        _fused_dcp_lse_combine_kernel,
        (grid_size,),
        recv,
        output,
        local_output if local_output is not None else recv,
        local_lse if local_lse is not None else recv,
        *(local_output.stride() if local_output is not None else (0, 0, 0)),
        *(local_lse.stride()[:2] if local_lse is not None else (0, 0)),
        *recv.stride(),
        *output.stride(),
        head_dim,
        num_heads,
        total_rows,
        DCP_SIZE=dcp_size,
        SCATTER_TOKENS=scatter_dim == 0,
        LSE_PACK_DIM=lse_pack_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        RETURN_LSE=return_lse,
        HAS_LOCAL=local_output is not None,
    )
    return


def _use_fast_exchange(partial_output, softmax_lse, scatter_size, scatter_dim, pcp_group, return_lse, defer_combine):
    return (
        pcp_group is None
        and not return_lse
        and not defer_combine
        and scatter_dim == 1
        and can_exchange(partial_output, softmax_lse, scatter_size)
    )


def dcp_a2a_fused_combine(
    partial_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    scatter_size: int,
    scatter_dim: int,
    scatter_group: dist.ProcessGroup | None,
    pcp_group: GroupCoordinator | None = None,
    return_lse: bool = False,
    defer_combine: bool = False,
    softmax_sum: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack, scatter over DCP or TP, optionally gather over PCP, then merge.

    scatter_size is the All2All group size, not the unified DCP size when
    stacking PCP and DCP. Use 1 when Q heads were not gathered over TP.
    """
    # Keep PCP composition and token-scatter paths on the upstream implementation.
    if _use_fast_exchange(partial_output, softmax_lse, scatter_size, scatter_dim, pcp_group, return_lse, defer_combine):
        if scatter_group is None:
            raise ValueError("SFA output scatter requires an explicit All2All group.")
        return exchange(partial_output, softmax_lse, scatter_group, softmax_sum=softmax_sum).to(partial_output.dtype)

    send = pack_dcp_output_lse(
        partial_output,
        softmax_lse,
        scatter_size,
        scatter_dim=scatter_dim,
        softmax_sum=softmax_sum,
    )
    if scatter_size > 1:
        if scatter_group is None:
            raise ValueError("DCP output scatter requires an explicit All2All group.")
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=scatter_group)
    else:
        recv = send

    if pcp_group is not None:
        # For head scatter, TP exchange gives [TP source, local H, T, packed D].
        # PCP gathers the same TP head slice into [PCP * TP, local H, T, packed D].
        # PCP2/TP4 thus supplies eight KV contributions per local head.
        # O and LSE travel together, so the source-rank order is immaterial
        # to their weighted sum; no logical-DCP permutation is needed here.
        recv = pcp_group.all_gather(recv, dim=0)
    if defer_combine:
        return recv
    return fused_dcp_lse_combine(recv, partial_output.shape[-1], scatter_dim=scatter_dim, return_lse=return_lse)


def dcp_a2a_fused(
    partial_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    dcp_size: int,
    scatter_dim: int,
    group_name: str,
    pcp_group_name: str | None = None,
    return_lse: bool = False,
    defer_combine: bool = False,
    softmax_sum: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused DCP output merge, optionally gathering contributions across PCP.

    With PCP enabled, dcp_size/group_name describe the TP scatter group,
    not the unified DCP group. A size of 1 skips scatter-group lookup and All2All.
    With defer_combine=True, return the packed rank contributions after communication.
    With return_lse=True, require FP32 input and return [..., head_dim + 1],
    with the merged LSE in the last element for a subsequent local merge.
    """
    if defer_combine and return_lse:
        raise ValueError("defer_combine and return_lse are mutually exclusive.")
    # In return_lse mode the last output element stores the merged FP32 LSE.
    if return_lse and partial_output.dtype != torch.float32:
        raise TypeError("Returning merged LSE requires FP32 attention output.")
    scatter_group = None
    if dcp_size > 1:
        group_ref = _groups.get(group_name)
        if group_ref is None:
            raise RuntimeError(f"DCP fused A2A group {group_name!r} is not registered.")
        group = group_ref()
        if group is None or group.device_group is None:
            raise RuntimeError(f"DCP fused A2A group {group_name!r} is unavailable.")
        if group.world_size != dcp_size:
            raise RuntimeError(
                f"DCP fused A2A group size does not match dcp_size: group={group.world_size}, dcp_size={dcp_size}."
            )
        scatter_group = group.device_group

    pcp_group = None
    if pcp_group_name is not None:
        pcp_group_ref = _groups.get(pcp_group_name)
        pcp_group = pcp_group_ref() if pcp_group_ref is not None else None
        if pcp_group is None or pcp_group.device_group is None:
            raise RuntimeError(f"PCP+DCP group {pcp_group_name!r} is unavailable.")
    return dcp_a2a_fused_combine(
        partial_output,
        softmax_lse,
        dcp_size,
        scatter_dim,
        scatter_group=scatter_group,
        pcp_group=pcp_group,
        return_lse=return_lse,
        defer_combine=defer_combine,
        softmax_sum=softmax_sum,
    )


def dcp_a2a_fused_fake(
    partial_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    dcp_size: int,
    scatter_dim: int,
    group_name: str,
    pcp_group_name: str | None = None,
    return_lse: bool = False,
    defer_combine: bool = False,
    softmax_sum: torch.Tensor | None = None,
) -> torch.Tensor:
    """Propagate output metadata for torch.compile without running HCCL.

    PyTorch invokes this FakeTensor implementation while tracing the custom
    operator. It must only describe the local output shape, dtype, and device;
    the real implementation performs the collective at execution time.
    """
    del softmax_lse, group_name
    if defer_combine and return_lse:
        raise ValueError("defer_combine and return_lse are mutually exclusive.")
    if defer_combine:
        rank_count = dcp_size
        if pcp_group_name is not None:
            group_ref = _groups.get(pcp_group_name)
            group = group_ref() if group_ref is not None else None
            if group is None:
                raise RuntimeError(f"PCP+DCP group {pcp_group_name!r} is unavailable.")
            rank_count *= group.world_size
        return torch.empty(
            (
                rank_count,
                partial_output.shape[scatter_dim] // dcp_size,
                partial_output.shape[1 - scatter_dim],
                partial_output.shape[-1] + _lse_pack_dim(partial_output.dtype),
            ),
            dtype=partial_output.dtype,
            device=partial_output.device,
        )
    if return_lse and partial_output.dtype != torch.float32:
        raise TypeError("Returning merged LSE requires FP32 attention output.")
    output_shape = list(partial_output.shape)
    output_shape[scatter_dim] //= dcp_size
    output_shape[-1] += int(return_lse)
    return torch.empty(output_shape, dtype=partial_output.dtype, device=partial_output.device)


direct_register_custom_op(
    op_name="dcp_a2a_fused",
    op_func=dcp_a2a_fused,
    fake_impl=dcp_a2a_fused_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
