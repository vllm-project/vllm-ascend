# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import _groups
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit
def _pack_sfa_dcp_output_lse_kernel(
    output_ptr,
    lse_or_max_ptr,
    sum_ptr,
    send_ptr,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    lse_or_max_stride_t,
    lse_or_max_stride_h,
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
    INPUT_MAX_SUM: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
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

        statistic_offset = token_idx * lse_or_max_stride_t + head_idx * lse_or_max_stride_h
        lse = tl.load(lse_or_max_ptr + statistic_offset).to(tl.float32)
        if INPUT_MAX_SUM:
            summation = tl.load(sum_ptr + token_idx * sum_stride_t + head_idx * sum_stride_h).to(tl.float32)
            finite_max = (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            finite_sum = (summation == summation) & (summation != float("inf")) & (summation != -float("inf"))
            valid_statistics = finite_max & finite_sum & (summation > 0.0)
            safe_sum = tl.where(valid_statistics, summation, 1.0)
            lse = tl.where(valid_statistics, lse + tl.log(safe_sum), -float("inf"))
        if LSE_PACK_DIM == 1:
            tl.store(
                send_ptr + send_base + head_dim * send_stride_d,
                lse.to(send_ptr.dtype.element_ty),
            )
        else:
            # Store a finite FP32 LSE as a signed exponent code plus three
            # base-256 significand digits. Every stored value is an integer in
            # [-255, 255], so FP16 and BF16 preserve it exactly.
            finite_lse = (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            abs_lse = tl.abs(lse)
            nonzero_lse = abs_lse != 0.0
            safe_abs_lse = tl.where(finite_lse & nonzero_lse, abs_lse, 1.0)
            lse_exponent = tl.floor(tl.log2(safe_abs_lse))
            lse_exponent = tl.maximum(-126.0, tl.minimum(lse_exponent, 127.0))
            lse_exponent = tl.where(nonzero_lse, lse_exponent, 0.0)
            significand = tl.where(
                finite_lse & nonzero_lse,
                abs_lse * tl.exp2(23.0 - lse_exponent),
                0.0,
            )
            significand_hi = tl.floor(significand / 65536.0)
            significand_remainder = significand - significand_hi * 65536.0
            significand_mid = tl.floor(significand_remainder / 256.0)
            significand_lo = significand_remainder - significand_mid * 256.0
            exponent_code = lse_exponent + 128.0
            exponent_code = tl.where(lse < 0.0, -exponent_code, exponent_code)
            exponent_code = tl.where(finite_lse, exponent_code, 0.0)
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
    num_heads,
    total_rows,
    DCP_SIZE: tl.constexpr,
    SCATTER_TOKENS: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
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

        denominator = tl.where(weight_sum > 0.0, weight_sum, 1.0)
        merged /= denominator
        output_offsets = token_idx * output_stride_t + head_idx * output_stride_h + d_offsets * output_stride_d
        tl.store(output_ptr + output_offsets, merged, mask=d_mask)


def _lse_pack_dim(output_dtype: torch.dtype) -> int:
    if output_dtype in (torch.bfloat16, torch.float16):
        return 4
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
    total_rows = num_tokens * num_heads
    init_device_properties_triton()
    grid_size = min(total_rows, get_vectorcore_num())
    _pack_sfa_dcp_output_lse_kernel[(grid_size,)](
        sfa_output,
        softmax_lse,
        softmax_lse,
        send,
        *sfa_output.stride(),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        *send.stride(),
        local_scatter_size,
        head_dim,
        num_heads,
        total_rows,
        SCATTER_TOKENS=scatter_dim == 0,
        INPUT_MAX_SUM=False,
        LSE_PACK_DIM=lse_pack_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return send


def pack_sfa_dcp_output_max_sum(
    sfa_output: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    dcp_size: int,
    scatter_dim: int,
) -> torch.Tensor:
    """Pack SFA output and PA_BSND FP32 max/sum without materializing LSE."""
    expected_statistics_shape = (1, *sfa_output.shape[:2])
    if softmax_max.shape != expected_statistics_shape or softmax_sum.shape != expected_statistics_shape:
        raise RuntimeError(
            "SFA DCP fused A2A expects PA_BSND max/sum [1, tokens, heads] "
            f"matching output, got {tuple(sfa_output.shape)}, "
            f"{tuple(softmax_max.shape)}, and {tuple(softmax_sum.shape)}."
        )
    if softmax_max.dtype != torch.float32 or softmax_sum.dtype != torch.float32:
        raise TypeError(
            f"SFA DCP fused A2A requires float32 PA_BSND max/sum, got {softmax_max.dtype} and {softmax_sum.dtype}."
        )
    if sfa_output.device != softmax_max.device or sfa_output.device != softmax_sum.device:
        raise RuntimeError(
            "SFA DCP fused A2A requires output and max/sum on the same device, "
            f"got {sfa_output.device}, {softmax_max.device}, and {softmax_sum.device}."
        )

    max_lse_view = softmax_max.permute(1, 2, 0)
    num_tokens, num_heads, head_dim, local_scatter_size, replicated_size = _validate_sfa_dcp_inputs(
        sfa_output, max_lse_view, dcp_size, scatter_dim
    )
    lse_pack_dim = _lse_pack_dim(sfa_output.dtype)
    send = torch.empty(
        (dcp_size, local_scatter_size, replicated_size, head_dim + lse_pack_dim),
        dtype=sfa_output.dtype,
        device=sfa_output.device,
    )
    total_rows = num_tokens * num_heads
    init_device_properties_triton()
    grid_size = min(total_rows, get_vectorcore_num())
    _pack_sfa_dcp_output_lse_kernel[(grid_size,)](
        sfa_output,
        softmax_max,
        softmax_sum,
        send,
        *sfa_output.stride(),
        softmax_max.stride(1),
        softmax_max.stride(2),
        softmax_sum.stride(1),
        softmax_sum.stride(2),
        *send.stride(),
        local_scatter_size,
        head_dim,
        num_heads,
        total_rows,
        SCATTER_TOKENS=scatter_dim == 0,
        INPUT_MAX_SUM=True,
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
    total_rows = num_tokens * num_heads
    init_device_properties_triton()
    grid_size = min(total_rows, get_vectorcore_num())
    _fused_sfa_dcp_lse_combine_kernel[(grid_size,)](
        recv,
        output,
        *recv.stride(),
        *output.stride(),
        head_dim,
        num_heads,
        total_rows,
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


def sfa_dcp_a2a_fused_max_sum_combine(
    sfa_output: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    dcp_size: int,
    scatter_dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Pack PA_BSND max/sum, run one HCCL All2All, and combine."""
    send = pack_sfa_dcp_output_max_sum(
        sfa_output,
        softmax_max,
        softmax_sum,
        dcp_size,
        scatter_dim,
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return fused_sfa_dcp_lse_combine(recv, sfa_output.shape[-1], scatter_dim)


def sfa_dcp_a2a_fused(
    sfa_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    dcp_size: int,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    """Custom-op entry point for fused SFA DCP output post-processing."""
    group_ref = _groups.get(group_name)
    if group_ref is None:
        raise RuntimeError(f"SFA DCP fused A2A group {group_name!r} is not registered.")
    group = group_ref()
    if group is None or group.device_group is None:
        raise RuntimeError(f"SFA DCP fused A2A group {group_name!r} is unavailable.")
    if group.world_size != dcp_size:
        raise RuntimeError(
            f"SFA DCP fused A2A group size does not match dcp_size: group={group.world_size}, dcp_size={dcp_size}."
        )
    return sfa_dcp_a2a_fused_combine(
        sfa_output,
        softmax_lse,
        dcp_size,
        scatter_dim,
        group.device_group,
    )


def sfa_dcp_a2a_fused_fake(
    sfa_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    dcp_size: int,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    """Propagate output metadata for torch.compile without running HCCL.

    PyTorch invokes this FakeTensor implementation while tracing the custom
    operator. It must only describe the local output shape, dtype, and device;
    the real implementation performs the collective at execution time.
    """
    del softmax_lse, group_name
    output_shape = list(sfa_output.shape)
    output_shape[scatter_dim] //= dcp_size
    return torch.empty(output_shape, dtype=sfa_output.dtype, device=sfa_output.device)


def sfa_dcp_a2a_fused_max_sum(
    sfa_output: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    dcp_size: int,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    """Custom-op entry point consuming PA_BSND max/sum statistics."""
    group_ref = _groups.get(group_name)
    if group_ref is None:
        raise RuntimeError(f"SFA DCP fused A2A group {group_name!r} is not registered.")
    group = group_ref()
    if group is None or group.device_group is None:
        raise RuntimeError(f"SFA DCP fused A2A group {group_name!r} is unavailable.")
    if group.world_size != dcp_size:
        raise RuntimeError(
            f"SFA DCP fused A2A group size does not match dcp_size: group={group.world_size}, dcp_size={dcp_size}."
        )
    return sfa_dcp_a2a_fused_max_sum_combine(
        sfa_output,
        softmax_max,
        softmax_sum,
        dcp_size,
        scatter_dim,
        group.device_group,
    )


def sfa_dcp_a2a_fused_max_sum_fake(
    sfa_output: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    dcp_size: int,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    """Propagate local output metadata for the PA_BSND max/sum custom op."""
    del softmax_max, softmax_sum, group_name
    output_shape = list(sfa_output.shape)
    output_shape[scatter_dim] //= dcp_size
    return torch.empty(output_shape, dtype=sfa_output.dtype, device=sfa_output.device)


direct_register_custom_op(
    op_name="sfa_dcp_a2a_fused",
    op_func=sfa_dcp_a2a_fused,
    fake_impl=sfa_dcp_a2a_fused_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="sfa_dcp_a2a_fused_max_sum",
    op_func=sfa_dcp_a2a_fused_max_sum,
    fake_impl=sfa_dcp_a2a_fused_max_sum_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
