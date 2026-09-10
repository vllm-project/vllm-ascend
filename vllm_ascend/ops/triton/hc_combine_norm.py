import torch

from vllm.triton_utils import tl, triton

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

@triton.jit
def _hc_combine_norm_kernel(
    residual_ptr,
    block_output_ptr,
    injection_logits_ptr,
    norm_weight_ptr,
    combined_ptr,
    normalized_ptr,
    residual_stride_0,
    residual_stride_1,
    block_stride_0,
    block_stride_1,
    injection_stride_0,
    injection_stride_1,
    H: tl.constexpr,
    HC_COUNT: tl.constexpr,
    EPS: tl.constexpr,
    WEIGHT_SHARED: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HC_COUNT
    branch = row % HC_COUNT

    offsets = tl.arange(0, BLOCK_H)
    mask = offsets < H

    residual_offsets = (
        token * residual_stride_0 + 
        (branch * H + offsets) * residual_stride_1
    )

    block_offsets = (token * block_stride_0 + 
                     offsets * block_stride_1)
    injection_offsets = (token * injection_stride_0 +
                         branch * injection_stride_1)

    residual = tl.load(residual_ptr + residual_offsets, 
                       mask=mask, 
                       other=0.0).to(tl.float32)
    
    block_output = tl.load(block_output_ptr + block_offsets,
                           mask=mask, 
                           other=0.0).to(tl.float32)

    injection_logits = tl.load(injection_logits_ptr + injection_offsets).to(tl.float32)

    injection = 2.0 * tl.sigmoid(injection_logits / HC_COUNT)
    combined_fp32 = residual + block_output * injection
    combined = combined_fp32.to(combined_ptr.dtype.element_ty)

    output_offsets = row * H + offsets
    tl.store(
        combined_ptr + output_offsets, 
        combined,
        mask=mask,
    )

    x = tl.where(mask, combined.to(tl.float32), 0.0)
    sum_squared = tl.sum(x*x, axis=0)
    inverse_rms = tl.rsqrt(sum_squared / H + EPS)

    if WEIGHT_SHARED:
        weight_offsets = offsets
    else:
        weight_offsets = branch * H + offsets

    weight = tl.load(
        norm_weight_ptr + weight_offsets,
        mask = mask,
        other = 0.0,
    ).to(tl.float32)

    normalized = (x * inverse_rms) * (1.0 + weight)

    tl.store(
            normalized_ptr + output_offsets,
            normalized.to(normalized_ptr.dtype.element_ty),
            mask=mask,
             )


def hc_combine_norm_fused(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Combine HC branches and apply grouped Gemma RMSNorm.

    Args:
        residual:
            Tensor shaped [N, hc_count * H].
        block_output:
            Tensor shaped [N, H].
        injection_logits:
            Tensor shaped [N, hc_count]. Noncontiguous views, such as
            outputs of split(), are supported.
        norm_weight:
            Contiguous tensor with H or hc_count * H elements.
            H elements means weights are shared across branches.
        eps:
            RMSNorm epsilon.
        hc_count:
            Positive number of HC branches.

    Returns:
        (combined, normalized), both contiguous and matching residual's
        shape, dtype, and device. 
    """
    if not isinstance(hc_count, int) or hc_count <= 0:
        raise ValueError("hc_count must be a positive integer")

    if residual.ndim != 2:
        raise ValueError(
            "residual must have shape [N, hc_count * H]"
        )
    n, total_hidden_size = residual.shape
    if total_hidden_size % hc_count != 0:
        raise ValueError(
            "residual hidden dimension must be divisible by hc_count"
        )

    hidden_size = total_hidden_size // hc_count
    if block_output.shape != (n, hidden_size):
        raise ValueError(
            f"block_output must have shape {(n, hidden_size)},"
            f"got {tuple(block_output.shape)}"
        )
    combined = torch.empty(
        (n, total_hidden_size),
        dtype=residual.dtype,
        device=residual.device
    )
    normalized = torch.empty_like(combined)

    if n == 0:
        return combined, normalized

    block_h = triton.next_power_of_2(hidden_size)

    with torch.npu.device(residual.device):
        _hc_combine_norm_kernel[(n * hc_count,)](
            residual,
            block_output,
            injection_logits,
            norm_weight,
            combined,
            normalized,
            residual.stride(0),
            residual.stride(1),
            block_output.stride(0),
            block_output.stride(1),
            injection_logits.stride(0),
            injection_logits.stride(1),
            H=hidden_size,
            HC_COUNT = hc_count,
            EPS = eps,
            WEIGHT_SHARED = norm_weight.numel() == hidden_size,
            BLOCK_H = block_h,
            enable_fp_fusion = False,
        )
    
    return combined, normalized

__all__ = ["hc_combine_norm"]
