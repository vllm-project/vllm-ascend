"""OSCAR INT2 Quantization for vllm-ascend.

Dynamic per-token-head affine INT2 quantization with clipping.

Quantization flow:
    1. Rotate: K_rot = K @ R_k (makes distribution more Gaussian)
    2. Clip: remove outliers via per-row quantile threshold
    3. Quantize: affine INT2 [0,3] with per-token-head scale+zero
    4. Pack: 4 values per uint8 byte

Dequantization:
    1. Unpack uint8 -> 4 x uint8 values [0,3]
    2. Dequantize: x = (q - zero) * scale
    3. Result is in rotated space (attention uses Q @ R_k too)
"""

import torch

# OSCAR clip ratios (from previous agent's tuning)
DEFAULT_K_CLIP_RATIO = 0.96
DEFAULT_V_CLIP_RATIO = 0.92


def oscar_quantize_int2(
    x: torch.Tensor,
    clip_ratio: float = 0.96,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize BF16 tensor to packed INT2 with dynamic per-token-head scales.

    Args:
        x: [num_tokens, num_heads, head_dim] BF16 tensor (already rotated)
        clip_ratio: fraction of values to keep (outliers clipped)

    Returns:
        packed: [num_tokens, num_heads, head_dim // 4] uint8
        scales: [num_tokens, num_heads] float32
        zeros:  [num_tokens, num_heads] float32
    """
    num_tokens, num_heads, head_dim = x.shape
    assert head_dim % 4 == 0, f"head_dim {head_dim} must be divisible by 4"

    # Step 1: Clip outliers per token-head
    clip_idx = min(int(clip_ratio * head_dim), head_dim)
    assert clip_idx >= 1, (
        f"clip_ratio={clip_ratio} yields clip_idx=0 for head_dim={head_dim}; "
        "need clip_ratio >= 1/head_dim so at least one value survives clipping"
    )
    abs_x = x.abs()
    # k-th smallest magnitude = clip threshold (same element a full sort
    # would select at position clip_idx - 1, without the O(N log N) sort)
    threshold, _ = torch.kthvalue(abs_x, clip_idx, dim=-1)
    threshold = threshold.unsqueeze(-1)  # [T, H, 1]
    x_clipped = torch.clamp(x, -threshold, threshold)

    # Step 2: Compute per-token-head affine scale and zero
    x_min = x_clipped.amin(dim=-1, keepdim=True)  # [T, H, 1]
    x_max = x_clipped.amax(dim=-1, keepdim=True)  # [T, H, 1]

    scale = (x_max - x_min).clamp(min=1e-8) / 3.0  # 4 levels -> 3 intervals
    zero = -x_min / scale  # affine offset: x_min maps to q=0

    # Step 3: Quantize to [0, 3]
    q = torch.clamp(torch.round(x_clipped / scale + zero), 0, 3).to(torch.uint8)

    # Step 4: Pack 4 values per byte
    # Layout: q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)
    quarter = head_dim // 4
    q = q.reshape(num_tokens, num_heads, 4, quarter)
    packed = q[:, :, 0, :] | (q[:, :, 1, :] << 2) | (q[:, :, 2, :] << 4) | (q[:, :, 3, :] << 6)

    return packed, scale.squeeze(-1), zero.squeeze(-1)


def oscar_dequantize_int2(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize packed INT2 to BF16.

    Args:
        packed: [num_tokens, num_heads, head_dim // 4] uint8
        scales: [num_tokens, num_heads] float32
        zeros:  [num_tokens, num_heads] float32

    Returns:
        x: [num_tokens, num_heads, head_dim] in target_dtype
    """
    num_tokens, num_heads, packed_dim = packed.shape
    head_dim = packed_dim * 4

    # Unpack 4 values per byte
    p = packed.to(torch.uint8)
    q0 = p & 0x03
    q1 = (p >> 2) & 0x03
    q2 = (p >> 4) & 0x03
    q3 = (p >> 6) & 0x03

    # Interleave back to original order
    q = torch.stack([q0, q1, q2, q3], dim=-2)  # [T, H, 4, packed_dim]
    q = q.reshape(num_tokens, num_heads, head_dim).to(torch.float32)

    # Dequantize: x = (q - zero) * scale
    x = (q - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)

    return x.to(target_dtype)
