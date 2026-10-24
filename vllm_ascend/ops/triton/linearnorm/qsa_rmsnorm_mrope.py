import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _qsa_rmsnorm_mrope_kernel(
    input_ptr,
    weight_ptr,
    cos_sin_ptr,
    output_ptr,
    positions_ptr,
    input_stride_token,
    input_stride_head,
    cos_sin_stride_axis,
    cos_sin_stride_token,
    output_stride_token,
    output_stride_head,
    positions_stride_axis,
    positions_stride_token,
    cache_num_positions,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    rope_dim: tl.constexpr,
    has_mrope: tl.constexpr,
    gather_positions: tl.constexpr,
    block_dim: tl.constexpr,
):
    program_id = tl.program_id(0)
    token_id = program_id // num_heads
    head_id = program_id % num_heads
    offsets = tl.arange(0, block_dim)
    mask = offsets < head_dim
    input_row = input_ptr + token_id * input_stride_token + head_id * input_stride_head
    values = tl.load(input_row + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32) + 1.0
    variance = tl.sum(values * values, axis=0) / head_dim
    reciprocal_std = tl.rsqrt(variance + eps)
    normalized = (values * reciprocal_std * weight).to(tl.bfloat16)

    half_rope_dim: tl.constexpr = rope_dim // 2
    rotary_offsets = tl.arange(0, half_rope_dim)
    if has_mrope:
        h_mask = (rotary_offsets % 3 == 1) & (rotary_offsets < mrope_section_h * 3)
        w_mask = (rotary_offsets % 3 == 2) & (rotary_offsets < mrope_section_w * 3)
        axis = tl.where(h_mask, 1, tl.where(w_mask, 2, 0))
    else:
        axis = 0
    if gather_positions:
        position = tl.load(positions_ptr + axis * positions_stride_axis + token_id * positions_stride_token).to(
            tl.int64
        )
        # Padding and unfinished compression groups can contain invalid
        # positions. Keep all addresses in bounds before loading the cache.
        position = tl.minimum(tl.maximum(position, 0), cache_num_positions - 1)
        cache_row = cos_sin_ptr + position * cos_sin_stride_token
    elif has_mrope:
        cache_row = cos_sin_ptr + axis * cos_sin_stride_axis + token_id * cos_sin_stride_token
    else:
        cache_row = cos_sin_ptr + token_id * cos_sin_stride_token

    cos = tl.load(cache_row + rotary_offsets).to(tl.float32)
    sin = tl.load(cache_row + rotary_offsets + half_rope_dim).to(tl.float32)
    first = tl.load(input_row + rotary_offsets).to(tl.float32)
    first_weight = tl.load(weight_ptr + rotary_offsets).to(tl.float32) + 1.0
    first = (first * reciprocal_std * first_weight).to(tl.bfloat16)
    second_offsets = rotary_offsets + half_rope_dim
    second = tl.load(input_row + second_offsets).to(tl.float32)
    second_weight = tl.load(weight_ptr + second_offsets).to(tl.float32) + 1.0
    second = (second * reciprocal_std * second_weight).to(tl.bfloat16)
    first_cos = (first * cos).to(tl.bfloat16)
    second_sin = (second * sin).to(tl.bfloat16)
    second_cos = (second * cos).to(tl.bfloat16)
    first_sin = (first * sin).to(tl.bfloat16)
    rotated_first = (first_cos - second_sin).to(tl.bfloat16)
    rotated_second = (second_cos + first_sin).to(tl.bfloat16)

    output_row = output_ptr + token_id * output_stride_token + head_id * output_stride_head
    tl.store(output_row + offsets, normalized, mask=mask)
    tl.store(output_row + rotary_offsets, rotated_first)
    tl.store(output_row + rotary_offsets + half_rope_dim, rotated_second)


def triton_qsa_rmsnorm_mrope(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
    mrope_section: list[int],
    rope_dim: int,
) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError("QSA norm+RoPE input must be [tokens, heads, head_dim]")
    if tensor.dtype != torch.bfloat16 or weight.dtype != tensor.dtype:
        raise ValueError("QSA norm+RoPE requires BF16 input and weight")
    if tensor.stride(-1) != 1 or weight.ndim != 1 or weight.shape[0] != tensor.shape[-1]:
        raise ValueError("QSA norm+RoPE requires contiguous head rows and matching weight")
    if cos_sin.ndim not in (2, 3) or cos_sin.shape[-1] != rope_dim:
        raise ValueError("QSA norm+RoPE received an invalid rotary cache view")
    if rope_dim <= 0 or rope_dim > tensor.shape[-1] or rope_dim % 2:
        raise ValueError("QSA norm+RoPE requires an even rotary width")
    if cos_sin.shape[-2] != tensor.shape[0]:
        raise ValueError("QSA norm+RoPE token and rotary rows must match")

    has_mrope = cos_sin.ndim == 3
    if has_mrope:
        if cos_sin.shape[0] != 3 or len(mrope_section) != 3:
            raise ValueError("QSA MRoPE requires three position axes and sections")
        cos_sin_stride_axis = cos_sin.stride(0)
    else:
        cos_sin_stride_axis = 0

    output = torch.empty_like(tensor)
    num_tokens, num_heads, head_dim = tensor.shape
    if not num_tokens:
        return output
    _qsa_rmsnorm_mrope_kernel[(num_tokens * num_heads,)](
        tensor,
        weight,
        cos_sin,
        output,
        None,
        tensor.stride(0),
        tensor.stride(1),
        cos_sin_stride_axis,
        cos_sin.stride(-2),
        output.stride(0),
        output.stride(1),
        0,
        0,
        0,
        num_heads=num_heads,
        head_dim=head_dim,
        eps=eps,
        mrope_section_h=mrope_section[1] if has_mrope else 0,
        mrope_section_w=mrope_section[2] if has_mrope else 0,
        rope_dim=rope_dim,
        has_mrope=has_mrope,
        gather_positions=False,
        block_dim=triton.next_power_of_2(head_dim),
        num_warps=1,
        num_stages=1,
    )
    return output


def triton_qsa_rmsnorm_mrope_fake(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
    mrope_section: list[int],
    rope_dim: int,
) -> torch.Tensor:
    return torch.empty_like(tensor)


direct_register_custom_op(
    op_name="triton_qsa_rmsnorm_mrope",
    op_func=triton_qsa_rmsnorm_mrope,
    fake_impl=triton_qsa_rmsnorm_mrope_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def triton_qsa_rmsnorm_mrope_from_cache(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    mrope_section: list[int],
    rope_dim: int,
) -> torch.Tensor:
    """Normalize QSA rows and gather rotary values from the persistent cache.

    Positions may be a strided [3, tokens] view, as emitted by compressed-key
    preparation. The kernel clamps them without materializing an intermediate
    position tensor or a per-token cos/sin tensor in the captured graph.
    """
    if tensor.ndim != 3 or tensor.stride(-1) != 1:
        raise ValueError("QSA norm+RoPE input must have contiguous [tokens, heads, head_dim] rows")
    if tensor.dtype != torch.bfloat16 or weight.dtype != tensor.dtype:
        raise ValueError("QSA norm+RoPE requires BF16 input and weight")
    if weight.ndim != 1 or weight.shape[0] != tensor.shape[-1] or weight.stride(0) != 1:
        raise ValueError("QSA norm+RoPE requires a contiguous matching weight")
    if rope_dim <= 0 or rope_dim > tensor.shape[-1] or rope_dim % 2:
        raise ValueError("QSA norm+RoPE requires an even rotary width")
    if rope_dim & (rope_dim - 1):
        raise ValueError("QSA norm+RoPE requires a power-of-two rotary width")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[0] == 0
        or cos_sin_cache.shape[1] != rope_dim
        or cos_sin_cache.stride(-1) != 1
        or cos_sin_cache.dtype != tensor.dtype
    ):
        raise ValueError("QSA norm+RoPE requires a nonempty matching rotary cache")
    if positions.dtype not in (torch.int32, torch.int64) or positions.ndim not in (1, 2):
        raise ValueError("QSA norm+RoPE positions must be a one- or three-axis integer tensor")
    if positions.shape[-1] != tensor.shape[0]:
        raise ValueError("QSA norm+RoPE token and position rows must match")
    has_mrope = positions.ndim == 2
    if has_mrope and (positions.shape[0] != 3 or len(mrope_section) != 3):
        raise ValueError("QSA MRoPE requires three position axes and sections")
    if any(value.device != tensor.device for value in (weight, cos_sin_cache, positions)):
        raise ValueError("QSA norm+RoPE inputs must be on the same device")

    output = torch.empty_like(tensor)
    num_tokens, num_heads, head_dim = tensor.shape
    if not num_tokens:
        return output
    _qsa_rmsnorm_mrope_kernel[(num_tokens * num_heads,)](
        tensor,
        weight,
        cos_sin_cache,
        output,
        positions,
        tensor.stride(0),
        tensor.stride(1),
        0,
        cos_sin_cache.stride(0),
        output.stride(0),
        output.stride(1),
        positions.stride(0) if has_mrope else 0,
        positions.stride(-1),
        cos_sin_cache.shape[0],
        num_heads=num_heads,
        head_dim=head_dim,
        eps=eps,
        mrope_section_h=mrope_section[1] if has_mrope else 0,
        mrope_section_w=mrope_section[2] if has_mrope else 0,
        rope_dim=rope_dim,
        has_mrope=has_mrope,
        gather_positions=True,
        block_dim=triton.next_power_of_2(head_dim),
        num_warps=1,
        num_stages=1,
    )
    return output


def triton_qsa_rmsnorm_mrope_from_cache_fake(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    mrope_section: list[int],
    rope_dim: int,
) -> torch.Tensor:
    return torch.empty_like(tensor)


direct_register_custom_op(
    op_name="triton_qsa_rmsnorm_mrope_from_cache",
    op_func=triton_qsa_rmsnorm_mrope_from_cache,
    fake_impl=triton_qsa_rmsnorm_mrope_from_cache_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
