import gc

import pytest
import torch

from vllm_ascend.ops.triton.rope import (
    rope_forward_triton,
    rope_forward_triton_siso,
)

IS_NEOX_STYLE = [True, False]
DTYPES = [torch.bfloat16, torch.float16]
MAX_POSITION_EMBEDDINGS = [262144]

# parameters for test_rotary_embedding_triton_kernel only
# (head_size, rotary_dim)
HEAD_ROTARY_DIMS = [
    (64, 32),
    (128, 128),
]
# (num_q_heads, num_k_heads)
NUM_QK_HEADS = [
    (64, 1),
    (96, 8),
]

# parameters for test_rotary_embedding_triton_kernel_siso only
SISO_HEAD_SIZES = [64, 128]
SISO_ROTARY_DIMS = [32, 64]
SISO_NUM_HEADS = [64]

NUM_TOKENS = [1, 4, 8, 16, 1024]
SEEDS = [0]
DEVICES = [f"npu:{0}"]
DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3
FP8_E4M3_MAX = 448.0
FP8_ROPE_CASES = [
    pytest.param(1, 2, 1, 128, 128, id="single-token-full-rope"),
    pytest.param(17, 8, 1, 128, 64, id="partial-rope"),
    pytest.param(1024, 8, 1, 128, 128, id="multi-row-grid"),
]


def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _rope_pytorch_native(query, key, cos, sin, rope_dim, is_neox_style) -> tuple[torch.Tensor, torch.Tensor | None]:
    """PyTorch-native implementation equivalent to forward()."""
    assert key is not None
    orig_dtype = query.dtype
    query_rot = query[..., :rope_dim].to(torch.float32)
    key_rot = key[..., :rope_dim].to(torch.float32)
    head_size = query.shape[-1]
    if rope_dim < head_size:
        query_pass = query[..., rope_dim:]
        key_pass = key[..., rope_dim:]

    if is_neox_style:
        cos = cos.repeat(1, 2).unsqueeze(-2).to(torch.float32)
        sin = sin.repeat(1, 2).unsqueeze(-2).to(torch.float32)
    else:
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2).to(torch.float32)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2).to(torch.float32)

    rotate_fn = rotate_neox if is_neox_style else rotate_gptj
    query_rot = query_rot * cos + rotate_fn(query_rot) * sin
    key_rot = key_rot * cos + rotate_fn(key_rot) * sin

    if rope_dim < head_size:
        query = torch.cat((query_rot.to(orig_dtype), query_pass), dim=-1)
        key = torch.cat((key_rot.to(orig_dtype), key_pass), dim=-1)
    else:
        query = query_rot.to(orig_dtype)
        key = key_rot.to(orig_dtype)
    return query, key


def _rope_siso_pytorch_native(query, cos, sin, rope_dim, is_neox_style) -> tuple[torch.Tensor, torch.Tensor | None]:
    """PyTorch-native implementation equivalent to forward()."""
    assert query is not None
    orig_dtype = query.dtype
    query_rot = query[..., :rope_dim].to(torch.float32)
    head_size = query.shape[-1]
    if rope_dim < head_size:
        query_pass = query[..., rope_dim:]

    if is_neox_style:
        cos = cos.repeat(1, 2).unsqueeze(-2).to(torch.float32)
        sin = sin.repeat(1, 2).unsqueeze(-2).to(torch.float32)
    else:
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2).to(torch.float32)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2).to(torch.float32)

    rotate_fn = rotate_neox if is_neox_style else rotate_gptj
    query_rot = query_rot * cos + rotate_fn(query_rot) * sin

    if rope_dim < head_size:
        query = torch.cat((query_rot.to(orig_dtype), query_pass), dim=-1)
    else:
        query = query_rot.to(orig_dtype)
    return query


def _rope_fp8_pytorch_native(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for NeoX RoPE with a direct E4M3 store."""
    half = rope_dim // 2
    cos_sin = cos_sin_cache.index_select(0, positions).to(torch.float32)
    cos = cos_sin[:, :half].unsqueeze(-2)
    sin = cos_sin[:, half:rope_dim].unsqueeze(-2)

    def apply_rope(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(torch.float32)
        first = tensor[..., :half]
        second = tensor[..., half:rope_dim]
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin),
            dim=-1,
        )
        if rope_dim < tensor.shape[-1]:
            rotated = torch.cat((rotated, tensor[..., rope_dim:]), dim=-1)
        return rotated.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)

    return apply_rope(query), apply_rope(key)


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads,num_k_heads", NUM_QK_HEADS)
@pytest.mark.parametrize("head_size,rotary_dim", HEAD_ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_rotary_embedding_triton_kernel(
    is_neox_style: bool,
    num_tokens: int,
    num_q_heads: int,
    num_k_heads: int,
    head_size: int,
    rotary_dim: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device(device)
    sin = torch.randn(num_tokens, rotary_dim // 2, dtype=dtype, device=device)
    cos = torch.randn(num_tokens, rotary_dim // 2, dtype=dtype, device=device)
    q_trt = torch.randn(num_tokens, num_q_heads, head_size, dtype=dtype, device=device)
    k_trt = torch.randn(num_tokens, num_k_heads, head_size, dtype=dtype, device=device)
    q_gold = torch.randn(num_tokens, num_q_heads, head_size, dtype=dtype, device=device)
    k_gold = torch.randn(num_tokens, num_k_heads, head_size, dtype=dtype, device=device)
    q_trt.copy_(q_gold)
    k_trt.copy_(k_gold)
    q_trt, k_trt = rope_forward_triton(q_trt, k_trt, cos, sin, rope_dim=rotary_dim, is_neox_style=is_neox_style)
    q_gold, k_gold = _rope_pytorch_native(q_gold, k_gold, cos, sin, rope_dim=rotary_dim, is_neox_style=is_neox_style)
    # Compare the results.
    torch.testing.assert_close(q_trt.view(q_gold.size()), q_gold, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    torch.testing.assert_close(k_trt.view(k_gold.size()), k_gold, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("max_position_embeddings", MAX_POSITION_EMBEDDINGS)
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads,num_k_heads", NUM_QK_HEADS)
@pytest.mark.parametrize("head_size,rotary_dim", HEAD_ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_rotary_embedding_triton_kernel_with_cos_sin_cache(
    max_position_embeddings: int,
    is_neox_style: bool,
    num_tokens: int,
    num_q_heads: int,
    num_k_heads: int,
    head_size: int,
    rotary_dim: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device(device)
    cos_sin_cache = torch.randn(max_position_embeddings, rotary_dim, dtype=dtype, device=device)
    positions = torch.randint(low=0, high=max_position_embeddings, size=(num_tokens,), dtype=torch.int64, device=device)
    q_trt = torch.randn(num_tokens, num_q_heads, head_size, dtype=dtype, device=device)
    k_trt = torch.randn(num_tokens, num_k_heads, head_size, dtype=dtype, device=device)
    q_gold = torch.randn(num_tokens, num_q_heads, head_size, dtype=dtype, device=device)
    k_gold = torch.randn(num_tokens, num_k_heads, head_size, dtype=dtype, device=device)
    q_trt.copy_(q_gold)
    k_trt.copy_(k_gold)
    q_trt, k_trt = rope_forward_triton(
        q_trt, k_trt, cos_sin_cache=cos_sin_cache, positions=positions, rope_dim=rotary_dim, is_neox_style=is_neox_style
    )
    cos, sin = cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)
    q_gold, k_gold = _rope_pytorch_native(q_gold, k_gold, cos, sin, rope_dim=rotary_dim, is_neox_style=is_neox_style)
    # Compare the results.
    torch.testing.assert_close(q_trt.view(q_gold.size()), q_gold, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    torch.testing.assert_close(k_trt.view(k_gold.size()), k_gold, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize(
    "num_tokens,num_q_heads,num_k_heads,head_size,rotary_dim",
    FP8_ROPE_CASES,
)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_rotary_embedding_triton_kernel_fp8(
    num_tokens: int,
    num_q_heads: int,
    num_k_heads: int,
    head_size: int,
    rotary_dim: int,
    device: str,
) -> None:
    torch.manual_seed(0)
    torch.set_default_device(device)

    max_positions = max(num_tokens, 32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    freqs = torch.outer(
        torch.arange(max_positions, dtype=torch.float32, device=device),
        inv_freq,
    )
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)

    query = torch.randn(num_tokens, num_q_heads, head_size, dtype=torch.bfloat16, device=device)
    key = torch.randn(num_tokens, num_k_heads, head_size, dtype=torch.bfloat16, device=device)
    # Position zero has cos=1 and sin=0, so these values exercise the
    # fixed-scale E4M3 clipping path without changing the expected sign.
    query[0, 0, 0] = FP8_E4M3_MAX * 2
    key[0, 0, 0] = -FP8_E4M3_MAX * 2

    expected_query, expected_key = _rope_fp8_pytorch_native(
        query,
        key,
        cos_sin_cache,
        positions,
        rotary_dim,
    )
    actual_query, actual_key = rope_forward_triton(
        query,
        key,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        rope_dim=rotary_dim,
        out_dtype=torch.float8_e4m3fn,
    )

    assert actual_query.dtype == torch.float8_e4m3fn
    assert actual_key.dtype == torch.float8_e4m3fn
    assert actual_query.shape == query.shape
    assert actual_key.shape == key.shape
    assert actual_query[0, 0, 0].to(torch.float32) == FP8_E4M3_MAX
    assert actual_key[0, 0, 0].to(torch.float32) == -FP8_E4M3_MAX
    torch.testing.assert_close(
        actual_query.to(torch.float32),
        expected_query.to(torch.float32),
        atol=0.125,
        rtol=0.125,
    )
    torch.testing.assert_close(
        actual_key.to(torch.float32),
        expected_key.to(torch.float32),
        atol=0.125,
        rtol=0.125,
    )
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads", SISO_NUM_HEADS)
@pytest.mark.parametrize("head_size", SISO_HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", SISO_ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_rotary_embedding_triton_kernel_siso(
    is_neox_style: bool,
    num_tokens: int,
    num_q_heads: int,
    head_size: int,
    rotary_dim: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device(device)

    if rotary_dim == -1:
        rotary_dim = head_size
    sin = torch.randn(num_tokens, rotary_dim // 2, dtype=dtype, device=device)
    cos = torch.randn(num_tokens, rotary_dim // 2, dtype=dtype, device=device)
    q_trt = torch.randn(num_tokens, num_q_heads, head_size, dtype=dtype, device=device)
    q_gold = torch.randn(num_tokens, num_q_heads, head_size, dtype=dtype, device=device)
    q_trt.copy_(q_gold)
    q_trt = rope_forward_triton_siso(q_trt, cos, sin, rope_dim=rotary_dim, is_neox_style=is_neox_style)
    q_gold = _rope_siso_pytorch_native(q_gold, cos, sin, rope_dim=rotary_dim, is_neox_style=is_neox_style)
    # Compare the results.
    torch.testing.assert_close(q_trt.view(q_gold.size()), q_gold, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
