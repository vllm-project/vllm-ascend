import gc

import pytest
import torch

from vllm_ascend.ops.triton.rope import rope_forward_triton

DEVICES = [f"npu:{0}"]
FP8_E4M3_MAX = 448.0
FP8_ROPE_CASES = [
    pytest.param(1, 2, 1, 128, 128, id="single-token-full-rope"),
    pytest.param(17, 8, 1, 128, 64, id="partial-rope"),
    pytest.param(1024, 8, 1, 128, 128, id="multi-row-grid"),
    # Non-power-of-two rotary_dim: triggers sin-offset fix in fp8 kernel
    # path; old code used pad_rope_dim//2=64 instead of rope_dim//2=48
    # for rotary_dim=96 (pad_rope_dim=128).
    pytest.param(17, 8, 1, 128, 96, id="non-pow2-rope"),
    # Large head_dim with non-power-of-two rotary_dim: exercises both the
    # sin-offset fix and UB-based dynamic tile sizing in fp8 path.
    pytest.param(17, 8, 1, 256, 192, id="large-head-non-pow2-rope"),
]


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
