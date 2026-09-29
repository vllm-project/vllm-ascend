import pytest
import torch
import torch_npu

from vllm_ascend.ops.rotary_embedding import rope_forward_oot


def _reference_rope(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    is_neox_style: bool,
) -> torch.Tensor:
    head_dim = tensor.shape[-1]
    rotated = tensor[..., :rotary_dim].float()
    if is_neox_style:
        cos = cos.repeat(1, 2).unsqueeze(1).float()
        sin = sin.repeat(1, 2).unsqueeze(1).float()
        first, second = rotated.chunk(2, dim=-1)
        shifted = torch.cat((-second, first), dim=-1)
    else:
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1).float()
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1).float()
        first, second = rotated[..., ::2], rotated[..., 1::2]
        shifted = torch.stack((-second, first), dim=-1).flatten(-2)
    result = (rotated * cos + shifted * sin).to(tensor.dtype)
    if rotary_dim < head_dim:
        result = torch.cat((result, tensor[..., rotary_dim:]), dim=-1)
    return result


@pytest.mark.parametrize("head_dim,rotary_dim", [(64, 32), (128, 96), (128, 128)])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("num_tokens", [1, 17])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_asc_rope_matches_reference(head_dim, rotary_dim, is_neox_style, num_tokens, dtype):
    device = "npu:0"
    torch.manual_seed(0)
    positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    freqs = torch.outer(positions.float(), inv_freq)
    cos, sin = freqs.cos().to(dtype), freqs.sin().to(dtype)
    cos_sin_cache = torch.cat((cos, sin), dim=-1)
    query = torch.randn(num_tokens, 8, head_dim, device=device, dtype=dtype)
    key = torch.randn(num_tokens, 2, head_dim, device=device, dtype=dtype)

    actual_q, actual_k = rope_forward_oot(
        positions,
        query.flatten(1),
        key.flatten(1),
        cos_sin_cache,
        head_dim,
        rotary_dim,
        is_neox_style,
    )
    expected_q = _reference_rope(query, cos, sin, rotary_dim, is_neox_style)
    expected_k = _reference_rope(key, cos, sin, rotary_dim, is_neox_style)
    tolerance = 1e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(actual_q.view_as(expected_q), expected_q, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_k.view_as(expected_k), expected_k, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("is_neox_style,rotary_mode", [(True, "half"), (False, "interleave")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_asc_indexer_rotary_mul_matches_reference(is_neox_style, rotary_mode, dtype):
    """SFA's partial Q/K RoPE must preserve both pair layouts."""
    num_tokens, head_dim, rotary_dim = 17, 128, 64
    torch.manual_seed(0)
    x = torch.randn(num_tokens, 8, head_dim, device="npu:0", dtype=dtype)
    angles = torch.randn(num_tokens, rotary_dim // 2, device="npu:0")
    cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
    if is_neox_style:
        expanded_cos = cos.repeat(1, 2)
        expanded_sin = sin.repeat(1, 2)
    else:
        expanded_cos = cos.repeat_interleave(2, dim=-1)
        expanded_sin = sin.repeat_interleave(2, dim=-1)

    rotated = torch_npu.npu_rotary_mul(
        x[..., :rotary_dim].unsqueeze(2),
        expanded_cos.view(num_tokens, 1, 1, rotary_dim),
        expanded_sin.view(num_tokens, 1, 1, rotary_dim),
        rotary_mode=rotary_mode,
    ).squeeze(2)
    actual = torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
    expected = _reference_rope(x, cos, sin, rotary_dim, is_neox_style)
    tolerance = 1e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
