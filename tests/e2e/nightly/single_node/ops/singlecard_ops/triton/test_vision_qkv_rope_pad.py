import pytest
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.ops.triton.vision_qkv_rope_pad import vision_qkv_rope_pad

TOKEN_COUNTS = [1, 4, 41, 257, 8160]
NUM_HEADS = 8
HEAD_DIM = 72
PADDED_DIM = 128


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def reference(
    qkv_proj: torch.Tensor,
    cos_half: torch.Tensor,
    sin_half: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_count = qkv_proj.shape[0]
    qkv = qkv_proj.view(token_count, 3, NUM_HEADS, HEAD_DIM)
    q, k, v = qkv.unbind(dim=1)
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1).float()
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1).float()
    q = (q.float() * cos + rotate_half(q.float()) * sin).to(qkv_proj.dtype)
    k = (k.float() * cos + rotate_half(k.float()) * sin).to(qkv_proj.dtype)
    padding = (0, PADDED_DIM - HEAD_DIM)
    return F.pad(q, padding), F.pad(k, padding), F.pad(v, padding)


@pytest.mark.parametrize("token_count", TOKEN_COUNTS)
@torch.inference_mode()
def test_vision_qkv_rope_pad(token_count: int):
    torch.manual_seed(20260911)
    device = torch.device("npu:0")
    qkv_proj = torch.randn(
        token_count,
        3 * NUM_HEADS * HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    cos_half = torch.randn(
        token_count,
        HEAD_DIM // 2,
        dtype=torch.bfloat16,
        device=device,
    )
    sin_half = torch.randn_like(cos_half)

    actual = vision_qkv_rope_pad(qkv_proj, cos_half, sin_half)
    expected = reference(qkv_proj, cos_half, sin_half)

    for output, golden in zip(actual, expected):
        torch.testing.assert_close(output, golden, atol=1e-2, rtol=1e-2)
        assert torch.count_nonzero(output[..., HEAD_DIM:]).item() == 0


@torch.inference_mode()
def test_vision_qkv_rope_pad_fia_output():
    token_count = 257
    device = torch.device("npu:0")
    qkv_proj = torch.randn(
        token_count,
        3 * NUM_HEADS * HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    cos_half = torch.randn(
        token_count,
        HEAD_DIM // 2,
        dtype=torch.bfloat16,
        device=device,
    )
    sin_half = torch.randn_like(cos_half)
    actual = vision_qkv_rope_pad(qkv_proj, cos_half, sin_half)
    expected = reference(qkv_proj, cos_half, sin_half)

    kwargs = {
        "atten_mask": None,
        "block_table": None,
        "input_layout": "TND",
        "block_size": 128,
        "actual_seq_lengths": [token_count],
        "actual_seq_lengths_kv": [token_count],
        "num_key_value_heads": NUM_HEADS,
        "num_heads": NUM_HEADS,
        "scale": HEAD_DIM**-0.5,
        "sparse_mode": 0,
        "pre_tokens": 2147483647,
        "next_tokens": 2147483647,
    }
    fused_output, _ = torch_npu.npu_fused_infer_attention_score(
        query=actual[0],
        key=actual[1],
        value=actual[2],
        **kwargs,
    )
    reference_output, _ = torch_npu.npu_fused_infer_attention_score(
        query=expected[0],
        key=expected[1],
        value=expected[2],
        **kwargs,
    )
    torch.testing.assert_close(fused_output, reference_output, atol=1e-2, rtol=1e-2)
