import pytest
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.vision_qkv_rope_pad import vision_qkv_rope_pad

# The production adapter remains A2-only. The standalone operator is also
# validated on A3, while A5 and 310P have not been validated.
pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="vision_qkv_rope_pad is validated on Ascend A2/A3 only",
)

# 224x224 images yield 196 pre-merge vision tokens with patch size 16.
TOKEN_COUNTS = [1, 4, 41, 196, 257, 8160]
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

    for output, golden in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(output, golden, atol=1e-2, rtol=1e-2)
        assert torch.count_nonzero(output[..., HEAD_DIM:]).item() == 0
    torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)


MULTI_SEQUENCE_BOUNDARIES = [
    [0, 257],  # single packed sequence
    [0, 196, 392, 588],  # three 224x224 image or video temporal groups
    [0, 100, 240, 400],  # multi-frame video / multi-image batch
    [0, 50, 100, 150, 200, 257],  # many short sequences
]


def _fia(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    actual_seq_lengths: list[int],
) -> torch.Tensor:
    output, _ = torch_npu.npu_fused_infer_attention_score(
        query=q,
        key=k,
        value=v,
        atten_mask=None,
        block_table=None,
        input_layout="TND",
        block_size=128,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths,
        num_key_value_heads=NUM_HEADS,
        num_heads=NUM_HEADS,
        scale=HEAD_DIM**-0.5,
        sparse_mode=0,
        pre_tokens=2147483647,
        next_tokens=2147483647,
    )
    return output


@pytest.mark.parametrize("boundaries", MULTI_SEQUENCE_BOUNDARIES)
@torch.inference_mode()
def test_vision_qkv_rope_pad_multi_sequence_fia(boundaries: list[int]):
    """Packed multi-sequence inputs keep sequence boundaries through FIA."""
    token_count = boundaries[-1]
    torch.manual_seed(20260914)
    device = torch.device("npu:0")
    qkv_proj = torch.randn(token_count, 3 * NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device)
    cos_half = torch.randn(token_count, HEAD_DIM // 2, dtype=torch.bfloat16, device=device)
    sin_half = torch.randn_like(cos_half)

    actual = vision_qkv_rope_pad(qkv_proj, cos_half, sin_half)
    expected = reference(qkv_proj, cos_half, sin_half)

    lengths = boundaries[1:]
    fused_output = _fia(actual[0], actual[1], actual[2], lengths)
    reference_output = _fia(expected[0], expected[1], expected[2], lengths)
    torch.testing.assert_close(fused_output, reference_output, atol=1e-2, rtol=1e-2)
