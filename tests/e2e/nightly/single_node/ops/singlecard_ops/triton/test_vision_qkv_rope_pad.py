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


MULTI_SEQUENCE_BOUNDARIES = [
    [0, 257],  # single packed sequence
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


LAUNCH_MODE_MATRIX = [
    {"core_mode": "dynamic", "block_t": "4", "store_mode": "block", "alloc_mode": "fused"},
    {"core_mode": "dynamic", "block_t": "4", "store_mode": "block", "alloc_mode": "split"},
    {"core_mode": "dynamic", "block_t": "4", "store_mode": "masked", "alloc_mode": "fused"},
    {"core_mode": "dynamic", "block_t": "4", "store_mode": "single", "alloc_mode": "fused"},
    {"core_mode": "fixed", "block_t": "4", "store_mode": "block", "alloc_mode": "fused"},
    {"core_mode": "bucket", "block_t": "4", "store_mode": "block", "alloc_mode": "fused"},
    {"core_mode": "dynamic", "block_t": "1", "store_mode": "block", "alloc_mode": "fused"},
    {"core_mode": "dynamic", "block_t": "8", "store_mode": "masked", "alloc_mode": "fused"},
    {"core_mode": "dynamic", "block_t": "16", "store_mode": "single", "alloc_mode": "fused"},
    {"core_mode": "dynamic", "block_t": "2", "store_mode": "block", "alloc_mode": "fused"},
    {"core_mode": "dynamic", "block_t": "8", "store_mode": "block", "alloc_mode": "fused"},
]


@pytest.mark.parametrize("mode", LAUNCH_MODE_MATRIX)
@pytest.mark.parametrize("token_count", [1, 41, 257])
@torch.inference_mode()
def test_vision_qkv_rope_pad_launch_modes(mode: dict, token_count: int):
    """Every launch option combination matches the reference producer."""
    from vllm_ascend import envs

    torch.manual_seed(20260914)
    device = torch.device("npu:0")
    qkv_proj = torch.randn(token_count, 3 * NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device)
    cos_half = torch.randn(token_count, HEAD_DIM // 2, dtype=torch.bfloat16, device=device)
    sin_half = torch.randn_like(cos_half)
    expected = reference(qkv_proj, cos_half, sin_half)

    saved = {}
    for key, value in (
        ("VLLM_ASCEND_VIT_ROPE_PAD_CORE_MODE", mode["core_mode"]),
        ("VLLM_ASCEND_VIT_ROPE_PAD_BLOCK_T", mode["block_t"]),
        ("VLLM_ASCEND_VIT_ROPE_PAD_STORE_MODE", mode["store_mode"]),
        ("VLLM_ASCEND_VIT_ROPE_PAD_ALLOC_MODE", mode["alloc_mode"]),
    ):
        saved[key] = getattr(envs, key, None)
        setattr(envs, key, value)

    try:
        actual = vision_qkv_rope_pad(qkv_proj, cos_half, sin_half)
    finally:
        for key, value in saved.items():
            setattr(envs, key, value)

    for output, golden in zip(actual, expected):
        torch.testing.assert_close(output, golden, atol=1e-2, rtol=1e-2)
        assert torch.count_nonzero(output[..., HEAD_DIM:]).item() == 0


@pytest.mark.parametrize(
    "boundaries",
    [[0, 100, 240, 400], [0, 50, 100, 150, 200, 257]],
)
@pytest.mark.parametrize("store_mode", ["block", "masked", "single"])
@torch.inference_mode()
def test_vision_qkv_rope_pad_multi_sequence_launch_modes(boundaries: list[int], store_mode: str):
    """Multi-sequence FIA equivalence holds for every store layout."""
    from vllm_ascend import envs

    token_count = boundaries[-1]
    torch.manual_seed(20260914)
    device = torch.device("npu:0")
    qkv_proj = torch.randn(token_count, 3 * NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device)
    cos_half = torch.randn(token_count, HEAD_DIM // 2, dtype=torch.bfloat16, device=device)
    sin_half = torch.randn_like(cos_half)
    expected = reference(qkv_proj, cos_half, sin_half)

    saved = getattr(envs, "VLLM_ASCEND_VIT_ROPE_PAD_STORE_MODE", None)
    envs.VLLM_ASCEND_VIT_ROPE_PAD_STORE_MODE = store_mode
    try:
        actual = vision_qkv_rope_pad(qkv_proj, cos_half, sin_half)
    finally:
        envs.VLLM_ASCEND_VIT_ROPE_PAD_STORE_MODE = saved

    lengths = boundaries[1:]
    fused_output = _fia(actual[0], actual[1], actual[2], lengths)
    reference_output = _fia(expected[0], expected[1], expected[2], lengths)
    torch.testing.assert_close(fused_output, reference_output, atol=1e-2, rtol=1e-2)
