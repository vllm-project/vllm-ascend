from types import SimpleNamespace

import pytest
import torch
import torch_npu
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.qwen4_exp.qsa import AscendQSAIndexer
from vllm_ascend.ops.triton.linearnorm.qsa_rmsnorm_mrope import (
    triton_qsa_rmsnorm_mrope,
)

pytestmark = [
    pytest.mark.skipif(
        get_ascend_device_type() not in (AscendDeviceType.A3, AscendDeviceType.A5),
        reason="Requires Ascend A3/A5",
    ),
]


def _reference(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    values = tensor.float()
    reciprocal_std = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    normalized = (values * reciprocal_std * (weight.float() + 1.0)).to(torch.bfloat16)
    rotary_offsets = torch.arange(32)
    h_mask = (rotary_offsets % 3 == 1) & (rotary_offsets < 33)
    w_mask = (rotary_offsets % 3 == 2) & (rotary_offsets < 30)
    axis = torch.where(h_mask, 1, torch.where(w_mask, 2, 0))
    token = torch.arange(tensor.shape[0])[:, None]
    offset = rotary_offsets[None, :]
    cos = cos_sin[axis[None, :], token, offset].float()
    sin = cos_sin[axis[None, :], token, offset + 32].float()
    first = normalized[..., :32]
    second = normalized[..., 32:64]
    output = normalized.clone()
    output[..., :32] = (
        (first.float() * cos[:, None, :]).to(torch.bfloat16) - (second.float() * sin[:, None, :]).to(torch.bfloat16)
    ).to(torch.bfloat16)
    output[..., 32:64] = (
        (second.float() * cos[:, None, :]).to(torch.bfloat16) + (first.float() * sin[:, None, :]).to(torch.bfloat16)
    ).to(torch.bfloat16)
    return output


@pytest.mark.parametrize("num_heads", [4, 1, 3, 8], ids=["prefill", "decode", "padded-heads", "multiple-head-tiles"])
@torch.inference_mode()
def test_qsa_rmsnorm_mrope(num_heads: int) -> None:
    torch.manual_seed(17)
    num_tokens, head_dim, rope_dim = 4096, 128, 64
    tensor_cpu = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16)
    weight_cpu = torch.randn(head_dim, dtype=torch.bfloat16) * 0.05
    cos_sin_cpu = torch.randn(3, num_tokens, rope_dim, dtype=torch.bfloat16)
    expected = _reference(tensor_cpu, weight_cpu, cos_sin_cpu, 1e-6)

    actual = triton_qsa_rmsnorm_mrope(
        tensor_cpu.npu(),
        weight_cpu.npu(),
        cos_sin_cpu.npu(),
        1e-6,
        [11, 11, 10],
        rope_dim,
    )
    torch_npu.npu.synchronize()

    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("strided_weight", [False, True])
@pytest.mark.parametrize("rows", [0, 1, 32, 129])
@pytest.mark.parametrize("axes", [1, 3])
@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Requires Ascend A3")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available")
@torch.inference_mode()
def test_qsa_rmsnorm_mrope_with_positions(rows: int, axes: int, strided_weight: bool) -> None:
    torch.manual_seed(17)
    cache = torch.randn(129, 64, dtype=torch.bfloat16, device="npu")
    # Model-runner buffers can have much larger axis strides than active rows.
    storage = torch.randint(-4, 140, (3, 2 * rows + 17), device="npu", dtype=torch.int64)
    positions = storage[:, : 2 * rows : 2] if axes == 3 else storage[0, : 2 * rows : 2]
    gathered = cache[positions.clamp(0, 128)]
    packed = torch.randn(rows, 5, 128, dtype=torch.bfloat16, device="npu")
    query = packed[:, :4]  # Preserve the projection's non-contiguous token stride.
    weight_storage = torch.randn(257, dtype=torch.bfloat16, device="npu")
    weight = weight_storage[1::2] if strided_weight else weight_storage[1:129]
    expected = triton_qsa_rmsnorm_mrope(query, weight.contiguous(), gathered, 1e-6, [11, 11, 10], 64)
    actual = triton_qsa_rmsnorm_mrope(query, weight, cache, 1e-6, [11, 11, 10], 64, positions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Requires Ascend A3")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available")
@torch.inference_mode()
def test_qsa_rmsnorm_mrope_positions_replay() -> None:
    rows = 32
    query = torch.randn(rows, 4, 128, dtype=torch.bfloat16, device="npu")
    weight = torch.randn(128, dtype=torch.bfloat16, device="npu")
    cache = torch.randn(129, 64, dtype=torch.bfloat16, device="npu")
    positions = torch.zeros(3, rows + 7, dtype=torch.int64, device="npu")[:, :rows]

    def run():
        return torch.ops.vllm.triton_qsa_rmsnorm_mrope(query, weight, cache, 1e-6, [11, 11, 10], 64, positions)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    for offset in [-3, 37, 135]:
        positions.copy_(torch.arange(3 * rows, device="npu").reshape(3, rows) + offset)
        query.add_(0.125)
        graph.replay()
        torch.testing.assert_close(captured, run(), rtol=0, atol=0)


@pytest.mark.parametrize("with_position_ring", [False, True])
@torch.inference_mode()
def test_indexer_strided_weight_fallback_stores_reference_cache(with_position_ring):
    torch.manual_seed(42)
    rows, width = 8, 128
    layer = AscendQSAIndexer.__new__(AscendQSAIndexer)
    torch.nn.Module.__init__(layer)
    layer.index_head_dim, layer.compress_ratio = width, 4
    weight_storage = torch.randn(2 * width + 1, dtype=torch.bfloat16, device="npu")
    weight = weight_storage[1::2]
    layer.k_layernorm = SimpleNamespace(weight=weight, variance_epsilon=1e-6)
    rotary = torch.randn(32, 64, dtype=torch.bfloat16, device="npu")
    layer.rotary_emb = SimpleNamespace(
        rotary_dim=64,
        is_neox_style=True,
        mrope_interleaved=True,
        mrope_section=[11, 11, 10],
        _match_cos_sin_cache_dtype=lambda _: rotary,
    )
    raw = torch.randn(rows, 1, width, dtype=torch.bfloat16, device="npu")
    logical = torch.arange(rows, dtype=torch.int64, device="npu")
    positions = torch.stack((logical, logical + 1, logical + 2))
    ring = torch.zeros(1, 4, 1, width, dtype=torch.bfloat16, device="npu")
    position_ring = torch.zeros(1, 4, 1, 3, dtype=torch.int64, device="npu") if with_position_ring else None
    layer.raw_key_cache = SimpleNamespace(key_cache=ring, rope_position_cache=position_ring)
    cache = torch.full((1, 4, 1, width), 17, dtype=torch.bfloat16, device="npu")
    layer.compressed_key_cache = SimpleNamespace(kv_cache=cache)
    metadata = SimpleNamespace(
        num_actual_tokens=rows,
        logical_positions=logical,
        block_table=torch.zeros(1, 1, dtype=torch.int32, device="npu"),
        token_to_req=torch.zeros(rows, dtype=torch.int32, device="npu"),
        query_start_loc=torch.tensor([0, rows], dtype=torch.int32, device="npu"),
        slot_mapping=logical % 4,
    )
    compressed = SimpleNamespace(slot_mapping=torch.tensor([-1, -1, -1, 0, -1, -1, -1, 1], device="npu"))
    pooled = raw.float().reshape(2, 4, width).mean(dim=1).to(torch.bfloat16).unsqueeze(1)
    first = positions[:, ::4] if with_position_ring else logical[::4].expand(3, -1)
    expected = _reference(pooled.cpu(), weight.cpu(), rotary[first].cpu(), 1e-6)

    # A strided weight bypasses fused compression and must remain usable by
    # the separate normalization kernel, all the way through cache publication.
    layer._update_and_compress(raw, positions, metadata, compressed)
    torch.testing.assert_close(cache[0, :2].cpu(), expected, rtol=2e-2, atol=2e-2)
    assert torch.all(cache[0, 2:] == 17)

    # The same logical weights through the contiguous fused-compression branch
    # must publish exactly the same BF16 cache, including untouched slots.
    strided_result = cache.clone()
    cache.fill_(17)
    ring.zero_()
    if position_ring is not None:
        position_ring.zero_()
    layer.k_layernorm.weight = weight.contiguous()
    layer._update_and_compress(raw, positions, metadata, compressed)
    torch.testing.assert_close(cache, strided_result, rtol=0, atol=0)
