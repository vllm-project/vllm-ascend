# SPDX-License-Identifier: Apache-2.0
"""A5-only A8C4 producer accuracy against an independent CPU reference."""

import pytest
import torch

from vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache import minimax_qknorm_rope_cache
from vllm_ascend.ops.triton.msa.msa_m3_a8c4_cache import a8c4_cache_views, dual_cache_scatter
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

pytestmark = pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="A8C4 requires Ascend 950")


def reference_pack(values):
    grouped = values.float().reshape(*values.shape[:-1], 8, 16)
    amax = grouped.abs().amax(-1)
    scale = torch.where(amax > 0, amax / 6, 1).bfloat16()
    normalized = (grouped / scale.float()[..., None]).abs().clamp_max(6)
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    distance = (normalized[..., None] - levels).abs()
    # Exact halfway values choose the even code, independently of threshold sums.
    candidates = distance == distance.amin(-1, keepdim=True)
    priority = torch.tensor([0, 9, 2, 11, 4, 13, 6, 15])
    codes = torch.where(candidates, priority, 99).argmin(-1)
    codes |= (grouped < 0).long() << 3
    codes = codes.reshape(*values.shape[:-1], 64, 2)
    packed = (codes[..., 0] | (codes[..., 1] << 4)).byte()
    return packed, scale


def test_gbsa_mode8_matches_same_quantization_math():
    from importlib import import_module

    bridge = import_module(
        "cann_ops_transformer.ops.attention.generic_block_sparse_attention.generic_block_sparse_attention"
    )
    if not hasattr(bridge.QuantMode, "Q_FP8_PER_TENSOR_KV_FP4_PER_TOKEN_PER_GROUP"):
        pytest.skip("Requires CANN PR16153 mode-8 bridge and operators")
    from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import minimax_m3_sparse_attn_decode

    generator = torch.Generator().manual_seed(16155)
    key = torch.randn((128, 1, 128), dtype=torch.bfloat16, generator=generator)
    value = torch.randn((128, 1, 128), dtype=torch.bfloat16, generator=generator)
    pages = torch.zeros((2, 1, 128, 1, 256), dtype=torch.uint8, device="npu").view(torch.float8_e4m3fn)
    dual_cache_scatter(key.npu(), value.npu(), pages, torch.arange(128, dtype=torch.int64, device="npu"))
    views = a8c4_cache_views(pages)
    query = (torch.randn((8, 16, 128), generator=generator) * 0.2).to(torch.float8_e4m3fn)
    output = torch.empty((8, 16, 128), dtype=torch.bfloat16, device="npu")
    minimax_m3_sparse_attn_decode(
        query.npu(),
        (views[0][1], views[1][1]),
        torch.zeros((1, 8, 1), dtype=torch.int32, device="npu"),
        torch.zeros((1, 1), dtype=torch.int32, device="npu"),
        torch.tensor([128], dtype=torch.int32, device="npu"),
        1,
        128**-0.5,
        output,
        8,
        select_num_idx=torch.ones((1, 8), dtype=torch.int32, device="npu"),
        dequant_scale=(views[0][2], views[1][2]),
        indices_are_sorted=True,
    )

    def restore(values):
        packed, scales = reference_pack(values)
        codes = torch.stack((packed & 15, packed >> 4), -1).reshape(128, 1, 8, 16)
        levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        restored = levels[(codes & 7).long()] * torch.where((codes & 8) > 0, -1.0, 1.0)
        restored *= scales.float()[..., None]
        return restored.reshape(128, 128).bfloat16().float().to(torch.float8_e4m3fn).float()

    scores = torch.matmul(query.float(), restore(key).T) * 128**-0.5
    mask = torch.arange(128)[None, :] <= torch.arange(120, 128)[:, None]
    scores.masked_fill_(~mask[:, None, :], float("-inf"))
    exponent = (scores - scores.amax(-1, keepdim=True)).exp()
    expected = torch.matmul(exponent.to(torch.float8_e4m3fn).float(), restore(value))
    expected /= exponent.sum(-1, keepdim=True)
    actual = output.cpu().float()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2**-6, atol=2**-6)
    assert (actual - expected).abs().max() <= 1


@pytest.mark.parametrize("tokens,heads", [(1, 1), (8, 1), (224, 1), (8, 4)])
def test_scatter_fp4_and_scales(tokens, heads):
    generator = torch.Generator().manual_seed(16153)
    key = torch.randn((tokens, heads, 128), dtype=torch.bfloat16, generator=generator)
    value = torch.randn((tokens, heads, 128), dtype=torch.bfloat16, generator=generator)
    key[0].zero_()  # Zero groups use scale 1, not NaN/zero scale.
    slots = torch.arange(tokens, dtype=torch.int64) + 127
    slots[-1] = -1  # Padding must leave the destination untouched.
    pages = torch.full((2, 4, 128, heads, 256), 0xCD, dtype=torch.uint8, device="npu")
    views = a8c4_cache_views(pages.view(torch.float8_e4m3fn))
    dual_cache_scatter(key.npu(), value.npu(), pages.view(torch.float8_e4m3fn), slots.npu())
    torch.npu.synchronize()
    for values, view in zip((key, value), views):
        expected_data, expected_scale = reference_pack(values)
        for token, slot in enumerate(slots.tolist()):
            if slot >= 0:
                block, row = divmod(slot, 128)
                torch.testing.assert_close(view[1][block, row].cpu(), expected_data[token], rtol=0, atol=0)
                torch.testing.assert_close(view[2][block, row].cpu(), expected_scale[token], rtol=0, atol=0)
        assert torch.all(view[1][0, :127].cpu() == 0xCD)
    # Payload is 208 bytes per K/V token/head, with 48 bytes of accounted padding.
    flat = pages.cpu().reshape(2, 4, -1)
    assert torch.all(flat[..., 128 * heads * 208 :] == 0xCD)


@pytest.mark.parametrize("tokens", [1, 8, 224])
def test_fused_qknorm_produces_a8c4_without_standalone_pack(tokens):
    generator = torch.Generator().manual_seed(16154)
    heads, kv_heads, index_heads, dim = 16, 1, 1, 128
    packed = torch.randn(
        (tokens, (heads + 2 * kv_heads + index_heads + 1) * dim), dtype=torch.bfloat16, generator=generator
    )
    raw = torch.zeros((2, 4, 128, kv_heads, 256), dtype=torch.uint8, device="npu")
    views = a8c4_cache_views(raw.view(torch.float8_e4m3fn))
    index = torch.zeros((4, 128, dim), device="npu", dtype=torch.float8_e4m3fn)
    weight = torch.zeros(dim, device="npu", dtype=torch.bfloat16)
    cos = torch.cat((torch.ones((tokens, 32)), torch.zeros((tokens, 32))), 1).bfloat16().npu()
    slots = torch.arange(tokens, dtype=torch.int64, device="npu")
    query, _ = minimax_qknorm_rope_cache(
        packed.npu(),
        cos,
        slots,
        weight,
        weight,
        weight,
        weight,
        views[0][0],
        views[1][0],
        index,
        slots,
        slots,
        tokens,
        heads,
        kv_heads,
        index_heads,
        1e-6,
        a8c4_cache=(views[0][1], views[1][1], views[0][2], views[1][2]),
        query_fp8=True,
    )
    torch.npu.synchronize()
    main = packed[:, : (heads + 2 * kv_heads) * dim].reshape(tokens, heads + 2 * kv_heads, dim).float()
    norm = main[:, :heads] * torch.rsqrt(main[:, :heads].square().mean(-1, keepdim=True) + 1e-6)
    expected_q = norm.bfloat16().float().clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.testing.assert_close(query.cpu().float().reshape_as(expected_q), expected_q.float(), rtol=0, atol=0.25)
    norm_k = main[:, heads : heads + kv_heads]
    norm_k = (norm_k * torch.rsqrt(norm_k.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
    for values, view in zip((norm_k, main[:, heads + kv_heads :].bfloat16()), views):
        data, scale = reference_pack(values)
        actual = view[1].cpu().reshape(-1, kv_heads, 64)[:tokens]
        actual_scale = view[2].cpu().reshape(-1, kv_heads, 8)[:tokens]
        # Normalization can cross an FP4 boundary after a BF16 rounding difference.
        assert (actual != data).float().mean() < 0.02
        torch.testing.assert_close(actual_scale, scale, rtol=0.02, atol=0)
