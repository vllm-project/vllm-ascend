import gc

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _skip_if_mla_prolog_v3_k3_unavailable():
    if not hasattr(torch.ops, "_C_ascend") or not hasattr(torch.ops._C_ascend, "npu_mla_prolog_v3_k3"):
        pytest.skip("requires the npu_mla_prolog_v3_k3 custom operator")


@torch.inference_mode()
def test_cann_mla_prolog_v3_mxfp8_per_tile_with_k3_registered():
    """Loading K3's custom implementation must not shadow CANN's KV mode 3."""
    test_mla_prolog_v3_k3_rope_disabled()
    token_num, he, hcq, hckv, head_num, d, dr, tile_size = 2, 7168, 1536, 512, 32, 128, 64, 128

    def sample(shape):
        return (torch.randn(shape, dtype=torch.float32) * 0.02).to(torch.bfloat16).npu()

    def quantized_weight(out_features, in_features):
        value, scale = torch_npu.npu_dynamic_mx_quant(sample((out_features, in_features)), dst_type=torch.float8_e4m3fn)
        return (
            torch_npu.npu_format_cast(value.T.contiguous(), 29),
            scale.reshape(out_features, -1).view(torch.float8_e8m0fnu),
        )

    weight_dq, scale_dq = quantized_weight(hcq, he)
    weight_uq_qr, scale_uq_qr = quantized_weight(head_num * (d + dr), hcq)
    weight_dkv_kr, scale_dkv_kr = quantized_weight(hckv + dr, he)
    token_x, scale_x = torch_npu.npu_dynamic_mx_quant(sample((token_num, he)), dst_type=torch.float8_e4m3fn)
    # C8 packs quantized KV, BF16 RoPE and per-tile FP32 scales into one cache.
    packed_dim = hckv + dr * 2 + hckv // tile_size * 4
    kv_cache = torch.zeros((2, 128, 1, packed_dim), dtype=torch.float8_e4m3fn, device="npu")
    query, query_rope, *_ = torch_npu.npu_mla_prolog_v3(
        token_x=token_x,
        weight_dq=weight_dq,
        weight_uq_qr=weight_uq_qr,
        weight_uk=sample((head_num, d, hckv)),
        weight_dkv_kr=weight_dkv_kr,
        rmsnorm_gamma_cq=torch.ones(hcq, dtype=torch.bfloat16, device="npu"),
        rmsnorm_gamma_ckv=torch.ones(hckv, dtype=torch.bfloat16, device="npu"),
        rope_sin=torch.zeros((token_num, dr), dtype=torch.bfloat16, device="npu"),
        rope_cos=torch.ones((token_num, dr), dtype=torch.bfloat16, device="npu"),
        kv_cache=kv_cache,
        kr_cache=torch.empty(0, dtype=torch.bfloat16, device="npu"),
        cache_index=torch.arange(token_num, dtype=torch.int64, device="npu"),
        dequant_scale_x=scale_x.reshape(token_num, -1).view(torch.float8_e8m0fnu),
        dequant_scale_w_dq=scale_dq,
        dequant_scale_w_uq_qr=scale_uq_qr,
        dequant_scale_w_dkv_kr=scale_dkv_kr,
        cache_mode="PA_BSND",
        weight_quant_mode=3,
        kv_cache_quant_mode=3,
        query_quant_mode=0,
        ckvkr_repo_mode=1,
        quant_scale_repo_mode=1,
        tile_size=tile_size,
    )
    torch.npu.synchronize()
    assert query.shape == (token_num, head_num, hckv)
    assert query_rope.shape == (token_num, head_num, dr)
    assert torch.isfinite(query.float()).all()
    assert torch.isfinite(query_rope.float()).all()
    cache_bytes = kv_cache.view(torch.uint8)
    assert cache_bytes[0, :token_num].count_nonzero() > 0
    assert cache_bytes[0, token_num:].count_nonzero() == 0
    assert cache_bytes[1].count_nonzero() == 0


@torch.inference_mode()
def test_mla_prolog_v3_k3_native_bf16_head96():
    """Kimi K3 native bf16: head_num=96, q_lora=1536, kv_lora=512, D=128, Dr=64."""
    _skip_if_mla_prolog_v3_k3_unavailable()

    token_num = 1
    head_num = 96
    he = 7168
    hcq = 1536
    hckv = 512
    d = 128
    dr = 64
    block_num = 2
    block_size = 128
    dtype = torch.bfloat16

    token_x = torch.randn((token_num, he), dtype=dtype).npu()
    weight_dq = torch_npu.npu_format_cast(torch.randn((he, hcq), dtype=dtype).npu().contiguous(), 29)
    weight_uq_qr = torch_npu.npu_format_cast(
        torch.randn((hcq, head_num * (d + dr)), dtype=dtype).npu().contiguous(), 29
    )
    weight_uk = torch.randn((head_num, d, hckv), dtype=dtype).npu()
    weight_dkv_kr = torch_npu.npu_format_cast(torch.randn((he, hckv + dr), dtype=dtype).npu().contiguous(), 29)
    rmsnorm_gamma_cq = torch.ones((hcq,), dtype=dtype).npu()
    rmsnorm_gamma_ckv = torch.ones((hckv,), dtype=dtype).npu()
    rope_sin = torch.randn((token_num, dr), dtype=dtype).npu()
    rope_cos = torch.randn((token_num, dr), dtype=dtype).npu()
    kv_cache = torch.zeros((block_num, block_size, 1, hckv), dtype=dtype).npu()
    kr_cache = torch.zeros((block_num, block_size, 1, dr), dtype=dtype).npu()
    cache_index = torch.arange(token_num, dtype=torch.int64).npu()

    kv_old = kv_cache.clone()
    kr_old = kr_cache.clone()

    query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm = (
        torch.ops._C_ascend.npu_mla_prolog_v3_k3(
            token_x,
            weight_dq,
            weight_uq_qr,
            weight_uk,
            weight_dkv_kr,
            rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv,
            rope_sin,
            rope_cos,
            kv_cache,
            kr_cache,
            cache_index=cache_index,
            cache_mode="PA_BSND",
        )
    )

    assert query.shape == (token_num, head_num, hckv)
    assert query_rope.shape == (token_num, head_num, dr)
    assert query.dtype == dtype
    assert query_rope.dtype == dtype
    assert dequant_scale_q_nope.numel() == 0
    assert query_norm.numel() == 0
    assert dequant_scale_q_norm.numel() == 0
    assert not torch.equal(kv_cache, kv_old)
    assert not torch.equal(kr_cache, kr_old)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_mla_prolog_v3_k3_rope_disabled():
    _skip_if_mla_prolog_v3_k3_unavailable()

    token_num = 1
    head_num = 96
    he = 7168
    hcq = 1536
    hckv = 512
    d = 128
    dr = 64
    block_num = 2
    block_size = 128
    dtype = torch.bfloat16

    token_x = torch.randn((token_num, he), dtype=dtype).npu()
    weight_dq = torch_npu.npu_format_cast(torch.randn((he, hcq), dtype=dtype).npu().contiguous(), 29)
    weight_uq_qr = torch_npu.npu_format_cast(
        torch.randn((hcq, head_num * (d + dr)), dtype=dtype).npu().contiguous(), 29
    )
    weight_uk = torch.randn((head_num, d, hckv), dtype=dtype).npu()
    weight_dkv_kr = torch_npu.npu_format_cast(torch.randn((he, hckv + dr), dtype=dtype).npu().contiguous(), 29)
    rmsnorm_gamma_cq = torch.ones((hcq,), dtype=dtype).npu()
    rmsnorm_gamma_ckv = torch.ones((hckv,), dtype=dtype).npu()
    rope_sin = torch.empty((0, dr), dtype=dtype).npu()
    rope_cos = torch.empty((0, dr), dtype=dtype).npu()
    kv_cache = torch.zeros((block_num, block_size, 1, hckv), dtype=dtype).npu()
    kr_cache = torch.zeros((block_num, block_size, 1, dr), dtype=dtype).npu()
    cache_index = torch.arange(token_num, dtype=torch.int64).npu()

    query, query_rope, *_ = torch.ops._C_ascend.npu_mla_prolog_v3_k3(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope_sin,
        rope_cos,
        kv_cache,
        kr_cache,
        cache_index=cache_index,
        cache_mode="PA_BSND",
    )

    assert query.shape == (token_num, head_num, hckv)
    assert query_rope.shape == (token_num, head_num, dr)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_mla_prolog_v3_k3_query_norm_flag():
    _skip_if_mla_prolog_v3_k3_unavailable()

    token_num = 1
    head_num = 96
    he = 7168
    hcq = 1536
    hckv = 512
    d = 128
    dr = 64
    block_num = 2
    block_size = 128
    dtype = torch.bfloat16

    token_x = torch.randn((token_num, he), dtype=dtype).npu()
    weight_dq = torch_npu.npu_format_cast(torch.randn((he, hcq), dtype=dtype).npu().contiguous(), 29)
    weight_uq_qr = torch_npu.npu_format_cast(
        torch.randn((hcq, head_num * (d + dr)), dtype=dtype).npu().contiguous(), 29
    )
    weight_uk = torch.randn((head_num, d, hckv), dtype=dtype).npu()
    weight_dkv_kr = torch_npu.npu_format_cast(torch.randn((he, hckv + dr), dtype=dtype).npu().contiguous(), 29)
    rmsnorm_gamma_cq = torch.ones((hcq,), dtype=dtype).npu()
    rmsnorm_gamma_ckv = torch.ones((hckv,), dtype=dtype).npu()
    rope_sin = torch.randn((token_num, dr), dtype=dtype).npu()
    rope_cos = torch.randn((token_num, dr), dtype=dtype).npu()
    kv_cache = torch.zeros((block_num, block_size, 1, hckv), dtype=dtype).npu()
    kr_cache = torch.zeros((block_num, block_size, 1, dr), dtype=dtype).npu()
    cache_index = torch.arange(token_num, dtype=torch.int64).npu()

    query, query_rope, _, query_norm, dequant_scale_q_norm = torch.ops._C_ascend.npu_mla_prolog_v3_k3(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope_sin,
        rope_cos,
        kv_cache,
        kr_cache,
        cache_index=cache_index,
        cache_mode="PA_BSND",
        query_norm_flag=True,
    )

    assert query.shape == (token_num, head_num, hckv)
    assert query_rope.shape == (token_num, head_num, dr)
    assert query_norm.shape == (token_num, hcq)
    assert query_norm.dtype == dtype
    assert dequant_scale_q_norm.numel() == 0

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("cache_mode", ["PA_NZ", "TND"])
@torch.inference_mode()
def test_mla_prolog_v3_k3_cache_mode(cache_mode: str):
    _skip_if_mla_prolog_v3_k3_unavailable()

    token_num = 1
    head_num = 96
    he = 7168
    hcq = 1536
    hckv = 512
    d = 128
    dr = 64
    block_num = 2
    block_size = 128
    dtype = torch.bfloat16

    token_x = torch.randn((token_num, he), dtype=dtype).npu()
    weight_dq = torch_npu.npu_format_cast(torch.randn((he, hcq), dtype=dtype).npu().contiguous(), 29)
    weight_uq_qr = torch_npu.npu_format_cast(
        torch.randn((hcq, head_num * (d + dr)), dtype=dtype).npu().contiguous(), 29
    )
    weight_uk = torch.randn((head_num, d, hckv), dtype=dtype).npu()
    weight_dkv_kr = torch_npu.npu_format_cast(torch.randn((he, hckv + dr), dtype=dtype).npu().contiguous(), 29)
    rmsnorm_gamma_cq = torch.ones((hcq,), dtype=dtype).npu()
    rmsnorm_gamma_ckv = torch.ones((hckv,), dtype=dtype).npu()
    rope_sin = torch.randn((token_num, dr), dtype=dtype).npu()
    rope_cos = torch.randn((token_num, dr), dtype=dtype).npu()

    if cache_mode == "TND":
        kv_cache = torch.zeros((token_num, 1, hckv), dtype=dtype).npu()
        kr_cache = torch.zeros((token_num, 1, dr), dtype=dtype).npu()
        cache_index = None
    else:
        kv_cache = torch.zeros((block_num, block_size, 1, hckv), dtype=dtype).npu()
        kr_cache = torch.zeros((block_num, block_size, 1, dr), dtype=dtype).npu()
        cache_index = torch.arange(token_num, dtype=torch.int64).npu()

    query, query_rope, *_ = torch.ops._C_ascend.npu_mla_prolog_v3_k3(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope_sin,
        rope_cos,
        kv_cache,
        kr_cache,
        cache_index=cache_index,
        cache_mode=cache_mode,
    )

    assert query.shape == (token_num, head_num, hckv)
    assert query_rope.shape == (token_num, head_num, dr)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("cache_mode", ["PA_BSND", "PA_NZ"])
@torch.inference_mode()
def test_mla_prolog_v3_k3_noncontiguous_cache_dim0(cache_mode: str):
    """KV/KR cache views may have padding between physical blocks."""
    _skip_if_mla_prolog_v3_k3_unavailable()

    token_num = 2
    head_num = 96
    he = 7168
    hcq = 1536
    hckv = 512
    d = 128
    dr = 64
    block_num = 4
    block_size = 128
    dtype = torch.bfloat16

    token_x = torch.randn((token_num, he), dtype=dtype).npu()
    weight_dq = torch_npu.npu_format_cast(torch.randn((he, hcq), dtype=dtype).npu().contiguous(), 29)
    weight_uq_qr = torch_npu.npu_format_cast(
        torch.randn((hcq, head_num * (d + dr)), dtype=dtype).npu().contiguous(), 29
    )
    weight_uk = torch.randn((head_num, d, hckv), dtype=dtype).npu()
    weight_dkv_kr = torch_npu.npu_format_cast(torch.randn((he, hckv + dr), dtype=dtype).npu().contiguous(), 29)
    rmsnorm_gamma_cq = torch.ones((hcq,), dtype=dtype).npu()
    rmsnorm_gamma_ckv = torch.ones((hckv,), dtype=dtype).npu()
    rope_sin = torch.randn((token_num, dr), dtype=dtype).npu()
    rope_cos = torch.randn((token_num, dr), dtype=dtype).npu()
    cache_index = torch.tensor([block_size + 1, 2 * block_size + 3], dtype=torch.int64).npu()

    kv_contiguous = torch.zeros((block_num, block_size, 1, hckv), dtype=dtype).npu()
    kr_contiguous = torch.zeros((block_num, block_size, 1, dr), dtype=dtype).npu()
    query_ref, query_rope_ref, *_ = torch.ops._C_ascend.npu_mla_prolog_v3_k3(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope_sin,
        rope_cos,
        kv_contiguous,
        kr_contiguous,
        cache_index=cache_index,
        cache_mode=cache_mode,
    )

    # Keep one physical padding block between adjacent logical KV blocks and
    # two physical padding blocks between adjacent logical KR blocks.
    kv_storage = torch.full((block_num * 2, block_size, 1, hckv), 7.0, dtype=dtype).npu()
    kr_storage = torch.full((block_num * 3, block_size, 1, dr), -5.0, dtype=dtype).npu()
    kv_cache = kv_storage[::2]
    kr_cache = kr_storage[::3]
    kv_cache.zero_()
    kr_cache.zero_()

    assert not kv_cache.is_contiguous()
    assert not kr_cache.is_contiguous()
    assert kv_cache.stride(0) == 2 * block_size * hckv
    assert kr_cache.stride(0) == 3 * block_size * dr

    kv_before = kv_cache.clone()
    kr_before = kr_cache.clone()
    kv_padding_before = kv_storage[1::2].clone()
    kr_padding_1_before = kr_storage[1::3].clone()
    kr_padding_2_before = kr_storage[2::3].clone()

    # Both tokens target non-zero logical blocks, so an implementation that
    # ignores dim0 stride would overwrite physical padding blocks.
    query, query_rope, *_ = torch.ops._C_ascend.npu_mla_prolog_v3_k3(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope_sin,
        rope_cos,
        kv_cache,
        kr_cache,
        cache_index=cache_index,
        cache_mode=cache_mode,
    )

    assert query.shape == (token_num, head_num, hckv)
    assert query_rope.shape == (token_num, head_num, dr)
    assert not torch.equal(kv_cache, kv_before)
    assert not torch.equal(kr_cache, kr_before)
    torch.testing.assert_close(query, query_ref, rtol=0, atol=0)
    torch.testing.assert_close(query_rope, query_rope_ref, rtol=0, atol=0)
    assert torch.equal(kv_cache, kv_contiguous)
    assert torch.equal(kr_cache, kr_contiguous)
    assert torch.equal(kv_storage[1::2], kv_padding_before)
    assert torch.equal(kr_storage[1::3], kr_padding_1_before)
    assert torch.equal(kr_storage[2::3], kr_padding_2_before)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_mla_prolog_v3_k3_cache_mode_bsnd():
    _skip_if_mla_prolog_v3_k3_unavailable()

    batch = 1
    seq = 2
    head_num = 96
    he = 7168
    hcq = 1536
    hckv = 512
    d = 128
    dr = 64
    dtype = torch.bfloat16

    token_x = torch.randn((batch, seq, he), dtype=dtype).npu()
    weight_dq = torch_npu.npu_format_cast(torch.randn((he, hcq), dtype=dtype).npu().contiguous(), 29)
    weight_uq_qr = torch_npu.npu_format_cast(
        torch.randn((hcq, head_num * (d + dr)), dtype=dtype).npu().contiguous(), 29
    )
    weight_uk = torch.randn((head_num, d, hckv), dtype=dtype).npu()
    weight_dkv_kr = torch_npu.npu_format_cast(torch.randn((he, hckv + dr), dtype=dtype).npu().contiguous(), 29)
    rmsnorm_gamma_cq = torch.ones((hcq,), dtype=dtype).npu()
    rmsnorm_gamma_ckv = torch.ones((hckv,), dtype=dtype).npu()
    rope_sin = torch.randn((batch, seq, dr), dtype=dtype).npu()
    rope_cos = torch.randn((batch, seq, dr), dtype=dtype).npu()
    kv_cache = torch.zeros((batch, seq, 1, hckv), dtype=dtype).npu()
    kr_cache = torch.zeros((batch, seq, 1, dr), dtype=dtype).npu()

    query, query_rope, *_ = torch.ops._C_ascend.npu_mla_prolog_v3_k3(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope_sin,
        rope_cos,
        kv_cache,
        kr_cache,
        cache_mode="BSND",
    )

    assert query.shape == (batch, seq, head_num, hckv)
    assert query_rope.shape == (batch, seq, head_num, dr)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
