import gc

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _skip_if_mla_prolog_v3_unavailable():
    if not hasattr(torch.ops, "_C_ascend") or not hasattr(torch.ops._C_ascend, "npu_mla_prolog_v3"):
        pytest.skip("requires the npu_mla_prolog_v3 custom operator")


@torch.inference_mode()
def test_mla_prolog_v3_native_bf16_head96():
    """Kimi K3 native bf16: head_num=96, q_lora=1536, kv_lora=512, D=128, Dr=64."""
    _skip_if_mla_prolog_v3_unavailable()

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

    query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm = torch.ops._C_ascend.npu_mla_prolog_v3(
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
def test_mla_prolog_v3_rope_disabled():
    _skip_if_mla_prolog_v3_unavailable()

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

    query, query_rope, *_ = torch.ops._C_ascend.npu_mla_prolog_v3(
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
def test_mla_prolog_v3_query_norm_flag():
    _skip_if_mla_prolog_v3_unavailable()

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

    query, query_rope, _, query_norm, dequant_scale_q_norm = torch.ops._C_ascend.npu_mla_prolog_v3(
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
def test_mla_prolog_v3_cache_mode(cache_mode: str):
    _skip_if_mla_prolog_v3_unavailable()

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

    query, query_rope, *_ = torch.ops._C_ascend.npu_mla_prolog_v3(
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


@torch.inference_mode()
def test_mla_prolog_v3_cache_mode_bsnd():
    _skip_if_mla_prolog_v3_unavailable()

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

    query, query_rope, *_ = torch.ops._C_ascend.npu_mla_prolog_v3(
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


@torch.inference_mode()
def test_mla_prolog_v3_partial_quant():
    _skip_if_mla_prolog_v3_unavailable()

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
        torch.randint(-7, 8, (hcq, head_num * (d + dr)), dtype=torch.int8).npu().contiguous(), 29
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
    dequant_scale_w_uq_qr = torch.rand((1, head_num * (d + dr)), dtype=torch.float).npu()

    query, query_rope, *_ = torch.ops._C_ascend.npu_mla_prolog_v3(
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
        dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
        cache_mode="PA_BSND",
        weight_quant_mode=1,
        kv_cache_quant_mode=0,
    )

    assert query.shape == (token_num, head_num, hckv)
    assert query_rope.shape == (token_num, head_num, dr)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_mla_prolog_v3_full_int8_quant():
    _skip_if_mla_prolog_v3_unavailable()

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

    token_x = torch.randint(-7, 8, (token_num, he), dtype=torch.int8).npu()
    weight_dq = torch_npu.npu_format_cast(torch.randint(-7, 8, (he, hcq), dtype=torch.int8).npu().contiguous(), 29)
    weight_uq_qr = torch_npu.npu_format_cast(
        torch.randint(-7, 8, (hcq, head_num * (d + dr)), dtype=torch.int8).npu().contiguous(), 29
    )
    weight_uk = torch.randn((head_num, d, hckv), dtype=dtype).npu()
    weight_dkv_kr = torch_npu.npu_format_cast(
        torch.randint(-7, 8, (he, hckv + dr), dtype=torch.int8).npu().contiguous(), 29
    )
    rmsnorm_gamma_cq = torch.ones((hcq,), dtype=dtype).npu()
    rmsnorm_gamma_ckv = torch.ones((hckv,), dtype=dtype).npu()
    rope_sin = torch.randn((token_num, dr), dtype=dtype).npu()
    rope_cos = torch.randn((token_num, dr), dtype=dtype).npu()
    kv_cache = torch.zeros((block_num, block_size, 1, hckv), dtype=dtype).npu()
    kr_cache = torch.zeros((block_num, block_size, 1, dr), dtype=dtype).npu()
    cache_index = torch.arange(token_num, dtype=torch.int64).npu()
    dequant_scale_x = torch.rand((token_num, 1), dtype=torch.float).npu()
    dequant_scale_w_dq = torch.rand((1, hcq), dtype=torch.float).npu()
    dequant_scale_w_uq_qr = torch.rand((1, head_num * (d + dr)), dtype=torch.float).npu()
    dequant_scale_w_dkv_kr = torch.rand((1, hckv + dr), dtype=torch.float).npu()

    query, query_rope, *_ = torch.ops._C_ascend.npu_mla_prolog_v3(
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
        dequant_scale_x=dequant_scale_x,
        dequant_scale_w_dq=dequant_scale_w_dq,
        dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
        dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr,
        cache_mode="PA_BSND",
        weight_quant_mode=2,
        kv_cache_quant_mode=0,
    )

    assert query.shape == (token_num, head_num, hckv)
    assert query_rope.shape == (token_num, head_num, dr)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
