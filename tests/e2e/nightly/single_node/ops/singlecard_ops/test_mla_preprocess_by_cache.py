import os

import pytest

if os.environ.get("VLLM_ASCEND_RUN_MLA_PREPROCESS_BY_CACHE_OP") != "1":
    pytest.skip("set VLLM_ASCEND_RUN_MLA_PREPROCESS_BY_CACHE_OP=1 to run this NPU op smoke", allow_module_level=True)

import gc

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


@torch.inference_mode()
def test_mla_preprocess_by_cache_kernel():
    token_num = 1
    head_num = 2
    num_kv_heads = 1
    hidden_size = 7168
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    dtype = torch.bfloat16

    hidden_states = torch.randn((token_num, hidden_size), dtype=dtype).npu()
    quant_scale0 = torch.randn((1,), dtype=dtype).npu()
    quant_offset0 = torch.randint(0, 7, (1,), dtype=torch.int8).npu()

    wdqkv = torch.randint(0, 7, (1, 224, 2112, 32), dtype=torch.int8).npu()
    wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)

    de_scale0 = torch.rand((2112,), dtype=torch.float).npu()
    bias0 = torch.randint(0, 7, (2112,), dtype=torch.int32).npu()
    gamma1 = torch.randn((q_lora_rank,), dtype=dtype).npu()
    beta1 = torch.randn((q_lora_rank,), dtype=dtype).npu()
    quant_scale1 = torch.randn((1,), dtype=dtype).npu()
    quant_offset1 = torch.randint(0, 7, (1,), dtype=torch.int8).npu()

    wuq = torch.randint(0, 7, (1, 48, head_num * 192, 32), dtype=torch.int8).npu()
    wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)

    de_scale1 = torch.rand((head_num * 192,), dtype=torch.float).npu()
    bias1 = torch.randint(0, 7, (head_num * 192,), dtype=torch.int32).npu()
    gamma2 = torch.randn((kv_lora_rank,), dtype=dtype).npu()

    positions = torch.arange(token_num, dtype=torch.long).npu()

    wuk = torch.randn((head_num, qk_nope_head_dim, kv_lora_rank), dtype=dtype).npu()
    wuk = torch_npu.npu_format_cast(wuk, 29)
    k_c_normed = torch.zeros((token_num, num_kv_heads, kv_lora_rank), dtype=dtype).npu()
    k_pe = torch.zeros((token_num, num_kv_heads, qk_rope_head_dim), dtype=dtype).npu()
    cos_sin_cache = torch.randn((token_num + 8, qk_rope_head_dim), dtype=dtype).npu()

    slotmapping = torch.arange(token_num, dtype=torch.int32).npu()
    ctkv_scale = torch.randn((1,), dtype=dtype).npu()
    qnope_scale = torch.randn((head_num,), dtype=dtype).npu()

    ql_nope = torch.zeros(
        (hidden_states.shape[0], head_num, kv_lora_rank),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    q_pe = torch.zeros(
        (hidden_states.shape[0], head_num, qk_rope_head_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    inner_out = torch.empty(
        (0,),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    raw_q_out = torch.zeros(
        (hidden_states.shape[0], wuk.shape[0], wuk.shape[1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    torch.ops._C_ascend.mla_preprocess_by_cache(
        hidden_states,
        wdqkv,
        de_scale0,
        gamma1,
        beta1,
        wuq,
        de_scale1,
        gamma2,
        positions,
        cos_sin_cache,
        wuk,
        k_c_normed,
        k_pe,
        slotmapping,
        quant_scale0=quant_scale0,
        quant_offset0=quant_offset0,
        bias0=bias0,
        quant_scale1=quant_scale1,
        quant_offset1=quant_offset1,
        bias1=bias1,
        ctkv_scale=ctkv_scale,
        q_nope_scale=qnope_scale,
        cache_mode="krope_ctkv",
        quant_mode="per_tensor_quant_asymm",
        enable_inner_out=False,
        is_neox_style=True,
        q_out0=ql_nope,
        kv_cache_out0=k_c_normed,
        q_out1=q_pe,
        kv_cache_out1=k_pe,
        inner_out=inner_out,
        enable_raw_q_out=True,
        raw_q_out=raw_q_out,
    )
    torch.npu.synchronize()

    assert torch.isfinite(ql_nope.float()).all()
    assert torch.isfinite(q_pe.float()).all()
    assert torch.isfinite(k_c_normed.float()).all()
    assert torch.isfinite(k_pe.float()).all()
    assert torch.isfinite(raw_q_out.float()).all()
    assert torch.any(ql_nope != 0).item()
    assert torch.any(q_pe != 0).item()
    assert torch.any(k_c_normed != 0).item()
    assert torch.any(k_pe != 0).item()
    assert torch.any(raw_q_out != 0).item()

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
