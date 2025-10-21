import gc

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


@torch.inference_mode()
def test_mla_preprocess_kernel():
    token_num = 1
    head_num = 2
    N_7168 = 7168
    block_num = 1
    block_size = 128
    dtype = torch.bfloat16

    hidden_states = torch.randn((token_num, N_7168), dtype=dtype).npu()
    quant_scale0 = torch.randn((1, ), dtype=dtype).npu()
    quant_offset0 = torch.randint(0, 7, (1, ), dtype=torch.int8).npu()

    wdqkv = torch.randint(0, 7, (1, 224, 2112, 32), dtype=torch.int8).npu()
    wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)

    de_scale0 = torch.rand((2112, ), dtype=torch.float).npu()
    bias0 = torch.randint(0, 7, (2112, ), dtype=torch.int32).npu()
    gamma1 = torch.randn((1536), dtype=dtype).npu()
    beta1 = torch.randn((1536), dtype=dtype).npu()
    quant_scale1 = torch.randn((1, ), dtype=dtype).npu()
    quant_offset1 = torch.randint(0, 7, (1, ), dtype=torch.int8).npu()

    wuq = torch.randint(0, 7, (1, 48, head_num * 192, 32),
                        dtype=torch.int8).npu()
    wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)

    de_scale1 = torch.rand((head_num * 192, ), dtype=torch.float).npu()
    bias1 = torch.randint(0, 7, (head_num * 192, ), dtype=torch.int32).npu()

    gamma2 = torch.randn((512), dtype=dtype).npu()

    cos = torch.randn((token_num, 64), dtype=dtype).npu()
    sin = torch.randn((token_num, 64), dtype=dtype).npu()

    wuk = torch.randn((head_num, 128, 512), dtype=dtype).npu()
    wuk = torch_npu.npu_format_cast(wuk, 29)
    kv_cache = torch.randint(0,
                             7,
                             (block_num, head_num * 512 // 32, block_size, 32),
                             dtype=dtype).npu()
    kv_cache_rope = torch.randn(
        (block_num, head_num * 64 // 16, block_size, 16), dtype=dtype).npu()

    slotmapping = torch.randint(0, 7, (token_num, ), dtype=torch.int32).npu()

    ctkv_scale = torch.randn((1, ), dtype=dtype).npu()
    qnope_scale = torch.randn((head_num), dtype=dtype).npu()

    q_nope_out = torch.empty(
        (hidden_states.shape[0], wuk.shape[0], kv_cache.shape[-1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    q_rope_out = torch.empty(
        (hidden_states.shape[0], wuk.shape[0], kv_cache_rope.shape[-1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    q_nope_old = q_nope_out.clone()
    q_rope_old = q_rope_out.clone()

    torch.ops._C_ascend.mla_preprocess(
        hidden_states,
        wdqkv,
        de_scale0,
        gamma1,
        beta1,
        wuq,
        de_scale1,
        gamma2,
        cos,
        sin,
        wuk,
        kv_cache,
        kv_cache_rope,
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
        q_out0=q_nope_out,
        kv_cache_out0=kv_cache,
        q_out1=q_rope_out,
        kv_cache_out1=kv_cache_rope,
    )
    assert not torch.equal(q_nope_out, q_nope_old)
    assert not torch.equal(q_rope_out, q_rope_old)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
