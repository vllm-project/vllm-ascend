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

    wdqkv = torch.randint(0, 7, (1, 448, 2112, 16), dtype=dtype).npu()
    wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)
    gamma1 = torch.randn((1536), dtype=dtype).npu()

    wuq = torch.randint(0, 7, (1, 96, head_num * 192, 16), dtype=dtype).npu()
    wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)
    gamma2 = torch.randn((512), dtype=dtype).npu()

    cos = torch.randn((token_num, 64), dtype=dtype).npu()
    sin = torch.randn((token_num, 64), dtype=dtype).npu()

    wuk = torch.randn((head_num, 128, 512), dtype=dtype).npu()
    # wuk = torch_npu.npu_format_cast(wuk, 29)
    kv_cache = torch.randint(0,
                             7,
                             (block_num, head_num * 512 // 32, block_size, 32),
                             dtype=dtype).npu()
    kv_cache_rope = torch.randn(
        (block_num, head_num * 64 // 16, block_size, 16), dtype=dtype).npu()

    slotmapping = torch.randint(0, 7, (token_num, ), dtype=torch.int32).npu()

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
    q_down = torch.empty(
        (hidden_states.shape[0], 1536),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    q_nope_old = q_nope_out.clone()
    q_rope_old = q_rope_out.clone()

    torch.ops._C_ascend.mla_preprocess(
        hidden_states,
        wdqkv,
        None,
        gamma1,
        None,
        wuq,
        None,
        gamma2,
        cos,
        sin,
        wuk,
        kv_cache,
        kv_cache_rope,
        slotmapping,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        cache_mode="krope_ctkv",
        quant_mode="no_quant",
        enable_inner_out=False,
        q_out0=q_nope_out,
        kv_cache_out0=kv_cache,
        q_out1=q_rope_out,
        kv_cache_out1=kv_cache_rope,
        inner_out=q_down,
    )
    assert not torch.equal(q_nope_out, q_nope_old)
    assert not torch.equal(q_rope_out, q_rope_old)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
