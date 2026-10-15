import gc

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _tensor_written(tensor: torch.Tensor) -> bool:
    return bool((~torch.isnan(tensor)).any().item())


@pytest.mark.parametrize("enable_rope", [True, False])
@torch.inference_mode()
def test_mla_preprocess_kernel(enable_rope: bool):
    token_num = 1
    head_num = 2
    N_7168 = 7168
    q_lora_rank = 1536
    kv_lora_rank = 512
    rope_dim = 64
    block_num = 1
    block_size = 128
    dtype = torch.bfloat16

    hidden_states = torch.randn((token_num, N_7168), dtype=dtype).npu()

    wdqkv = torch.randint(0, 7, (1, N_7168 // 16, 2112, 16), dtype=dtype).npu()
    wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)
    gamma1 = torch.randn((q_lora_rank), dtype=dtype).npu()

    wuq = torch.randint(0, 7, (1, q_lora_rank // 16, head_num * 192, 16), dtype=dtype).npu()
    wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)
    gamma2 = torch.randn((kv_lora_rank), dtype=dtype).npu()

    cos = torch.randn((token_num, rope_dim), dtype=dtype).npu()
    sin = torch.randn((token_num, rope_dim), dtype=dtype).npu()

    # The no_quant kernel is instantiated with weightFormat3 == DataFormat::ND,
    # so wuk stays ND here.
    wuk = torch.randn((head_num, 128, kv_lora_rank), dtype=dtype).npu()
    # cache_mode="krope_ctkv" is an ND cache mode: the caches are
    # [block_num, block_size, dim], not the 4-D NZ layout. The host tiling reads
    # blockSize from dim1 and the rope head dim from the last axis, so an NZ
    # shape here yields kvCacheBlockSize=32 / qkRopeHeadDim=16 and makes the
    # kernel write past the end of q_nope_out.
    kv_cache = torch.full((block_num, block_size, kv_lora_rank), float("nan"), dtype=dtype, device="npu")
    kv_cache_rope = torch.full((block_num, block_size, rope_dim), float("nan"), dtype=dtype, device="npu")

    slotmapping = torch.randint(0, block_size, (token_num,), dtype=torch.int32).npu()

    q_nope_out = torch.full(
        (hidden_states.shape[0], wuk.shape[0], kv_cache.shape[-1]),
        float("nan"),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    q_rope_out = torch.full(
        (hidden_states.shape[0], wuk.shape[0], kv_cache_rope.shape[-1]),
        float("nan"),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    q_down = torch.empty(
        (hidden_states.shape[0], q_lora_rank),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    torch.ops._C_ascend.mla_preprocess(
        hidden_states,
        wdqkv,
        None,
        gamma1,
        None,
        wuq,
        None,
        gamma2,
        cos if enable_rope else None,
        sin if enable_rope else None,
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
    torch.npu.synchronize()

    assert _tensor_written(q_nope_out), "q_nope was not written"
    assert _tensor_written(q_rope_out), "q_rope was not written"
    assert _tensor_written(kv_cache), "kv_cache was not written"
    assert _tensor_written(kv_cache_rope), "kv_cache_rope was not written"

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
