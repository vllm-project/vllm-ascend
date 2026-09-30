import gc

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _tensor_written(tensor: torch.Tensor) -> bool:
    return bool((~torch.isnan(tensor)).any().item())


def _build_mode_caches(
    cache_mode: str,
    block_num: int,
    block_size: int,
    kv_lora_rank: int,
    rope_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the ctkv/rope caches in the layout each cache_mode expects.

    The host tiling reads blockSize and the rope head dim straight off these
    shapes (GetKvCacheBlockSize / GetRopeHeadDim in
    csrc/mla_preprocess/op_host/mla_preprocess.h), so the layout has to match
    the mode: "krope_ctkv" is an ND mode with 3-D [block_num, block_size, dim]
    caches, while "nzcache" is a physical NZ mode with 4-D
    [block_num, dim // C0, block_size, C0] caches.
    """
    ctkv_shape: tuple[int, ...]
    rope_shape: tuple[int, ...]
    if cache_mode == "krope_ctkv":
        ctkv_shape = (block_num, block_size, kv_lora_rank)
        rope_shape = (block_num, block_size, rope_dim)
    elif cache_mode == "nzcache":
        ctkv_shape = (block_num, kv_lora_rank // 16, block_size, 16)
        rope_shape = (block_num, rope_dim // 16, block_size, 16)
    else:
        raise ValueError(f"unsupported cache_mode: {cache_mode}")

    kv_cache = torch.full(ctkv_shape, float("nan"), dtype=dtype, device="npu")
    kv_cache_rope = torch.full(rope_shape, float("nan"), dtype=dtype, device="npu")
    return kv_cache, kv_cache_rope


@pytest.mark.parametrize("cache_mode", ["krope_ctkv", "nzcache"])
@pytest.mark.parametrize("enable_rope", [True, False])
@torch.inference_mode()
def test_mla_preprocess_kernel(cache_mode: str, enable_rope: bool):
    """Exercise MLA QDown cache modes with RoPE enabled and disabled."""
    torch.manual_seed(0)
    token_num = 1
    head_num = 2
    N_7168 = 7168
    mm1_out = 2112
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    rope_dim = 64
    block_num = 1
    block_size = 128
    dtype = torch.bfloat16

    hidden_states = torch.randn((token_num, N_7168), dtype=dtype).npu()
    quant_scale0 = torch.randn((1,), dtype=dtype).npu()
    quant_offset0 = torch.randint(0, 7, (1,), dtype=torch.int8).npu()

    wdqkv = torch.randint(0, 7, (1, N_7168 // 32, mm1_out, 32), dtype=torch.int8).npu()
    wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)

    de_scale0 = torch.rand((mm1_out,), dtype=torch.float).npu()
    bias0 = torch.randint(0, 7, (mm1_out,), dtype=torch.int32).npu()
    gamma1 = torch.randn((q_lora_rank), dtype=dtype).npu()
    beta1 = torch.randn((q_lora_rank), dtype=dtype).npu()
    quant_scale1 = torch.randn((1,), dtype=dtype).npu()
    quant_offset1 = torch.randint(0, 7, (1,), dtype=torch.int8).npu()

    wuq = torch.randint(0, 7, (1, q_lora_rank // 32, head_num * 192, 32), dtype=torch.int8).npu()
    wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)

    de_scale1 = torch.rand((head_num * 192,), dtype=torch.float).npu()
    bias1 = torch.randint(0, 7, (head_num * 192,), dtype=torch.int32).npu()

    gamma2 = torch.randn((kv_lora_rank), dtype=dtype).npu()

    cos = torch.randn((token_num, rope_dim), dtype=dtype).npu()
    sin = torch.randn((token_num, rope_dim), dtype=dtype).npu()

    wuk = torch.randn((head_num, qk_nope_head_dim, kv_lora_rank), dtype=dtype).npu()
    wuk = torch_npu.npu_format_cast(wuk, 29)
    kv_cache, kv_cache_rope = _build_mode_caches(cache_mode, block_num, block_size, kv_lora_rank, rope_dim, dtype)

    slotmapping = torch.randint(0, block_size, (token_num,), dtype=torch.int32).npu()

    ctkv_scale = torch.randn((1,), dtype=dtype).npu()
    qnope_scale = torch.randn((head_num), dtype=dtype).npu()

    # q_nope_out / q_rope_out are laid out per token as [head_num, dim]; the dim
    # comes from wuk / the rope head dim, never from the cache's trailing axis,
    # which is C0 (16) in the NZ modes.
    q_nope_out = torch.full(
        (token_num, head_num, kv_lora_rank),
        float("nan"),
        dtype=dtype,
        device=hidden_states.device,
    )
    q_rope_out = torch.full(
        (token_num, head_num, rope_dim),
        float("nan"),
        dtype=dtype,
        device=hidden_states.device,
    )
    q_down = torch.full(
        (token_num, q_lora_rank),
        float("nan"),
        dtype=dtype,
        device=hidden_states.device,
    )

    torch.ops._C_ascend.mla_preprocess(
        hidden_states,
        wdqkv,
        de_scale0,
        gamma1,
        beta1,
        wuq,
        de_scale1,
        gamma2,
        cos if enable_rope else None,
        sin if enable_rope else None,
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
        cache_mode=cache_mode,
        quant_mode="per_tensor_quant_asymm",
        enable_inner_out=True,
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
    assert _tensor_written(q_down), "inner_out (q_down) was not written"

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
