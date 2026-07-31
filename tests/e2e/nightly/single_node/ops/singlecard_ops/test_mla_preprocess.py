import gc

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _make_dim0_strided_cache(
    shape: tuple[int, ...],
    stride_factor: int,
    dtype: torch.dtype,
    fill_value: float | int,
) -> tuple[torch.Tensor, torch.Tensor]:
    backing_shape = (shape[0] * stride_factor, *shape[1:])
    backing = torch.full(backing_shape, fill_value, dtype=dtype, device="npu")
    contiguous_strides = backing.stride()
    cache = backing.as_strided(
        shape,
        (stride_factor * contiguous_strides[0], *contiguous_strides[1:]),
    )
    return cache, backing


def _cache_to_logical(cache: torch.Tensor, cache_mode: str) -> torch.Tensor:
    if cache_mode == "krope_ctkv":
        return cache
    # Physical NZ [block, C1, block_size, C0] -> logical ND
    # [block, block_size, C1 * C0].
    return cache.permute(0, 2, 1, 3).reshape(cache.shape[0], cache.shape[2], -1)


def _assert_dim0_holes_untouched(
    backing: torch.Tensor,
    stride_factor: int,
    fill_value: float | int,
) -> None:
    for remainder in range(1, stride_factor):
        holes = backing[remainder::stride_factor]
        if backing.dtype.is_floating_point:
            assert torch.isnan(holes).all()
        else:
            assert (holes == fill_value).all()


@pytest.mark.skip(reason="Failure of an individual operator use case causes failures of other operators.")
@pytest.mark.parametrize("cache_mode", ["krope_ctkv", "nzcache"])
@torch.inference_mode()
def test_mla_preprocess_kernel(cache_mode: str):
    token_num = 1
    head_num = 2
    N_7168 = 7168
    block_num = 1
    block_size = 128
    dtype = torch.bfloat16

    hidden_states = torch.randn((token_num, N_7168), dtype=dtype).npu()
    quant_scale0 = torch.randn((1,), dtype=dtype).npu()
    quant_offset0 = torch.randint(0, 7, (1,), dtype=torch.int8).npu()

    wdqkv = torch.randint(0, 7, (1, 224, 2112, 32), dtype=torch.int8).npu()
    wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)

    de_scale0 = torch.rand((2112,), dtype=torch.float).npu()
    bias0 = torch.randint(0, 7, (2112,), dtype=torch.int32).npu()
    gamma1 = torch.randn((1536), dtype=dtype).npu()
    beta1 = torch.randn((1536), dtype=dtype).npu()
    quant_scale1 = torch.randn((1,), dtype=dtype).npu()
    quant_offset1 = torch.randint(0, 7, (1,), dtype=torch.int8).npu()

    wuq = torch.randint(0, 7, (1, 48, head_num * 192, 32), dtype=torch.int8).npu()
    wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)

    de_scale1 = torch.rand((head_num * 192,), dtype=torch.float).npu()
    bias1 = torch.randint(0, 7, (head_num * 192,), dtype=torch.int32).npu()

    gamma2 = torch.randn((512), dtype=dtype).npu()

    cos = torch.randn((token_num, 64), dtype=dtype).npu()
    sin = torch.randn((token_num, 64), dtype=dtype).npu()

    wuk = torch.randn((head_num, 128, 512), dtype=dtype).npu()
    wuk = torch_npu.npu_format_cast(wuk, 29)
    kv_cache = torch.randint(0, 7, (block_num, head_num * 512 // 32, block_size, 32), dtype=dtype).npu()
    kv_cache_rope = torch.randn((block_num, head_num * 64 // 16, block_size, 16), dtype=dtype).npu()

    slotmapping = torch.randint(0, 7, (token_num,), dtype=torch.int32).npu()

    ctkv_scale = torch.randn((1,), dtype=dtype).npu()
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
        cache_mode=cache_mode,
        quant_mode="per_tensor_quant_asymm",
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


@pytest.mark.parametrize(
    "cache_mode",
    ["krope_ctkv", "int8_nzcache"],
    ids=["cachemode1", "cachemode2"],
)
@torch.inference_mode()
def test_mla_preprocess_qm1_noncontiguous_cache(cache_mode: str):
    """Compare contiguous and dim0-strided caches for cacheMode 1/2 + quantMode 1."""
    torch.manual_seed(0)
    token_num = 200
    head_num = 2
    hidden_size = 7168
    block_num = 2
    block_size = 128
    stride_factor = 2
    dtype = torch.bfloat16
    quantized_cache = cache_mode == "int8_nzcache"

    hidden_states = torch.randn((token_num, hidden_size), dtype=dtype, device="npu")
    quant_scale0 = torch.tensor([0.25], dtype=dtype, device="npu")
    quant_offset0 = torch.zeros((1,), dtype=torch.int8, device="npu")
    wdqkv = torch.randint(
        0, 7, (1, 224, 2112, 32), dtype=torch.int8, device="npu"
    )
    wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)
    de_scale0 = torch.rand((2112,), dtype=torch.float32, device="npu")
    bias0 = torch.randint(0, 7, (2112,), dtype=torch.int32, device="npu")
    gamma1 = torch.randn((1536,), dtype=dtype, device="npu")
    beta1 = torch.randn((1536,), dtype=dtype, device="npu")

    quant_scale1 = torch.tensor([0.25], dtype=dtype, device="npu")
    quant_offset1 = torch.zeros((1,), dtype=torch.int8, device="npu")
    wuq = torch.randint(
        0, 7, (1, 48, head_num * 192, 32), dtype=torch.int8, device="npu"
    )
    wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)
    de_scale1 = torch.rand(
        (head_num * 192,), dtype=torch.float32, device="npu"
    )
    bias1 = torch.randint(
        0, 7, (head_num * 192,), dtype=torch.int32, device="npu"
    )
    gamma2 = torch.randn((512,), dtype=dtype, device="npu")
    cos = torch.randn((token_num, 64), dtype=dtype, device="npu")
    sin = torch.randn((token_num, 64), dtype=dtype, device="npu")
    wuk = torch.randn((head_num, 128, 512), dtype=dtype, device="npu")
    wuk = torch_npu.npu_format_cast(wuk, 29)
    slotmapping = torch.arange(token_num, dtype=torch.int32, device="npu")

    ctkv_scale = torch.tensor([1.0], dtype=dtype, device="npu")
    qnope_scale = torch.ones((head_num,), dtype=dtype, device="npu")
    ctkv_shape = (
        (block_num, 512 // 32, block_size, 32)
        if quantized_cache
        else (block_num, block_size, 512)
    )
    rope_shape = (
        (block_num, 64 // 16, block_size, 16)
        if quantized_cache
        else (block_num, block_size, 64)
    )
    ctkv_dtype = torch.int8 if quantized_cache else dtype
    ctkv_fill = -85 if quantized_cache else float("nan")

    def run(stride: int):
        kv_cache, kv_backing = _make_dim0_strided_cache(
            ctkv_shape, stride, ctkv_dtype, ctkv_fill
        )
        kv_cache_rope, rope_backing = _make_dim0_strided_cache(
            rope_shape, stride, dtype, float("nan")
        )
        q_nope_out = torch.full(
            (token_num, head_num, 512),
            ctkv_fill,
            dtype=ctkv_dtype,
            device="npu",
        )
        q_rope_out = torch.full(
            (token_num, head_num, 64), float("nan"), dtype=dtype, device="npu"
        )
        q_down = torch.empty((token_num, 1536), dtype=dtype, device="npu")

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
            cache_mode=cache_mode,
            quant_mode="per_token_quant_symm",
            enable_inner_out=False,
            q_out0=q_nope_out,
            kv_cache_out0=kv_cache,
            q_out1=q_rope_out,
            kv_cache_out1=kv_cache_rope,
            inner_out=q_down,
        )
        torch.npu.synchronize()
        return (
            q_nope_out,
            q_rope_out,
            _cache_to_logical(kv_cache, cache_mode),
            _cache_to_logical(kv_cache_rope, cache_mode),
            kv_backing,
            rope_backing,
        )

    contiguous = run(1)
    noncontiguous = run(stride_factor)

    for contiguous_output, noncontiguous_output in zip(
        contiguous[:4], noncontiguous[:4]
    ):
        torch.testing.assert_close(
            noncontiguous_output,
            contiguous_output,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    _assert_dim0_holes_untouched(
        noncontiguous[4], stride_factor, ctkv_fill
    )
    _assert_dim0_holes_untouched(
        noncontiguous[5], stride_factor, float("nan")
    )
