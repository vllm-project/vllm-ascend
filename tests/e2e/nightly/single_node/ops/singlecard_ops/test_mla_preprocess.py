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


def _nz_memory_to_logical(cache: torch.Tensor, block_size: int, width: int, c0: int) -> torch.Tensor:
    # An NZ cache always holds [block][C1][block_size][C0] in memory, whatever
    # shape it was declared with. Re-read it as logical [block, block_size, width].
    block_num = cache.numel() // (block_size * width)
    return (
        cache.reshape(block_num, width // c0, block_size, c0).permute(0, 2, 1, 3).reshape(block_num, block_size, width)
    )


def _cache_to_logical(cache: torch.Tensor, cache_mode: str) -> torch.Tensor:
    if cache_mode in ("kvcache", "krope_ctkv"):
        return cache
    if cache_mode in ("int8_nzcache", "nzcache"):
        # Physical NZ [block, C1, block_size, C0] -> logical ND
        # [block, block_size, C1 * C0].
        return cache.permute(0, 2, 1, 3).reshape(cache.shape[0], cache.shape[2], -1)
    raise ValueError(f"unsupported cache_mode: {cache_mode}")


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


def _tensor_written(tensor: torch.Tensor, fill_value: float | int) -> bool:
    if tensor.dtype.is_floating_point:
        return bool((~torch.isnan(tensor)).any().item())
    return bool((tensor != fill_value).any().item())


def _build_mode_caches(
    cache_mode: str,
    block_num: int,
    block_size: int,
    kv_lora_rank: int,
    rope_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.dtype, float | int]:
    """Build caches in the layout each cache_mode expects."""
    # ND modes use 3-D shapes; NZ modes use 4-D shapes.
    ctkv_shape: tuple[int, ...]
    rope_shape: tuple[int, ...]
    if cache_mode == "kvcache":
        # Fused ND cache: [block, block_size, kv_lora_rank + rope_dim].
        ctkv_shape = (block_num, block_size, kv_lora_rank + rope_dim)
        rope_shape = (block_num, block_size, rope_dim)
        ctkv_dtype, fill = dtype, float("nan")
    elif cache_mode == "krope_ctkv":
        ctkv_shape = (block_num, block_size, kv_lora_rank)
        rope_shape = (block_num, block_size, rope_dim)
        ctkv_dtype, fill = dtype, float("nan")
    elif cache_mode == "int8_nzcache":
        ctkv_shape = (block_num, kv_lora_rank // 32, block_size, 32)
        rope_shape = (block_num, rope_dim // 16, block_size, 16)
        ctkv_dtype, fill = torch.int8, -85
    elif cache_mode == "nzcache":
        ctkv_shape = (block_num, kv_lora_rank // 16, block_size, 16)
        rope_shape = (block_num, rope_dim // 16, block_size, 16)
        ctkv_dtype, fill = dtype, float("nan")
    else:
        raise ValueError(f"unsupported cache_mode: {cache_mode}")

    kv_cache = torch.full(ctkv_shape, fill, dtype=ctkv_dtype, device="npu")
    kv_cache_rope = torch.full(rope_shape, float("nan"), dtype=dtype, device="npu")
    return kv_cache, kv_cache_rope, ctkv_dtype, fill


# Combinations confirmed to write outputs on the current kernel instantiations.
# (cache_mode, quant_mode, dtype)
SUPPORTED_MODE_COMBOS = [
    pytest.param("kvcache", "per_tensor_quant_asymm", torch.bfloat16, id="cm0_qm0_bf16"),
    pytest.param("kvcache", "per_token_quant_symm", torch.bfloat16, id="cm0_qm1_bf16"),
    pytest.param("kvcache", "per_tensor_quant_asymm", torch.float16, id="cm0_qm0_fp16"),
    pytest.param("krope_ctkv", "per_tensor_quant_asymm", torch.bfloat16, id="cm1_qm0_bf16"),
    pytest.param("krope_ctkv", "per_token_quant_symm", torch.bfloat16, id="cm1_qm1_bf16"),
    pytest.param("krope_ctkv", "no_quant", torch.bfloat16, id="cm1_qm3_bf16"),
    pytest.param("krope_ctkv", "per_tensor_quant_asymm", torch.float16, id="cm1_qm0_fp16"),
    pytest.param("int8_nzcache", "per_token_quant_symm", torch.bfloat16, id="cm2_qm1_bf16"),
    pytest.param("nzcache", "per_tensor_quant_asymm", torch.bfloat16, id="cm3_qm0_bf16"),
    pytest.param("nzcache", "per_token_quant_symm", torch.bfloat16, id="cm3_qm1_bf16"),
]


@pytest.mark.parametrize("cache_mode, quant_mode, dtype", SUPPORTED_MODE_COMBOS)
@pytest.mark.parametrize("enable_rope", [True, False])
@torch.inference_mode()
def test_mla_preprocess_supported_modes(
    cache_mode: str,
    quant_mode: str,
    dtype: torch.dtype,
    enable_rope: bool,
):
    """Smoke-test supported modes with RoPE enabled and disabled."""
    torch.manual_seed(0)
    token_num = 8
    head_num = 2
    hidden_size = 7168
    mm1_out = 2112
    q_lora = 1536
    kv_lora_rank = 512
    rope_dim = 64
    block_num = 2
    block_size = 128
    no_quant = quant_mode == "no_quant"

    hidden_states = torch.randn((token_num, hidden_size), dtype=dtype, device="npu")
    gamma1 = torch.randn((q_lora,), dtype=dtype, device="npu")
    gamma2 = torch.randn((kv_lora_rank,), dtype=dtype, device="npu")
    cos = torch.randn((token_num, rope_dim), dtype=dtype, device="npu")
    sin = torch.randn((token_num, rope_dim), dtype=dtype, device="npu")
    wuk = torch.randn((head_num, 128, kv_lora_rank), dtype=dtype, device="npu")
    slotmapping = torch.arange(token_num, dtype=torch.int32, device="npu")
    ctkv_scale = torch.tensor([1.0], dtype=dtype, device="npu")
    qnope_scale = torch.ones((head_num,), dtype=dtype, device="npu")

    if no_quant:
        wdqkv = torch.randn((1, hidden_size // 16, mm1_out, 16), dtype=dtype, device="npu")
        wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)
        wuq = torch.randn((1, q_lora // 16, head_num * 192, 16), dtype=dtype, device="npu")
        wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)
        beta1 = de_scale0 = de_scale1 = bias0 = bias1 = None
        quant_scale0 = quant_offset0 = quant_scale1 = quant_offset1 = None
    else:
        wuk = torch_npu.npu_format_cast(wuk, 29)
        wdqkv = torch.randint(0, 7, (1, hidden_size // 32, mm1_out, 32), dtype=torch.int8, device="npu")
        wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)
        wuq = torch.randint(0, 7, (1, q_lora // 32, head_num * 192, 32), dtype=torch.int8, device="npu")
        wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)
        beta1 = torch.randn((q_lora,), dtype=dtype, device="npu")
        de_scale0 = torch.rand((mm1_out,), dtype=torch.float32, device="npu")
        de_scale1 = torch.rand((head_num * 192,), dtype=torch.float32, device="npu")
        bias0 = torch.randint(0, 7, (mm1_out,), dtype=torch.int32, device="npu")
        bias1 = torch.randint(0, 7, (head_num * 192,), dtype=torch.int32, device="npu")
        quant_scale0 = torch.tensor([0.25], dtype=dtype, device="npu")
        quant_offset0 = torch.zeros((1,), dtype=torch.int8, device="npu")
        quant_scale1 = torch.tensor([0.25], dtype=dtype, device="npu")
        quant_offset1 = torch.zeros((1,), dtype=torch.int8, device="npu")

    kv_cache, kv_cache_rope, ctkv_dtype, ctkv_fill = _build_mode_caches(
        cache_mode, block_num, block_size, kv_lora_rank, rope_dim, dtype
    )
    q_nope_out = torch.full((token_num, head_num, kv_lora_rank), ctkv_fill, dtype=ctkv_dtype, device="npu")
    q_rope_out = torch.full((token_num, head_num, rope_dim), float("nan"), dtype=dtype, device="npu")
    q_down = torch.empty((token_num, q_lora), dtype=dtype, device="npu")

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
        quant_mode=quant_mode,
        enable_inner_out=False,
        q_out0=q_nope_out,
        kv_cache_out0=kv_cache,
        q_out1=q_rope_out,
        kv_cache_out1=kv_cache_rope,
        inner_out=q_down,
    )
    torch.npu.synchronize()

    assert _tensor_written(q_nope_out, ctkv_fill), "q_nope was not written"
    assert _tensor_written(q_rope_out, float("nan")), "q_rope was not written"
    assert _tensor_written(kv_cache, ctkv_fill), "kv_cache was not written"
    # kvcache fuses rope into kv_cache; other modes write a separate rope cache.
    if cache_mode != "kvcache":
        assert _tensor_written(kv_cache_rope, float("nan")), "kv_cache_rope was not written"

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
    wdqkv = torch.randint(0, 7, (1, 224, 2112, 32), dtype=torch.int8, device="npu")
    wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)
    de_scale0 = torch.rand((2112,), dtype=torch.float32, device="npu")
    bias0 = torch.randint(0, 7, (2112,), dtype=torch.int32, device="npu")
    gamma1 = torch.randn((1536,), dtype=dtype, device="npu")
    beta1 = torch.randn((1536,), dtype=dtype, device="npu")

    quant_scale1 = torch.tensor([0.25], dtype=dtype, device="npu")
    quant_offset1 = torch.zeros((1,), dtype=torch.int8, device="npu")
    wuq = torch.randint(0, 7, (1, 48, head_num * 192, 32), dtype=torch.int8, device="npu")
    wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)
    de_scale1 = torch.rand((head_num * 192,), dtype=torch.float32, device="npu")
    bias1 = torch.randint(0, 7, (head_num * 192,), dtype=torch.int32, device="npu")
    gamma2 = torch.randn((512,), dtype=dtype, device="npu")
    cos = torch.randn((token_num, 64), dtype=dtype, device="npu")
    sin = torch.randn((token_num, 64), dtype=dtype, device="npu")
    wuk = torch.randn((head_num, 128, 512), dtype=dtype, device="npu")
    wuk = torch_npu.npu_format_cast(wuk, 29)
    slotmapping = torch.arange(token_num, dtype=torch.int32, device="npu")

    ctkv_scale = torch.tensor([1.0], dtype=dtype, device="npu")
    qnope_scale = torch.ones((head_num,), dtype=dtype, device="npu")
    ctkv_shape = (block_num, 512 // 32, block_size, 32) if quantized_cache else (block_num, block_size, 512)
    rope_shape = (block_num, 64 // 16, block_size, 16) if quantized_cache else (block_num, block_size, 64)
    ctkv_dtype = torch.int8 if quantized_cache else dtype
    ctkv_fill = -85 if quantized_cache else float("nan")

    def run(stride: int):
        kv_cache, kv_backing = _make_dim0_strided_cache(ctkv_shape, stride, ctkv_dtype, ctkv_fill)
        kv_cache_rope, rope_backing = _make_dim0_strided_cache(rope_shape, stride, dtype, float("nan"))
        q_nope_out = torch.full(
            (token_num, head_num, 512),
            ctkv_fill,
            dtype=ctkv_dtype,
            device="npu",
        )
        q_rope_out = torch.full((token_num, head_num, 64), float("nan"), dtype=dtype, device="npu")
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

    for contiguous_output, noncontiguous_output in zip(contiguous[:4], noncontiguous[:4]):
        torch.testing.assert_close(
            noncontiguous_output,
            contiguous_output,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    _assert_dim0_holes_untouched(noncontiguous[4], stride_factor, ctkv_fill)
    _assert_dim0_holes_untouched(noncontiguous[5], stride_factor, float("nan"))


@torch.inference_mode()
def test_mla_preprocess_nzcache_accepts_logical_cache_shape():
    """nzcache must accept vLLM's [blocks, block_size, num_kv_heads, dim] caches.

    vLLM allocates its MLA caches with that logical shape and lets the kernel
    lay the bytes out as NZ, while the ATB-style harnesses declare the physical
    [blocks, C1, block_size, C0] shape. Both describe the same bytes, so the
    written slots must come out identical.
    """
    torch.manual_seed(0)
    token_num = 200
    head_num = 2
    hidden_size = 7168
    block_num = 2
    block_size = 128
    kv_lora_rank = 512
    rope_dim = 64
    c0 = 16
    dtype = torch.bfloat16

    hidden_states = torch.randn((token_num, hidden_size), dtype=dtype, device="npu")
    quant_scale0 = torch.tensor([0.25], dtype=dtype, device="npu")
    quant_offset0 = torch.zeros((1,), dtype=torch.int8, device="npu")
    wdqkv = torch.randint(0, 7, (1, 224, 2112, 32), dtype=torch.int8, device="npu")
    wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)
    de_scale0 = torch.rand((2112,), dtype=torch.float32, device="npu")
    bias0 = torch.randint(0, 7, (2112,), dtype=torch.int32, device="npu")
    gamma1 = torch.randn((1536,), dtype=dtype, device="npu")
    beta1 = torch.randn((1536,), dtype=dtype, device="npu")
    quant_scale1 = torch.tensor([0.25], dtype=dtype, device="npu")
    quant_offset1 = torch.zeros((1,), dtype=torch.int8, device="npu")
    wuq = torch.randint(0, 7, (1, 48, head_num * 192, 32), dtype=torch.int8, device="npu")
    wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)
    de_scale1 = torch.rand((head_num * 192,), dtype=torch.float32, device="npu")
    bias1 = torch.randint(0, 7, (head_num * 192,), dtype=torch.int32, device="npu")
    gamma2 = torch.randn((512,), dtype=dtype, device="npu")
    cos = torch.randn((token_num, rope_dim), dtype=dtype, device="npu")
    sin = torch.randn((token_num, rope_dim), dtype=dtype, device="npu")
    wuk = torch.randn((head_num, 128, kv_lora_rank), dtype=dtype, device="npu")
    wuk = torch_npu.npu_format_cast(wuk, 29)
    slotmapping = torch.arange(token_num, dtype=torch.int32, device="npu")
    ctkv_scale = torch.tensor([1.0], dtype=dtype, device="npu")
    qnope_scale = torch.ones((head_num,), dtype=dtype, device="npu")

    def run(logical: bool):
        if logical:
            ctkv_shape = (block_num, block_size, 1, kv_lora_rank)
            rope_shape = (block_num, block_size, 1, rope_dim)
        else:
            ctkv_shape = (block_num, kv_lora_rank // c0, block_size, c0)
            rope_shape = (block_num, rope_dim // c0, block_size, c0)
        kv_cache = torch.full(ctkv_shape, float("nan"), dtype=dtype, device="npu")
        kv_cache_rope = torch.full(rope_shape, float("nan"), dtype=dtype, device="npu")
        q_nope_out = torch.full((token_num, head_num, kv_lora_rank), float("nan"), dtype=dtype, device="npu")
        q_rope_out = torch.full((token_num, head_num, rope_dim), float("nan"), dtype=dtype, device="npu")
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
            cache_mode="nzcache",
            quant_mode="per_tensor_quant_asymm",
            enable_inner_out=False,
            q_out0=q_nope_out,
            kv_cache_out0=kv_cache,
            q_out1=q_rope_out,
            kv_cache_out1=kv_cache_rope,
            inner_out=q_down,
        )
        torch.npu.synchronize()
        block_indices = slotmapping.long() // block_size
        block_offsets = slotmapping.long() % block_size
        ctkv = _nz_memory_to_logical(kv_cache, block_size, kv_lora_rank, c0)
        rope = _nz_memory_to_logical(kv_cache_rope, block_size, rope_dim, c0)
        return (
            q_nope_out,
            q_rope_out,
            ctkv[block_indices, block_offsets],
            rope[block_indices, block_offsets],
        )

    physical = run(logical=False)
    logical = run(logical=True)

    for name, expected, actual in zip(("q_nope", "q_rope", "cache_ctkv", "cache_rope"), physical, logical):
        assert not torch.isnan(actual).any(), f"{name} has unwritten slots"
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
