import gc
import socket

import pytest
import torch
import torch_npu

from vllm_ascend.compilation.updatable_graph import UpdatableGraph
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import scale_from_float_to_int64
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
    csrc/mla_preprocess/op_host/mla_preprocess_tiling.cpp), so the layout has to match
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


# Non-default dimensions exercise the shared host/kernel path, not a promise
# that every shape warned about by original MLAPO is supported. FP16 retains
# the original 64-wide RoPE BMM layout; BF16 also checks a 128-wide RoPE.
DIRECT_CASES = [
    (torch.bfloat16, 512, 64),
    (torch.bfloat16, 256, 128),
    (torch.float16, 512, 64),
    (torch.float16, 256, 64),
]


def _assert_same_bits(actual, expected, name):
    actual, expected = actual.cpu().contiguous(), expected.cpu().contiguous()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), name


@pytest.fixture(scope="module")
def mapped_offload():
    # Use the same registered SHARED allocation as the existing equivalence
    # harness. Ordinary CPU tensors are not device-accessible writeback pools.
    from memfabric_hybrid import offload

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    cfg = offload.OffloadConfig()
    cfg.store_url = f"tcp://127.0.0.1:{port}"
    cfg.device_id = torch.npu.current_device()
    cfg.reserve_size = cfg.alloc_size = 1 << 30
    cfg.world_size, cfg.rank_id, cfg.scene = 1, 0, offload.Scene.SHARED
    assert offload.initialize(cfg) == 0
    try:
        yield offload
    finally:
        torch.npu.synchronize()
        gc.collect()
        offload.uninitialize()


@pytest.mark.parametrize(
    "slots_kind,k_base,rope_base,capacity,error",
    [
        ("none", 0, 0, 0, None),
        ("valid", 0, 0, 0, None),
        ("valid", 4096, 8192, 2, None),
        ("none", 4096, 8192, 2, "require writeback_slots"),
        ("dtype", 0, 0, 0, "writeback_slots"),
        ("shape", 0, 0, 0, "writeback_slots"),
        ("noncontiguous", 0, 0, 0, "writeback_slots"),
        ("device", 0, 0, 0, "writeback_slots"),
        ("valid", 0, 8192, 2, "positive"),
        ("valid", 4096, 0, 2, "positive"),
        ("valid", 4096, 8192, 0, "positive"),
        ("valid", -32, 8192, 2, "positive"),
        ("valid", 4096, -32, 2, "positive"),
        ("valid", 4096, 8192, -1, "positive"),
        ("valid", 4097, 8192, 2, "32-byte aligned"),
        ("valid", 2**63 - 32, 8192, 2, "overflows int64"),
        ("valid", 4096, 4128, 2, "must not overlap"),
    ],
)
def test_mla_direct_writeback_meta_contract(slots_kind, k_base, rope_base, capacity, error):
    # Meta calls the same C++ checker as eager without submitting invalid
    # CPU pointers to a device. All-zero bases must not bypass slot validation.
    def tensor(shape):
        return torch.empty(shape, dtype=torch.bfloat16, device="meta")

    slots = torch.empty(2, dtype=torch.int32, device="meta")
    if slots_kind == "none":
        slots = None
    elif slots_kind == "dtype":
        slots = slots.to(torch.int64)
    elif slots_kind == "shape":
        slots = slots.view(2, 1)
    elif slots_kind == "noncontiguous":
        slots = torch.empty((2, 2), dtype=torch.int32, device="meta")[:, 0]
    elif slots_kind == "device":
        slots = torch.empty(2, dtype=torch.int32)
    outputs = tuple(tensor(shape) for shape in ((2, 2, 512), (2, 2, 64), (2, 1536), (2, 512), (2, 64)))

    def run():
        return torch.ops._C_ascend.mla_preprocess_with_kv_offload(
            tensor((2, 7168)),
            tensor((1, 448, 2112, 16)),
            None,
            tensor((1536,)),
            None,
            tensor((1, 96, 384, 16)),
            None,
            tensor((512,)),
            tensor((2, 1, 1, 64)),
            tensor((2, 1, 1, 64)),
            tensor((2, 128, 512)),
            quant_scale0=None,
            quant_offset0=None,
            bias0=None,
            quant_scale1=None,
            quant_offset1=None,
            bias1=None,
            ctkv_scale=None,
            q_nope_scale=None,
            quant_mode="no_quant",
            q_out0=outputs[0],
            q_out1=outputs[1],
            inner_out=outputs[2],
            current_k=outputs[3],
            current_rope=outputs[4],
            writeback_slots=slots,
            writeback_k_base=k_base,
            writeback_rope_base=rope_base,
            writeback_capacity=capacity,
        )

    if error is not None:
        with pytest.raises(RuntimeError, match=error):
            run()
    else:
        result = run()
        assert [value.shape for value in result] == [value.shape for value in outputs]


@pytest.mark.parametrize(
    "cache_mode,dtype,kv_lora_rank,rope_dim",
    [("krope_ctkv", *case) for case in DIRECT_CASES] + [("nzcache", torch.bfloat16, 512, 64)],
)
@pytest.mark.parametrize("enable_rope", [True, False])
@pytest.mark.parametrize("token_num", [1, 6])
@torch.inference_mode()
def test_mla_preprocess_kernel(cache_mode, dtype, kv_lora_rank, rope_dim, enable_rope, token_num, monkeypatch):
    monkeypatch.setattr(torch.npu.config, "allow_internal_format", True)
    _exercise_qdown_case(cache_mode, dtype, kv_lora_rank, rope_dim, enable_rope, token_num)


@pytest.mark.parametrize(
    "dtype,kv_lora_rank,rope_dim,quant_mode",
    [(*case, "per_tensor_quant_asymm") for case in DIRECT_CASES] + [(torch.bfloat16, 512, 64, "per_token_quant_symm")],
)
@pytest.mark.parametrize("enable_rope", [True, False])
@pytest.mark.parametrize("writeback_enabled", [True, False], ids=["writeback", "peer"])
@torch.inference_mode()
def test_mla_direct_mapped_writeback_graph(
    dtype, kv_lora_rank, rope_dim, quant_mode, enable_rope, writeback_enabled, mapped_offload, monkeypatch
):
    """Independent operator checks; the offload manager still registers BF16 KV."""
    monkeypatch.setattr(torch.npu.config, "allow_internal_format", True)
    _exercise_qdown_case(
        "krope_ctkv", dtype, kv_lora_rank, rope_dim, enable_rope, 6, mapped_offload, quant_mode, writeback_enabled
    )


def _exercise_qdown_case(
    cache_mode,
    dtype,
    kv_lora_rank,
    rope_dim,
    enable_rope,
    token_num,
    offload=None,
    quant_mode="per_tensor_quant_asymm",
    writeback_enabled=True,
):
    torch.manual_seed(0)
    head_num, hidden_size, q_lora_rank, qk_nope_head_dim = 2, 7168, 1536, 128
    mm1_out = q_lora_rank + kv_lora_rank + rope_dim
    mm2_out = head_num * (qk_nope_head_dim + rope_dim)
    block_size = 128
    # Bounded values keep both FP16 and BF16 outputs finite. Nonzero offsets,
    # biases and beta also exercise the original asymmetric quantization path.
    hidden = torch.randn(token_num, hidden_size, dtype=dtype, device="npu") * 0.1
    wdqkv = torch.randint(-3, 4, (1, hidden_size // 32, mm1_out, 32), dtype=torch.int8, device="npu")
    wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)
    wuq = torch.randint(-3, 4, (1, q_lora_rank // 32, mm2_out, 32), dtype=torch.int8, device="npu")
    wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)
    cos = torch.rand(token_num, 1, 1, rope_dim, dtype=dtype, device="npu")
    sin = torch.rand_like(cos)
    de_scales = tuple(torch.full((width,), 0.005, dtype=torch.float32, device="npu") for width in (mm1_out, mm2_out))
    if dtype == torch.float16:
        # Match W8A8 FP16 checkpoints: FP32 scale bits are stored in INT64,
        # which the original FP16 matmul reads as uint64_t dequant metadata.
        de_scales = tuple(scale_from_float_to_int64(value) for value in de_scales)
    inputs = (
        hidden,
        wdqkv,
        de_scales[0],
        torch.ones(q_lora_rank, dtype=dtype, device="npu"),
        torch.full((q_lora_rank,), 0.1, dtype=dtype, device="npu"),
        wuq,
        de_scales[1],
        torch.ones(kv_lora_rank, dtype=dtype, device="npu"),
        cos.view(token_num, rope_dim) if enable_rope else None,
        sin.view(token_num, rope_dim) if enable_rope else None,
        torch.randn(head_num, qk_nope_head_dim, kv_lora_rank, dtype=dtype, device="npu") * 0.01,
    )
    kwargs = {
        "quant_scale0": torch.full((1,), 0.25, dtype=dtype, device="npu"),
        "quant_offset0": torch.ones(1, dtype=torch.int8, device="npu"),
        "bias0": torch.ones(mm1_out, dtype=torch.int32, device="npu"),
        "quant_scale1": torch.full((1,), 0.25, dtype=dtype, device="npu"),
        "quant_offset1": torch.ones(1, dtype=torch.int8, device="npu"),
        "bias1": torch.ones(mm2_out, dtype=torch.int32, device="npu"),
        "ctkv_scale": torch.ones(1, dtype=dtype, device="npu"),
        "q_nope_scale": torch.ones(head_num, dtype=dtype, device="npu"),
        "quant_mode": quant_mode,
    }
    caches = _build_mode_caches(cache_mode, 1, block_size, kv_lora_rank, rope_dim, dtype)
    compact_slots = torch.arange(token_num, dtype=torch.int32, device="npu")
    q_outputs = tuple(
        torch.full(shape, float("nan"), dtype=dtype, device="npu")
        for shape in ((token_num, head_num, kv_lora_rank), (token_num, head_num, rope_dim), (token_num, q_lora_rank))
    )

    def original(inner=True):
        torch.ops._C_ascend.mla_preprocess(
            *inputs,
            *caches,
            compact_slots,
            **kwargs,
            cache_mode=cache_mode,
            enable_inner_out=inner,
            q_out0=q_outputs[0],
            kv_cache_out0=caches[0],
            q_out1=q_outputs[1],
            kv_cache_out1=caches[1],
            inner_out=q_outputs[2],
        )

    original()
    torch.npu.synchronize()
    assert all(_tensor_written(value) for value in (*q_outputs, *caches))
    if cache_mode != "krope_ctkv":
        return
    reference = (*q_outputs, caches[0].view(-1, kv_lora_rank)[:token_num], caches[1].view(-1, rope_dim)[:token_num])
    if dtype == torch.float16:
        # The newly exposed q_c output must not alter the original FP16
        # no-inner-output kernel's four previously available outputs.
        before = [reference[i].clone() for i in (0, 1, 3, 4)]
        for index in (0, 1, 3, 4):
            reference[index].fill_(float("nan"))
        original(inner=False)
        torch.npu.synchronize()
        for index, expected in zip((0, 1, 3, 4), before):
            _assert_same_bits(reference[index], expected, f"FP16 original inner on/off output {index}")
        original()
    outputs = tuple(torch.full_like(value, float("nan")) for value in reference)

    direct_inputs = (*inputs[:8], cos if enable_rope else None, sin if enable_rope else None, *inputs[10:])

    def direct(writeback=None, call_inputs=direct_inputs):
        torch.ops._C_ascend.mla_preprocess_with_kv_offload(
            *call_inputs,
            **kwargs,
            q_out0=outputs[0],
            q_out1=outputs[1],
            inner_out=outputs[2],
            current_k=outputs[3],
            current_rope=outputs[4],
            **(writeback or {}),
        )

    def check_outputs():
        torch.npu.synchronize()
        for name, actual, expected in zip(("q_nope", "q_rope", "q_c", "current_k", "current_rope"), outputs, reference):
            _assert_same_bits(actual, expected, name)

    # Keep the external no-slots/2D-RoPE contract, then exercise the production
    # metadata tensors directly with CPU writeback disabled by zero defaults.
    direct(call_inputs=inputs)
    check_outputs()
    direct({"writeback_slots": compact_slots})
    check_outputs()
    if offload is None:
        return

    pools = tuple(
        offload.empty([block_size, width], dtype=dtype, pin_memory=True) for width in (kv_lora_rank, rope_dim)
    )
    slots = torch.empty(token_num, dtype=torch.int32, device="npu")
    writeback = {
        "writeback_slots": slots,
        "writeback_k_base": pools[0].data_ptr() if writeback_enabled else 0,
        "writeback_rope_base": pools[1].data_ptr() if writeback_enabled else 0,
        "writeback_capacity": block_size if writeback_enabled else 0,
    }
    # Read the registered CPU pools through the actual offload API on another
    # stream. Comparing only host reads would not exercise the consumer path.
    readback = tuple(torch.empty(value.shape, dtype=dtype, device="npu") for value in pools)
    src = torch.tensor([value.data_ptr() for value in pools], dtype=torch.int64, device="npu")
    dst = torch.tensor([value.data_ptr() for value in readback], dtype=torch.int64, device="npu")
    sizes = torch.tensor([value.numel() * value.element_size() for value in pools], dtype=torch.int32, device="npu")
    count = torch.tensor(2, dtype=torch.int32, device="npu")
    consumer, done = torch.npu.Stream(), torch.npu.Event()
    slot_ids = []

    def update(iteration):
        torch.manual_seed(100 + iteration)
        hidden.copy_(torch.randn_like(hidden) * 0.1)
        cos.copy_(torch.rand_like(cos))
        sin.copy_(torch.rand_like(sin))
        # Valid, negative and capacity-boundary slots; change destinations on
        # replay while retaining every graph-bound tensor address.
        slot_ids[:] = [3 + iteration, 7 + iteration, -1, 11 + iteration, 15 + iteration, block_size]
        slots.copy_(torch.tensor(slot_ids, dtype=slots.dtype, device=slots.device))
        for pool in pools:
            pool.fill_(-17)
        for value in outputs:
            value.fill_(float("nan"))
        original()
        torch.npu.synchronize()

    def launch():
        direct(writeback)
        producer = torch.npu.current_stream()
        done.record(producer)
        consumer.wait_event(done)
        with torch.npu.stream(consumer):
            assert offload.sparse_copy(src, dst, sizes, count, hidden.device) in (None, 0)
        producer.wait_stream(consumer)

    def check():
        check_outputs()
        for pool, read, value in zip(pools, readback, reference[3:]):
            expected = torch.full_like(pool, -17)
            value = value.cpu()
            for row, slot in enumerate(slot_ids):
                if writeback_enabled and 0 <= slot < block_size:
                    expected[slot] = value[row]
            _assert_same_bits(pool, expected, "CPU pool including untouched slots")
            _assert_same_bits(read, expected, "CPU-to-NPU consumer")

    update(0)
    launch()
    check()
    streams = (torch.npu.Stream(), torch.npu.Stream())
    graphs = []
    for stream in streams:
        graph = UpdatableGraph()
        with torch.npu.graph(graph, stream=stream, capture_error_mode="global"):
            launch()
        graphs.append(graph)
        check()
    prior_q = outputs[0].cpu().clone()
    for iteration in (1, 2, 3):
        update(iteration)
        with torch.npu.stream(streams[iteration % 2]):
            graphs[(iteration + 1) % 2].replay()
        # Reference is a fresh eager original, so two stale captured outputs
        # cannot pass together. No throughput or accuracy dataset is measured.
        check()
        assert not torch.equal(outputs[0].cpu().view(torch.int16), prior_q.view(torch.int16))
        prior_q = outputs[0].cpu().clone()


@torch.inference_mode()
def test_mla_preprocess_whole_batch_matches_split_mm2_tiles():
    """A 257-token call (MM2 with m_loop > 1) must match a 256-token run.

    With more tokens than the MM2 tile size (256) the Cube core picks tile
    indices from the tiling-selected swizzle direction. If the Cube traversal
    order diverges from the Vector core's, the dequantized MM2 outputs
    (q_nope / q_rope) are paired with the wrong tile: q_nope and q_rope are
    corrupted while q_c, which bypasses that pairing, stays correct. A
    256-token call avoids
    the multi-tile traversal, so it serves as the reference for the first 256
    tokens. The weights and per-token inputs are identical across calls, so
    the outputs must match bit-exactly.
    """
    torch.manual_seed(0)
    # The int8 weights must keep their FRACTAL_NZ layout for the support
    # check to pass, which requires allowing internal formats.
    torch.npu.config.allow_internal_format = True
    token_num = 257
    head_num = 64
    hidden_size = 7168
    mm1_out = 2112
    q_lora_rank = 1536
    kv_lora_rank = 512
    rope_dim = 64
    block_num = 4
    block_size = 128
    dtype = torch.bfloat16

    # Small, well-conditioned magnitudes (same spirit as the MM2 accuracy
    # reproducer): large activations would blow up the dequantized outputs to
    # ~1e5 where a single bf16 ULP is ~1e3, drowning the comparison signal.
    hidden_states = torch.randn((token_num, hidden_size), dtype=dtype, device="npu") * 0.1
    quant_scale0 = torch.full((1,), 0.25, dtype=dtype, device="npu")
    quant_offset0 = torch.zeros((1,), dtype=torch.int8, device="npu")
    wdqkv = torch.randint(-3, 4, (1, hidden_size // 32, mm1_out, 32), dtype=torch.int8, device="npu")
    wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)
    de_scale0 = torch.full((mm1_out,), 0.005, dtype=torch.float32, device="npu")
    bias0 = torch.zeros((mm1_out,), dtype=torch.int32, device="npu")
    gamma1 = torch.ones((q_lora_rank,), dtype=dtype, device="npu")
    beta1 = torch.zeros((q_lora_rank,), dtype=dtype, device="npu")
    quant_scale1 = torch.full((1,), 0.25, dtype=dtype, device="npu")
    quant_offset1 = torch.zeros((1,), dtype=torch.int8, device="npu")
    wuq = torch.randint(-3, 4, (1, q_lora_rank // 32, head_num * 192, 32), dtype=torch.int8, device="npu")
    wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)
    de_scale1 = torch.full((head_num * 192,), 0.005, dtype=torch.float32, device="npu")
    bias1 = torch.zeros((head_num * 192,), dtype=torch.int32, device="npu")
    gamma2 = torch.ones((kv_lora_rank,), dtype=dtype, device="npu")
    cos = torch.ones((token_num, rope_dim), dtype=dtype, device="npu")
    sin = torch.zeros((token_num, rope_dim), dtype=dtype, device="npu")
    wuk = torch.randn((head_num, 128, kv_lora_rank), dtype=dtype, device="npu") * 0.01
    ctkv_scale = torch.ones((1,), dtype=dtype, device="npu")
    qnope_scale = torch.ones((head_num,), dtype=dtype, device="npu")

    def run(num_tokens: int, slot_start: int):
        kv_cache = torch.full((block_num, block_size, kv_lora_rank), float("nan"), dtype=dtype, device="npu")
        kv_cache_rope = torch.full((block_num, block_size, rope_dim), float("nan"), dtype=dtype, device="npu")
        q_nope_out = torch.full((num_tokens, head_num, kv_lora_rank), float("nan"), dtype=dtype, device="npu")
        q_rope_out = torch.full((num_tokens, head_num, rope_dim), float("nan"), dtype=dtype, device="npu")
        q_down = torch.empty((num_tokens, q_lora_rank), dtype=dtype, device="npu")
        slotmapping = torch.arange(slot_start, slot_start + num_tokens, dtype=torch.int32, device="npu")

        torch.ops._C_ascend.mla_preprocess(
            hidden_states[:num_tokens],
            wdqkv,
            de_scale0,
            gamma1,
            beta1,
            wuq,
            de_scale1,
            gamma2,
            cos[:num_tokens],
            sin[:num_tokens],
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
            enable_inner_out=True,
            q_out0=q_nope_out,
            kv_cache_out0=kv_cache,
            q_out1=q_rope_out,
            kv_cache_out1=kv_cache_rope,
            inner_out=q_down,
        )
        torch.npu.synchronize()
        return q_nope_out, q_rope_out, q_down

    whole = run(token_num, 0)
    part0 = run(256, 0)

    # The regression signal: the first 256 tokens of the whole-batch run use
    # the multi-tile (m_loop > 1) MM2 traversal, the 256-token run does not.
    # With the Cube/Vector pairing bug they diverge; with the fix they are
    # bit-identical (same per-token tiling path, deterministic). The pairing
    # error corrupts every tile from the second traversal step on, so the
    # first 256 tokens are enough to detect it.
    for name, whole_out, expected in zip(("q_nope", "q_rope", "q_c"), whole[:3], part0):
        assert not torch.isnan(whole_out).any(), f"{name} has unwritten slots"
        torch.testing.assert_close(whole_out[:256], expected, rtol=0, atol=0, msg=name)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
