# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import scipy.linalg
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.attention import indexer as indexer_module
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

ROWS, WIDTH, CAPACITY = 6, 128, 257


def assert_bytes(actual, expected):
    assert torch.equal(
        actual.detach().cpu().contiguous().view(torch.uint8), expected.cpu().contiguous().view(torch.uint8)
    )


def native(x):
    key, scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
    return key, scale.to(torch.float16).unsqueeze(-1)


def cache_pair(capacity=CAPACITY):
    key = torch.randint(-128, 128, (capacity, WIDTH), dtype=torch.int8, device="npu")
    scale = torch.randint(-32768, 32768, (capacity, 1), dtype=torch.int16, device="npu").view(torch.float16)
    return key, scale


def expected_cache(initial, slots, outputs):
    key, scale = [t.cpu().clone() for t in initial]
    for row, slot in enumerate(slots.cpu().tolist()):
        if 0 <= slot < key.shape[0]:
            key[slot] = outputs[0].cpu()[row]
            scale[slot] = outputs[1].cpu()[row]
    return key, scale


def special_inputs(dtype):
    yield torch.randn(ROWS, WIDTH, dtype=dtype)
    yield torch.zeros(ROWS, WIDTH, dtype=dtype)
    yield torch.full((ROWS, WIDTH), -0.0, dtype=dtype)
    yield torch.full((ROWS, WIDTH), torch.finfo(dtype).max, dtype=dtype)
    halfway = torch.zeros(ROWS, WIDTH, dtype=dtype)
    halfway[:, :6] = torch.tensor([127, -127, 63.5, -63.5, 0.5, -0.5], dtype=dtype)
    yield halfway
    for maximum in (1.0, 2.0, 0.125, 128.0):
        # BF16 half-max cases distinguish native vector division from the
        # mathematically similar host reciprocal/rounding sequence.
        halfway = torch.zeros(ROWS, WIDTH, dtype=dtype)
        halfway[:, :4] = torch.tensor([maximum, -maximum, maximum / 2, -maximum / 2], dtype=dtype)
        yield halfway
    for value in [float("nan"), float("inf"), -float("inf"), torch.finfo(dtype).tiny]:
        x = torch.zeros(ROWS, WIDTH, dtype=dtype)
        x[:, 0], x[:, 1] = value, -value
        yield x
    for column in (1, 31, 127):
        x = torch.ones(ROWS, WIDTH, dtype=dtype)
        x[:, column] = float("nan")
        yield x
    # Smallest subnormal, not just the smallest normal returned by finfo.
    yield torch.ones(ROWS, WIDTH, dtype=torch.int16).view(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_outputs_and_every_cache_byte(dtype, slot_dtype):
    torch.manual_seed(42)
    patterns = [
        [0, 1, 2, 3, 4, 5],
        [256, 0, -1, 3, 2, 1],
        [-1] * ROWS,
        [17, 16, 18, 31, 32, 33],
        [257, -2, -1, 0, 256, 32],
    ]
    for host in special_inputs(dtype):
        x = host.to("npu")
        reference = native(x)
        for pattern in patterns:
            slots = torch.tensor(pattern, dtype=slot_dtype, device="npu")
            cache = cache_pair()
            initial = tuple(t.clone() for t in cache)
            outputs = torch.ops._C_ascend.li_quant_cache(x, slots, *cache)
            for actual, expected in zip(outputs, reference):
                assert_bytes(actual, expected)
            for actual, expected in zip(cache, expected_cache(initial, slots, reference)):
                assert_bytes(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_all_finite_input_bit_patterns(dtype, slot_dtype):
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    values = values[torch.isfinite(values)]
    slots = torch.tensor([0, 256, -1, 1, 2, 128], dtype=slot_dtype, device="npu")
    for start in range(0, values.numel(), ROWS * WIDTH):
        host = torch.zeros(ROWS * WIDTH, dtype=dtype)
        part = values[start : start + ROWS * WIDTH]
        host[: part.numel()] = part
        x = host.reshape(ROWS, WIDTH).to("npu")
        reference = native(x)
        cache = cache_pair()
        initial = tuple(t.clone() for t in cache)
        outputs = torch.ops._C_ascend.li_quant_cache(x, slots, *cache)
        for actual, expected in zip(outputs, reference):
            assert_bytes(actual, expected)
        for actual, expected in zip(cache, expected_cache(initial, slots, reference)):
            assert_bytes(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_graph_replay_reads_changed_values_slots_and_allocations(dtype, slot_dtype):
    allocations = []
    for _ in range(2):
        x = torch.zeros(ROWS, WIDTH, dtype=dtype, device="npu")
        slots = torch.arange(ROWS, dtype=slot_dtype, device="npu")
        cache = cache_pair()
        for _ in range(3):
            torch.ops._C_ascend.li_quant_cache(x, slots, *cache)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            output = torch.ops._C_ascend.li_quant_cache(x, slots, *cache)
        allocations.append((x, slots, cache, graph, output))
    inputs = list(special_inputs(dtype))
    for iteration, host in enumerate(inputs):
        for x, slots, cache, graph, output in reversed(allocations) if iteration % 2 else allocations:
            x.copy_(host)
            slots.copy_(torch.tensor([256 - iteration, -1, iteration, 17, 32, 64], dtype=slot_dtype, device="npu"))
            fresh = cache_pair()
            for tensor, initial in zip(cache, fresh):
                tensor.copy_(initial)
            reference = native(x)
            graph.replay()
            for actual, expected in zip(output, reference):
                assert_bytes(actual, expected)
            for actual, expected in zip(cache, expected_cache(fresh, slots, reference)):
                assert_bytes(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_real_capacity_preserves_untouched_bytes(dtype, slot_dtype):
    capacity = 1_127_168
    x = torch.randn(ROWS, WIDTH, dtype=dtype, device="npu")
    slots = torch.tensor([0, 1, capacity - 1, -1, 123456, 654321], dtype=slot_dtype, device="npu")
    cache = cache_pair(capacity)
    initial = tuple(t.clone() for t in cache)
    reference = native(x)
    torch.ops._C_ascend.li_quant_cache(x, slots, *cache)
    for actual, expected in zip(cache, expected_cache(initial, slots, reference)):
        assert_bytes(actual, expected)


def test_meta_output_contract():
    key, scale = torch.ops._C_ascend.li_quant_cache(
        torch.empty(ROWS, WIDTH, dtype=torch.bfloat16, device="meta"),
        torch.empty(ROWS, dtype=torch.int64, device="meta"),
        torch.empty(CAPACITY, WIDTH, dtype=torch.int8, device="meta"),
        torch.empty(CAPACITY, 1, dtype=torch.float16, device="meta"),
    )
    assert key.shape == (ROWS, WIDTH) and key.dtype == torch.int8
    assert scale.shape == (ROWS, 1) and scale.dtype == torch.float16


@pytest.mark.parametrize("shape", [(0, WIDTH), (5, WIDTH), (ROWS, 64)])
def test_meta_rejects_the_same_unsupported_shape_as_npu(shape):
    with pytest.raises(RuntimeError, match=r"requires x\[6,128\]"):
        torch.ops._C_ascend.li_quant_cache(
            torch.empty(shape, dtype=torch.bfloat16, device="meta"),
            torch.empty(ROWS, dtype=torch.int64, device="meta"),
            torch.empty(CAPACITY, WIDTH, dtype=torch.int8, device="meta"),
            torch.empty(CAPACITY, 1, dtype=torch.float16, device="meta"),
        )


@torch.inference_mode()
def test_real_indexer_k_pipeline_matches_native_in_changing_graphs(monkeypatch, record_property):
    """Exercise the actual main backend, including projection, norm and RoPE."""
    init_device_properties_triton()
    torch.manual_seed(73)
    original_fusion = indexer_module.try_li_quant_cache
    fusion_calls = []

    def traced_fusion(*args):
        outputs = original_fusion(*args)
        assert outputs is not None, "The candidate must run the fused op, not its native fallback"
        fusion_calls.append(tuple(args[0].shape))
        return outputs

    monkeypatch.setattr(indexer_module, "try_li_quant_cache", traced_fusion)

    class Projection(torch.nn.Linear):
        def forward(self, values):
            return super().forward(values), None

    hidden = torch.randn(ROWS, WIDTH, dtype=torch.bfloat16, device="npu")
    slots = torch.arange(ROWS, dtype=torch.int32, device="npu")
    backend = SimpleNamespace(
        _use_fused_cache_store=True,
        _use_fused_quant_cache=False,
        head_dim=WIDTH,
        qk_rope_head_dim=64,
        is_rope_neox_style=False,
        enable_sparse_li_quant=True,
        enable_sparse_li_c8=True,
        enable_sparse_li_c4=False,
        c8_k_cache_dtype=torch.int8,
        c8_k_scale_cache_dtype=torch.float16,
        wk_weights_proj=Projection(WIDTH, WIDTH + 8, bias=False, dtype=torch.bfloat16, device="npu"),
        k_norm=torch.nn.LayerNorm(WIDTH, dtype=torch.bfloat16, device="npu"),
        q_hadamard=torch.tensor(scipy.linalg.hadamard(WIDTH), dtype=torch.bfloat16, device="npu") / WIDTH**0.5,
        use_pcp=False,
    )
    backend._quantize_li_tensor = lambda *a, **kw: AscendSFAIndexerBackend._quantize_li_tensor(backend, *a, **kw)
    backend.forward_k = lambda *a, **kw: AscendSFAIndexerBackend.forward_k(backend, *a, **kw)
    backend._gather_cache_inputs = lambda *a: AscendSFAIndexerBackend._gather_cache_inputs(backend, *a)
    metadata = SimpleNamespace(
        cos=torch.ones(ROWS, 64, dtype=torch.bfloat16, device="npu"),
        sin=torch.zeros(ROWS, 64, dtype=torch.bfloat16, device="npu"),
        num_actual_tokens=ROWS,
        num_decode_tokens=ROWS,
        slot_mapping=slots,
    )
    variants = []
    for enabled in (False, True):
        cache = cache_pair(256)
        backend.k_cache = SimpleNamespace(kv_cache=(cache[0].view(2, 128, 1, WIDTH), cache[1].view(2, 128, 1, 1)))
        backend._use_fused_quant_cache = enabled
        for _ in range(3):
            AscendSFAIndexerBackend.forward(backend, hidden, hidden, hidden, metadata, compute_topk=False)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            AscendSFAIndexerBackend.forward(backend, hidden, hidden, hidden, metadata, compute_topk=False)
        variants.append((cache, graph))
    assert fusion_calls and all(shape == (ROWS, WIDTH) for shape in fusion_calls)
    record_property("fused_n6_invocations", len(fusion_calls))
    for iteration in range(8):
        hidden.copy_(torch.randn_like(hidden))
        slots.copy_(torch.tensor([255 - iteration, -1, iteration, 17, 32, 64], dtype=torch.int32, device="npu"))
        fresh = cache_pair(256)
        for cache, graph in variants:
            for tensor, initial in zip(cache, fresh):
                tensor.copy_(initial)
            graph.replay()
        for actual, expected in zip(variants[1][0], variants[0][0]):
            assert_bytes(actual, expected)
