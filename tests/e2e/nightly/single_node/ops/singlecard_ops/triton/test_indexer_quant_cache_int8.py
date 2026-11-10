# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 correctness/graph checks; execute this file directly for a microbenchmark."""

import argparse
import importlib
import json
import statistics

import pytest
import torch
import torch_npu
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton import indexer_quant_cache_int8 as kernels
from vllm_ascend.ops.triton.indexer_quant_cache_int8 import indexer_quant_cache_int8

pytestmark = pytest.mark.skipif(
    not get_current_hardware_profile().supports(HardwareCapability.NATIVE_INT8_SPARSE), reason="A3 native INT8 indexer"
)


def make_cache(paired=True):
    pages, page_size, head_dim = 8, 128, 128
    if paired:
        # Nonzero storage offset, per-page padding and guards around both planes.
        prefix, gap = 32, 64
        key_bytes = page_size * head_dim
        page_bytes = key_bytes + gap + page_size * 2 + gap
        backing = torch.full((prefix + pages * page_bytes + gap,), 17, dtype=torch.uint8, device="npu")
        key = backing.view(torch.int8).as_strided(
            (pages, page_size, 1, head_dim), (page_bytes, head_dim, head_dim, 1), prefix
        )
        scale = backing.view(torch.float16).as_strided(
            (pages, page_size, 1, 1), (page_bytes // 2, 1, 1, 1), (prefix + key_bytes + gap) // 2
        )
        return key, scale, backing
    key = torch.full((pages, page_size, 1, head_dim), 17, dtype=torch.int8, device="npu")
    scale = torch.full((pages, page_size, 1, 1), 0.25, dtype=torch.float16, device="npu")
    return key, scale, None


def make_inputs(tokens, slot_dtype=torch.int64, strided=False, seed=7):
    generator = torch.Generator().manual_seed(seed)
    width = 256 if strided else 128
    x = torch.randn(tokens, width, generator=generator).bfloat16().npu()
    if strided:
        x = x[:, ::2]
    # Zero, near rounding boundaries, very small and large finite BF16 values.
    if tokens >= 4:
        special = torch.zeros(4, 128)
        special[1] = torch.arange(128) - 63.5
        special[1, -1] = 127.0
        special[2] = torch.linspace(-1e-7, 1e-7, 128)
        special[3] = torch.linspace(-1e4, 1e4, 128)
        x[:4].copy_(special.bfloat16().npu())
    # Interleave consecutive destination slots across row tiles. This makes
    # neighboring FP16 scale writes originate from different input row groups.
    slots = torch.arange(tokens, dtype=slot_dtype) + 119
    slots = slots[torch.randperm(tokens, generator=generator)]
    if tokens >= 3:
        slots[-2:] = -1
    if strided:
        storage = torch.full((tokens * 2,), -1, dtype=slot_dtype, device="npu")
        device_slots = storage[::2]
        device_slots.copy_(slots)
    else:
        device_slots = slots.npu()
    return x, device_slots, slots


def assert_matches_native(cache, x, host_slots, initial):
    key, scale, backing = cache
    quantized, fp32_scale = torch_npu.npu_dynamic_quant(x.contiguous(), dst_type=torch.int8)
    quantized, fp16_scale = quantized.cpu(), fp32_scale.half().cpu()
    if backing is None:
        expected_key, expected_scale = (value.clone() for value in initial)
    else:
        expected_backing = initial.clone()
        expected_key = expected_backing.view(torch.int8).as_strided(key.shape, key.stride(), key.storage_offset())
        expected_scale = expected_backing.view(torch.float16).as_strided(
            scale.shape, scale.stride(), scale.storage_offset()
        )
    for row, slot in enumerate(host_slots.tolist()):
        if 0 <= slot < key.shape[0] * key.shape[1]:
            page, token = divmod(slot, key.shape[1])
            expected_key[page, token, 0].copy_(quantized[row])
            expected_scale[page, token, 0, 0] = fp16_scale[row]
    torch.testing.assert_close(key.cpu(), expected_key, rtol=0, atol=0)
    torch.testing.assert_close(scale.cpu(), expected_scale, rtol=0, atol=0)
    if backing is not None:
        # Includes untouched slots, both planes, page gaps and allocation guards.
        torch.testing.assert_close(backing.cpu(), expected_backing, rtol=0, atol=0)


def snapshot(cache):
    key, scale, backing = cache
    return backing.cpu().clone() if backing is not None else (key.cpu().clone(), scale.cpu().clone())


@pytest.mark.parametrize("tokens", [1, 3, 16, 17, 48, 127, 128])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_native_quantization_and_cache_contents(tokens, paired, slot_dtype, strided):
    cache = make_cache(paired)
    x, slots, host_slots = make_inputs(tokens, slot_dtype, strided)
    initial = snapshot(cache)
    for _ in range(5):
        indexer_quant_cache_int8(cache[0], cache[1], x, slots)
        assert_matches_native(cache, x, host_slots, initial)


@pytest.mark.parametrize("tokens", [1, 3, 16, 17, 48, 127, 128])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_graph_replay_refreshes_inputs_and_slots(tokens, slot_dtype):
    cache = make_cache()
    x, slots, _ = make_inputs(tokens, slot_dtype)
    indexer_quant_cache_int8(cache[0], cache[1], x, slots)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    # Explicit GLOBAL capture catches per-launch synchronous H2D lock
    # initialization even after an eager warmup successfully ran the kernel.
    with torch.npu.graph(graph, capture_error_mode="global"):
        for _ in range(3):
            indexer_quant_cache_int8(cache[0], cache[1], x, slots)
    for seed in range(10, 14):
        next_x, _, next_slots = make_inputs(tokens, slot_dtype, seed=seed)
        valid = next_slots >= 0
        next_slots[valid] += (seed - 9) * 128
        initial = snapshot(cache)
        x.copy_(next_x)
        slots.copy_(next_slots.npu())
        graph.replay()
        assert_matches_native(cache, x, next_slots, initial)


@pytest.mark.parametrize("tokens", [1, 48, 128])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_compiler_does_not_allocate_cross_block_locks(monkeypatch, tokens, strided):
    """Detect lock-generating store lowering before testing graph capture."""
    compiled = []
    original_run = kernels._indexer_quant_cache_int8_kernel.run

    def record_kernel(*args, **kwargs):
        result = original_run(*args, **kwargs)
        compiled.append(result)
        return result

    monkeypatch.setattr(kernels._indexer_quant_cache_int8_kernel, "run", record_kernel)
    cache = make_cache()
    x, slots, _ = make_inputs(tokens, strided=strided)
    indexer_quant_cache_int8(cache[0], cache[1], x, slots)
    torch.npu.synchronize()
    assert len(compiled) == 1
    metadata = compiled[0].metadata
    assert metadata.sync_block_lock_layout == 0, "Cache writes must not require launcher lock initialization"
    # Some older compiler versions expose the initialization value separately.
    assert getattr(metadata, "lock_init_val", 0) == 0
    assert getattr(metadata, "lock_init_value", 0) == 0


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_whole_block_writes_preserve_holes_and_page_boundaries(slot_dtype):
    cache = make_cache()
    x, slots, host_slots = make_inputs(48, slot_dtype)
    # Touch both sides of page boundaries, several lanes of one DMA block, and
    # the last cache block. Leave holes in every block that must remain intact.
    host_slots[:] = -1
    selected = torch.tensor([0, 7, 15, 16, 31, 127, 128, 129, 143, 255, 256, 1023], dtype=slot_dtype)
    host_slots[::4] = selected.flip(0)
    slots.copy_(host_slots.npu())
    initial = snapshot(cache)
    for _ in range(5):
        indexer_quant_cache_int8(cache[0], cache[1], x, slots)
    assert_matches_native(cache, x, host_slots, initial)


@torch.inference_mode()
def test_invalid_and_empty_slots_do_not_write():
    cache = make_cache()
    x, slots, _ = make_inputs(48)
    initial = snapshot(cache)
    slots.fill_(-1)
    indexer_quant_cache_int8(cache[0], cache[1], x, slots)
    slots.fill_(cache[0].shape[0] * cache[0].shape[1])
    indexer_quant_cache_int8(cache[0], cache[1], x, slots)
    indexer_quant_cache_int8(cache[0], cache[1], x[:0], slots[:0])
    torch.testing.assert_close(cache[2].cpu(), initial, rtol=0, atol=0)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_registry_warmup_prevents_runtime_compilation(paired, slot_dtype, monkeypatch):
    """Run alone with a fresh TRITON_CACHE_DIR to test cold compile-only startup."""
    cache = make_cache(paired)
    initial = snapshot(cache)
    registry = JitWarmupRegistry(None)
    with registry.activate():
        kernels.register_indexer_quant_cache_int8_warmup(cache[0], cache[1])
    registry.warmup()
    after_warmup = snapshot(cache)
    torch.testing.assert_close(after_warmup, initial, rtol=0, atol=0)

    def forbid_compile(*args, **kwargs):
        raise AssertionError("INT8 indexer compiled after registry warmup")

    for name in ("triton", "triton.compiler", "triton.compiler.compiler", "triton.runtime.jit"):
        module = importlib.import_module(name)
        if callable(getattr(module, "compile", None)):
            monkeypatch.setattr(module, "compile", forbid_compile)
    for count in range(1, 129):
        x, slots, host_slots = make_inputs(count, slot_dtype)
        for offset in (0, 1):
            storage = torch.empty(count + offset, dtype=slot_dtype, device="npu")
            mapped_slots = storage[offset:]
            mapped_slots.copy_(slots)
            initial = snapshot(cache)
            indexer_quant_cache_int8(cache[0], cache[1], x, mapped_slots)
            assert_matches_native(cache, x, host_slots, initial)

    for count in (1, 17, 65, 128):
        x, slots, host_slots = make_inputs(count, slot_dtype)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, capture_error_mode="global"):
            indexer_quant_cache_int8(cache[0], cache[1], x, slots)
        next_x, _, next_slots = make_inputs(count, slot_dtype, seed=11)
        next_slots[next_slots >= 0] += 128
        x.copy_(next_x)
        slots.copy_(next_slots.npu())
        initial = snapshot(cache)
        graph.replay()
        assert_matches_native(cache, x, next_slots, initial)

    initial = snapshot(cache)
    indexer_quant_cache_int8(cache[0], cache[1], x[:0], slots[:0])
    torch.testing.assert_close(snapshot(cache), initial, rtol=0, atol=0)
    assert not kernels.supports_indexer_quant_cache_int8(cache[0], cache[1], 129)


@torch.inference_mode()
def benchmark(tokens, trials):
    """Measure separate graphs to exclude Python launch gaps and JIT warmup."""
    assert get_current_hardware_profile().supports(HardwareCapability.NATIVE_INT8_SPARSE)
    x, slots, host_slots = make_inputs(tokens, slot_dtype=torch.int32)
    baseline, candidate = make_cache(), make_cache()
    cache_scatter_indices = torch.stack((slots // baseline[0].shape[1], slots % baseline[0].shape[1]), dim=-1)

    def split():
        k, scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
        scale = scale.half()
        AscendSFAIndexerBackend._write_cache_tensor(baseline[0], k, slots, cache_scatter_indices)
        AscendSFAIndexerBackend._write_cache_tensor(baseline[1], scale.unsqueeze(-1), slots, cache_scatter_indices)

    def fused():
        indexer_quant_cache_int8(candidate[0], candidate[1], x, slots)

    initial = snapshot(candidate)
    split()
    fused()
    assert_matches_native(candidate, x, host_slots, initial)
    torch.testing.assert_close(candidate[2].cpu(), baseline[2].cpu(), rtol=0, atol=0)
    repeats = 100
    graphs = {}
    for name, fn in (("split", split), ("fused", fused)):
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(repeats):
                fn()
        for _ in range(5):
            graph.replay()
        graphs[name] = graph
    samples: dict[str, list[float]] = {name: [] for name in graphs}
    for trial in range(trials):
        names = ("split", "fused") if trial % 2 == 0 else ("fused", "split")
        for name in names:
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / repeats)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    print(json.dumps({"tokens": tokens, "trials": trials, "median_us": medians, "samples_us": samples}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, choices=(1, 3, 16, 48, 128), default=48)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    benchmark(args.tokens, args.trials)
