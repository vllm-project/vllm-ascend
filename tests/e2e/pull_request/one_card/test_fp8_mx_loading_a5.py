# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare chunked checkpoint conversion with the full dense intermediate.

Run on Ascend 950 with the serving torch_npu/CANN bundle::

    pytest -sv tests/e2e/pull_request/one_card/test_fp8_mx_loading_a5.py

Both implementations use the real MX operator and the same checkpoint inputs.
The memory/latency test measures conversion of one weight matrix, including
Python dispatch and final synchronization, not model startup or inference.
"""

from statistics import median
from time import perf_counter

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_config import check_ascend_device_type, is_950
from vllm_ascend.quantization.methods.w8a8.fp8_block import (
    _mx_quantize,
    requantize_block_scales,
    resolve_block_scales,
)

MX_GROUP_SIZE = 32
BENCHMARK_SHAPE = (8192, 8192)
BENCHMARK_BLOCK_SIZE = 128
WARMUP_ITERS = 2
BENCHMARK_REPEATS = 5
BYTES_PER_MIB = 1024**2


@pytest.fixture(scope="module", autouse=True)
def require_a5():
    if not torch.npu.is_available():
        pytest.skip("MXFP8 checkpoint conversion requires Ascend 950")
    check_ascend_device_type()
    if not is_950():
        pytest.skip("MXFP8 checkpoint conversion requires Ascend 950")


def _checkpoint(out_features, in_features, block_n, block_k):
    generator = torch.Generator(device="cpu").manual_seed(73)
    # Exercise all finite E4M3 encodings, including subnormals and signed zero.
    # Exclude the two NaN encodings so equality tests do not mask payload loss.
    bits = torch.randint(0, 256, (out_features, in_features), dtype=torch.uint8, generator=generator)
    bits[(bits & 0x7F) == 0x7F] = 0
    # Include a complete all-zero MX group, independently of checkpoint blocks.
    bits[0, :MX_GROUP_SIZE] = 0
    weight = bits.view(torch.float8_e4m3fn).npu()
    scale_shape = ((out_features + block_n - 1) // block_n, (in_features + block_k - 1) // block_k)
    # Non-power-of-two FP32 scales exercise rounding to the model dtype before
    # requantization. This range also keeps the resolved values finite in FP16.
    scales = torch.exp2(torch.rand(scale_shape, generator=generator) * 12 - 6).npu()
    return weight, scales


def _full_matrix_conversion(weight, scales, block_n, block_k, dtype, scale_alg):
    # The former linear loader path, retaining the complete model-dtype matrix
    # while the native MX operator allocates its quantized weights and scales.
    resolved = resolve_block_scales(weight, scales, block_n, block_k, dtype)
    quantized, mx_scale = _mx_quantize(resolved, scale_alg)
    return quantized, mx_scale.view(torch.uint8).reshape(weight.shape[0], -1)


def _cpu_bytes(outputs):
    return tuple(tensor.view(torch.uint8).cpu() for tensor in outputs)


def _assert_byte_identical(actual, expected):
    for name, observed, reference in zip(("weights", "scales"), actual, expected):
        assert observed.shape == reference.shape, name
        assert observed.dtype == reference.dtype == torch.uint8, name
        assert torch.equal(observed, reference), f"MXFP8 {name} differ from full-matrix conversion"


@pytest.mark.e2e_coverage(
    arch="",
    feature="",
    parallel="",
    deploy="",
    hardware="A5",
    quantization="",
    graph_mode="eager",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("scale_alg", [0, 1])
@pytest.mark.parametrize(
    "out_features,in_features,block_n,block_k",
    [
        pytest.param(128, 256, 128, 128, id="aligned"),
        pytest.param(129, 160, 128, 128, id="partial-row-and-column"),
        pytest.param(1121, 96, 96, 128, id="odd-mx-groups-block-aligned-chunks"),
        pytest.param(2049, 4160, 128, 128, id="multiple-chunks-and-tail"),
        pytest.param(1537, 2048, 1536, 128, id="checkpoint-block-larger-than-chunk-target"),
    ],
)
@torch.inference_mode()
def test_fp8_mx_loading_byte_identical_a5(out_features, in_features, block_n, block_k, dtype, scale_alg):
    weight, scales = _checkpoint(out_features, in_features, block_n, block_k)
    expected = _cpu_bytes(_full_matrix_conversion(weight, scales, block_n, block_k, dtype, scale_alg))
    actual = requantize_block_scales(weight, scales, block_n, block_k, dtype, scale_alg)
    assert actual[0].dtype == torch.float8_e4m3fn
    assert actual[1].dtype == torch.uint8
    actual_bytes = _cpu_bytes(actual)
    _assert_byte_identical(actual_bytes, expected)

    num_groups = in_features // MX_GROUP_SIZE
    # Native scales are paired, including zero padding when K/32 is odd.
    assert actual[1].shape == (out_features, (num_groups + 1) // 2 * 2)
    if num_groups % 2:
        assert torch.count_nonzero(actual_bytes[1][:, -1]).item() == 0


def _measure_conversion(convert):
    wall_times_ms = []
    peak_bytes = []
    for _ in range(BENCHMARK_REPEATS):
        torch.npu.synchronize()
        baseline_bytes = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        started = perf_counter()
        outputs = convert()
        torch.npu.synchronize()
        wall_times_ms.append((perf_counter() - started) * 1000)
        peak_bytes.append(torch.npu.max_memory_allocated() - baseline_bytes)
        del outputs
    # Output validation and the CPU copy are deliberately outside timing and
    # peak-memory sampling. No previous NPU result survives into the next run.
    outputs_cpu = _cpu_bytes(convert())
    return median(wall_times_ms), max(peak_bytes), outputs_cpu


@pytest.mark.e2e_coverage(
    arch="",
    feature="",
    parallel="",
    deploy="",
    hardware="A5",
    quantization="",
    graph_mode="eager",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@torch.inference_mode()
def test_fp8_mx_loading_memory_and_latency_a5(dtype, record_property):
    weight, scales = _checkpoint(*BENCHMARK_SHAPE, BENCHMARK_BLOCK_SIZE, BENCHMARK_BLOCK_SIZE)

    def baseline():
        return _full_matrix_conversion(weight, scales, BENCHMARK_BLOCK_SIZE, BENCHMARK_BLOCK_SIZE, dtype, 0)

    def chunked():
        return requantize_block_scales(weight, scales, BENCHMARK_BLOCK_SIZE, BENCHMARK_BLOCK_SIZE, dtype, 0)

    # Warm up both paths on the same inputs before resetting allocator peaks.
    # Report live allocated bytes rather than allocator-reserved pool capacity.
    for _ in range(WARMUP_ITERS):
        for convert in (baseline, chunked):
            outputs = convert()
            torch.npu.synchronize()
            del outputs

    baseline_ms, baseline_peak, expected = _measure_conversion(baseline)
    chunked_ms, chunked_peak, actual = _measure_conversion(chunked)
    _assert_byte_identical(actual, expected)
    record_property("device", torch.npu.get_device_name())
    record_property("torch_version", str(torch.__version__))
    record_property("torch_npu_version", str(torch_npu.__version__))
    record_property("dtype", str(dtype))
    record_property("weight_shape", str(BENCHMARK_SHAPE))
    record_property("baseline_wall_ms", baseline_ms)
    record_property("chunked_wall_ms", chunked_ms)
    record_property("baseline_peak_additional_bytes", baseline_peak)
    record_property("chunked_peak_additional_bytes", chunked_peak)
    print(
        f"shape={BENCHMARK_SHAPE}, dtype={dtype}: "
        f"full={baseline_ms:.3f} ms / {baseline_peak / BYTES_PER_MIB:.2f} MiB, "
        f"chunked={chunked_ms:.3f} ms / {chunked_peak / BYTES_PER_MIB:.2f} MiB "
        "(median synchronized wall time / peak additional allocated memory)"
    )
    assert baseline_ms > 0 and chunked_ms > 0
    # The optimization promises lower peak allocation for large weights. Do
    # not impose a speed threshold: extra launches can trade latency for memory.
    assert chunked_peak < baseline_peak, "Chunked conversion did not reduce peak allocated memory"
