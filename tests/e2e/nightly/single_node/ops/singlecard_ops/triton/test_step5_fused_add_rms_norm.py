# SPDX-License-Identifier: Apache-2.0

from statistics import median

import pytest
import torch
import torch_npu
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.triton.step5.fused_add_rms_norm import (
    can_use_fused_add_rms_norm_cast,
    fused_add_rms_norm_cast,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

pytestmark = pytest.mark.skipif(not (HAS_TRITON and is_950()), reason="Requires Ascend A5 with Triton")

STEP5_RMS_NORM_EPS = 1e-5
STEP5_HIDDEN_SIZE = 4096
# Observed in the Step-5-Preview TP8 "Who are you" prefill trace.
STEP5_PROFILED_PREFILL_TOKENS = 14
BF16_TOLERANCE = 2**-6
MIN_WITHIN_TOLERANCE = 0.99
MAX_ABSOLUTE_ERROR = 1.0
WARMUP_RUNS = 5
CALLS_PER_GRAPH = 100
TIMING_TRIALS = 10


@pytest.fixture(scope="module", autouse=True)
def init_triton_device_properties():
    init_device_properties_triton()


def assert_bf16_output_matches(output, expected):
    assert output.dtype == expected.dtype == torch.bfloat16
    assert output.shape == expected.shape
    expected = expected.float()
    diff = (output.float() - expected).abs()
    assert (diff <= BF16_TOLERANCE * (1 + expected.abs())).float().mean().item() >= MIN_WITHIN_TOLERANCE
    assert diff.max().item() <= MAX_ABSOLUTE_ERROR


@pytest.mark.parametrize(
    "tokens, input_dtype, zero_centered, has_residual",
    [
        pytest.param(STEP5_PROFILED_PREFILL_TOKENS, torch.float32, True, False, id="prefill-no-residual"),
        pytest.param(STEP5_PROFILED_PREFILL_TOKENS, torch.float32, True, True, id="prefill-fp32-with-residual"),
        pytest.param(STEP5_PROFILED_PREFILL_TOKENS, torch.bfloat16, True, True, id="prefill-bf16-with-residual"),
        pytest.param(1, torch.bfloat16, True, True, id="single-token"),
        pytest.param(0, torch.bfloat16, True, True, id="empty-with-residual"),
        pytest.param(0, torch.float32, True, False, id="empty-no-residual"),
        pytest.param(STEP5_PROFILED_PREFILL_TOKENS, torch.float16, False, True, id="fp16-plain-weight"),
    ],
)
def test_fused_add_rms_norm_matches_decoder_path(tokens, input_dtype, zero_centered, has_residual):
    torch.manual_seed(0)
    hidden_states = torch.randn(tokens, STEP5_HIDDEN_SIZE, device="npu", dtype=input_dtype)
    residual = torch.randn_like(hidden_states, dtype=torch.float32) if has_residual else None
    weight = torch.randn(STEP5_HIDDEN_SIZE, device="npu", dtype=torch.float32) * 0.1
    expected_residual = hidden_states if residual is None else hidden_states.float() + residual
    assert can_use_fused_add_rms_norm_cast(hidden_states, residual, weight, fp32_residual=True)
    output, residual_out = fused_add_rms_norm_cast(hidden_states, residual, weight, STEP5_RMS_NORM_EPS, zero_centered)

    torch.testing.assert_close(residual_out, expected_residual, atol=0, rtol=0)
    assert output.dtype == torch.bfloat16
    assert output.shape == hidden_states.shape
    if residual is None:
        assert residual_out is hidden_states
    elif tokens:
        assert residual_out.data_ptr() != residual.data_ptr()
    if tokens == 0:
        return

    assert output.data_ptr() != hidden_states.data_ptr()

    if zero_centered:
        expected, _ = torch_npu.npu_rms_norm(expected_residual, 1.0 + weight, STEP5_RMS_NORM_EPS)
    else:
        inv_rms = torch.rsqrt(expected_residual.square().mean(-1, keepdim=True) + STEP5_RMS_NORM_EPS)
        expected = expected_residual * inv_rms * weight
    assert_bf16_output_matches(output, expected.bfloat16())


def test_fused_add_rms_norm_rejects_inputs_outside_fp32_path():
    shape = (STEP5_PROFILED_PREFILL_TOKENS, STEP5_HIDDEN_SIZE)
    hidden_states = torch.ones(shape, device="npu", dtype=torch.bfloat16)
    residual = torch.ones(shape, device="npu", dtype=torch.float32)
    weight = torch.ones(STEP5_HIDDEN_SIZE, device="npu", dtype=torch.float32)

    assert not can_use_fused_add_rms_norm_cast(hidden_states, None, weight, fp32_residual=True)
    assert not can_use_fused_add_rms_norm_cast(hidden_states, residual, weight, fp32_residual=False)
    assert not can_use_fused_add_rms_norm_cast(hidden_states, residual, weight.bfloat16(), fp32_residual=True)
    assert not can_use_fused_add_rms_norm_cast(hidden_states, residual[:1], weight, fp32_residual=True)
    assert not can_use_fused_add_rms_norm_cast(hidden_states, residual.T.contiguous().T, weight, fp32_residual=True)


@pytest.mark.parametrize("tokens", [1, 4, 16, 128, 512, 2048])
@torch.inference_mode()
def test_fused_add_rms_norm_cast_vs_npu_add_rms_norm(tokens):
    torch.manual_seed(0)
    shape = (tokens, STEP5_HIDDEN_SIZE)
    x = torch.randn(shape, device="npu", dtype=torch.bfloat16)
    residual = torch.randn(shape, device="npu", dtype=torch.float32)
    weight = torch.randn(STEP5_HIDDEN_SIZE, device="npu", dtype=torch.float32) * 0.1
    # Give the baseline a precomputed scale so timing covers only the three requested ops.
    scale = 1.0 + weight

    def baseline():
        normed, _, residual_out = torch_npu.npu_add_rms_norm(x.float(), residual, scale, STEP5_RMS_NORM_EPS)
        return normed.bfloat16(), residual_out

    def fused():
        return fused_add_rms_norm_cast(x, residual, weight, STEP5_RMS_NORM_EPS, zero_centered=True)

    expected, expected_residual = baseline()
    output, residual_out = fused()
    assert_bf16_output_matches(output, expected)
    torch.testing.assert_close(residual_out, expected_residual, atol=0, rtol=0)

    graphs = {}
    for name, run in (("baseline", baseline), ("fused", fused)):
        for _ in range(WARMUP_RUNS):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(CALLS_PER_GRAPH):
                run()
        for _ in range(WARMUP_RUNS):
            graph.replay()
        torch.npu.synchronize()
        graphs[name] = graph

    samples: dict[str, list[float]] = {"baseline": [], "fused": []}
    for trial in range(TIMING_TRIALS):
        order = ("baseline", "fused") if trial % 2 == 0 else ("fused", "baseline")
        for name in order:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / CALLS_PER_GRAPH)

    baseline_us = median(samples["baseline"])
    fused_us = median(samples["fused"])
    print(f"{tokens:4} tokens | Cast + AddRMSNorm + Cast: {baseline_us:.3f} us | fused: {fused_us:.3f} us")
