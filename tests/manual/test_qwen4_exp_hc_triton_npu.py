# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Standalone NPU correctness checks for Qwen4Exp HC Triton kernels."""

from __future__ import annotations

import argparse
import time

import torch
import torch_npu  # noqa: F401
from vllm.models.qwen4_exp.amd import hyperconnection as upstream_hc

import vllm_ascend.models.qwen4_exp.model  # noqa: F401
from vllm_ascend.ops.triton.qwen4_exp.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)

EPS = 1e-6
HC_COUNT = 4
HIDDEN_SIZE = 2560
HYPER_HIDDEN_SIZE = HC_COUNT * HIDDEN_SIZE
LOWRANK_SIZE = 320
DTYPE = torch.bfloat16
ATOL = 3e-2
RTOL = 3e-2


def verify_model_binding() -> None:
    functions = {
        "grouped_gemma_rmsnorm": grouped_gemma_rmsnorm,
        "hc_silu": hc_silu,
        "hc_gate_mix": hc_gate_mix,
        "hc_combine": hc_combine,
        "hc_combine_norm": hc_combine_norm,
    }
    for name, function in functions.items():
        if getattr(upstream_hc, name) is not function:
            raise AssertionError(f"Qwen4Exp model did not bind Ascend {name}")

    op_names = (
        "qwen4_exp_ascend_grouped_gemma_rmsnorm",
        "qwen4_exp_ascend_hc_silu",
        "qwen4_exp_ascend_hc_gate_mix",
        "qwen4_exp_ascend_hc_combine",
        "qwen4_exp_ascend_hc_combine_norm",
    )
    for op_name in op_names:
        if not hasattr(torch.ops.vllm, op_name):
            raise AssertionError(f"missing custom op vllm::{op_name}")
    print("PASS Qwen4Exp model binding and custom-op registration")


def reference_grouped_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, groups: int) -> torch.Tensor:
    grouped = x.float().view(x.shape[0], groups, x.shape[1] // groups)
    normalized = grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
    if weight.numel() == x.shape[1] // groups:
        affine = weight.float().view(1, 1, -1)
    else:
        affine = weight.float().view(1, groups, -1)
    return (normalized * (1.0 + affine)).flatten(1).to(x.dtype)


def reference_silu(x: torch.Tensor, groups: int) -> torch.Tensor:
    values = x.float() / groups
    return (values * torch.sigmoid(values)).to(x.dtype)


def reference_gate_mix(x: torch.Tensor, gate: torch.Tensor, groups: int) -> torch.Tensor:
    grouped_x = x.float().view(x.shape[0], groups, x.shape[1] // groups)
    grouped_gate = gate.float().view_as(grouped_x)
    return (grouped_x * torch.sigmoid(grouped_gate)).mean(dim=1).to(x.dtype)


def reference_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    groups: int,
) -> torch.Tensor:
    grouped = residual.float().view(residual.shape[0], groups, residual.shape[1] // groups)
    block = block_output.float().unsqueeze(1)
    if injection_logits is not None:
        block = block * (2.0 * torch.sigmoid(injection_logits.float() / groups)).unsqueeze(-1)
    return (grouped + block).flatten(1).to(residual.dtype)


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.npu.synchronize()
    actual_cpu = actual.float().cpu()
    expected_cpu = expected.float().cpu()
    difference = (actual_cpu - expected_cpu).abs()
    max_abs = difference.max().item() if difference.numel() else 0.0
    denominator = expected_cpu.abs().clamp_min(1e-6)
    max_rel = (difference / denominator).max().item() if difference.numel() else 0.0
    torch.testing.assert_close(actual_cpu, expected_cpu, atol=ATOL, rtol=RTOL)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"{name}: metadata mismatch actual={actual.shape}/{actual.dtype}, "
            f"expected={expected.shape}/{expected.dtype}"
        )
    print(f"PASS {name:<42} shape={tuple(actual.shape)!s:<16} max_abs={max_abs:.6g} max_rel={max_rel:.6g}")


def make_input(rows: int, width: int, *, padded: bool = False) -> torch.Tensor:
    if not padded:
        return torch.randn((rows, width), device="npu", dtype=DTYPE) * 0.5
    backing = torch.randn((rows, width + 16), device="npu", dtype=DTYPE) * 0.5
    result = backing[:, :width]
    assert result.stride(1) == 1 and not result.is_contiguous()
    return result


def run_correctness(rows: int, *, padded_input: bool) -> None:
    suffix = f"M={rows},row_strided={padded_input}"
    residual = make_input(rows, HYPER_HIDDEN_SIZE, padded=padded_input)
    block_output = make_input(rows, HIDDEN_SIZE, padded=padded_input)
    injection = make_input(rows, HC_COUNT, padded=padded_input)
    gate = make_input(rows, HYPER_HIDDEN_SIZE, padded=padded_input)
    lowrank = make_input(rows, LOWRANK_SIZE, padded=padded_input)
    shared_weight = torch.randn(HIDDEN_SIZE, device="npu", dtype=DTYPE) * 0.1
    branch_weight = torch.randn(HYPER_HIDDEN_SIZE, device="npu", dtype=DTYPE) * 0.1

    for weight_name, weight in (
        ("shared_weight", shared_weight),
        ("branch_weight", branch_weight),
    ):
        expected = reference_grouped_norm(residual, weight, EPS, HC_COUNT)
        actual = grouped_gemma_rmsnorm(residual, weight, EPS, HC_COUNT)
        assert_close(f"grouped_gemma_rmsnorm/{weight_name}/{suffix}", actual, expected)

    assert_close(
        f"hc_silu/{suffix}",
        hc_silu(lowrank, HC_COUNT),
        reference_silu(lowrank, HC_COUNT),
    )
    assert_close(
        f"hc_gate_mix/{suffix}",
        hc_gate_mix(residual, gate, HC_COUNT),
        reference_gate_mix(residual, gate, HC_COUNT),
    )

    for injection_name, injection_value in (
        ("with_injection", injection),
        ("without_injection", None),
    ):
        expected_combined = reference_combine(residual, block_output, injection_value, HC_COUNT)
        actual_combined = hc_combine(residual, block_output, injection_value, HC_COUNT)
        assert_close(
            f"hc_combine/{injection_name}/{suffix}",
            actual_combined,
            expected_combined,
        )

        for weight_name, weight in (
            ("shared_weight", shared_weight),
            ("branch_weight", branch_weight),
        ):
            expected_normalized = reference_grouped_norm(expected_combined, weight, EPS, HC_COUNT)
            actual_combined, actual_normalized = hc_combine_norm(
                residual,
                block_output,
                injection_value,
                weight,
                EPS,
                HC_COUNT,
            )
            assert_close(
                f"hc_combine_norm.combine/{injection_name}/{weight_name}/{suffix}",
                actual_combined,
                expected_combined,
            )
            assert_close(
                f"hc_combine_norm.norm/{injection_name}/{weight_name}/{suffix}",
                actual_normalized,
                expected_normalized,
            )


def verify_fullgraph_capture() -> None:
    rows = 2
    residual = make_input(rows, HYPER_HIDDEN_SIZE)
    block_output = make_input(rows, HIDDEN_SIZE)
    injection = make_input(rows, HC_COUNT)
    gate = make_input(rows, HYPER_HIDDEN_SIZE)
    lowrank = make_input(rows, LOWRANK_SIZE)
    weight = torch.randn(HIDDEN_SIZE, device="npu", dtype=DTYPE) * 0.1

    def full_hc_path(
        residual: torch.Tensor,
        gate: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor,
        weight: torch.Tensor,
        lowrank: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        normalized = upstream_hc.grouped_gemma_rmsnorm(residual, weight, EPS, HC_COUNT)
        activated = upstream_hc.hc_silu(lowrank, HC_COUNT)
        mixed = upstream_hc.hc_gate_mix(residual, gate, HC_COUNT)
        combined = upstream_hc.hc_combine(residual, block_output, injection, HC_COUNT)
        fused_combined, fused_norm = upstream_hc.hc_combine_norm(
            residual,
            block_output,
            injection,
            weight,
            EPS,
            HC_COUNT,
        )
        return (
            normalized,
            activated,
            mixed,
            combined,
            fused_combined,
            fused_norm,
        )

    expected = full_hc_path(residual, gate, block_output, injection, weight, lowrank)
    captured_targets: list[str] = []

    def inspect_backend(graph_module, example_inputs):
        del example_inputs
        captured_targets.extend(str(node.target) for node in graph_module.graph.nodes if node.op == "call_function")
        return graph_module.forward

    compiled = torch.compile(full_hc_path, backend=inspect_backend, fullgraph=True)
    actual = compiled(residual, gate, block_output, injection, weight, lowrank)
    torch.npu.synchronize()
    for compiled_output, eager_output in zip(actual, expected):
        torch.testing.assert_close(compiled_output, eager_output, atol=0, rtol=0)

    expected_targets = {
        "vllm.qwen4_exp_ascend_grouped_gemma_rmsnorm",
        "vllm.qwen4_exp_ascend_hc_silu",
        "vllm.qwen4_exp_ascend_hc_gate_mix",
        "vllm.qwen4_exp_ascend_hc_combine",
        "vllm.qwen4_exp_ascend_hc_combine_norm",
    }
    captured = {target for target in captured_targets if "qwen4_exp_ascend" in target}
    if not expected_targets.issubset(captured):
        raise AssertionError(f"fullgraph is missing custom ops: {expected_targets - captured}")
    print("PASS torch.compile fullgraph captured all five Ascend HC custom ops")


def benchmark() -> None:
    rows = 32
    residual = make_input(rows, HYPER_HIDDEN_SIZE)
    block_output = make_input(rows, HIDDEN_SIZE)
    injection = make_input(rows, HC_COUNT)
    gate = make_input(rows, HYPER_HIDDEN_SIZE)
    lowrank = make_input(rows, LOWRANK_SIZE)
    weight = torch.randn(HYPER_HIDDEN_SIZE, device="npu", dtype=DTYPE) * 0.1
    cases = {
        "grouped_gemma_rmsnorm": lambda: grouped_gemma_rmsnorm(residual, weight, EPS, HC_COUNT),
        "hc_silu": lambda: hc_silu(lowrank, HC_COUNT),
        "hc_gate_mix": lambda: hc_gate_mix(residual, gate, HC_COUNT),
        "hc_combine": lambda: hc_combine(residual, block_output, injection, HC_COUNT),
        "hc_combine_norm": lambda: hc_combine_norm(residual, block_output, injection, weight, EPS, HC_COUNT),
    }
    for name, function in cases.items():
        for _ in range(5):
            function()
        torch.npu.synchronize()
        start = time.perf_counter()
        iterations = 20
        for _ in range(iterations):
            function()
        torch.npu.synchronize()
        elapsed_us = (time.perf_counter() - start) * 1e6 / iterations
        print(f"BENCH {name:<28} M={rows} average={elapsed_us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if not torch.npu.is_available():
        raise RuntimeError("No NPU is available")
    torch.npu.set_device(0)
    torch.manual_seed(20261010)
    verify_model_binding()
    print(
        "device=",
        torch.npu.get_device_name(0),
        "dtype=",
        DTYPE,
        "hidden_size=",
        HIDDEN_SIZE,
        "hc_count=",
        HC_COUNT,
    )
    for rows, padded in ((1, False), (17, False), (7, True), (257, False)):
        run_correctness(rows, padded_input=padded)
    verify_fullgraph_capture()
    if args.benchmark:
        benchmark()
    print("ALL QWEN4_EXP HC TRITON NPU CHECKS PASSED")


if __name__ == "__main__":
    main()
