# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU accuracy tests for DFlash/DSpark anchor-union routing."""

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.dflash_topm_state import DFlashTopMState

DEVICE = "npu"
SUPPORTED_DEVICES = (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5)


def _make_state(num_experts, route_top_k, scoring_func, rows):
    return DFlashTopMState.from_dict(
        {
            "backend": "anchor_union_fused_native",
            "num_experts": num_experts,
            "route_top_k": route_top_k,
            "scoring_func": scoring_func,
            "protected_rows": 3,
            "suffix_pool_top_k": 2,
            "fused_rows": [rows],
            "expected_router_layers": None,
        }
    )


def _make_roles(rows, device):
    roles = torch.zeros(rows, dtype=torch.int32, device=device)
    roles[:3] = 1
    if rows > 3:
        roles[3:] = 2
    return roles


def _report_precision(name, actual, expected, *, rtol, atol, max_abs_limit):
    """Print mixed-tolerance metrics before applying the test assertion."""
    actual_cpu = actual.detach().float().cpu()
    expected_cpu = expected.detach().float().cpu()
    abs_error = (actual_cpu - expected_cpu).abs()
    relative_error = abs_error / expected_cpu.abs().clamp_min(torch.finfo(torch.float32).eps)
    tolerance = atol + rtol * expected_cpu.abs()
    within_tolerance = abs_error <= tolerance
    pass_ratio = within_tolerance.float().mean().item()
    print(
        f"[precision] {name}: max_abs_error={abs_error.max().item():.8e}, "
        f"max_rel_error={relative_error.max().item():.8e}, "
        f"within_tolerance={pass_ratio:.6%}, "
        f"rtol={rtol:.8e}, atol={atol:.8e}, "
        f"max_abs_limit={max_abs_limit:.8e}"
    )
    assert abs_error.max().item() <= max_abs_limit


def _report_exact(name, actual, expected):
    """Print exact-match metrics for discrete routing outputs."""
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    mismatch = torch.count_nonzero(actual_cpu != expected_cpu).item()
    total = expected_cpu.numel()
    print(f"[precision] {name}: mismatches={mismatch}, total={total}")


@pytest.mark.skipif(
    get_ascend_device_type() not in SUPPORTED_DEVICES,
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("rows", [8, 16])
@pytest.mark.parametrize("num_experts", [128, 256])
def test_qwen_fused_matches_reference(rows, num_experts):
    torch.manual_seed(0)
    state = _make_state(num_experts, 8, "softmax", rows)
    logits = torch.randn(rows, num_experts, dtype=torch.bfloat16, device=DEVICE)
    roles = _make_roles(rows, DEVICE)
    state.prepare(rows, [(0, rows)], DEVICE)

    reference = state.route_reference(
        logits.clone(),
        roles,
        top_k=8,
        renormalize=True,
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
    )
    fused = state.route(
        logits.clone(),
        roles,
        top_k=8,
        renormalize=True,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
    )

    _report_exact("qwen_expert_ids", fused[1], reference[1])
    _report_precision(
        "qwen_weights",
        fused[0],
        reference[0],
        rtol=2**-6,
        atol=2**-6,
        max_abs_limit=1.0,
    )
    torch.testing.assert_close(fused[1], reference[1], rtol=0, atol=0)
    torch.testing.assert_close(fused[0], reference[0], rtol=2**-6, atol=2**-6)


@pytest.mark.skipif(
    get_ascend_device_type() not in SUPPORTED_DEVICES,
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("rows", [8, 16])
def test_dspark_fused_matches_reference_and_handles_strides(rows):
    torch.manual_seed(1)
    state = _make_state(256, 6, "sqrtsoftplus", rows)
    base = torch.randn(rows, 512, dtype=torch.float32, device=DEVICE)
    logits = base[:, ::2]
    bias = torch.randn(256, dtype=torch.float32, device=DEVICE)
    roles = _make_roles(rows, DEVICE)
    state.prepare(rows, [(0, rows)], DEVICE)

    reference = state.route_reference(
        logits.clone(),
        roles,
        top_k=6,
        renormalize=True,
        routed_scaling_factor=1.5,
        e_score_correction_bias=bias,
    )
    fused = state.route(
        logits.clone(),
        roles,
        top_k=6,
        renormalize=True,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        e_score_correction_bias=bias,
    )

    _report_exact("dspark_expert_ids", fused[1], reference[1])
    _report_precision(
        "dspark_weights",
        fused[0],
        reference[0],
        rtol=2**-16,
        atol=2**-10,
        max_abs_limit=0.01,
    )
    torch.testing.assert_close(fused[1], reference[1], rtol=0, atol=0)
    torch.testing.assert_close(fused[0], reference[0], rtol=2**-16, atol=2**-10)
