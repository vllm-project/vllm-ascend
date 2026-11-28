# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Launch-argument contract for the fused_kda_gate lower_bound path.

The dispatcher forwards plain calls verbatim, so the impl must accept
upstream's ``lower_bound`` keyword and translate it to the kernel
contract: ``lower_bound or 0.0`` positionally plus the ``USE_LOWER_BOUND``
constexpr switch. The kernel itself needs an NPU, so a fake kernel
captures the launch and these tests run on CPU.
"""

import pytest
import torch

import vllm_ascend.ops.triton.kda.kda as kda_module


class _FakeKernel:
    """Captures ``kernel[grid](*args, **kwargs)`` launches."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((args, kwargs))

        return launch


@pytest.fixture()
def fake_gate_kernel(monkeypatch):
    fake = _FakeKernel()
    monkeypatch.setattr(kda_module, "kda_gate_fwd_kernel", fake)
    return fake


def test_lower_bound_call_no_longer_raises(fake_gate_kernel):
    g = torch.zeros(4, 2 * 8)
    A = torch.zeros(2)

    # bailing_moe_v3's call shape: lower_bound passed by keyword. Before
    # the fix this raised TypeError at the verbatim forward.
    kda_module.fused_kda_gate(g, A, 8, lower_bound=-5.0)

    assert len(fake_gate_kernel.launches) == 1


def test_lower_bound_forwards_the_bound_and_switch(fake_gate_kernel):
    g = torch.zeros(4, 2 * 8)
    A = torch.zeros(2)

    kda_module.fused_kda_gate(g, A, 8, lower_bound=-5.0)

    args, kwargs = fake_gate_kernel.launches[-1]
    # Positional order mirrors the kernel signature: g, A, y, g_bias,
    # lower_bound, beta, threshold, T, H, head_k_dim.
    assert args[4] == -5.0
    assert kwargs["USE_LOWER_BOUND"] is True


def test_none_lower_bound_forwards_placeholder_and_switch_off(fake_gate_kernel):
    g = torch.zeros(4, 2 * 8)
    A = torch.zeros(2)

    kda_module.fused_kda_gate(g, A, 8)

    args, kwargs = fake_gate_kernel.launches[-1]
    assert args[4] == 0.0
    assert kwargs["USE_LOWER_BOUND"] is False
