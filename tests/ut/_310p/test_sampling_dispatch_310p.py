# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exercise the original upstream wrappers after 310P kernel registration."""

import importlib
import inspect
from unittest.mock import patch

import pytest
import torch
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.sample.logit_bias import apply_logit_bias
from vllm.v1.worker.gpu.sample.min_p import apply_min_p

from vllm_ascend._310p.worker.v2.sample.dispatch import (
    bias_kernel,
    min_p_kernel,
    register_sampling_kernel_overrides,
    temperature_kernel,
)
from vllm_ascend._310p.worker.v2.sample.logit_bias import LogitBiasState310
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.platform import NPUPlatform

pytestmark = pytest.mark.usefixtures("sampling_kernel_overrides")


@pytest.mark.parametrize(
    ("module", "name", "implementation"),
    [
        ("gumbel", "_temperature_kernel", temperature_kernel),
        ("min_p", "_min_p_kernel", min_p_kernel),
        ("logit_bias", "_bias_kernel", bias_kernel),
    ],
)
def test_override_matches_upstream_kernel_arguments(module, name, implementation):
    kernel = getattr(importlib.import_module(f"vllm.v1.worker.gpu.sample.{module}"), name)
    assert kernel.arg_names == tuple(inspect.signature(implementation).parameters)


def test_upstream_temperature_and_min_p_use_registered_kernels():
    # Row-strided logits and repeated/nonidentity request slots exercise the
    # upstream launch ABI rather than calling a Torch helper directly.
    storage = torch.tensor([[6.0, 2.0, 0.0, 99.0], [2.0, 4.0, 0.0, 99.0], [3.0, 1.0, 0.0, 99.0]])
    logits = storage[:, :3]
    mapping = torch.tensor([2, 0, 2])
    apply_temperature(logits, mapping, torch.tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(logits, torch.tensor([[3.0, 1.0, 0.0], [2.0, 4.0, 0.0], [1.5, 0.5, 0.0]]))
    apply_min_p(logits, mapping, torch.tensor([0.0, 0.0, 0.5]))
    torch.testing.assert_close(
        logits, torch.tensor([[3.0, -torch.inf, -torch.inf], [2.0, 4.0, 0.0], [1.5, -torch.inf, -torch.inf]])
    )
    assert storage[:, 3].tolist() == [99.0] * 3


def test_upstream_whitelist_bias_and_slot_reuse():
    state = LogitBiasState310(3, torch.device("cpu"))
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    state.apply(logits, torch.tensor([2, 0, 2]), {2: [0, 2]}, {2: {0: 5.0}})
    torch.testing.assert_close(logits, torch.tensor([[6.0, -torch.inf, 3.0], [4.0, 5.0, 6.0], [12.0, -torch.inf, 9.0]]))
    state.invalidate()
    reused = torch.tensor([[1.0, 2.0, 3.0]])
    state.apply(reused, torch.tensor([2]), {}, {2: {1: 2.0}})
    torch.testing.assert_close(reused, torch.tensor([[1.0, 4.0, 3.0]]))


@pytest.mark.parametrize("restore", [False, True])
def test_upstream_bias_kernel_preserves_stop_token_restore(restore):
    logits = torch.tensor([[2.0, -torch.inf, -torch.inf]])
    zeros = torch.zeros(1, dtype=torch.int32)
    ones = torch.ones_like(zeros)
    ids = zeros.view(1, 1)
    apply_logit_bias(
        logits,
        zeros,
        zeros,
        zeros,
        ids,
        zeros,
        ids,
        torch.zeros(1, 1),
        torch.tensor([3]),
        ones,
        ones,
        ids,
        check_all_masked_rows=restore,
    )
    expected = torch.tensor([[2.0 if restore else -torch.inf, -torch.inf, -torch.inf]])
    torch.testing.assert_close(logits, expected)


def test_registration_is_idempotent():
    with patch("vllm_ascend._310p.worker.v2.sample.dispatch.register_kernels") as register:
        register_sampling_kernel_overrides()
        register_sampling_kernel_overrides()
    register.assert_not_called()  # The module fixture already registered once.


@pytest.mark.parametrize("device_type", [AscendDeviceType._310P, AscendDeviceType.A2, AscendDeviceType.A3])
def test_platform_only_registers_compatibility_sampling_kernels(device_type):
    with (
        patch("vllm_ascend.platform.get_current_hardware_profile", return_value=get_hardware_profile(device_type)),
        patch("vllm_ascend._310p.worker.v2.sample.dispatch.register_sampling_kernel_overrides") as register,
    ):
        NPUPlatform.register_triton_kernel_overrides()
    assert register.call_count == int(device_type is AscendDeviceType._310P)
