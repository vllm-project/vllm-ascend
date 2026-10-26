# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.device.device_op import BaseDeviceAdaptor
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_hardware_profile
from vllm_ascend.ops.triton.fused_gdn_decode import fused_gdn_decode


@pytest.mark.parametrize("device_type", list(AscendDeviceType))
def test_fused_decode_is_limited_to_validated_hardware(device_type):
    profile = get_hardware_profile(device_type)
    assert profile.supports(HardwareCapability.GDN_FUSED_DECODE) == (device_type == AscendDeviceType.A3)


@pytest.mark.parametrize(
    "fallback",
    [None, "padded_state", "hardware", "dtype", "dimension", "batch", "state_stride", "overlapping_state", "cpu"],
)
def test_adaptor_dispatch_and_fallback(fallback):
    q = Mock(dtype=torch.bfloat16, shape=(1, 2, 128), device=SimpleNamespace(type="npu"))
    k = Mock(dtype=torch.bfloat16)
    v = Mock(dtype=torch.bfloat16, shape=(1, 4, 128))
    a, b = Mock(dtype=torch.bfloat16), Mock(dtype=torch.bfloat16)
    state = torch.empty(2, 4, 128, 128)
    if fallback == "padded_state":
        state = torch.empty(2, 4 * 128 * 128 + 256)[:, 128:-128].view(2, 4, 128, 128)
    elif fallback == "overlapping_state":
        state = state.as_strided(state.shape, (1, 128 * 128, 128, 1))
    elif fallback == "dtype":
        q.dtype = torch.float16
    elif fallback == "dimension":
        q.shape = (1, 2, 64)
    elif fallback == "batch":
        q.shape = (33, 2, 128)
    elif fallback == "state_stride":
        state = state.transpose(-1, -2)
    elif fallback == "cpu":
        q.device.type = "cpu"
    args = (q, k, v, a, b, Mock(), Mock(), state, Mock(), Mock())
    with (
        patch("vllm_ascend.device.device_op.get_current_hardware_profile") as profile,
        patch("vllm_ascend.device.device_op.fused_gdn_decode") as kernel,
    ):
        profile.return_value.supports.return_value = fallback != "hardware"
        result = BaseDeviceAdaptor.try_fused_gdn_decode(*args)
    if fallback in (None, "padded_state"):
        kernel.assert_called_once_with(*args)
        assert result is kernel.return_value
    else:
        kernel.assert_not_called()
        assert result is None


@pytest.mark.parametrize("invalid", ["gates", "state", "indices", "boundaries", "head_stride", "parameters"])
def test_invalid_kernel_shapes_fail_before_launch(invalid):
    q = torch.empty(1, 2, 128)
    args = [
        q,
        torch.empty_like(q),
        torch.empty(1, 4, 128),
        torch.empty(1, 4),
        torch.empty(1, 4),
        torch.empty(4),
        torch.empty(4),
        torch.empty(2, 4, 128, 128),
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
    ]
    if invalid == "gates":
        args[3] = torch.empty(2, 4)
    elif invalid == "state":
        args[7] = args[7].transpose(-1, -2)
    elif invalid == "indices":
        args[8] = torch.zeros(2, dtype=torch.int32)
    elif invalid == "boundaries":
        args[9] = torch.empty(1, dtype=torch.int32)
    elif invalid == "head_stride":
        args[0] = torch.empty(1, 2, 256)[:, :, ::2]
    elif invalid == "parameters":
        args[5] = torch.empty(2)
    args = [x.to(torch.bfloat16) if x.is_floating_point() else x for x in args]
    if invalid == "head_stride":
        args[0] = torch.empty(1, 2, 256, dtype=torch.bfloat16)[:, :, ::2]
    with pytest.raises(ValueError):
        fused_gdn_decode(*args)
