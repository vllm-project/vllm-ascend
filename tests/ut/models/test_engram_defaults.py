# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


@pytest.mark.parametrize("device", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("explicit", [None, False, True])
def test_engram_overlap_hardware_default_and_explicit_override(monkeypatch, device, explicit):
    monkeypatch.setattr("vllm_ascend.ascend_config.get_current_hardware_profile", lambda: get_hardware_profile(device))
    kwargs = {} if explicit is None else {"multistream_engram_overlap": explicit}
    config = AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False), **kwargs)
    expected = device == AscendDeviceType.A5 if explicit is None else explicit
    assert config.multistream_engram_overlap is expected
