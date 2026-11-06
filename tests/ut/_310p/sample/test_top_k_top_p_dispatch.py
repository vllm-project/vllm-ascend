# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ensure 310P MRV2 follows the shared sampling capability dispatch."""

import importlib
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType._310P, None])
def test_top_k_top_p_capability_dispatch(device_type):
    # Load an isolated module to exercise import-time selection without changing
    # the implementation already imported by other samplers in this process.
    shared = importlib.import_module("vllm_ascend.worker.v2.sample.apply_top_k_top_p")
    spec = importlib.util.spec_from_file_location("_top_k_top_p_dispatch_test", shared.__file__)
    module = importlib.util.module_from_spec(spec)
    profile = (
        get_hardware_profile(device_type) if device_type is not None else SimpleNamespace(supports=lambda _: False)
    )
    with patch("vllm_ascend.device.hardware_profile.get_current_hardware_profile", return_value=profile):
        spec.loader.exec_module(module)
    expected = module._apply_top_k_top_p_pytorch if device_type is None else module._apply_top_k_top_p_npu
    assert module.apply_top_k_top_p is expected


def test_310p_reexports_shared_top_k_top_p():
    shared = importlib.import_module("vllm_ascend.worker.v2.sample.apply_top_k_top_p")
    compatibility = importlib.import_module("vllm_ascend._310p.worker.v2.sample")
    assert compatibility.apply_top_k_top_p is shared.apply_top_k_top_p
