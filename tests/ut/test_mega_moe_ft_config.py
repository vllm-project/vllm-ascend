# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.ascend_config import AscendConfig, clear_ascend_config, is_mega_moe_supported
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


@pytest.mark.parametrize(
    "device,ft,quant,hidden,intermediate,expected",
    [
        (AscendDeviceType.A3, True, "w8a8_dynamic", 2048, 768, True),
        (AscendDeviceType.A3, False, "w8a8_dynamic", 2048, 768, False),
        (AscendDeviceType.A3, True, "w4a8_dynamic", 2048, 768, False),
        (AscendDeviceType.A3, True, None, 2048, 768, False),
        (AscendDeviceType.A3, True, "w8a8_dynamic", 256, 768, False),
        (AscendDeviceType.A3, True, "w8a8_dynamic", 2048, 896, False),
        (AscendDeviceType.A2, True, "w8a8_dynamic", 2048, 768, False),
        (AscendDeviceType.A5, True, "w8a8_dynamic", 2048, 768, False),
        (AscendDeviceType.A3, False, "w8a8_dynamic", 2048, 1024, True),
    ],
)
def test_only_validated_768_ft_shape_is_allowed(device, ft, quant, hidden, intermediate, expected):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=hidden, moe_intermediate_size=intermediate, quantize=quant)
        ),
        parallel_config=SimpleNamespace(enable_fault_tolerance=ft),
    )
    with patch("vllm_ascend.ascend_config.get_current_hardware_profile", return_value=get_hardware_profile(device)):
        assert AscendConfig._is_megamoe_supported_by_config(config) is expected


def test_mega_moe_mode_normalization():
    try:
        with patch("vllm_ascend.ascend_config.importlib.util.find_spec", return_value=True):
            config = AscendConfig(enable_fused_mc2=2, sparse_kv_offload_config=SimpleNamespace(enabled=False))
            assert config.enable_fused_mc2 == 1
            assert is_mega_moe_supported()
    finally:
        clear_ascend_config()
