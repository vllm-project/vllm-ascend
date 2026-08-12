# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import cast

import pytest

from vllm_ascend.device.device_config import DeviceConfig
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import (
    AttentionBackendFamily,
    CPUBindingMode,
    DeviceAdaptorFamily,
    DeviceAddressingMode,
    HardwareCapability,
    QuantizationBackendFamily,
    WeightLayoutPolicy,
    get_current_hardware_profile,
    get_hardware_profile,
)

_STANDARD_CAPABILITIES = frozenset(
    {
        HardwareCapability.AUTO_ENABLE_CUSTOM_OPS,
        HardwareCapability.BGMV_SGMV_META_REGISTRATION,
        HardwareCapability.IRQ_CPU_RESERVATION,
        HardwareCapability.LORA_CUSTOM_OPS,
        HardwareCapability.NPUGRAPH_EX,
        HardwareCapability.RUNTIME_CUSTOM_OPS,
        HardwareCapability.SFA_DCP_REPLICATED_INDEXER,
        HardwareCapability.STANDARD_WORKER_PATCHES,
    }
)

_EXPECTED_CAPABILITIES = {
    AscendDeviceType.A2: _STANDARD_CAPABILITIES | {HardwareCapability.SKIP_REMOTE_H2D_BUFFER_REGISTRATION},
    AscendDeviceType.A3: _STANDARD_CAPABILITIES,
    AscendDeviceType._310P: frozenset(
        {
            HardwareCapability.COMPATIBILITY_OP_IMPLEMENTATIONS,
            HardwareCapability.IRQ_CPU_RESERVATION,
            HardwareCapability.RC_DEVICE_DISCOVERY,
            HardwareCapability.RUNTIME_CUSTOM_OPS,
        }
    ),
    AscendDeviceType.A5: frozenset(
        {
            HardwareCapability.AUTO_ENABLE_CUSTOM_OPS,
            HardwareCapability.BGMV_SGMV_META_REGISTRATION,
            HardwareCapability.CLUSTER_CPU_TOPOLOGY,
            HardwareCapability.DYNAMIC_MX_QUANT_FUSION,
            HardwareCapability.LORA_CUSTOM_OPS,
            HardwareCapability.NPUGRAPH_EX,
            HardwareCapability.REDUCED_CUDAGRAPH_CAPTURE_SIZES,
            HardwareCapability.STANDARD_WORKER_PATCHES,
        }
    ),
}


@pytest.mark.parametrize(
    (
        "device_type",
        "attention_backend_family",
        "cpu_binding_mode",
        "default_worker_cls",
        "device_adaptor_family",
        "device_addressing_mode",
        "weight_layout_policy",
        "quantization_backend_family",
    ),
    [
        (
            AscendDeviceType.A2,
            AttentionBackendFamily.STANDARD,
            CPUBindingMode.TOPO_AFFINITY,
            "vllm_ascend.worker.worker.NPUWorker",
            DeviceAdaptorFamily.STANDARD,
            DeviceAddressingMode.DIRECT,
            WeightLayoutPolicy.CONFIGURABLE,
            QuantizationBackendFamily.STANDARD,
        ),
        (
            AscendDeviceType.A3,
            AttentionBackendFamily.STANDARD,
            CPUBindingMode.GLOBAL_SLICE,
            "vllm_ascend.worker.worker.NPUWorker",
            DeviceAdaptorFamily.STANDARD,
            DeviceAddressingMode.DUAL_CHIP_CARD,
            WeightLayoutPolicy.CONFIGURABLE,
            QuantizationBackendFamily.STANDARD,
        ),
        (
            AscendDeviceType._310P,
            AttentionBackendFamily.COMPATIBILITY,
            CPUBindingMode.TOPO_AFFINITY,
            "vllm_ascend._310p.worker_310p.NPUWorker310",
            DeviceAdaptorFamily.COMPATIBILITY,
            DeviceAddressingMode.DIRECT,
            WeightLayoutPolicy.FORCE_NZ,
            QuantizationBackendFamily.COMPATIBILITY,
        ),
        (
            AscendDeviceType.A5,
            AttentionBackendFamily.STANDARD,
            CPUBindingMode.TOPO_AFFINITY,
            "vllm_ascend.worker.worker.NPUWorker",
            DeviceAdaptorFamily.FP8_OPTIMIZED,
            DeviceAddressingMode.DIRECT,
            WeightLayoutPolicy.CONFIGURABLE,
            QuantizationBackendFamily.STANDARD,
        ),
    ],
)
def test_hardware_profile_implementation_matrix(
    device_type: AscendDeviceType,
    attention_backend_family: AttentionBackendFamily,
    cpu_binding_mode: CPUBindingMode,
    default_worker_cls: str,
    device_adaptor_family: DeviceAdaptorFamily,
    device_addressing_mode: DeviceAddressingMode,
    weight_layout_policy: WeightLayoutPolicy,
    quantization_backend_family: QuantizationBackendFamily,
) -> None:
    profile = get_hardware_profile(device_type)

    assert profile._device_type is device_type
    assert profile.attention_backend_family is attention_backend_family
    assert profile.cpu_binding_mode is cpu_binding_mode
    assert profile.default_worker_cls == default_worker_cls
    assert profile.device_adaptor_family is device_adaptor_family
    assert profile.device_addressing_mode is device_addressing_mode
    assert profile.weight_layout_policy is weight_layout_policy
    assert profile.quantization_backend_family is quantization_backend_family


@pytest.mark.parametrize("device_type", list(AscendDeviceType))
def test_hardware_profile_capability_matrix(device_type: AscendDeviceType) -> None:
    profile = get_hardware_profile(device_type)
    expected_capabilities = _EXPECTED_CAPABILITIES[device_type]

    assert profile.capabilities == expected_capabilities
    for capability in HardwareCapability:
        assert profile.supports(capability) is (capability in expected_capabilities)


def test_current_hardware_profile_uses_device_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_ascend.device.hardware_profile as profile_module

    monkeypatch.setattr(
        profile_module,
        "get_device_config",
        lambda: DeviceConfig(_device_type=AscendDeviceType.A5),
    )

    assert get_current_hardware_profile() is get_hardware_profile(AscendDeviceType.A5)


def test_unknown_device_type_is_rejected() -> None:
    unknown_device_type = cast(AscendDeviceType, object())

    with pytest.raises(RuntimeError, match="No hardware profile is registered"):
        get_hardware_profile(unknown_device_type)


def test_every_device_type_has_a_profile() -> None:
    for device_type in AscendDeviceType:
        assert get_hardware_profile(device_type)._device_type is device_type
