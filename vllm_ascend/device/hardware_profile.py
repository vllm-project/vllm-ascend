# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Capability-oriented profiles for supported Ascend hardware families."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType

from vllm_ascend.device.device_config import get_device_config
from vllm_ascend.device.hardware import AscendDeviceType


class HardwareCapability(Enum):
    """Independent SoC capabilities consumed by shared business logic."""

    AUTO_ENABLE_CUSTOM_OPS = auto()
    BGMV_SGMV_META_REGISTRATION = auto()
    CLUSTER_CPU_TOPOLOGY = auto()
    COMPATIBILITY_OP_IMPLEMENTATIONS = auto()
    DYNAMIC_MX_QUANT_FUSION = auto()
    IRQ_CPU_RESERVATION = auto()
    LORA_CUSTOM_OPS = auto()
    NPUGRAPH_EX = auto()
    RC_DEVICE_DISCOVERY = auto()
    REDUCED_CUDAGRAPH_CAPTURE_SIZES = auto()
    RUNTIME_CUSTOM_OPS = auto()
    SFA_DCP_REPLICATED_INDEXER = auto()
    SKIP_REMOTE_H2D_BUFFER_REGISTRATION = auto()
    STANDARD_WORKER_PATCHES = auto()


class AttentionBackendFamily(Enum):
    """Attention backend implementation families selected by the platform."""

    STANDARD = auto()
    COMPATIBILITY = auto()


class CPUBindingMode(Enum):
    """CPU binding policies selected for worker processes."""

    TOPO_AFFINITY = "topo_affinity"
    GLOBAL_SLICE = "global_slice"


class DeviceAdaptorFamily(Enum):
    """Device operation adaptor implementation families."""

    STANDARD = auto()
    FP8_OPTIMIZED = auto()
    COMPATIBILITY = auto()


class DeviceAddressingMode(Enum):
    """PCIe device addressing policies used by CPU binding."""

    DIRECT = auto()
    DUAL_CHIP_CARD = auto()


class QuantizationBackendFamily(Enum):
    """Quantization configuration implementation families."""

    STANDARD = auto()
    COMPATIBILITY = auto()


class WeightLayoutPolicy(Enum):
    """Weight layout selection policies for supported hardware families."""

    CONFIGURABLE = auto()
    FORCE_NZ = auto()


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Immutable capabilities and implementation choices for one SoC family."""

    _device_type: AscendDeviceType
    attention_backend_family: AttentionBackendFamily
    cpu_binding_mode: CPUBindingMode
    default_worker_cls: str
    device_adaptor_family: DeviceAdaptorFamily
    device_addressing_mode: DeviceAddressingMode
    weight_layout_policy: WeightLayoutPolicy
    quantization_backend_family: QuantizationBackendFamily
    capabilities: frozenset[HardwareCapability]

    def supports(self, capability: HardwareCapability) -> bool:
        """Return whether this hardware family provides ``capability``."""

        return capability in self.capabilities


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
_DEFAULT_WORKER_CLS = "vllm_ascend.worker.worker.NPUWorker"
_HARDWARE_PROFILES: Mapping[AscendDeviceType, HardwareProfile] = MappingProxyType(
    {
        AscendDeviceType.A2: HardwareProfile(
            _device_type=AscendDeviceType.A2,
            attention_backend_family=AttentionBackendFamily.STANDARD,
            cpu_binding_mode=CPUBindingMode.TOPO_AFFINITY,
            default_worker_cls=_DEFAULT_WORKER_CLS,
            device_adaptor_family=DeviceAdaptorFamily.STANDARD,
            device_addressing_mode=DeviceAddressingMode.DIRECT,
            weight_layout_policy=WeightLayoutPolicy.CONFIGURABLE,
            quantization_backend_family=QuantizationBackendFamily.STANDARD,
            capabilities=_STANDARD_CAPABILITIES | {HardwareCapability.SKIP_REMOTE_H2D_BUFFER_REGISTRATION},
        ),
        AscendDeviceType.A3: HardwareProfile(
            _device_type=AscendDeviceType.A3,
            attention_backend_family=AttentionBackendFamily.STANDARD,
            cpu_binding_mode=CPUBindingMode.GLOBAL_SLICE,
            default_worker_cls=_DEFAULT_WORKER_CLS,
            device_adaptor_family=DeviceAdaptorFamily.STANDARD,
            device_addressing_mode=DeviceAddressingMode.DUAL_CHIP_CARD,
            weight_layout_policy=WeightLayoutPolicy.CONFIGURABLE,
            quantization_backend_family=QuantizationBackendFamily.STANDARD,
            capabilities=_STANDARD_CAPABILITIES,
        ),
        AscendDeviceType._310P: HardwareProfile(
            _device_type=AscendDeviceType._310P,
            attention_backend_family=AttentionBackendFamily.COMPATIBILITY,
            cpu_binding_mode=CPUBindingMode.TOPO_AFFINITY,
            default_worker_cls="vllm_ascend._310p.worker_310p.NPUWorker310",
            device_adaptor_family=DeviceAdaptorFamily.COMPATIBILITY,
            device_addressing_mode=DeviceAddressingMode.DIRECT,
            weight_layout_policy=WeightLayoutPolicy.FORCE_NZ,
            quantization_backend_family=QuantizationBackendFamily.COMPATIBILITY,
            capabilities=frozenset(
                {
                    HardwareCapability.COMPATIBILITY_OP_IMPLEMENTATIONS,
                    HardwareCapability.IRQ_CPU_RESERVATION,
                    HardwareCapability.RC_DEVICE_DISCOVERY,
                    HardwareCapability.RUNTIME_CUSTOM_OPS,
                }
            ),
        ),
        AscendDeviceType.A5: HardwareProfile(
            _device_type=AscendDeviceType.A5,
            attention_backend_family=AttentionBackendFamily.STANDARD,
            cpu_binding_mode=CPUBindingMode.TOPO_AFFINITY,
            default_worker_cls=_DEFAULT_WORKER_CLS,
            device_adaptor_family=DeviceAdaptorFamily.FP8_OPTIMIZED,
            device_addressing_mode=DeviceAddressingMode.DIRECT,
            weight_layout_policy=WeightLayoutPolicy.CONFIGURABLE,
            quantization_backend_family=QuantizationBackendFamily.STANDARD,
            capabilities=frozenset(
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
        ),
    }
)


def get_hardware_profile(device_type: AscendDeviceType) -> HardwareProfile:
    """Return the immutable profile registered for ``device_type``."""

    try:
        return _HARDWARE_PROFILES[device_type]
    except KeyError as exc:
        raise RuntimeError(f"No hardware profile is registered for device type: {device_type}.") from exc


def get_current_hardware_profile() -> HardwareProfile:
    """Return the profile selected by the current device configuration."""

    return get_hardware_profile(get_device_config()._device_type)
