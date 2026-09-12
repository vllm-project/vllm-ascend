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
    ATB_EXTENSIONS = auto()
    ATB_WARMUP = auto()
    BGMV_SGMV_META_REGISTRATION = auto()
    CANN_MEGAMOE = auto()
    CHUNKED_PREFILL_PHASE_SPLIT = auto()
    CLUSTER_CPU_TOPOLOGY = auto()
    COMPATIBILITY_OP_IMPLEMENTATIONS = auto()
    DISTRIBUTED_COMMUNICATION_ADAPTATION = auto()
    DSA_C128_STATE_SMALL_BLOCK_SIZES = auto()
    DSV4_COMPRESSED_CACHE = auto()
    DYNAMIC_MX_QUANT_FUSION = auto()
    DYNAMIC_MX_QUANT_SCALE_ALG_ONE = auto()
    FP8_ATTENTION = auto()
    FUSED_MOE_COMPATIBILITY = auto()
    FUSED_SWIGLU_TUNING_ARGS = auto()
    GDN_COMPATIBILITY = auto()
    GRAPH_MULS_ADD_FUSION = auto()
    GRAPH_NORM_QUANT_FUSION = auto()
    IRQ_CPU_RESERVATION = auto()
    LOCAL_KV_COMM_RESOURCE = auto()
    LORA_CUSTOM_OPS = auto()
    MLA_DECODE_PROLOG_WITHOUT_ROPE = auto()
    MLAPO_NATIVE_WEIGHTS = auto()
    MC2_FULLMESH_V2_COMM = auto()
    MC2_HIERARCHY_COMM = auto()
    MOE_DISPATCH_EXTRA_ARGS = auto()
    MOE_DISPATCH_SHARED_EXPERT_ARGS = auto()
    NPUGRAPH_EX = auto()
    NPU_TOP_K_TOP_P = auto()
    PAGED_ATTENTION = auto()
    RC_DEVICE_DISCOVERY = auto()
    REDUCED_CUDAGRAPH_CAPTURE_SIZES = auto()
    RUNTIME_CUSTOM_OPS = auto()
    SFA_DCP_REPLICATED_INDEXER = auto()
    STANDARD_WORKER_PATCHES = auto()
    STANDARD_MAMBA_PATCH = auto()
    SWIGLU_OAI_MX_QUANT = auto()
    TRITON_BATCH_MEMCPY = auto()
    UNRESTRICTED_MLAPO = auto()


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


class MoECommPolicy(Enum):
    """MoE communication selection policies."""

    CAPACITY_AND_EXPERT_DENSITY = auto()
    FUSED_OR_CAPACITY = auto()
    CAPACITY_AND_WORLD_SIZE = auto()
    ALLGATHER = auto()


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
    moe_comm_policy: MoECommPolicy
    quantization_backend_family: QuantizationBackendFamily
    capabilities: frozenset[HardwareCapability]

    def supports(self, capability: HardwareCapability) -> bool:
        """Return whether this hardware family provides ``capability``."""

        return capability in self.capabilities


_STANDARD_CAPABILITIES = frozenset(
    {
        HardwareCapability.AUTO_ENABLE_CUSTOM_OPS,
        HardwareCapability.ATB_EXTENSIONS,
        HardwareCapability.ATB_WARMUP,
        HardwareCapability.BGMV_SGMV_META_REGISTRATION,
        HardwareCapability.FUSED_SWIGLU_TUNING_ARGS,
        HardwareCapability.GRAPH_MULS_ADD_FUSION,
        HardwareCapability.GRAPH_NORM_QUANT_FUSION,
        HardwareCapability.IRQ_CPU_RESERVATION,
        HardwareCapability.LORA_CUSTOM_OPS,
        HardwareCapability.MC2_HIERARCHY_COMM,
        HardwareCapability.NPUGRAPH_EX,
        HardwareCapability.PAGED_ATTENTION,
        HardwareCapability.RUNTIME_CUSTOM_OPS,
        HardwareCapability.SFA_DCP_REPLICATED_INDEXER,
        HardwareCapability.STANDARD_MAMBA_PATCH,
        HardwareCapability.STANDARD_WORKER_PATCHES,
        HardwareCapability.TRITON_BATCH_MEMCPY,
    }
)
_A3_CAPABILITIES = _STANDARD_CAPABILITIES | {HardwareCapability.MC2_FULLMESH_V2_COMM}
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
            moe_comm_policy=MoECommPolicy.CAPACITY_AND_EXPERT_DENSITY,
            quantization_backend_family=QuantizationBackendFamily.STANDARD,
            capabilities=_STANDARD_CAPABILITIES | {HardwareCapability.NPU_TOP_K_TOP_P},
        ),
        AscendDeviceType.A3: HardwareProfile(
            _device_type=AscendDeviceType.A3,
            attention_backend_family=AttentionBackendFamily.STANDARD,
            cpu_binding_mode=CPUBindingMode.GLOBAL_SLICE,
            default_worker_cls=_DEFAULT_WORKER_CLS,
            device_adaptor_family=DeviceAdaptorFamily.STANDARD,
            device_addressing_mode=DeviceAddressingMode.DUAL_CHIP_CARD,
            weight_layout_policy=WeightLayoutPolicy.CONFIGURABLE,
            moe_comm_policy=MoECommPolicy.FUSED_OR_CAPACITY,
            quantization_backend_family=QuantizationBackendFamily.STANDARD,
            capabilities=_A3_CAPABILITIES
            | {
                HardwareCapability.CANN_MEGAMOE,
                HardwareCapability.MOE_DISPATCH_EXTRA_ARGS,
                HardwareCapability.NPU_TOP_K_TOP_P,
            },
        ),
        AscendDeviceType._310P: HardwareProfile(
            _device_type=AscendDeviceType._310P,
            attention_backend_family=AttentionBackendFamily.COMPATIBILITY,
            cpu_binding_mode=CPUBindingMode.TOPO_AFFINITY,
            default_worker_cls="vllm_ascend._310p.worker_310p.NPUWorker310",
            device_adaptor_family=DeviceAdaptorFamily.COMPATIBILITY,
            device_addressing_mode=DeviceAddressingMode.DIRECT,
            weight_layout_policy=WeightLayoutPolicy.FORCE_NZ,
            moe_comm_policy=MoECommPolicy.ALLGATHER,
            quantization_backend_family=QuantizationBackendFamily.COMPATIBILITY,
            capabilities=frozenset(
                {
                    HardwareCapability.COMPATIBILITY_OP_IMPLEMENTATIONS,
                    HardwareCapability.DISTRIBUTED_COMMUNICATION_ADAPTATION,
                    HardwareCapability.FUSED_MOE_COMPATIBILITY,
                    HardwareCapability.FUSED_SWIGLU_TUNING_ARGS,
                    HardwareCapability.GDN_COMPATIBILITY,
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
            moe_comm_policy=MoECommPolicy.CAPACITY_AND_WORLD_SIZE,
            quantization_backend_family=QuantizationBackendFamily.STANDARD,
            capabilities=frozenset(
                {
                    HardwareCapability.AUTO_ENABLE_CUSTOM_OPS,
                    HardwareCapability.BGMV_SGMV_META_REGISTRATION,
                    HardwareCapability.CANN_MEGAMOE,
                    HardwareCapability.CHUNKED_PREFILL_PHASE_SPLIT,
                    HardwareCapability.CLUSTER_CPU_TOPOLOGY,
                    HardwareCapability.DSA_C128_STATE_SMALL_BLOCK_SIZES,
                    HardwareCapability.DSV4_COMPRESSED_CACHE,
                    HardwareCapability.DYNAMIC_MX_QUANT_FUSION,
                    HardwareCapability.DYNAMIC_MX_QUANT_SCALE_ALG_ONE,
                    HardwareCapability.FP8_ATTENTION,
                    HardwareCapability.GRAPH_MULS_ADD_FUSION,
                    HardwareCapability.GRAPH_NORM_QUANT_FUSION,
                    HardwareCapability.LOCAL_KV_COMM_RESOURCE,
                    HardwareCapability.LORA_CUSTOM_OPS,
                    HardwareCapability.MLA_DECODE_PROLOG_WITHOUT_ROPE,
                    HardwareCapability.MLAPO_NATIVE_WEIGHTS,
                    HardwareCapability.MOE_DISPATCH_EXTRA_ARGS,
                    HardwareCapability.MOE_DISPATCH_SHARED_EXPERT_ARGS,
                    HardwareCapability.NPUGRAPH_EX,
                    HardwareCapability.REDUCED_CUDAGRAPH_CAPTURE_SIZES,
                    HardwareCapability.STANDARD_MAMBA_PATCH,
                    HardwareCapability.STANDARD_WORKER_PATCHES,
                    HardwareCapability.SWIGLU_OAI_MX_QUANT,
                    HardwareCapability.TRITON_BATCH_MEMCPY,
                    HardwareCapability.UNRESTRICTED_MLAPO,
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


_CURRENT_HARDWARE_PROFILE = get_hardware_profile(get_device_config()._device_type)


def get_current_hardware_profile() -> HardwareProfile:
    """Return the profile selected by the current device configuration."""

    return _CURRENT_HARDWARE_PROFILE
