# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Semantic profiles for supported Ascend hardware families.

This module is the boundary between hardware identity and shared business
logic. Detection maps a SoC to one immutable :class:`HardwareProfile`; code
outside the device package then asks that profile about a semantic contract
instead of branching on an A2, A3, A5, or 310P name.

The profile uses three kinds of fields:

* A :class:`HardwareCapability` is a binary, independently testable contract
  that says a code path, operator ABI, runtime facility, or kernel variant is
  available. It describes the supported software-and-hardware stack, not just
  a physical instruction in the silicon. A caller must still check its model,
  shape, dtype, user configuration, and runtime preconditions.
* A ``*Policy`` or ``*Mode`` selects behavior or a default. It answers "which
  rule applies?", not "is the device capable?".
* A ``*Family`` selects one implementation from several equivalent interfaces.

The absence of a capability means that its consumer must use its documented
fallback or reject that feature. It does not, by itself, mean that every
related operation is unsupported on the device.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType

from vllm_ascend.device.device_config import get_device_config
from vllm_ascend.device.hardware import AscendDeviceType


class HardwareCapability(Enum):
    """Binary contracts consumed by shared business logic.

    ``profile.supports(X)`` answers only whether the exact contract documented
    for ``X`` is available. Capabilities are intentionally narrow so callers
    do not infer unrelated behavior from a device generation. New default or
    implementation-selection fields should normally be modeled as a policy,
    mode, or family rather than added here.

    A few historical members still describe a default or implementation
    selection. Their comments call this out explicitly; do not interpret them
    as broad claims about the underlying silicon.
    """

    # Legacy default: replace vLLM's custom-op list with ``["all"]`` during
    # platform configuration. Runtime extension loading is a separate contract.
    AUTO_ENABLE_CUSTOM_OPS = auto()
    # Register ATB extension entry points eagerly during worker construction.
    ATB_EXTENSIONS = auto()
    # Run the ATB matmul warmup that avoids first-request cache-write latency.
    ATB_WARMUP = auto()
    # Register fake/meta implementations for the custom BGMV and SGMV LoRA ops.
    BGMV_SGMV_META_REGISTRATION = auto()
    # Allow the CANN MegaMoe fused-MC2 path when its model, EP, and config checks pass.
    CANN_MEGAMOE = auto()
    # Split a mixed chunked-prefill batch into separate prefill and decode FIA calls.
    CHUNKED_PREFILL_PHASE_SPLIT = auto()
    # Build CPU-affinity pools from the cluster-aware CPU topology.
    CLUSTER_CPU_TOPOLOGY = auto()
    # Legacy implementation selector: install the 310P-specific operator overrides.
    COMPATIBILITY_OP_IMPLEMENTATIONS = auto()
    # Install distributed-communication adaptations required by the compatibility path.
    DISTRIBUTED_COMMUNICATION_ADAPTATION = auto()
    # Use DSA C128-state kernel block sizes ``[4, 8, 16]`` instead of ``[8, 16, 32]``.
    DSA_C128_STATE_SMALL_BLOCK_SIZES = auto()
    # Use the DeepSeek-V4/DSA compressed-KV-cache layout and compressor/indexer flow.
    DSV4_COMPRESSED_CACHE = auto()
    # Enable dynamic-MX norm fusion and the associated ``wo_a`` weight-layout contract.
    DYNAMIC_MX_QUANT_FUSION = auto()
    # Select DynamicMxQuantV3 ``scale_alg=1`` for model paths that require it.
    DYNAMIC_MX_QUANT_SCALE_ALG_ONE = auto()
    # Enable the FP8/C8 attention KV-cache ABI and matching attention preprocess paths.
    # This is not a general statement that every FP8 operation is supported.
    FP8_ATTENTION = auto()
    # Select the compatibility grouped-top-k router used by the fused-MoE path.
    FUSED_MOE_COMPATIBILITY = auto()
    # Pass ``glu_alpha`` and ``glu_bias`` to the fused dequant-SwiGLU-quant operator.
    FUSED_SWIGLU_TUNING_ARGS = auto()
    # Select the compatibility GatedDeltaNet core and state-dtype implementation.
    GDN_COMPATIBILITY = auto()
    # Register the FX graph rewrite that fuses the supported muls-plus-add pattern.
    GRAPH_MULS_ADD_FUSION = auto()
    # Register the FX graph rewrites for supported RMSNorm-plus-quant patterns.
    GRAPH_NORM_QUANT_FUSION = auto()
    # Reserve host CPUs for interrupt handling when constructing worker CPU pools.
    IRQ_CPU_RESERVATION = auto()
    # Create the local communication resource used by supported KV-transfer deployments.
    LOCAL_KV_COMM_RESOURCE = auto()
    # Use vLLM-Ascend's custom BGMV/SGMV LoRA kernels when rank constraints also pass.
    LORA_CUSTOM_OPS = auto()
    # Allow the fused MLA decode prolog even when the model does not use MLA RoPE.
    MLA_DECODE_PROLOG_WITHOUT_ROPE = auto()
    # Allow MLAPO with native floating-point projection weights, not only quantized weights.
    MLAPO_NATIVE_WEIGHTS = auto()
    # Accept ``fullmesh_v2`` as the MC2 communication algorithm.
    MC2_FULLMESH_V2_COMM = auto()
    # Accept hierarchical MC2 communication, subject to its expert-count constraints.
    MC2_HIERARCHY_COMM = auto()
    # Pass TP group/rank metadata required by the newer MoE dispatch/combine ABI.
    MOE_DISPATCH_EXTRA_ARGS = auto()
    # Pass shared-expert, expert-scale, quant-mode, and output-dtype metadata to MoE dispatch.
    MOE_DISPATCH_SHARED_EXPERT_ARGS = auto()
    # Allow the extended NPU graph backend; static-kernel mode depends on this contract.
    NPUGRAPH_EX = auto()
    # Use ``torch_npu.npu_top_k_top_p`` for sampling instead of the PyTorch fallback.
    NPU_TOP_K_TOP_P = auto()
    # Allow shared paged-attention decode when graph mode, shape, and model checks pass.
    PAGED_ATTENTION = auto()
    # Inspect PCIe topology to distinguish 310P Root-Complex and endpoint deployments.
    RC_DEVICE_DISCOVERY = auto()
    # Apply the temporary reduced ACL-graph capture-size set required by affected drivers.
    REDUCED_CUDAGRAPH_CAPTURE_SIZES = auto()
    # Import and register the compiled vLLM-Ascend custom-op library at runtime.
    # This is independent of whether custom ops are enabled by default.
    RUNTIME_CUSTOM_OPS = auto()
    # Allow SFA decode-context parallelism with a replicated indexer.
    SFA_DCP_REPLICATED_INDEXER = auto()
    # Legacy implementation selector: install the standard worker patch bundle.
    STANDARD_WORKER_PATCHES = auto()
    # Legacy implementation selector: install the standard Mamba platform patch.
    STANDARD_MAMBA_PATCH = auto()
    # Reserved contract for the A5 SwiGLU OAI MX-quant path; it currently has no consumer.
    SWIGLU_OAI_MX_QUANT = auto()
    # Use the Triton batch-memcpy kernel for Mamba state copies.
    TRITON_BATCH_MEMCPY = auto()
    # Honor MLAPO enablement on any pipeline role; other profiles limit it to decode consumers.
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
    """Immutable semantic contracts registered for one SoC family.

    ``_device_type`` is registration metadata and must not escape into business
    logic. Family, mode, and policy fields select implementations or behavior;
    ``capabilities`` records additive binary support contracts.
    """

    # Identity is private to detection and profile registration.
    _device_type: AscendDeviceType
    # Mutually exclusive implementation and behavior selections.
    attention_backend_family: AttentionBackendFamily
    cpu_binding_mode: CPUBindingMode
    default_worker_cls: str
    device_adaptor_family: DeviceAdaptorFamily
    device_addressing_mode: DeviceAddressingMode
    weight_layout_policy: WeightLayoutPolicy
    moe_comm_policy: MoECommPolicy
    quantization_backend_family: QuantizationBackendFamily
    # Independent, additive contracts queried with ``supports``.
    capabilities: frozenset[HardwareCapability]

    def supports(self, capability: HardwareCapability) -> bool:
        """Return whether the exact documented ``capability`` contract is available."""

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
