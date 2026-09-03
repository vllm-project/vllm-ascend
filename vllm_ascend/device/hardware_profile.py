# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Capability-oriented profiles for supported Ascend hardware families.

``HardwareCapability`` contains only positive, independently testable hardware
or runtime support. Profile families select implementation contracts; policies
and explicitly named scalar fields select defaults or bounded parameters. New
names state their scope, action, and object; a device codename is never the
contract.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType

from vllm_ascend.device.device_config import get_device_config
from vllm_ascend.device.hardware import AscendDeviceType


class HardwareCapability(Enum):
    """Independent, positive features consumed by shared business logic."""

    # Registers the ATB-backed model execution extensions during worker setup.
    ATB_MODEL_EXECUTION_EXTENSIONS = auto()
    # Provides meta kernels for the LoRA BGMV and SGMV custom operators.
    BGMV_SGMV_META_KERNELS = auto()
    # Requantizes loaded block-FP8 weights to the MXFP8 execution format.
    BLOCK_FP8_TO_MXFP8_REQUANTIZATION = auto()
    # Provides the CANN MegaMoe fused-MC2 execution path.
    CANN_MEGAMOE_FUSED_MC2 = auto()
    # Exposes cluster topology used to build CPU affinity groups.
    CLUSTER_CPU_TOPOLOGY = auto()
    # Accepts glu_alpha/glu_bias in the dequant-SwiGLU operator ABI.
    DEQUANT_SWIGLU_GLU_ALPHA_BIAS_ARGS = auto()
    # Supports tensor-parallel O projection in DSA context parallelism.
    DSA_O_PROJ_TP = auto()
    # Supports the compressed attention KV-cache representation required by DSA.
    DSA_COMPRESSED_KV_CACHE = auto()
    # Supports overflow-safe E8M0 scale selection for DynamicMxQuantV3.
    DYNAMIC_MX_QUANT_E8M0_OVERFLOW_SAFE_SCALING = auto()
    # Provides FP8 attention kernels and their quantized KV-cache ABI.
    FP8_ATTENTION = auto()
    # Provides graph patterns for DynamicMxQuant fusion.
    GRAPH_DYNAMIC_MX_QUANT_FUSION = auto()
    # Provides the muls-add graph fusion pattern.
    GRAPH_MULS_ADD_FUSION = auto()
    # Provides graph patterns that fuse normalization with quantization.
    GRAPH_NORM_QUANT_FUSION = auto()
    # Creates a process-local communication resource for KV transfer.
    LOCAL_KV_TRANSFER_COMM_RESOURCE = auto()
    # Provides the custom BGMV/SGMV kernels used by LoRA Punica.
    LORA_BGMV_SGMV_CUSTOM_OPS = auto()
    # Supports MLA decode-prolog execution without rotary embeddings.
    MLA_DECODE_PROLOG_WITHOUT_ROPE = auto()
    # Allows MLAPO to consume unquantized projection weights directly.
    MLAPO_UNQUANTIZED_PROJECTION_WEIGHTS = auto()
    # Supports the fullmesh_v2 MC2 communication algorithm.
    MC2_FULLMESH_V2 = auto()
    # Supports hierarchical MC2 dispatch and combine communication.
    MC2_HIERARCHY_COMM = auto()
    # Uses paged scatter updates for the MiniMax-M3 index-K cache.
    MINIMAX_M3_PAGED_INDEX_CACHE_SCATTER = auto()
    # Provides the fused MiniMax-M3 QKV RMSNorm and RoPE operator.
    MINIMAX_M3_FUSED_QKV_RMSNORM_ROPE = auto()
    # Provides the fused clipped-SwiGLU operator used by MiniMax-M3.
    MINIMAX_M3_FUSED_CLIPPED_SWIGLU = auto()
    # Quantizes MiniMax-M3 SwiGLU output for MXFP8 projections.
    MINIMAX_M3_SWIGLU_OAI_MXFP8_OUTPUT_QUANTIZATION = auto()
    # Supports TP-sharded sparse-index decode for MiniMax-M3.
    MINIMAX_M3_TP_SHARDED_INDEX_DECODE = auto()
    # Enables the NPUGraph extended compilation backend.
    NPUGRAPH_EX_COMPILATION_BACKEND = auto()
    # Provides the native NPU top-k/top-p sampling operator.
    NATIVE_TOP_K_TOP_P_SAMPLING = auto()
    # Provides the full paged-attention decode kernel.
    PAGED_ATTENTION_FULL_DECODE = auto()
    # Detects PCIe root-complex mode from the host's NPU topology.
    PCIE_ROOT_COMPLEX_MODE_DETECTION = auto()
    # Supports lazy registration and dispatch of the Ascend custom-op library.
    ASCEND_CUSTOM_OP_RUNTIME_REGISTRATION = auto()
    # Replicates the SFA indexer cache under decode-context parallelism.
    SFA_DCP_REPLICATED_INDEXER = auto()
    # Provides Triton batch memcpy for Mamba state movement.
    TRITON_MAMBA_STATE_BATCH_MEMCPY = auto()
    # Allows the Xlite graph worker implementation.
    XLITE_GRAPH_WORKER = auto()


class AttentionBackendFamily(Enum):
    """Attention backend implementation families selected by the platform."""

    DENSE_MLA_SFA_DSA = auto()
    DENSE_ONLY = auto()


class CompilationCustomOpPolicy(Enum):
    """How platform setup updates ``compilation_config.custom_ops``."""

    PRESERVE_CONFIGURATION = auto()
    FORCE_ENABLE_ALL = auto()


class DistributedCollectiveFamily(Enum):
    """Distributed collective implementations used by runtime patches."""

    TORCH_NPU_NATIVE = auto()
    BROADCAST_AND_INT64_ALLREDUCE_VIA_ALLGATHER = auto()


class CPUBindingMode(Enum):
    """CPU binding policies selected for worker processes."""

    TOPO_AFFINITY = "topo_affinity"
    GLOBAL_SLICE = "global_slice"


class DeviceAdaptorFamily(Enum):
    """Device operation adaptor implementation families."""

    BASE_DEVICE_OPERATIONS = auto()
    # Adds FP8/MXFP attention and MoE, DSA cache/indexer, MC2/GDN, and
    # model-specific fused operator implementations.
    FP8_MXFP_DSA_AND_FUSED_MODEL_OPERATIONS = auto()
    RESHAPE_CACHE_AND_INDEX_FILL_OPERATIONS = auto()


class DeviceAddressingMode(Enum):
    """PCIe device addressing policies used by CPU binding."""

    DIRECT = auto()
    DUAL_CHIP_CARD = auto()


class DSAOProjWeightLayoutPolicy(Enum):
    """Layout conversion applied while loading DSA ``wo_a`` weights."""

    ALWAYS_TRANSPOSE_WO_A = auto()
    TRANSPOSE_UNQUANTIZED_BF16_WO_A_ONLY = auto()


class GDNPatchFamily(Enum):
    """Gated-delta-net monkey-patch coverage."""

    FULL_FORWARD_PATCH = auto()
    CORE_AND_DTYPE_PATCH = auto()


class MambaConfigPatchFamily(Enum):
    """Mamba cache-layout validation implementations."""

    MLA_OR_DENSE_CACHE_LAYOUT = auto()
    DENSE_ATTENTION_CACHE_ONLY = auto()


class MC2FullmeshAliasPolicy(Enum):
    """Operator argument emitted for the configured ``fullmesh`` alias."""

    PASSTHROUGH = auto()
    MAP_FULLMESH_TO_V1 = auto()


class MiniMaxM3IndexerFamily(Enum):
    """MiniMax-M3 index-score, top-k, and decode implementation bundle."""

    ASCENDC_SCORE_AND_DECODE_WITH_TRITON_TOPK = auto()
    TRITON_SCORE_TOPK_DECODE = auto()


class MiniMaxM3SparseAttentionPrefillFamily(Enum):
    """MiniMax-M3 sparse-attention prefill metadata contract."""

    SELECT_INDEX_AND_COUNT_SCORE = auto()
    K2Q_CSR_SCORE = auto()


class MLAPOEnablementPolicy(Enum):
    """Runtime instances on which configured MLAPO may be enabled."""

    DECODE_KV_CONSUMER_ONLY = auto()
    ANY_CONFIGURED_INSTANCE = auto()


class ModelRunnerV2ImplementationFamily(Enum):
    """Model Runner V2 validation, state, and block-table implementation."""

    DEFAULT_TRITON_BACKED = auto()
    TRITON_FREE_HOST_METADATA = auto()


class MoEDistributeApiFamily(Enum):
    """Baseline CANN MoE distribute-v2 metadata contract.

    A selected communication algorithm may add its own operands, such as
    ``expert_scales`` for hierarchical communication.
    """

    EP_METADATA = auto()
    EP_AND_TP_METADATA = auto()
    # Adds always-on expert scales plus MXFP quant mode and output dtype.
    EP_TP_WITH_EXPERT_SCALES_MXFP_MODE_AND_QUANT_OUTPUT_DTYPE = auto()


class MoECommPolicy(Enum):
    """MoE communication selection policies."""

    # MC2 requires capacity plus a dense-enough expert placement; otherwise
    # use all-gather.
    MC2_IF_CAPACITY_AND_EXPERT_DENSITY_ELSE_ALLGATHER = auto()
    # Prefer fused MC2, fall back to capacity-bounded MC2, then all-to-all.
    FUSED_MC2_THEN_CAPACITY_MC2_ELSE_ALLTOALL = auto()
    # Use MC2 within capacity; otherwise choose all-gather when world size is
    # no greater than top-k, and all-to-all for larger worlds.
    MC2_IF_CAPACITY_ELSE_ALLGATHER_OR_ALLTOALL_BY_WORLD_SIZE = auto()
    ALLGATHER = auto()


class MoERouterFamily(Enum):
    """Top-k routing implementation used by fused MoE layers."""

    FUSED_OR_GROUPED_TOP_K = auto()
    CHUNKED_SOFTMAX_TOP_K = auto()


class OperatorRegistryFamily(Enum):
    """Custom-op implementation set registered for shared business code."""

    BASE_ASCEND_IMPLEMENTATIONS = auto()
    # Overrides activation, rotary, normalization, embedding/LM-head,
    # multimodal attention, Conv3d, GDN, and MoE implementations.
    BASE_WITH_TRITON_FREE_KERNEL_OVERRIDES = auto()


class QuantizationBackendFamily(Enum):
    """Quantization configuration implementation families."""

    COMPRESSED_TENSORS_FP8_MXFP8_AND_MODELSLIM = auto()
    MODELSLIM_ONLY = auto()


class WorkerPatchFamily(Enum):
    """Worker-time model patch bundles."""

    QWEN3_5_DFLASH_AND_VL = auto()
    TRITON_FREE_GDN_QWEN3_VL_AND_SPEC_DECODE = auto()


class WeightLayoutPolicy(Enum):
    """Weight layout selection policies for supported hardware families."""

    CONFIGURABLE = auto()
    FORCE_NZ = auto()


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Immutable capabilities and implementation choices for one SoC family."""

    _device_type: AscendDeviceType
    attention_backend_family: AttentionBackendFamily
    atb_matmul_warmup_required: bool
    compilation_custom_op_policy: CompilationCustomOpPolicy
    cpu_binding_mode: CPUBindingMode
    cudagraph_capture_size_limit: int | None
    default_worker_cls: str
    device_adaptor_family: DeviceAdaptorFamily
    device_addressing_mode: DeviceAddressingMode
    distributed_collective_family: DistributedCollectiveFamily
    dsa_c128_block_sizes: tuple[int, ...]
    dsa_o_proj_weight_layout_policy: DSAOProjWeightLayoutPolicy
    gdn_patch_family: GDNPatchFamily
    mamba_config_patch_family: MambaConfigPatchFamily
    mc2_fullmesh_alias_policy: MC2FullmeshAliasPolicy
    minimax_m3_indexer_family: MiniMaxM3IndexerFamily
    minimax_m3_sparse_attention_prefill_family: MiniMaxM3SparseAttentionPrefillFamily
    mlapo_enablement_policy: MLAPOEnablementPolicy
    model_runner_v2_implementation_family: ModelRunnerV2ImplementationFamily
    moe_comm_policy: MoECommPolicy
    moe_distribute_api_family: MoEDistributeApiFamily
    moe_router_family: MoERouterFamily
    operator_registry_family: OperatorRegistryFamily
    quantization_backend_family: QuantizationBackendFamily
    reserve_irq_cpus: bool
    split_fia_chunked_prefill_by_default: bool
    weight_layout_policy: WeightLayoutPolicy
    worker_patch_family: WorkerPatchFamily
    capabilities: frozenset[HardwareCapability]

    def supports(self, capability: HardwareCapability) -> bool:
        """Return whether this hardware family provides ``capability``."""

        return capability in self.capabilities


_A2_A3_COMMON_CAPABILITIES = frozenset(
    {
        HardwareCapability.ATB_MODEL_EXECUTION_EXTENSIONS,
        HardwareCapability.BGMV_SGMV_META_KERNELS,
        HardwareCapability.DEQUANT_SWIGLU_GLU_ALPHA_BIAS_ARGS,
        HardwareCapability.GRAPH_MULS_ADD_FUSION,
        HardwareCapability.GRAPH_NORM_QUANT_FUSION,
        HardwareCapability.LORA_BGMV_SGMV_CUSTOM_OPS,
        HardwareCapability.MC2_HIERARCHY_COMM,
        HardwareCapability.NPUGRAPH_EX_COMPILATION_BACKEND,
        HardwareCapability.PAGED_ATTENTION_FULL_DECODE,
        HardwareCapability.ASCEND_CUSTOM_OP_RUNTIME_REGISTRATION,
        HardwareCapability.SFA_DCP_REPLICATED_INDEXER,
        HardwareCapability.TRITON_MAMBA_STATE_BATCH_MEMCPY,
        HardwareCapability.XLITE_GRAPH_WORKER,
    }
)
_MINIMAX_M3_FALLBACK_CAPABILITIES = frozenset(
    {
        # Preserve the pre-profile ``device != A5`` behavior.  These entries
        # do not by themselves declare MiniMax-M3 support on every family.
        HardwareCapability.MINIMAX_M3_FUSED_CLIPPED_SWIGLU,
        HardwareCapability.MINIMAX_M3_FUSED_QKV_RMSNORM_ROPE,
        HardwareCapability.MINIMAX_M3_TP_SHARDED_INDEX_DECODE,
    }
)
_DEFAULT_WORKER_CLS = "vllm_ascend.worker.worker.NPUWorker"
_HARDWARE_PROFILES: Mapping[AscendDeviceType, HardwareProfile] = MappingProxyType(
    {
        AscendDeviceType.A2: HardwareProfile(
            _device_type=AscendDeviceType.A2,
            attention_backend_family=AttentionBackendFamily.DENSE_MLA_SFA_DSA,
            atb_matmul_warmup_required=True,
            compilation_custom_op_policy=CompilationCustomOpPolicy.FORCE_ENABLE_ALL,
            cpu_binding_mode=CPUBindingMode.TOPO_AFFINITY,
            cudagraph_capture_size_limit=None,
            default_worker_cls=_DEFAULT_WORKER_CLS,
            device_adaptor_family=DeviceAdaptorFamily.BASE_DEVICE_OPERATIONS,
            device_addressing_mode=DeviceAddressingMode.DIRECT,
            distributed_collective_family=DistributedCollectiveFamily.TORCH_NPU_NATIVE,
            dsa_c128_block_sizes=(8, 16, 32),
            dsa_o_proj_weight_layout_policy=DSAOProjWeightLayoutPolicy.ALWAYS_TRANSPOSE_WO_A,
            gdn_patch_family=GDNPatchFamily.FULL_FORWARD_PATCH,
            mamba_config_patch_family=MambaConfigPatchFamily.MLA_OR_DENSE_CACHE_LAYOUT,
            mc2_fullmesh_alias_policy=MC2FullmeshAliasPolicy.PASSTHROUGH,
            minimax_m3_indexer_family=MiniMaxM3IndexerFamily.ASCENDC_SCORE_AND_DECODE_WITH_TRITON_TOPK,
            minimax_m3_sparse_attention_prefill_family=MiniMaxM3SparseAttentionPrefillFamily.SELECT_INDEX_AND_COUNT_SCORE,
            mlapo_enablement_policy=MLAPOEnablementPolicy.DECODE_KV_CONSUMER_ONLY,
            model_runner_v2_implementation_family=ModelRunnerV2ImplementationFamily.DEFAULT_TRITON_BACKED,
            moe_comm_policy=MoECommPolicy.MC2_IF_CAPACITY_AND_EXPERT_DENSITY_ELSE_ALLGATHER,
            moe_distribute_api_family=MoEDistributeApiFamily.EP_METADATA,
            moe_router_family=MoERouterFamily.FUSED_OR_GROUPED_TOP_K,
            operator_registry_family=OperatorRegistryFamily.BASE_ASCEND_IMPLEMENTATIONS,
            quantization_backend_family=QuantizationBackendFamily.COMPRESSED_TENSORS_FP8_MXFP8_AND_MODELSLIM,
            reserve_irq_cpus=True,
            split_fia_chunked_prefill_by_default=False,
            weight_layout_policy=WeightLayoutPolicy.CONFIGURABLE,
            worker_patch_family=WorkerPatchFamily.QWEN3_5_DFLASH_AND_VL,
            capabilities=_A2_A3_COMMON_CAPABILITIES
            | _MINIMAX_M3_FALLBACK_CAPABILITIES
            | {HardwareCapability.NATIVE_TOP_K_TOP_P_SAMPLING},
        ),
        AscendDeviceType.A3: HardwareProfile(
            _device_type=AscendDeviceType.A3,
            attention_backend_family=AttentionBackendFamily.DENSE_MLA_SFA_DSA,
            atb_matmul_warmup_required=True,
            compilation_custom_op_policy=CompilationCustomOpPolicy.FORCE_ENABLE_ALL,
            cpu_binding_mode=CPUBindingMode.GLOBAL_SLICE,
            cudagraph_capture_size_limit=None,
            default_worker_cls=_DEFAULT_WORKER_CLS,
            device_adaptor_family=DeviceAdaptorFamily.BASE_DEVICE_OPERATIONS,
            device_addressing_mode=DeviceAddressingMode.DUAL_CHIP_CARD,
            distributed_collective_family=DistributedCollectiveFamily.TORCH_NPU_NATIVE,
            dsa_c128_block_sizes=(8, 16, 32),
            dsa_o_proj_weight_layout_policy=DSAOProjWeightLayoutPolicy.ALWAYS_TRANSPOSE_WO_A,
            gdn_patch_family=GDNPatchFamily.FULL_FORWARD_PATCH,
            mamba_config_patch_family=MambaConfigPatchFamily.MLA_OR_DENSE_CACHE_LAYOUT,
            mc2_fullmesh_alias_policy=MC2FullmeshAliasPolicy.MAP_FULLMESH_TO_V1,
            minimax_m3_indexer_family=MiniMaxM3IndexerFamily.ASCENDC_SCORE_AND_DECODE_WITH_TRITON_TOPK,
            minimax_m3_sparse_attention_prefill_family=MiniMaxM3SparseAttentionPrefillFamily.SELECT_INDEX_AND_COUNT_SCORE,
            mlapo_enablement_policy=MLAPOEnablementPolicy.DECODE_KV_CONSUMER_ONLY,
            model_runner_v2_implementation_family=ModelRunnerV2ImplementationFamily.DEFAULT_TRITON_BACKED,
            moe_comm_policy=MoECommPolicy.FUSED_MC2_THEN_CAPACITY_MC2_ELSE_ALLTOALL,
            moe_distribute_api_family=MoEDistributeApiFamily.EP_AND_TP_METADATA,
            moe_router_family=MoERouterFamily.FUSED_OR_GROUPED_TOP_K,
            operator_registry_family=OperatorRegistryFamily.BASE_ASCEND_IMPLEMENTATIONS,
            quantization_backend_family=QuantizationBackendFamily.COMPRESSED_TENSORS_FP8_MXFP8_AND_MODELSLIM,
            reserve_irq_cpus=True,
            split_fia_chunked_prefill_by_default=False,
            weight_layout_policy=WeightLayoutPolicy.CONFIGURABLE,
            worker_patch_family=WorkerPatchFamily.QWEN3_5_DFLASH_AND_VL,
            capabilities=_A2_A3_COMMON_CAPABILITIES
            | _MINIMAX_M3_FALLBACK_CAPABILITIES
            | {
                HardwareCapability.CANN_MEGAMOE_FUSED_MC2,
                HardwareCapability.MC2_FULLMESH_V2,
                HardwareCapability.NATIVE_TOP_K_TOP_P_SAMPLING,
            },
        ),
        AscendDeviceType._310P: HardwareProfile(
            _device_type=AscendDeviceType._310P,
            attention_backend_family=AttentionBackendFamily.DENSE_ONLY,
            atb_matmul_warmup_required=False,
            compilation_custom_op_policy=CompilationCustomOpPolicy.PRESERVE_CONFIGURATION,
            cpu_binding_mode=CPUBindingMode.TOPO_AFFINITY,
            cudagraph_capture_size_limit=None,
            default_worker_cls="vllm_ascend._310p.worker_310p.NPUWorker310",
            device_adaptor_family=DeviceAdaptorFamily.RESHAPE_CACHE_AND_INDEX_FILL_OPERATIONS,
            device_addressing_mode=DeviceAddressingMode.DIRECT,
            distributed_collective_family=DistributedCollectiveFamily.BROADCAST_AND_INT64_ALLREDUCE_VIA_ALLGATHER,
            dsa_c128_block_sizes=(8, 16, 32),
            dsa_o_proj_weight_layout_policy=DSAOProjWeightLayoutPolicy.ALWAYS_TRANSPOSE_WO_A,
            gdn_patch_family=GDNPatchFamily.CORE_AND_DTYPE_PATCH,
            mamba_config_patch_family=MambaConfigPatchFamily.DENSE_ATTENTION_CACHE_ONLY,
            mc2_fullmesh_alias_policy=MC2FullmeshAliasPolicy.PASSTHROUGH,
            minimax_m3_indexer_family=MiniMaxM3IndexerFamily.ASCENDC_SCORE_AND_DECODE_WITH_TRITON_TOPK,
            minimax_m3_sparse_attention_prefill_family=MiniMaxM3SparseAttentionPrefillFamily.SELECT_INDEX_AND_COUNT_SCORE,
            mlapo_enablement_policy=MLAPOEnablementPolicy.DECODE_KV_CONSUMER_ONLY,
            model_runner_v2_implementation_family=ModelRunnerV2ImplementationFamily.TRITON_FREE_HOST_METADATA,
            moe_comm_policy=MoECommPolicy.ALLGATHER,
            moe_distribute_api_family=MoEDistributeApiFamily.EP_METADATA,
            moe_router_family=MoERouterFamily.CHUNKED_SOFTMAX_TOP_K,
            operator_registry_family=OperatorRegistryFamily.BASE_WITH_TRITON_FREE_KERNEL_OVERRIDES,
            quantization_backend_family=QuantizationBackendFamily.MODELSLIM_ONLY,
            reserve_irq_cpus=True,
            split_fia_chunked_prefill_by_default=False,
            weight_layout_policy=WeightLayoutPolicy.FORCE_NZ,
            worker_patch_family=WorkerPatchFamily.TRITON_FREE_GDN_QWEN3_VL_AND_SPEC_DECODE,
            capabilities=_MINIMAX_M3_FALLBACK_CAPABILITIES
            | frozenset(
                {
                    HardwareCapability.DEQUANT_SWIGLU_GLU_ALPHA_BIAS_ARGS,
                    HardwareCapability.PCIE_ROOT_COMPLEX_MODE_DETECTION,
                    HardwareCapability.ASCEND_CUSTOM_OP_RUNTIME_REGISTRATION,
                }
            ),
        ),
        AscendDeviceType.A5: HardwareProfile(
            _device_type=AscendDeviceType.A5,
            attention_backend_family=AttentionBackendFamily.DENSE_MLA_SFA_DSA,
            atb_matmul_warmup_required=False,
            compilation_custom_op_policy=CompilationCustomOpPolicy.FORCE_ENABLE_ALL,
            cpu_binding_mode=CPUBindingMode.TOPO_AFFINITY,
            cudagraph_capture_size_limit=4,
            default_worker_cls=_DEFAULT_WORKER_CLS,
            device_adaptor_family=DeviceAdaptorFamily.FP8_MXFP_DSA_AND_FUSED_MODEL_OPERATIONS,
            device_addressing_mode=DeviceAddressingMode.DIRECT,
            distributed_collective_family=DistributedCollectiveFamily.TORCH_NPU_NATIVE,
            dsa_c128_block_sizes=(4, 8, 16),
            dsa_o_proj_weight_layout_policy=DSAOProjWeightLayoutPolicy.TRANSPOSE_UNQUANTIZED_BF16_WO_A_ONLY,
            gdn_patch_family=GDNPatchFamily.FULL_FORWARD_PATCH,
            mamba_config_patch_family=MambaConfigPatchFamily.MLA_OR_DENSE_CACHE_LAYOUT,
            mc2_fullmesh_alias_policy=MC2FullmeshAliasPolicy.PASSTHROUGH,
            minimax_m3_indexer_family=MiniMaxM3IndexerFamily.TRITON_SCORE_TOPK_DECODE,
            minimax_m3_sparse_attention_prefill_family=MiniMaxM3SparseAttentionPrefillFamily.K2Q_CSR_SCORE,
            mlapo_enablement_policy=MLAPOEnablementPolicy.ANY_CONFIGURED_INSTANCE,
            model_runner_v2_implementation_family=ModelRunnerV2ImplementationFamily.DEFAULT_TRITON_BACKED,
            moe_comm_policy=MoECommPolicy.MC2_IF_CAPACITY_ELSE_ALLGATHER_OR_ALLTOALL_BY_WORLD_SIZE,
            moe_distribute_api_family=MoEDistributeApiFamily.EP_TP_WITH_EXPERT_SCALES_MXFP_MODE_AND_QUANT_OUTPUT_DTYPE,
            moe_router_family=MoERouterFamily.FUSED_OR_GROUPED_TOP_K,
            operator_registry_family=OperatorRegistryFamily.BASE_ASCEND_IMPLEMENTATIONS,
            quantization_backend_family=QuantizationBackendFamily.COMPRESSED_TENSORS_FP8_MXFP8_AND_MODELSLIM,
            reserve_irq_cpus=False,
            split_fia_chunked_prefill_by_default=True,
            weight_layout_policy=WeightLayoutPolicy.CONFIGURABLE,
            worker_patch_family=WorkerPatchFamily.QWEN3_5_DFLASH_AND_VL,
            capabilities=frozenset(
                {
                    HardwareCapability.BGMV_SGMV_META_KERNELS,
                    HardwareCapability.BLOCK_FP8_TO_MXFP8_REQUANTIZATION,
                    HardwareCapability.CLUSTER_CPU_TOPOLOGY,
                    HardwareCapability.DSA_O_PROJ_TP,
                    HardwareCapability.DSA_COMPRESSED_KV_CACHE,
                    HardwareCapability.DYNAMIC_MX_QUANT_E8M0_OVERFLOW_SAFE_SCALING,
                    HardwareCapability.FP8_ATTENTION,
                    HardwareCapability.GRAPH_DYNAMIC_MX_QUANT_FUSION,
                    HardwareCapability.GRAPH_MULS_ADD_FUSION,
                    HardwareCapability.GRAPH_NORM_QUANT_FUSION,
                    HardwareCapability.LOCAL_KV_TRANSFER_COMM_RESOURCE,
                    HardwareCapability.LORA_BGMV_SGMV_CUSTOM_OPS,
                    HardwareCapability.MLA_DECODE_PROLOG_WITHOUT_ROPE,
                    HardwareCapability.MLAPO_UNQUANTIZED_PROJECTION_WEIGHTS,
                    HardwareCapability.MINIMAX_M3_PAGED_INDEX_CACHE_SCATTER,
                    HardwareCapability.MINIMAX_M3_SWIGLU_OAI_MXFP8_OUTPUT_QUANTIZATION,
                    HardwareCapability.NPUGRAPH_EX_COMPILATION_BACKEND,
                    HardwareCapability.TRITON_MAMBA_STATE_BATCH_MEMCPY,
                    HardwareCapability.XLITE_GRAPH_WORKER,
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
