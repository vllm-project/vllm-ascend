# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import fields
from typing import cast

import pytest
import torch

from vllm_ascend.device.device_config import get_device_config
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import (
    AttentionBackendFamily,
    CompilationCustomOpPolicy,
    CPUBindingMode,
    DeviceAdaptorFamily,
    DeviceAddressingMode,
    DistributedCollectiveFamily,
    DSAOProjWeightLayoutPolicy,
    GDNPatchFamily,
    HardwareCapability,
    HardwareProfile,
    MambaConfigPatchFamily,
    MC2FullmeshAliasPolicy,
    MiniMaxM3IndexerFamily,
    MiniMaxM3SparseAttentionPrefillFamily,
    MLAPOEnablementPolicy,
    ModelRunnerV2ImplementationFamily,
    MoECommPolicy,
    MoEDistributeApiFamily,
    MoERouterFamily,
    OperatorRegistryFamily,
    QuantizationBackendFamily,
    WeightLayoutPolicy,
    WorkerPatchFamily,
    get_current_hardware_profile,
    get_hardware_profile,
)

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
        # These preserve the previous ``device != A5`` behavior; they do not
        # independently declare MiniMax-M3 support on every listed family.
        HardwareCapability.MINIMAX_M3_FUSED_CLIPPED_SWIGLU,
        HardwareCapability.MINIMAX_M3_FUSED_QKV_RMSNORM_ROPE,
        HardwareCapability.MINIMAX_M3_TP_SHARDED_INDEX_DECODE,
    }
)

_EXPECTED_CAPABILITIES = {
    AscendDeviceType.A2: _A2_A3_COMMON_CAPABILITIES
    | _MINIMAX_M3_FALLBACK_CAPABILITIES
    | {HardwareCapability.NATIVE_TOP_K_TOP_P_SAMPLING},
    AscendDeviceType.A3: _A2_A3_COMMON_CAPABILITIES
    | _MINIMAX_M3_FALLBACK_CAPABILITIES
    | {
        HardwareCapability.CANN_MEGAMOE_FUSED_MC2,
        HardwareCapability.MC2_FULLMESH_V2,
        HardwareCapability.NATIVE_TOP_K_TOP_P_SAMPLING,
    },
    AscendDeviceType._310P: _MINIMAX_M3_FALLBACK_CAPABILITIES
    | frozenset(
        {
            HardwareCapability.DEQUANT_SWIGLU_GLU_ALPHA_BIAS_ARGS,
            HardwareCapability.PCIE_ROOT_COMPLEX_MODE_DETECTION,
            HardwareCapability.ASCEND_CUSTOM_OP_RUNTIME_REGISTRATION,
        }
    ),
    AscendDeviceType.A5: frozenset(
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
}

_EXPECTED_PROFILE_FIELDS: dict[AscendDeviceType, dict[str, object]] = {
    AscendDeviceType.A2: {
        "attention_backend_family": AttentionBackendFamily.DENSE_MLA_SFA_DSA,
        "atb_matmul_warmup_required": True,
        "cpu_binding_mode": CPUBindingMode.TOPO_AFFINITY,
        "cudagraph_capture_size_limit": None,
        "default_worker_cls": "vllm_ascend.worker.worker.NPUWorker",
        "device_adaptor_family": DeviceAdaptorFamily.BASE_DEVICE_OPERATIONS,
        "device_addressing_mode": DeviceAddressingMode.DIRECT,
        "distributed_collective_family": DistributedCollectiveFamily.TORCH_NPU_NATIVE,
        "dsa_c128_block_sizes": (8, 16, 32),
        "dsa_o_proj_weight_layout_policy": DSAOProjWeightLayoutPolicy.ALWAYS_TRANSPOSE_WO_A,
        "compilation_custom_op_policy": CompilationCustomOpPolicy.FORCE_ENABLE_ALL,
        "gdn_patch_family": GDNPatchFamily.FULL_FORWARD_PATCH,
        "mamba_config_patch_family": MambaConfigPatchFamily.MLA_OR_DENSE_CACHE_LAYOUT,
        "mc2_fullmesh_alias_policy": MC2FullmeshAliasPolicy.PASSTHROUGH,
        "minimax_m3_indexer_family": MiniMaxM3IndexerFamily.ASCENDC_SCORE_AND_DECODE_WITH_TRITON_TOPK,
        "minimax_m3_sparse_attention_prefill_family": (
            MiniMaxM3SparseAttentionPrefillFamily.SELECT_INDEX_AND_COUNT_SCORE
        ),
        "mlapo_enablement_policy": MLAPOEnablementPolicy.DECODE_KV_CONSUMER_ONLY,
        "model_runner_v2_implementation_family": ModelRunnerV2ImplementationFamily.DEFAULT_TRITON_BACKED,
        "moe_comm_policy": MoECommPolicy.MC2_IF_CAPACITY_AND_EXPERT_DENSITY_ELSE_ALLGATHER,
        "moe_distribute_api_family": MoEDistributeApiFamily.EP_METADATA,
        "moe_router_family": MoERouterFamily.FUSED_OR_GROUPED_TOP_K,
        "operator_registry_family": OperatorRegistryFamily.BASE_ASCEND_IMPLEMENTATIONS,
        "quantization_backend_family": QuantizationBackendFamily.COMPRESSED_TENSORS_FP8_MXFP8_AND_MODELSLIM,
        "reserve_irq_cpus": True,
        "split_fia_chunked_prefill_by_default": False,
        "weight_layout_policy": WeightLayoutPolicy.CONFIGURABLE,
        "worker_patch_family": WorkerPatchFamily.QWEN3_5_DFLASH_AND_VL,
    },
    AscendDeviceType.A3: {
        "attention_backend_family": AttentionBackendFamily.DENSE_MLA_SFA_DSA,
        "atb_matmul_warmup_required": True,
        "cpu_binding_mode": CPUBindingMode.GLOBAL_SLICE,
        "cudagraph_capture_size_limit": None,
        "default_worker_cls": "vllm_ascend.worker.worker.NPUWorker",
        "device_adaptor_family": DeviceAdaptorFamily.BASE_DEVICE_OPERATIONS,
        "device_addressing_mode": DeviceAddressingMode.DUAL_CHIP_CARD,
        "distributed_collective_family": DistributedCollectiveFamily.TORCH_NPU_NATIVE,
        "dsa_c128_block_sizes": (8, 16, 32),
        "dsa_o_proj_weight_layout_policy": DSAOProjWeightLayoutPolicy.ALWAYS_TRANSPOSE_WO_A,
        "compilation_custom_op_policy": CompilationCustomOpPolicy.FORCE_ENABLE_ALL,
        "gdn_patch_family": GDNPatchFamily.FULL_FORWARD_PATCH,
        "mamba_config_patch_family": MambaConfigPatchFamily.MLA_OR_DENSE_CACHE_LAYOUT,
        "mc2_fullmesh_alias_policy": MC2FullmeshAliasPolicy.MAP_FULLMESH_TO_V1,
        "minimax_m3_indexer_family": MiniMaxM3IndexerFamily.ASCENDC_SCORE_AND_DECODE_WITH_TRITON_TOPK,
        "minimax_m3_sparse_attention_prefill_family": (
            MiniMaxM3SparseAttentionPrefillFamily.SELECT_INDEX_AND_COUNT_SCORE
        ),
        "mlapo_enablement_policy": MLAPOEnablementPolicy.DECODE_KV_CONSUMER_ONLY,
        "model_runner_v2_implementation_family": ModelRunnerV2ImplementationFamily.DEFAULT_TRITON_BACKED,
        "moe_comm_policy": MoECommPolicy.FUSED_MC2_THEN_CAPACITY_MC2_ELSE_ALLTOALL,
        "moe_distribute_api_family": MoEDistributeApiFamily.EP_AND_TP_METADATA,
        "moe_router_family": MoERouterFamily.FUSED_OR_GROUPED_TOP_K,
        "operator_registry_family": OperatorRegistryFamily.BASE_ASCEND_IMPLEMENTATIONS,
        "quantization_backend_family": QuantizationBackendFamily.COMPRESSED_TENSORS_FP8_MXFP8_AND_MODELSLIM,
        "reserve_irq_cpus": True,
        "split_fia_chunked_prefill_by_default": False,
        "weight_layout_policy": WeightLayoutPolicy.CONFIGURABLE,
        "worker_patch_family": WorkerPatchFamily.QWEN3_5_DFLASH_AND_VL,
    },
    AscendDeviceType._310P: {
        "attention_backend_family": AttentionBackendFamily.DENSE_ONLY,
        "atb_matmul_warmup_required": False,
        "cpu_binding_mode": CPUBindingMode.TOPO_AFFINITY,
        "cudagraph_capture_size_limit": None,
        "default_worker_cls": "vllm_ascend._310p.worker_310p.NPUWorker310",
        "device_adaptor_family": DeviceAdaptorFamily.RESHAPE_CACHE_AND_INDEX_FILL_OPERATIONS,
        "device_addressing_mode": DeviceAddressingMode.DIRECT,
        "distributed_collective_family": DistributedCollectiveFamily.BROADCAST_AND_INT64_ALLREDUCE_VIA_ALLGATHER,
        "dsa_c128_block_sizes": (8, 16, 32),
        "dsa_o_proj_weight_layout_policy": DSAOProjWeightLayoutPolicy.ALWAYS_TRANSPOSE_WO_A,
        "compilation_custom_op_policy": CompilationCustomOpPolicy.PRESERVE_CONFIGURATION,
        "gdn_patch_family": GDNPatchFamily.CORE_AND_DTYPE_PATCH,
        "mamba_config_patch_family": MambaConfigPatchFamily.DENSE_ATTENTION_CACHE_ONLY,
        "mc2_fullmesh_alias_policy": MC2FullmeshAliasPolicy.PASSTHROUGH,
        "minimax_m3_indexer_family": MiniMaxM3IndexerFamily.ASCENDC_SCORE_AND_DECODE_WITH_TRITON_TOPK,
        "minimax_m3_sparse_attention_prefill_family": (
            MiniMaxM3SparseAttentionPrefillFamily.SELECT_INDEX_AND_COUNT_SCORE
        ),
        "mlapo_enablement_policy": MLAPOEnablementPolicy.DECODE_KV_CONSUMER_ONLY,
        "model_runner_v2_implementation_family": ModelRunnerV2ImplementationFamily.TRITON_FREE_HOST_METADATA,
        "moe_comm_policy": MoECommPolicy.ALLGATHER,
        "moe_distribute_api_family": MoEDistributeApiFamily.EP_METADATA,
        "moe_router_family": MoERouterFamily.CHUNKED_SOFTMAX_TOP_K,
        "operator_registry_family": OperatorRegistryFamily.BASE_WITH_TRITON_FREE_KERNEL_OVERRIDES,
        "quantization_backend_family": QuantizationBackendFamily.MODELSLIM_ONLY,
        "reserve_irq_cpus": True,
        "split_fia_chunked_prefill_by_default": False,
        "weight_layout_policy": WeightLayoutPolicy.FORCE_NZ,
        "worker_patch_family": WorkerPatchFamily.TRITON_FREE_GDN_QWEN3_VL_AND_SPEC_DECODE,
    },
    AscendDeviceType.A5: {
        "attention_backend_family": AttentionBackendFamily.DENSE_MLA_SFA_DSA,
        "atb_matmul_warmup_required": False,
        "cpu_binding_mode": CPUBindingMode.TOPO_AFFINITY,
        "cudagraph_capture_size_limit": 4,
        "default_worker_cls": "vllm_ascend.worker.worker.NPUWorker",
        "device_adaptor_family": DeviceAdaptorFamily.FP8_MXFP_DSA_AND_FUSED_MODEL_OPERATIONS,
        "device_addressing_mode": DeviceAddressingMode.DIRECT,
        "distributed_collective_family": DistributedCollectiveFamily.TORCH_NPU_NATIVE,
        "dsa_c128_block_sizes": (4, 8, 16),
        "dsa_o_proj_weight_layout_policy": DSAOProjWeightLayoutPolicy.TRANSPOSE_UNQUANTIZED_BF16_WO_A_ONLY,
        "compilation_custom_op_policy": CompilationCustomOpPolicy.FORCE_ENABLE_ALL,
        "gdn_patch_family": GDNPatchFamily.FULL_FORWARD_PATCH,
        "mamba_config_patch_family": MambaConfigPatchFamily.MLA_OR_DENSE_CACHE_LAYOUT,
        "mc2_fullmesh_alias_policy": MC2FullmeshAliasPolicy.PASSTHROUGH,
        "minimax_m3_indexer_family": MiniMaxM3IndexerFamily.TRITON_SCORE_TOPK_DECODE,
        "minimax_m3_sparse_attention_prefill_family": MiniMaxM3SparseAttentionPrefillFamily.K2Q_CSR_SCORE,
        "mlapo_enablement_policy": MLAPOEnablementPolicy.ANY_CONFIGURED_INSTANCE,
        "model_runner_v2_implementation_family": ModelRunnerV2ImplementationFamily.DEFAULT_TRITON_BACKED,
        "moe_comm_policy": MoECommPolicy.MC2_IF_CAPACITY_ELSE_ALLGATHER_OR_ALLTOALL_BY_WORLD_SIZE,
        "moe_distribute_api_family": MoEDistributeApiFamily.EP_TP_WITH_EXPERT_SCALES_MXFP_MODE_AND_QUANT_OUTPUT_DTYPE,
        "moe_router_family": MoERouterFamily.FUSED_OR_GROUPED_TOP_K,
        "operator_registry_family": OperatorRegistryFamily.BASE_ASCEND_IMPLEMENTATIONS,
        "quantization_backend_family": QuantizationBackendFamily.COMPRESSED_TENSORS_FP8_MXFP8_AND_MODELSLIM,
        "reserve_irq_cpus": False,
        "split_fia_chunked_prefill_by_default": True,
        "weight_layout_policy": WeightLayoutPolicy.CONFIGURABLE,
        "worker_patch_family": WorkerPatchFamily.QWEN3_5_DFLASH_AND_VL,
    },
}


@pytest.mark.parametrize("device_type", list(AscendDeviceType))
def test_hardware_profile_implementation_matrix(device_type: AscendDeviceType) -> None:
    profile = get_hardware_profile(device_type)
    expected_fields = _EXPECTED_PROFILE_FIELDS[device_type]
    implementation_field_names = {field.name for field in fields(HardwareProfile)} - {
        "_device_type",
        "capabilities",
    }

    assert profile._device_type is device_type
    assert set(expected_fields) == implementation_field_names
    for field_name, expected in expected_fields.items():
        assert getattr(profile, field_name) == expected


@pytest.mark.parametrize("device_type", list(AscendDeviceType))
def test_hardware_profile_capability_matrix(device_type: AscendDeviceType) -> None:
    profile = get_hardware_profile(device_type)
    expected_capabilities = _EXPECTED_CAPABILITIES[device_type]

    assert profile.capabilities == expected_capabilities
    for capability in HardwareCapability:
        assert profile.supports(capability) is (capability in expected_capabilities)


def test_current_hardware_profile_uses_device_config() -> None:
    expected_profile = get_hardware_profile(get_device_config()._device_type)

    assert get_current_hardware_profile() is expected_profile


def test_current_hardware_profile_is_dynamo_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    # CPU UTs expose a mocked NPU backend that is not importable as
    # ``torch.npu``. Keep accelerator stream discovery out of this test so it
    # only exercises hardware-profile tracing.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)

    def use_profile_capability(value: torch.Tensor) -> torch.Tensor:
        if get_current_hardware_profile().supports(HardwareCapability.ASCEND_CUSTOM_OP_RUNTIME_REGISTRATION):
            return value + 1
        return value

    value = torch.ones(1)
    expected = use_profile_capability(value)
    compiled = torch.compile(use_profile_capability, backend="eager", fullgraph=True)

    assert torch.equal(compiled(value), expected)


def test_unknown_device_type_is_rejected() -> None:
    unknown_device_type = cast(AscendDeviceType, object())

    with pytest.raises(RuntimeError, match="No hardware profile is registered"):
        get_hardware_profile(unknown_device_type)


def test_every_device_type_has_a_profile() -> None:
    for device_type in AscendDeviceType:
        assert get_hardware_profile(device_type)._device_type is device_type
