# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Runtime and capability validation for the CatCCOS MoE backend."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
import torch_npu
from vllm.distributed import get_ep_group
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEFusedExpertsInput
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, AscendDeviceType, get_ascend_device_type

CATCCOS_TOKEN_ALIGNMENT = 64
CATCCOS_MAX_LOCAL_EXPERTS = 16
CATCCOS_SUPPORTED_TOP_K = 8
CATCCOS_SHMEM_BUFFER_BYTES = 1004 * 1024 * 1024


def catccos_moe_enabled() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_CATCCOS_MOE


@dataclass(frozen=True)
class CatCCOSRuntimeConfig:
    library_path: str
    store_addr: str
    local_mem_size: int

    @classmethod
    def from_env(cls) -> CatCCOSRuntimeConfig:
        library_path = envs_ascend.VLLM_ASCEND_CATCCOS_LIBRARY_PATH.strip()
        store_addr = envs_ascend.VLLM_ASCEND_CATCCOS_STORE_ADDR.strip()
        local_mem_size = envs_ascend.VLLM_ASCEND_CATCCOS_LOCAL_MEM_SIZE
        if not library_path:
            raise ValueError("VLLM_ASCEND_CATCCOS_LIBRARY_PATH is required when CatCCOS is enabled")
        library = Path(library_path)
        if not library.is_absolute():
            raise ValueError("VLLM_ASCEND_CATCCOS_LIBRARY_PATH must be absolute")
        if not library.is_file():
            raise FileNotFoundError(f"CatCCOS extension does not exist: {library}")
        if not store_addr.startswith("tcp://"):
            raise ValueError("VLLM_ASCEND_CATCCOS_STORE_ADDR must be a tcp:// rendezvous address")
        if not CATCCOS_SHMEM_BUFFER_BYTES <= local_mem_size <= 2**31 - 1:
            raise ValueError(f"VLLM_ASCEND_CATCCOS_LOCAL_MEM_SIZE must be in [{CATCCOS_SHMEM_BUFFER_BYTES}, 2^31 - 1]")
        return cls(str(library), store_addr, local_mem_size)


def _moe_config_fingerprint(moe_config: FusedMoEConfig) -> tuple[object, ...]:
    return (
        moe_config.ep_size,
        moe_config.num_experts,
        moe_config.num_local_experts,
        moe_config.experts_per_token,
        moe_config.hidden_dim,
        moe_config.intermediate_size_per_partition,
        moe_config.in_dtype,
        getattr(moe_config, "global_redundant_expert_num", 0),
    )


def validate_catccos_moe_config(moe_config: FusedMoEConfig) -> None:
    if get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3):
        raise ValueError("CatCCOS dispatch-FFN-combine requires an Atlas A2 or A3 device")
    if moe_config.in_dtype != torch.bfloat16:
        raise ValueError(f"CatCCOS requires BF16 MoE input, got {moe_config.in_dtype}")
    if moe_config.ep_size <= 1:
        raise ValueError("CatCCOS requires expert parallelism")
    if moe_config.num_experts % moe_config.ep_size:
        raise ValueError("CatCCOS requires the expert count to be divisible by the EP size")
    if moe_config.num_local_experts != moe_config.num_experts // moe_config.ep_size:
        raise ValueError("CatCCOS requires linear, non-redundant expert placement")
    if not 0 < moe_config.num_local_experts <= CATCCOS_MAX_LOCAL_EXPERTS:
        raise ValueError(
            f"CatCCOS supports at most {CATCCOS_MAX_LOCAL_EXPERTS} local experts, got {moe_config.num_local_experts}"
        )
    if moe_config.experts_per_token != CATCCOS_SUPPORTED_TOP_K:
        raise ValueError(f"CatCCOS supports TopK={CATCCOS_SUPPORTED_TOP_K}, got {moe_config.experts_per_token}")
    if moe_config.hidden_dim % 256:
        raise ValueError("CatCCOS requires the hidden dimension to be divisible by 256")
    merged_intermediate = 2 * moe_config.intermediate_size_per_partition
    if merged_intermediate % 512:
        raise ValueError("CatCCOS requires the merged intermediate dimension to be divisible by 512")
    if getattr(moe_config, "global_redundant_expert_num", 0):
        raise ValueError("CatCCOS does not support redundant experts")
    if getattr(moe_config, "has_bias", False):
        raise ValueError("CatCCOS does not support expert bias")
    if getattr(moe_config, "is_lora_enabled", False):
        raise ValueError("CatCCOS does not support MoE LoRA")
    if getattr(moe_config, "swiglu_limit", 0):
        raise ValueError("CatCCOS does not support SwiGLU clamp")
    activation = str(getattr(moe_config.activation, "value", moe_config.activation)).lower()
    if activation not in {"silu", "swiglu", "moeactivation.silu"}:
        raise ValueError(f"CatCCOS requires SiLU/SwiGLU activation, got {activation}")


def validate_catccos_selection(vllm_config, is_draft_model: bool) -> None:
    if get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3):
        raise ValueError("CatCCOS dispatch-FFN-combine requires an Atlas A2 or A3 device")
    if is_draft_model:
        raise ValueError("CatCCOS does not support draft/MTP models")
    parallel_config = vllm_config.parallel_config
    if not parallel_config.enable_expert_parallel:
        raise ValueError("CatCCOS requires --enable-expert-parallel")
    if parallel_config.pipeline_parallel_size != 1:
        raise ValueError("CatCCOS does not support pipeline parallelism")
    ep_size = get_ep_group().world_size
    if ep_size <= 1:
        raise ValueError("CatCCOS requires an EP group larger than one rank")
    num_experts = vllm_config.model_config.get_num_experts()
    if num_experts % ep_size:
        raise ValueError("CatCCOS requires experts to be divisible by the EP size")
    local_experts = num_experts // ep_size
    if local_experts > CATCCOS_MAX_LOCAL_EXPERTS:
        raise ValueError(f"CatCCOS supports at most {CATCCOS_MAX_LOCAL_EXPERTS} local experts, got {local_experts}")
    hf_config = vllm_config.model_config.hf_text_config
    top_k = getattr(
        hf_config,
        "num_experts_per_tok",
        getattr(hf_config, "top_k_experts", None),
    )
    if top_k != CATCCOS_SUPPORTED_TOP_K:
        raise ValueError(f"CatCCOS supports TopK={CATCCOS_SUPPORTED_TOP_K}, got {top_k}")
    quant_type = getattr(
        hf_config,
        "moe_quantize",
        getattr(hf_config, "quantize", None),
    )
    model_quantization = getattr(vllm_config.model_config, "quantization", None)
    if quant_type is not None or model_quantization is not None:
        quantization = model_quantization if model_quantization is not None else quant_type
        raise ValueError(f"CatCCOS requires unquantized BF16 MoE, got {quantization}")
    if getattr(vllm_config, "lora_config", None) is not None:
        raise ValueError("CatCCOS does not support LoRA")
    if getattr(parallel_config, "enable_eplb", False):
        raise ValueError("CatCCOS does not support vLLM EPLB")
    ascend_config = get_ascend_config()
    if ascend_config.enable_fused_mc2:
        raise ValueError("VLLM_ASCEND_ENABLE_CATCCOS_MOE and VLLM_ASCEND_ENABLE_FUSED_MC2 cannot both be enabled")
    if getattr(ascend_config, "mix_placement", False):
        raise ValueError("CatCCOS does not support mixed shared-expert placement")
    eplb_config = ascend_config.eplb_config
    if (
        eplb_config.dynamic_eplb
        or eplb_config.expert_map_path is not None
        or eplb_config.expert_map_record_path is not None
        or eplb_config.num_redundant_experts
    ):
        raise ValueError("CatCCOS does not support Ascend EPLB")


def validate_catccos_fused_input(
    fused_input: MoEFusedExpertsInput,
    moe_config: FusedMoEConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = fused_input.hidden_states
    weights = fused_input.weights
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError(f"CatCCOS requires BF16 hidden states, got {hidden_states.dtype}")
    if hidden_states.ndim != 2 or hidden_states.shape[1] != moe_config.hidden_dim:
        raise ValueError(
            f"CatCCOS hidden-state shape must be [M, {moe_config.hidden_dim}], got {tuple(hidden_states.shape)}"
        )
    if fused_input.topk_ids.shape != fused_input.topk_weights.shape:
        raise ValueError("CatCCOS expert ids and gate weights must have the same shape")
    if fused_input.topk_ids.ndim != 2:
        raise ValueError("CatCCOS expert ids must be a 2-D tensor")
    expected_routing_shape = (hidden_states.shape[0], CATCCOS_SUPPORTED_TOP_K)
    if tuple(fused_input.topk_ids.shape) != expected_routing_shape:
        raise ValueError(f"CatCCOS routing tensors must have shape {expected_routing_shape}")
    if fused_input.quant.quant_type != QuantType.NONE:
        raise ValueError("CatCCOS supports only unquantized BF16 MoE weights")
    if fused_input.need_trans:
        raise ValueError("CatCCOS requires pre-transposed ND expert weights")
    if fused_input.dynamic_eplb:
        raise ValueError("CatCCOS does not support dynamic EPLB")
    if fused_input.routing.log2phy is not None:
        raise ValueError("CatCCOS does not support non-linear expert placement")
    if fused_input.routing.global_redundant_expert_num:
        raise ValueError("CatCCOS does not support redundant experts")
    if fused_input.routing.apply_router_weight_on_input:
        raise ValueError("CatCCOS requires router weights to be applied by the operator")
    if fused_input.lora_context is not None:
        raise ValueError("CatCCOS does not support MoE LoRA")
    if fused_input.swiglu_limit:
        raise ValueError("CatCCOS does not support SwiGLU clamp")
    activation = str(getattr(fused_input.activation, "value", fused_input.activation)).lower()
    if activation not in {"silu", "swiglu", "moeactivation.silu"}:
        raise ValueError(f"CatCCOS requires SiLU/SwiGLU activation, got {activation}")
    if weights.w1_bias is not None or weights.w2_bias is not None:
        raise ValueError("CatCCOS does not support expert bias")
    if any(
        value is not None
        for value in (
            weights.w1_scale,
            weights.w2_scale,
            weights.w1_scale_bias,
            weights.w2_scale_bias,
            weights.w1_offset,
            weights.w2_offset,
        )
    ):
        raise ValueError("CatCCOS does not support quantization side tensors")
    if not isinstance(weights.w1, torch.Tensor) or not isinstance(weights.w2, torch.Tensor):
        raise ValueError("CatCCOS requires packed expert weights as tensors")

    w1, w2 = weights.w1, weights.w2
    merged_intermediate = 2 * moe_config.intermediate_size_per_partition
    expected_w1 = (
        moe_config.num_local_experts,
        moe_config.hidden_dim,
        merged_intermediate,
    )
    expected_w2 = (
        moe_config.num_local_experts,
        moe_config.intermediate_size_per_partition,
        moe_config.hidden_dim,
    )
    if tuple(w1.shape) != expected_w1 or tuple(w2.shape) != expected_w2:
        raise ValueError(
            f"CatCCOS weight shapes must be {expected_w1} and {expected_w2}, "
            f"got {tuple(w1.shape)} and {tuple(w2.shape)}"
        )
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise ValueError("CatCCOS requires BF16 expert weights")
    if not w1.is_contiguous() or not w2.is_contiguous():
        raise ValueError("CatCCOS requires contiguous row-major expert weights")
    if w1.device.type == "npu":
        if torch_npu.get_npu_format(w1) != ACL_FORMAT_FRACTAL_ND:
            raise ValueError("CatCCOS w1 must use ND format")
        if torch_npu.get_npu_format(w2) != ACL_FORMAT_FRACTAL_ND:
            raise ValueError("CatCCOS w2 must use ND format")
    return w1, w2


def pad_catccos_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    original_tokens = hidden_states.shape[0]
    if original_tokens <= 0:
        raise ValueError("CatCCOS requires at least one token")
    padded_tokens = (original_tokens + CATCCOS_TOKEN_ALIGNMENT - 1) // CATCCOS_TOKEN_ALIGNMENT * CATCCOS_TOKEN_ALIGNMENT
    pad_size = padded_tokens - original_tokens
    if pad_size:
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
        topk_ids = torch.nn.functional.pad(topk_ids, (0, 0, 0, pad_size))
        topk_weights = torch.nn.functional.pad(topk_weights, (0, 0, 0, pad_size))
    return hidden_states, topk_ids, topk_weights, original_tokens


class CatCCOSRuntime:
    """One process-local CatCCOS runtime owned by the MoE registry."""

    def __init__(self, moe_config: FusedMoEConfig):
        validate_catccos_moe_config(moe_config)
        self.config = CatCCOSRuntimeConfig.from_env()
        self.moe_fingerprint = _moe_config_fingerprint(moe_config)
        ep_group = get_ep_group()
        self.rank = ep_group.rank_in_group
        self.world_size = ep_group.world_size
        if self.world_size != moe_config.ep_size:
            raise ValueError(f"CatCCOS EP group size {self.world_size} does not match MoE EP size {moe_config.ep_size}")

        torch.ops.load_library(self.config.library_path)
        try:
            _ = torch.ops.catccos.dispatch_ffn_combine
        except AttributeError as error:
            raise RuntimeError("CatCCOS extension does not register dispatch_ffn_combine") from error
        ep_group.barrier()
        status = torch.ops.catccos.init(
            self.rank,
            self.world_size,
            self.config.local_mem_size,
            self.config.store_addr,
        )
        if status != 0:
            raise RuntimeError(f"CatCCOS initialization failed on EP rank {self.rank}/{self.world_size}: {status}")
        logger.info(
            "Initialized CatCCOS MoE backend: rank=%d/%d, library=%s, sha256=%s",
            self.rank,
            self.world_size,
            self.config.library_path,
            self._library_sha256(),
        )

    def _library_sha256(self) -> str:
        digest = hashlib.sha256()
        with Path(self.config.library_path).open("rb") as library:
            for chunk in iter(lambda: library.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def validate_compatible(self, moe_config: FusedMoEConfig) -> None:
        validate_catccos_moe_config(moe_config)
        if CatCCOSRuntimeConfig.from_env() != self.config:
            raise RuntimeError("CatCCOS runtime configuration changed after initialization")
        fingerprint = _moe_config_fingerprint(moe_config)
        if fingerprint != self.moe_fingerprint:
            raise RuntimeError(
                "All CatCCOS MoE layers must use the same runtime configuration: "
                f"initialized={self.moe_fingerprint}, requested={fingerprint}"
            )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.catccos.dispatch_ffn_combine(
            hidden_states.contiguous(),
            topk_ids.to(torch.int32).contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            w1,
            w2,
        )
