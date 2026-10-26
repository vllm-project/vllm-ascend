# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from vllm.config import VllmConfig
from vllm.config.load import LoadConfig

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.device.hardware_profile import WeightLayoutPolicy, get_current_hardware_profile


def validate_preload_config(vllm_config: VllmConfig, *, mode: str = "zero_copy") -> None:
    """Reject unsupported layouts before allocating or importing weights."""
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    ascend = init_ascend_config(vllm_config)
    reasons = []
    if not vllm_config.use_v2_model_runner:
        reasons.append("Model Runner V2 is required")
    if ascend.weight_nz_mode != 0:
        reasons.append("set --additional-config '{\"weight_nz_mode\": 0}' on both preload and serve")
    if get_current_hardware_profile().weight_layout_policy is WeightLayoutPolicy.FORCE_NZ:
        reasons.append("hardware requiring NZ weights is unsupported")
    if model.dtype not in (torch.bfloat16, torch.float16):
        reasons.append("only BF16 and FP16 weights are supported")
    if model.quantization is not None or getattr(model.hf_config, "quantization_config", None):
        reasons.append("quantized weights are unsupported")
    if model.is_moe or model.use_mla or model.is_multimodal_model:
        reasons.append("only non-MoE, non-MLA text Dense models are supported")
    if getattr(model, "is_hybrid", False) or getattr(model, "is_attention_free", False):
        reasons.append("hybrid and attention-free models are unsupported")
    if model.runner_type != "generate":
        reasons.append("only the generate runner is supported")
    if vllm_config.speculative_config is not None:
        reasons.append("speculative decoding is unsupported")
    if vllm_config.lora_config is not None:
        reasons.append("LoRA is unsupported")
    if getattr(vllm_config, "kv_transfer_config", None) or getattr(vllm_config, "ec_transfer_config", None):
        reasons.append("KV and encoder cache transfer are unsupported")
    finegrained = getattr(ascend, "finegrained_tp_config", None)
    if any(
        getattr(finegrained, field, 0) > 0
        for field in (
            "oproj_tensor_parallel_size",
            "lmhead_tensor_parallel_size",
            "embedding_tensor_parallel_size",
            "mlp_tensor_parallel_size",
        )
    ):
        # Upstream's cache key describes global TP, not per-module TP shards.
        reasons.append("fine-grained tensor parallelism is unsupported")
    if (
        parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.enable_expert_parallel
        or parallel.enable_eplb
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
    ):
        reasons.append("PP, DP, EP, EPLB and context parallelism are unsupported")
    if mode == "zero_copy" and model.enable_sleep_mode:
        reasons.append("zero_copy weights cannot use sleep mode")
    offload = vllm_config.offload_config
    if offload.uva.cpu_offload_gb > 0 or offload.prefetch.offload_group_size > 0:
        reasons.append("CPU weight offloading is unsupported")
    if getattr(vllm_config, "weight_transfer_config", None) is not None:
        reasons.append("weight transfer and in-place weight updates are unsupported")
    if reasons:
        raise ValueError("Unsupported Ascend preload configuration: " + "; ".join(reasons))


def require_private_weights(load_config: LoadConfig, operation: str) -> None:
    extra = load_config.model_loader_extra_config or {}
    if load_config.load_format == "ipc_cache" and extra.get("mode", "zero_copy") == "zero_copy":
        raise ValueError(f"{operation} is unsupported with daemon-owned zero_copy weights")
