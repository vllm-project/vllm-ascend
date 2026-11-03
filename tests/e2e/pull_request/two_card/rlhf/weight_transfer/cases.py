# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Reduced models and coverage selection for the two-card weight-transfer suite."""

from typing import Any

from tests.e2e.pull_request.rlhf.weight_transfer.adapters.deepseek import deepseek_v4_checkpoint_name
from tests.e2e.pull_request.rlhf.weight_transfer.adapters.minimax import minimax_m3_checkpoint_name
from tests.e2e.pull_request.rlhf.weight_transfer.model import WeightUpdateModelCase

QWEN35_CASE = WeightUpdateModelCase(
    id="qwen3.5-35b-a3b-moe-layout",
    model="Qwen/Qwen3.5-35B-A3B",
    full_transfer_matrix=True,
    hf_overrides={
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "text_config": {
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        },
    },
    # The public checkpoint is multimodal, while this test serves its
    # language model architecture. Enumerate every parameter from the text
    # config so visual parameters are neither skipped nor sent by mistake.
    meta_config_attribute="text_config",
    checkpoint_model_prefix="model.language_model.",
)


QWEN_CASES = (QWEN35_CASE,)


DEEPSEEK_V4_CASE = WeightUpdateModelCase(
    id="deepseek-v4-flash-bf16-derived-gate",
    model="kylesayrs/DeepSeek-V4-Flash-bf16",
    hf_overrides={
        "num_hidden_layers": 4,
        "n_routed_experts": 8,
    },
    checkpoint_name_map=deepseek_v4_checkpoint_name,
    # The HF ``from_config`` model sizes every expert MLP by
    # ``intermediate_size`` (18432), while the checkpoint - and therefore the
    # served model - use ``moe_intermediate_size`` (2048); verified against
    # the real safetensors header of ``experts.0.w1.weight`` == [2048, 4096].
    expert_intermediate_size=2048,
    extra_server_args=("--tokenizer-mode", "deepseek_v4"),
    skip_reason=(
        "DeepSeek-V4-Flash is skipped in both lanes for now: its reload needs "
        "the attention-sink fix that lives in #16355, and the two-card HCCL "
        "lane has an open problem with the model as well"
    ),
)


DEEPSEEK_CASES = (DEEPSEEK_V4_CASE,)


GLM51_CASE = WeightUpdateModelCase(
    id="glm-5.1-sfa-derived-kv",
    model="zai-org/GLM-5.1",
    hf_overrides={
        "num_hidden_layers": 4,
        # Keep the trainer payload and TP workers within the test memory budget.
        "n_routed_experts": 8,
    },
)

GLM_CASES = (GLM51_CASE,)


# Reserve HBM for M3's FIA graph workspace rather than sizing the KV cache
# from the memory-utilization fraction. The four-layer, 1024-token test only
# needs a small cache; keep startup and live-update servers identical.
MINIMAX_M3_KV_CACHE_BYTES = 128 * 2**20


MINIMAX_M3_CASE = WeightUpdateModelCase(
    id="minimax-m3-sparse-moe-layout",
    model="MiniMax/MiniMax-M3",
    hf_overrides={
        "architectures": ["MiniMaxM3SparseForCausalLM"],
        "text_config": {
            "num_hidden_layers": 4,
            "num_local_experts": 8,
            # Explicit schedules for the fixed four-layer test model;
            # native config derives these before overrides are applied.
            "layer_types": ["full_attention", "full_attention", "full_attention", "minimax_m3_sparse"],
            "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
        },
    },
    meta_config_attribute="text_config",
    checkpoint_model_prefix="language_model.model.",
    checkpoint_name_map=minimax_m3_checkpoint_name,
    # Disable the checkpoint's auto_map remote config and use Transformers'
    # native MiniMaxM3VLConfig, whose text_config is MiniMaxM3VLTextConfig.
    # AutoModelForCausalLM resolves that text config to MiniMaxM3VLForCausalLM
    # for meta-model construction. The legacy remote config instead yields
    # a generic text PreTrainedConfig that AutoModel cannot resolve.
    trust_remote_code=False,
    split_fused_mlp=True,
    extra_server_args=("--block-size", "128", "--kv-cache-memory-bytes", str(MINIMAX_M3_KV_CACHE_BYTES)),
)


MINIMAX_CASES = (MINIMAX_M3_CASE,)


# Keep packed lanes skipped until #18763 fixes cross-stream chunk loading.
# HCCL's second update diverged in #18726 (38570793020, attempts 1/2) and
# main daily runs 38186343470, 38325024027, 38657430708 and 38679145716;
# unpacked passed. Remove the shared skip after validating the fix.
PACKED_LANE_SKIP_REASON = (
    "packed weight transfer is flaky: the receive buffer of a chunk is filled and "
    "loaded on different NPU streams, so the layerwise reload can read tensors "
    "retained from an earlier chunk without ordering against their allocation "
    "(#18726; fix proposed in #18763)"
)


MODEL_CASES = (*QWEN_CASES, *DEEPSEEK_CASES, *GLM_CASES, *MINIMAX_CASES)


def pytest_model_cases(backend: str, *, model_cases: tuple[WeightUpdateModelCase, ...] = MODEL_CASES) -> list[Any]:
    """Select coverage and mark known limitations before entering a runner."""
    # Case data and adapters remain usable without importing pytest.
    import pytest

    if backend not in ("npu_ipc", "hccl"):
        raise ValueError(f"Unsupported weight transfer backend: {backend}")
    parameters = []
    for packed in (False, True):
        for case in model_cases:
            if not ((backend == "npu_ipc" and not packed) or case.full_transfer_matrix):
                continue
            marks = [pytest.mark.e2e_model(case.model)]
            reason = case.skip_reason or (PACKED_LANE_SKIP_REASON if packed else None)
            if reason:
                marks.append(pytest.mark.skip(reason=f"{case.id}: {reason}"))
            parameters.append(
                pytest.param(case, packed, id=f"{'packed' if packed else 'unpacked'}-{case.id}", marks=marks)
            )
    return parameters
