# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Model descriptions, reduced configurations and weight-transfer test matrix."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tests.e2e.common.rlhf.weight_transfer.checkpoint_adapters import deepseek_v4_checkpoint_name


@dataclass(frozen=True)
class WeightUpdateModelCase:
    """A layer-reduced model whose complete parameter set is reloaded."""

    id: str
    model: str
    hf_overrides: dict[str, Any]
    meta_config_attribute: str | None = None
    checkpoint_model_prefix: str | None = None
    checkpoint_name_map: Callable[[str], str] | None = None
    expert_intermediate_size: int | None = None
    extra_server_args: tuple[str, ...] = ()
    # None preserves the existing single-file startup checkpoint.
    checkpoint_shard_size_bytes: int | None = None
    # One representative model covers both backends and packing modes.
    full_transfer_matrix: bool = False
    skip_reason: str | None = None
    """Why neither lane can carry this case yet.

    A case whose model is not ready for the transaction is skipped in both lanes
    instead of being reported as a transfer regression; the reason is attached to
    the case so the report says which prerequisite is missing.
    """

    def server_args(self) -> list[str]:
        # Run the worker out-of-process. With a single-process executor the
        # worker shares the engine core's ``VllmConfig``, and ``EngineCoreProc``
        # rewrites ``cache_config.block_size`` to the minimum block size across
        # the KV cache groups before the worker recomputes the KV cache specs.
        # DeepSeek-V4's block-size tables are keyed by the user-facing 32/64/128,
        # so the rewritten value (8/4/2) raises ``KeyError`` on every lookup. An
        # out-of-process worker gets its own config copy, taken before that
        # rewrite. This applies to both TP1 and multi-worker TP2 servers.
        return [
            "--hf-overrides",
            json.dumps(self.hf_overrides),
            "--distributed-executor-backend",
            "mp",
            *self.extra_server_args,
        ]


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


QWEN38_CASE = WeightUpdateModelCase(
    id="qwen3.8-flash-next-4layers-small-ple-mrope-head128",
    model="/mnt/weight/Qwen3.8-Flash-Next",
    hf_overrides={
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "language_model_only": False,
        "text_config": {
            "num_hidden_layers": 4,
            # Overrides run after config normalization; use the runtime QSA name.
            "layer_types": ["linear_attention"] * 3 + ["indexed_attention"],
            "ple_layer_ids": [2],
            "ngram_vocab_size_base": 4096,
            "head_dim": 128,
            "partial_rotary_factor": 0.5,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.5,
            },
            "mtp_num_hidden_layers": 0,
            "mtp": {
                "hybrid": True,
                "layer_types": [],
                "mtp_use_hidden_state_from_layer": None,
                "num_hidden_layers": 0,
                "rope_theta": 10000000,
            },
        },
    },
    # Enumerate the text model on meta while retaining the conditional-generation
    # architecture/MRoPE in both inference servers.
    meta_config_attribute="text_config",
    checkpoint_model_prefix="model.language_model.",
    checkpoint_shard_size_bytes=2 * 2**30,
    extra_server_args=(
        "--max-num-seqs",
        "4",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
        "--engram-config",
        '{"cpu_offload":false}',
        "--kv-cache-memory-bytes",
        str(4 * 2**30),
    ),
)


QWEN38_CASES = (QWEN38_CASE,)


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


MODEL_CASES = (*QWEN_CASES, *QWEN38_CASES, *DEEPSEEK_CASES, *GLM_CASES)


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
