# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Model descriptions, reduced configurations and weight-transfer test matrix."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from vllm import envs

from tests.e2e.common.rlhf.weight_transfer.checkpoint_adapters import (
    deepseek_v4_checkpoint_name,
    deepseek_v4_vision_params,
    expand_fused_expert_params,
    expand_minimax_m3_params,
    minimax_m3_checkpoint_name,
)

if TYPE_CHECKING:
    import torch
    from vllm.distributed.weight_transfer.base import WeightSource


@dataclass(frozen=True)
class WeightUpdateModelCase:
    """A layer-reduced model whose complete parameter set is reloaded."""

    id: str
    model: str
    hf_overrides: dict[str, Any]
    # Build this case's payload source instead of ``FixedRandomWeightSource``.
    # DeepSeek-V4.1 overrides it: Transformers has no ``deepseek_v41``, so its
    # layout comes from the published checkpoint's safetensors headers
    # (``DeepseekV41WeightSource`` in ``weight_source``).
    weight_source_factory: Callable[[Any, torch.device], WeightSource] | None = None
    meta_config_attribute: str | None = None
    checkpoint_model_prefix: str | None = None
    checkpoint_name_map: Callable[[str], str] | None = None
    checkpoint_param_expander: Callable[[str, tuple[int, ...]], list[tuple[str, tuple[int, ...]]]] = (
        expand_fused_expert_params
    )
    expert_intermediate_size: int | None = None
    trust_remote_code: bool = True
    extra_server_args: tuple[str, ...] = ()
    prompts: tuple[str | tuple[int, ...], ...] = (
        "The capital of France is",
        "Explain why the sky is blue in one sentence:",
    )
    extra_param_builder: Callable[[Any], list[tuple[str, tuple[int, ...], Any]]] | None = None
    image_prompt: bool = False
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


# Turning Engram back on needs all of: `engram_layer_ids` pointing at a kept layer,
# the vocabulary reductions (`engram_vocab_size`, `engram_num_embeddings`), and
# `("--engram-config", '{"cpu_offload": false}')` in `extra_server_args` - plus the
# loader change that installs a transferred table (see the scope note above).


def _deepseek_v41_weight_source(case: WeightUpdateModelCase, device: torch.device) -> WeightSource:
    """Payload source for DeepSeek-V4.1, which has no Transformers meta model."""
    # Imported lazily: ``weight_source`` imports this module, and the header
    # reader it needs is not part of every lane's import path.
    from tests.e2e.common.rlhf.weight_transfer.weight_source import DeepseekV41WeightSource

    return DeepseekV41WeightSource(case, device)


# DeepSeek-V4.1 weight-transfer coverage, without Engram.
#
# Engram is disabled here on purpose (`engram_layer_ids: []`): on this branch
# `AscendParallelEngramEmbedding._weight_loader` ignores `loaded_weight` and re-reads
# the rows from the indexed shard bound by `bind_checkpoint`, so a lane whose payload
# differs from that checkpoint cannot reproduce it (the update would either stream
# the stale rows or fail when the shard is absent). Serving the table from a
# synthesised checkpoint instead would make the lane green without ever exercising
# the update, which is the opposite of what a regression guard should do.
#
# Engram coverage therefore waits for the loader change that installs the
# transferred tensor inside the reload window; until then this lane keeps the
# non-Engram half honest: every parameter is generated from the published
# safetensors headers and must survive sleep(level 2) -> wake weights -> two update
# rounds -> wake KV cache against a reference started from the same payload.
#
# `full_transfer_matrix=False` keeps the case on the unpacked NPU IPC path only:
# `pytest_model_cases` selects a non-matrix case for `npu_ipc` with
# `packed=False` and drops it from every other lane, so the packed buffer
# lifecycle cannot mask a reload regression here.
DEEPSEEK_V41_ENGRAM_CASE = WeightUpdateModelCase(
    id="deepseek-v4.1-flash-bf16-6l",
    # Served from the published repo like the V4 lanes (config/tokenizer via HF cache).
    model="deepseek-ai/DeepSeek-V4.1-Flash",
    # Flat keys: ``DeepseekV41Config`` flattens ``text_config`` before vLLM applies
    # overrides, so a nested ``{"text_config": {...}}`` would replace the sub-config
    # instead (see ``_dsv41_text_config``).
    #
    # Six layers, as a plain prefix of the released stack. The payload comes from the
    # published safetensors headers, so a kept layer has to keep the role the
    # checkpoint gives it: promoting a layer to an ``index``/``kv`` *source* owns
    # ``compressor``/``indexer`` weights that only the released sources (2, 8, 14,
    # 20, ...) have, and asking for them fails with "Following weights were not
    # initialized from checkpoint: ...layers.4.self_attn.compressor.wkv.weight".
    # The prefix covers ratio 0 at 0-1, the first source at 2 and its ratio-2
    # consumers at 3-5. The ratio-1 group and candidate-block emission start
    # at layer 20, i.e. 21 kept layers - a nightly-sized lane rather than this one.
    hf_overrides={
        "num_hidden_layers": 6,
        "compress_ratios": [0, 0, 2, 2, 2, 2],
        "index_source_layer_ids": [2],
        "kv_source_layer_ids": [2],
        "n_routed_experts": 8,
        "num_nextn_predict_layers": 0,
        "dspark_noise_token_id": None,
        # No Engram table: see the scope note above.
        "engram_layer_ids": [],
        "max_position_embeddings": 4096,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 1.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 4096,
        },
        "image_token_id": 0,
        # The lane is served in bf16; the released descriptor would quantize it.
        "dtype": "bfloat16",
        "quantization_config": None,
    },
    extra_server_args=("--tokenizer-mode", "deepseek_v41"),
    weight_source_factory=_deepseek_v41_weight_source,
)


DEEPSEEK_V41_CASES = (DEEPSEEK_V41_ENGRAM_CASE,)


DEEPSEEK_V4_VISION_CASE = WeightUpdateModelCase(
    id="deepseek-v4-flash-vision-bf16-4l-vit2-e8-hash3",
    model="deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
    full_transfer_matrix=True,
    hf_overrides={
        "architectures": ["DeepseekV4ForConditionalGeneration"],
        "num_hidden_layers": 4,
        "vision_n_layers": 2,
        "n_routed_experts": 8,
        "num_hash_layers": 3,
        "num_nextn_predict_layers": 0,
        "dspark_target_layer_ids": [],
        "quantization_config": None,
    },
    checkpoint_name_map=deepseek_v4_checkpoint_name,
    expert_intermediate_size=2048,
    extra_param_builder=deepseek_v4_vision_params,
    image_prompt=True,
    extra_server_args=("--tokenizer-mode", "deepseek_v4"),
)


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


# Bound KV allocation to leave room for the sparse-attention graph workspace.
MINIMAX_M3_KV_CACHE_BYTES = 128 * 2**20


MINIMAX_M3_CASE = WeightUpdateModelCase(
    id="minimax-m3-sparse-moe-layout",
    # CI uses ModelScope's namespace; Hugging Face publishes under MiniMaxAI.
    model="MiniMax/MiniMax-M3" if envs.VLLM_USE_MODELSCOPE else "MiniMaxAI/MiniMax-M3",
    hf_overrides={
        "architectures": ["MiniMaxM3SparseForCausalLM"],
        "text_config": {
            "num_hidden_layers": 4,
            "num_local_experts": 8,
            # Explicit schedules for the fixed four-layer test model;
            # native config derives these before overrides are applied.
            "layer_types": ["full_attention", "full_attention", "full_attention", "minimax_m3_sparse"],
            "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
            "sparse_attention_config": {
                "use_sparse_attention": True,
                "sparse_index_dim": 128,
                "sparse_num_index_heads": 4,
                # One local block plus one score-selected block; the prompt spans five.
                "sparse_topk_blocks": 2,
                "sparse_block_size": 128,
                "sparse_score_type": "max",
                "sparse_init_block": 0,
                "sparse_local_block": 1,
                "sparse_attention_freq": [0, 0, 0, 1],
                "sparse_disable_index_value": [0, 0, 0, 1],
            },
        },
    },
    # Token IDs make the multi-block length independent of tokenizer segmentation.
    prompts=(tuple(range(1000, 1640)),),
    meta_config_attribute="text_config",
    checkpoint_model_prefix="language_model.model.",
    checkpoint_name_map=minimax_m3_checkpoint_name,
    checkpoint_param_expander=expand_minimax_m3_params,
    # The legacy remote config creates a generic text config; the native
    # Transformers config resolves to the registered text model instead.
    trust_remote_code=False,
    extra_server_args=("--block-size", "128", "--kv-cache-memory-bytes", str(MINIMAX_M3_KV_CACHE_BYTES)),
)


MINIMAX_M31_FLASH_CASE = replace(
    MINIMAX_M3_CASE,
    id="minimax-m3.1-flash-sparse-moe-layout",
    model="MiniMax/MiniMax-M3.1-Flash" if envs.VLLM_USE_MODELSCOPE else "MiniMaxAI/MiniMax-M3.1-Flash",
    hf_overrides={
        **MINIMAX_M3_CASE.hf_overrides,
        # The published checkpoint is quantized; this lane generates BF16 weights.
        "quantization_config": None,
        "text_config": {
            **MINIMAX_M3_CASE.hf_overrides["text_config"],
            # Flash uses sparse attention in every layer, including the dense MLP layers.
            "layer_types": ["minimax_m3_sparse"] * 4,
            "sparse_attention_config": {
                **MINIMAX_M3_CASE.hf_overrides["text_config"]["sparse_attention_config"],
                "sparse_attention_freq": [1] * 4,
                "sparse_disable_index_value": [1] * 4,
            },
        },
    },
)


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


MODEL_CASES = (*QWEN_CASES, *DEEPSEEK_CASES, *DEEPSEEK_V41_CASES, *GLM_CASES, MINIMAX_M3_CASE, MINIMAX_M31_FLASH_CASE)


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
