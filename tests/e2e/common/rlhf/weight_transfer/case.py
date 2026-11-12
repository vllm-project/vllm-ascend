# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Model descriptions, reduced configurations and weight-transfer test matrix."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tests.e2e.common.rlhf.weight_transfer.checkpoint_adapters import deepseek_v4_checkpoint_name

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
    expert_intermediate_size: int | None = None
    extra_server_args: tuple[str, ...] = ()
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


def _dsv41_engram_args() -> tuple[str, ...]:
    """``--engram-config`` for the V4.1 lane, or nothing when engram is off.

    Gated because ``EngramConfig.verify_model_config()`` rejects an engram config
    for a model with no n-gram embeddings, so a lane served from a config with
    ``engram_layer_ids == []`` must not pass it at all. ``cpu_offload`` stays
    switchable to tell the host/UVA-backed table apart from a device one.
    """
    if os.environ.get("DSV41_ENGRAM", "1") != "1":
        return ()
    offload = "true" if os.environ.get("DSV41_ENGRAM_CPU_OFFLOAD", "1") == "1" else "false"
    return ("--engram-config", f'{{"cpu_offload": {offload}}}')


def _deepseek_v41_weight_source(case: WeightUpdateModelCase, device: torch.device) -> WeightSource:
    """Payload source for DeepSeek-V4.1, which has no Transformers meta model."""
    # Imported lazily: ``weight_source`` imports this module, and the header
    # reader it needs is not part of every lane's import path.
    from tests.e2e.common.rlhf.weight_transfer.weight_source import DeepseekV41WeightSource

    return DeepseekV41WeightSource(case, device)


# Engram host-offload coverage. The published V4.1 checkpoint ships two Engram
# tables of 384M x 256 BF16 rows (`text_config.engram_num_embeddings`), i.e.
# ~183 GiB each; reloading one must update the registered host range instead of
# materializing the table on the device, which is the OOM this lane guards.
#
# `full_transfer_matrix=False` keeps the case on the unpacked NPU IPC path only:
# `pytest_model_cases` selects a non-matrix case for `npu_ipc` with
# `packed=False` and drops it from every other lane, so the packed buffer
# lifecycle cannot mask a reload regression here.
DEEPSEEK_V41_ENGRAM_CASE = WeightUpdateModelCase(
    id="deepseek-v4.1-flash-engram-cpu-offload",
    # V4.1 cannot be enumerated through Transformers, so the lane is served from
    # a local copy of the released config (plus its tokenizer) rather than a repo
    # id. All 40 layers and the released shape parameters are kept so the
    # published headers stay authoritative; only the expert count, the engram
    # vocabulary and the MTP stages are reduced, and that reduction lives in the
    # model directory's own config.json rather than in ``--hf-overrides``.
    model=os.environ.get("DSV41_MODEL_DIR", ""),
    # The reduction is already in that config.json: vLLM flattens the released
    # ``text_config``, so a nested override would replace the sub-config and drop
    # every field it does not restate (``num_attention_heads`` among them).
    hf_overrides={},
    extra_server_args=(
        "--tokenizer-mode",
        "deepseek_v41",
        # Engram stays part of the model, but its tables live in pinned host
        # memory: vLLM only offloads when an ``EngramConfig`` is present at all,
        # so the flag - not the default of ``cpu_offload`` - is what turns it on.
        *_dsv41_engram_args(),
    ),
    weight_source_factory=_deepseek_v41_weight_source,
    skip_reason=(
        None if os.environ.get("DSV41_MODEL_DIR") else "set DSV41_MODEL_DIR to a DeepSeek-V4.1 checkpoint directory"
    ),
)


DEEPSEEK_V41_CASES = (DEEPSEEK_V41_ENGRAM_CASE,)


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


MODEL_CASES = (*QWEN_CASES, *DEEPSEEK_CASES, *DEEPSEEK_V41_CASES, *GLM_CASES)


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
