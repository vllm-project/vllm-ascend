# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared configuration contract for reduced weight-transfer models."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


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
    # Options for enumerating native HF state in checkpoint format.
    trust_remote_code: bool = True
    split_fused_mlp: bool = False
    extra_server_args: tuple[str, ...] = ()
    meta_model_factory: Callable[[Any], torch.nn.Module] | None = None
    tensor_initializer: Callable[[str, torch.Tensor], torch.Tensor] | None = None
    server_env: tuple[tuple[str, str], ...] = ()
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
        # rewrite, and the single-worker topology of these cases is unchanged.
        return [
            "--hf-overrides",
            json.dumps(self.hf_overrides),
            "--distributed-executor-backend",
            "mp",
            *self.extra_server_args,
        ]
