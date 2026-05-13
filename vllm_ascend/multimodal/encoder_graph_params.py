#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Per–encoder-budget ACL graph bookkeeping for MM ViT FIA tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class EncoderGraphParams:
    """Mirrors :class:`vllm_ascend.compilation.acl_graph.GraphParams` but keyed by encoder token budget."""

    events: dict[int, list[torch.npu.ExternalEvent]] = field(default_factory=dict)
    workspaces: dict[int, torch.Tensor | None] = field(default_factory=dict)
    handles: dict[int, list[Any]] = field(default_factory=dict)
    # Flattened per-forward insertion order (one entry per ViT block invocation).
    attn_params: dict[int, list[tuple]] = field(default_factory=dict)


_encoder_graph_params: EncoderGraphParams | None = None


def set_encoder_graph_params(token_budgets: list[int]) -> None:
    global _encoder_graph_params
    budgets_sorted_unique = sorted(token_budgets)
    if _encoder_graph_params is not None:
        existing = sorted(_encoder_graph_params.events.keys())
        if existing == budgets_sorted_unique:
            return
        raise RuntimeError(
            "Encoder graph params already initialized with different budgets "
            f"(existing={existing}, new={budgets_sorted_unique})"
        )
    _encoder_graph_params = EncoderGraphParams(
        events={b: [] for b in budgets_sorted_unique},
        workspaces={b: None for b in budgets_sorted_unique},
        handles={b: [] for b in budgets_sorted_unique},
        attn_params={b: [] for b in budgets_sorted_unique},
    )


def get_encoder_graph_params() -> EncoderGraphParams | None:
    return _encoder_graph_params


def reset_encoder_graph_params_for_testing() -> None:
    global _encoder_graph_params
    _encoder_graph_params = None


def update_encoder_graph_workspace(token_budget: int, workspace: torch.Tensor) -> None:
    if _encoder_graph_params is None:
        return
    _encoder_graph_params.workspaces[token_budget] = workspace

