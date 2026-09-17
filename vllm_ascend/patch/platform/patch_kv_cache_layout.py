# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Negotiate KV layouts across PP stages with different attention backends."""

import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.v1.attention.backends import utils as attention_utils

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheSpec


_original_resolve_kv_cache_layout = attention_utils.resolve_kv_cache_layout


def resolve_kv_cache_layout(
    vllm_config: "VllmConfig",
    supported_layouts: list[list[str]],
    kv_cache_specs: Iterable["KVCacheSpec"] | None = None,
):
    # PP stages need not run identical backends: for example, only the last
    # stage hosts the DSpark draft. Keep the first stage's preference order,
    # but require every selected layout to be supported by every worker.
    if not supported_layouts or not all(supported_layouts):
        raise ValueError("No worker reported supported KV cache layouts.")
    worker_layouts = [[attention_utils._layout_from_name(name) for name in names] for names in supported_layouts]
    common = set(worker_layouts[0]).intersection(*worker_layouts[1:])
    candidates = list(dict.fromkeys(layout.name for layout in worker_layouts[0] if layout in common))
    if not candidates:
        raise ValueError(f"No KV cache layout satisfies every supported set: {supported_layouts}.")
    # The upstream resolver still validates mixed cache shapes, explicit
    # layout requests and connector preferences before recording the result.
    return _original_resolve_kv_cache_layout(vllm_config, [candidates], kv_cache_specs)


attention_utils.resolve_kv_cache_layout = resolve_kv_cache_layout
# EngineCore imports the function directly. Cover either import order without
# eagerly importing it while platform initialization may still be in progress.
if (core := sys.modules.get("vllm.v1.engine.core")) is not None:
    if getattr(core, "resolve_kv_cache_layout", None) is _original_resolve_kv_cache_layout:
        core.resolve_kv_cache_layout = resolve_kv_cache_layout
