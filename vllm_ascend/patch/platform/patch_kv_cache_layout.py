# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Keep GLM's private cache pools out of generic shared-backing checks."""

from collections.abc import Iterable
from functools import wraps
from typing import Any

import vllm.envs as envs
import vllm.v1.attention.backends.utils as attention_utils
import vllm.v1.engine.core as engine_core
from vllm.config import VllmConfig

_orig_resolve_kv_cache_layout = attention_utils.resolve_kv_cache_layout


@wraps(_orig_resolve_kv_cache_layout)
def resolve_kv_cache_layout(
    vllm_config: VllmConfig,
    supported_layouts: list[list[str]],
    kv_cache_specs: Iterable[Any] | None = None,
):
    speculative_config = vllm_config.speculative_config
    if (
        speculative_config is None
        or speculative_config.method != "dflash"
        or envs.VLLM_KV_CACHE_LAYOUT not in ("NHD", "HND", "LBNHC", "LBHNC")
    ):
        return _orig_resolve_kv_cache_layout(vllm_config, supported_layouts, kv_cache_specs)

    # Core passes specs on the older API and layer configs on the newer API.
    # GLM's planner owns its MLA/indexer/tail pools and allocates each draft
    # layer separately. These specs do not share the generic packed backing,
    # so their different page sizes / DCP sharding must not force a BL layout.
    # Keep the original specs for grouping, allocation and slot mapping; only
    # omit GLM-owned pools from the generic layout check. Draft/state specs and
    # every worker's backend-supported layouts still go through upstream.
    specs = tuple(kv_cache_specs or ())
    layout_specs = tuple(
        entry for entry in specs if getattr(getattr(entry, "spec", entry), "model_version", None) != "glm5_next"
    )
    return _orig_resolve_kv_cache_layout(vllm_config, supported_layouts, layout_specs)


attention_utils.resolve_kv_cache_layout = resolve_kv_cache_layout
# EngineCore imports the function directly, before platform patches run.
engine_core.resolve_kv_cache_layout = resolve_kv_cache_layout
