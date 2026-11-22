# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Negotiate DFlash layer layouts against backend support on Ascend."""

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

    # Ascend uses per-layer storage, paired replicated draft pages, or native
    # model pools rather than upstream's mixed-page shared backing. Do not
    # constrain this choice by target model name. Keep backend intersections
    # and the real specs used by grouping, accounting and slot mapping intact.
    return _orig_resolve_kv_cache_layout(vllm_config, supported_layouts, ())


attention_utils.resolve_kv_cache_layout = resolve_kv_cache_layout
# EngineCore imports the function directly, before platform patches run.
engine_core.resolve_kv_cache_layout = resolve_kv_cache_layout
