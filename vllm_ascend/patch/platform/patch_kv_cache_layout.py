# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Align effective page sizes before upstream ``resolve_kv_cache_layout``."""

from collections.abc import Iterable

import vllm.v1.attention.backends.utils as _attn_backends_utils
import vllm.v1.engine.core as _engine_core
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.core.kv_cache_planning import effective_kv_cache_specs_for_layout

_ORIGINAL_RESOLVE_KV_CACHE_LAYOUT = getattr(
    _attn_backends_utils.resolve_kv_cache_layout,
    "_ascend_orig",
    _attn_backends_utils.resolve_kv_cache_layout,
)


def resolve_kv_cache_layout(
    vllm_config: VllmConfig,
    supported_layouts: list[list[str]],
    kv_cache_specs: Iterable[KVCacheSpec] | None = None,
) -> KVCacheLayout:
    """Resolve KV cache layout against Ascend's planned page sizes."""
    effective_specs = effective_kv_cache_specs_for_layout(vllm_config, kv_cache_specs)
    return _ORIGINAL_RESOLVE_KV_CACHE_LAYOUT(vllm_config, supported_layouts, effective_specs)


resolve_kv_cache_layout._ascend_orig = _ORIGINAL_RESOLVE_KV_CACHE_LAYOUT  # type: ignore[attr-defined]
_attn_backends_utils.resolve_kv_cache_layout = resolve_kv_cache_layout
_engine_core.resolve_kv_cache_layout = resolve_kv_cache_layout
