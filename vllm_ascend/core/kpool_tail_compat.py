# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish ``KpoolTailSpec`` on vLLM wheels that do not yet export it.

cpu-ut installs the verified vLLM pin, which predates ``KpoolTailSpec``.
Importing ``model_runner`` then fails at collection. Attach a
``SlidingWindowSpec`` subclass onto ``vllm.v1.kv_cache_interface`` so
``from vllm.v1.kv_cache_interface import KpoolTailSpec`` keeps working.

The fallback is not a ``class KpoolTailSpec`` statement in this tree, so
GLM grouping still treats the upstream type as canonical when it exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import vllm.v1.kv_cache_interface as _kv_mod
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowSpec


def _install_fallback() -> None:
    @dataclass(frozen=True, kw_only=True)
    class _FallbackKpoolTailSpec(SlidingWindowSpec):
        """One-block circular scratch cache for a kpool indexer's raw tail."""

        def max_admission_blocks_per_request(self, max_in_flight_tokens: int, max_model_len: int) -> int:
            return 1

        def max_num_blocks_per_req(self, vllm_config: VllmConfig | None, max_len: int) -> int:
            return 1

        def is_uniform_with_collection(self, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
            return all(isinstance(spec, type(self)) for spec in kv_cache_specs.values())

        @property
        def prefix_cacheable(self) -> bool:
            return False

        @property
        def participates_in_prefix_caching(self) -> bool:
            return False

    _kv_mod.KpoolTailSpec = _FallbackKpoolTailSpec  # type: ignore[attr-defined]


if not hasattr(_kv_mod, "KpoolTailSpec"):
    try:
        _install_fallback()
    except TypeError:
        _kv_mod.KpoolTailSpec = type("KpoolTailSpec", (SlidingWindowSpec,), {})  # type: ignore[attr-defined]
