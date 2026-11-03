# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Regression tests for the Ascend kv-cache-dtype widening patch.

``--kv-cache-dtype int8`` (DeepSeek V4 MLA) and ``--kv-cache-dtype mxfp8``
(C8-MXFP on the A5 QFA path) must survive ``CacheConfig`` validation. The
gate differs by vLLM version: current main gates ``cache_dtype`` on the
mutable ``vllm.config.cache.KV_CACHE_DTYPES`` mirror list
(``is_known_kv_cache_dtype`` in ``_validate_cache_dtype``; the same list
object backs the ``get_attn_backend`` assertion in the attention selector),
while releases (e.g. v0.32.0) gate on the ``CacheDType`` Literal alone.
The patch widens both, so both gates are exercised here where they exist.
"""

import typing

import pytest
from pydantic import ValidationError
from vllm.config import cache as cache_mod
from vllm.config.cache import CacheConfig

import vllm_ascend.patch.platform.patch_kv_cache_dtype  # noqa: F401

# Ascend-only ``--kv-cache-dtype`` values the patch widens with (kept in
# sync with the patch files).
_ASCEND_CACHE_DTYPES = ("int8", "mxfp8")

# The patch derives the widened Literal from the running vLLM's members, so
# every member of the running Literal must still construct.
_UPSTREAM_DTYPES = tuple(d for d in typing.get_args(cache_mod.CacheDType) if d not in _ASCEND_CACHE_DTYPES)

# The KV_CACHE_DTYPES mirror list only exists on newer vLLM main; releases
# (e.g. v0.32.0) gate on the Literal alone.
_HAS_MIRROR_LIST = hasattr(cache_mod, "KV_CACHE_DTYPES")


class TestCacheConfigAcceptsAscendDtypes:
    @pytest.mark.parametrize("cache_dtype", _ASCEND_CACHE_DTYPES)
    def test_cache_config_constructs(self, cache_dtype):
        """The regression: pydantic ValidationError before the patch widened
        the accepted dtype set (Literal on releases, the KV_CACHE_DTYPES
        mirror list on current main)."""
        config = CacheConfig(block_size=32, cache_dtype=cache_dtype)
        assert config.cache_dtype == cache_dtype

    @pytest.mark.parametrize("cache_dtype", _UPSTREAM_DTYPES)
    def test_upstream_dtypes_still_construct(self, cache_dtype):
        """The widened Literal must be derived, never hardcoded: no member of
        the running vLLM's Literal may be dropped."""
        config = CacheConfig(block_size=32, cache_dtype=cache_dtype)
        assert config.cache_dtype == cache_dtype

    def test_unknown_dtypes_are_still_rejected(self):
        with pytest.raises(ValidationError):
            CacheConfig(block_size=32, cache_dtype="not-a-dtype")


@pytest.mark.skipif(
    not _HAS_MIRROR_LIST,
    reason="the KV_CACHE_DTYPES mirror list only exists on newer vLLM main",
)
class TestMirrorListWidening:
    def test_mirror_list_carries_the_ascend_dtypes(self):
        for dtype in _ASCEND_CACHE_DTYPES:
            assert dtype in cache_mod.KV_CACHE_DTYPES

    def test_selector_reads_the_same_live_list(self):
        from vllm.v1.attention import selector

        assert selector.KV_CACHE_DTYPES is cache_mod.KV_CACHE_DTYPES
        for dtype in _ASCEND_CACHE_DTYPES:
            assert selector.is_known_kv_cache_dtype(dtype)

    def test_widened_literal_keeps_every_upstream_member(self):
        args = set(typing.get_args(cache_mod.CacheDType))
        assert set(_UPSTREAM_DTYPES) <= args
        assert set(_ASCEND_CACHE_DTYPES) <= args

    def test_reapplying_the_patch_does_not_duplicate(self):
        from vllm_ascend.patch.platform.patch_kv_cache_dtype import (
            _apply_kv_cache_dtype_int8_patch,
        )

        before = list(cache_mod.KV_CACHE_DTYPES)
        _apply_kv_cache_dtype_int8_patch()
        assert before == cache_mod.KV_CACHE_DTYPES
