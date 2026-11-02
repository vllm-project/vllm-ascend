# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Regression tests for the Ascend kv-cache-dtype widening patch.

``--kv-cache-dtype int8`` (DeepSeek V4 MLA) and ``--kv-cache-dtype mxfp8``
(C8-MXFP on the A5 QFA path) must survive ``CacheConfig`` validation:
current vLLM main gates ``cache_dtype`` on the mutable
``vllm.config.cache.KV_CACHE_DTYPES`` mirror list
(``is_known_kv_cache_dtype`` in ``_validate_cache_dtype``; the same list
object backs the ``get_attn_backend`` assertion in the attention selector),
not only on the ``CacheDType`` Literal the patch rebuilds. Widening the
Literal alone left ``--kv-cache-dtype mxfp8`` dying with pydantic
``ValidationError: Invalid kv_cache_dtype: 'mxfp8'``.
"""

import typing

import pytest
from pydantic import ValidationError
from vllm.config import cache as cache_mod
from vllm.config.cache import CacheConfig

import vllm_ascend.patch.platform.patch_kv_cache_dtype  # noqa: F401

_UPSTREAM_DTYPES = (
    "auto",
    "float16",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
    "nvfp4_ds_mla",
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
    "ultraquant_4bit",
    "int4_per_token_head",
    "int8_per_token_head",
    "fp8_per_token_head",
    "nvfp4",
    "nvfp4_4over6",
)


class TestCacheConfigAcceptsAscendDtypes:
    @pytest.mark.parametrize("cache_dtype", ["int8", "mxfp8"])
    def test_cache_config_constructs(self, cache_dtype):
        """The regression: pydantic ValidationError before the mirror list
        was widened."""
        config = CacheConfig(block_size=32, cache_dtype=cache_dtype)
        assert config.cache_dtype == cache_dtype

    @pytest.mark.parametrize("cache_dtype", _UPSTREAM_DTYPES)
    def test_upstream_dtypes_still_construct(self, cache_dtype):
        """The widened Literal must be derived, never hardcoded: no upstream
        member (e.g. nvfp4_ds_mla, ultraquant_4bit) may be dropped."""
        config = CacheConfig(block_size=32, cache_dtype=cache_dtype)
        assert config.cache_dtype == cache_dtype

    def test_unknown_dtypes_are_still_rejected(self):
        with pytest.raises(ValidationError):
            CacheConfig(block_size=32, cache_dtype="not-a-dtype")


class TestMirrorListWidening:
    def test_mirror_list_carries_the_ascend_dtypes(self):
        assert "int8" in cache_mod.KV_CACHE_DTYPES
        assert "mxfp8" in cache_mod.KV_CACHE_DTYPES

    def test_selector_reads_the_same_live_list(self):
        from vllm.v1.attention import selector

        assert selector.KV_CACHE_DTYPES is cache_mod.KV_CACHE_DTYPES
        assert selector.is_known_kv_cache_dtype("mxfp8")

    def test_widened_literal_keeps_every_upstream_member(self):
        args = set(typing.get_args(cache_mod.CacheDType))
        assert set(_UPSTREAM_DTYPES) <= args
        assert {"int8", "mxfp8"} <= args

    def test_reapplying_the_patch_does_not_duplicate(self):
        from vllm_ascend.patch.platform.patch_kv_cache_dtype import (
            _apply_kv_cache_dtype_int8_patch,
        )

        before = list(cache_mod.KV_CACHE_DTYPES)
        _apply_kv_cache_dtype_int8_patch()
        assert before == cache_mod.KV_CACHE_DTYPES
