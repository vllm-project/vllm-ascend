# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RFC #21732: mode resolution for ``--kv-cache-dtype int8``.

The same flag must cover the existing static C8 path (checkpoint carries
recipe-calibrated scales) and the dynamic path (plain checkpoint, K/V
quantized at runtime). Resolution is data-driven and must not depend on any
new switch, environment variable, or checkpoint regeneration.
"""

from types import SimpleNamespace

from vllm_ascend.utils import resolve_kv_cache_int8_mode


def _cfg(cache_dtype="auto", enable_c8_quant=False, has_quant_config=True):
    quant_config = SimpleNamespace(enable_c8_quant=enable_c8_quant) if has_quant_config else None
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        quant_config=quant_config,
    )


def test_non_int8_resolves_to_none():
    assert resolve_kv_cache_int8_mode(_cfg("auto")) is None
    assert resolve_kv_cache_int8_mode(_cfg("fp8")) is None
    assert resolve_kv_cache_int8_mode(_cfg("mxfp8")) is None
    assert resolve_kv_cache_int8_mode(_cfg("bfloat16")) is None


def test_int8_with_recipe_scales_is_static():
    assert resolve_kv_cache_int8_mode(_cfg("int8", enable_c8_quant=True)) == "static"


def test_int8_plain_checkpoint_is_dynamic():
    # No quant config at all (vanilla bf16 checkpoint).
    assert resolve_kv_cache_int8_mode(_cfg("int8", has_quant_config=False)) == "dynamic"
    # Quant config present but C8 not enabled by the recipe.
    assert resolve_kv_cache_int8_mode(_cfg("int8", enable_c8_quant=False)) == "dynamic"


def test_partially_constructed_config_is_tolerated():
    # Mirrors the SimpleNamespace configs used across the unit tests.
    assert resolve_kv_cache_int8_mode(SimpleNamespace(quant_config=None)) is None
    assert resolve_kv_cache_int8_mode(SimpleNamespace()) is None
