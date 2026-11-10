# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
# ruff: noqa: E402
"""Dispatch contract of ``AscendKVCacheConfigBuilder`` (main-only).

Locks the grouping precedence (GLM-Next, Ascend shared-tuple layouts, Kimi-K3
DSpark, then vLLM's generic grouping) and the TurboQuant pool accounting: the
surface most likely to drift between the pluggable builder on main and the
release-lane monkey patches.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

pytest.importorskip(
    "vllm.v1.core.kv_cache_config_builder",
    reason="the pluggable KV cache builder is main-only",
)

import torch
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core import kv_cache_config_builder as builder_module
from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import (
    AscendSlidingWindowMLASpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.quantization.methods.kv_cache.turboquant import (
    SLOT_BYTES,
    TURBOQUANT_CACHE_DTYPE,
)
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache as tq_cache


@pytest.fixture(autouse=True)
def registered_specs():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()


def make_runtime():
    return SimpleNamespace(
        model_config=SimpleNamespace(supports_kv_cache_grouping=False),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
    )


def generic_specs():
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16)
    return {"layer.a": spec, "layer.b": spec}


def test_generic_uniform_spec_falls_back_to_default_grouping():
    runtime = make_runtime()
    specs = generic_specs()
    result = AscendKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs)
    expected = DefaultKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs)
    assert result == expected
    assert len(result) == 1
    assert set(result[0].layer_names) == {"layer.a", "layer.b"}


def test_glm_next_layout_takes_precedence_over_ascend_grouping(monkeypatch):
    sentinel = [KVCacheGroupSpec(["glm"], generic_specs()["layer.a"])]
    glm_mock = Mock(return_value=sentinel)
    ascend_mock = Mock()
    monkeypatch.setattr(builder_module, "is_glm5_next_cache_spec", lambda spec: True)
    monkeypatch.setattr(builder_module, "get_glm5_next_kv_cache_groups", glm_mock)
    monkeypatch.setattr(builder_module, "group_and_unify_kv_cache_specs", ascend_mock)
    assert AscendKVCacheConfigBuilder().get_kv_cache_groups(make_runtime(), generic_specs()) is sentinel
    ascend_mock.assert_not_called()


def test_ascend_layout_takes_precedence_over_kimi_dspark(monkeypatch):
    sentinel = [KVCacheGroupSpec(["tuple"], generic_specs()["layer.a"])]
    ascend_mock = Mock(return_value=sentinel)
    kimi_mock = Mock(return_value=None)
    monkeypatch.setattr(builder_module, "is_glm5_next_cache_spec", lambda spec: False)
    monkeypatch.setattr(builder_module, "is_deepseek_v41_cache", lambda spec: True)
    monkeypatch.setattr(builder_module, "group_cache_specs", Mock(return_value=[]))
    monkeypatch.setattr(builder_module, "make_cache_groups", ascend_mock)
    monkeypatch.setattr(builder_module, "_get_kimi_k3_dspark_mixed_kv_cache_groups", kimi_mock)
    assert AscendKVCacheConfigBuilder().get_kv_cache_groups(make_runtime(), generic_specs()) is sentinel
    kimi_mock.assert_not_called()


def test_turboquant_groups_use_packed_pool_bytes():
    tq_spec = AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=SLOT_BYTES,
        dtype=torch.uint8,
        sliding_window=128,
        cache_dtype_str=TURBOQUANT_CACHE_DTYPE,
        model_version="deepseek_v4",
    )
    uniform = UniformTypeKVCacheSpecs.from_specs({"l0": tq_spec, "l1": tq_spec})
    groups = [KVCacheGroupSpec(layer_names=["l0", "l1"], kv_cache_spec=uniform)]
    builder = AscendKVCacheConfigBuilder()
    assert tq_cache.uses_turboquant_groups(groups)
    pool_bytes = builder._get_pool_bytes_per_block(make_runtime(), groups)
    assert pool_bytes == tq_cache.pool_bytes_per_block(groups)
    assert pool_bytes >= sum(spec.page_size_bytes for spec in uniform.kv_cache_specs.values())


def _builder() -> AscendKVCacheConfigBuilder:
    # Skip __init__: the override under test does not read instance state,
    # and the parent constructor requires a populated VllmConfig.
    return AscendKVCacheConfigBuilder.__new__(AscendKVCacheConfigBuilder)


class TestPoolBytesDispatch(unittest.TestCase):
    """The three-branch dispatch of ``_get_pool_bytes_per_block``."""

    def _groups(self):
        return [MagicMock(), MagicMock()]

    def test_glm_layout_wins_and_short_circuits(self):
        builder = _builder()
        groups = self._groups()
        with (
            patch.object(builder_module, "_get_glm5_next_cache_layout", return_value=object()) as layout,
            patch.object(builder_module, "get_glm5_next_pool_bytes_per_block", return_value=4096) as glm_bytes,
            patch.object(builder_module, "get_qwen4_exp_pool_bytes_per_block", return_value=8192) as qwen_bytes,
            patch.object(
                builder_module.DefaultKVCacheConfigBuilder, "_get_pool_bytes_per_block", return_value=1024
            ) as default,
        ):
            self.assertEqual(builder._get_pool_bytes_per_block(MagicMock(), groups), 4096)
        layout.assert_called_once_with(groups)
        glm_bytes.assert_called_once_with(groups)
        qwen_bytes.assert_not_called()
        default.assert_not_called()

    def test_glm_byte_count_result_is_returned_verbatim(self):
        # When a GLM layout is detected, the GLM helper's return value is
        # forwarded without further filtering (including None).
        builder = _builder()
        groups = self._groups()
        with (
            patch.object(builder_module, "_get_glm5_next_cache_layout", return_value=object()),
            patch.object(builder_module, "get_glm5_next_pool_bytes_per_block", return_value=None),
            patch.object(builder_module, "get_qwen4_exp_pool_bytes_per_block", return_value=8192) as qwen_bytes,
            patch.object(builder_module.DefaultKVCacheConfigBuilder, "_get_pool_bytes_per_block", return_value=1024),
        ):
            self.assertIsNone(builder._get_pool_bytes_per_block(MagicMock(), groups))
        qwen_bytes.assert_not_called()

    def test_qwen_layout_used_when_glm_layout_absent(self):
        builder = _builder()
        groups = self._groups()
        with (
            patch.object(builder_module, "_get_glm5_next_cache_layout", return_value=None),
            patch.object(builder_module, "get_glm5_next_pool_bytes_per_block", return_value=4096) as glm_bytes,
            patch.object(builder_module, "get_qwen4_exp_pool_bytes_per_block", return_value=1234) as qwen_bytes,
            patch.object(
                builder_module.DefaultKVCacheConfigBuilder, "_get_pool_bytes_per_block", return_value=1024
            ) as default,
        ):
            self.assertEqual(builder._get_pool_bytes_per_block(MagicMock(), groups), 1234)
        glm_bytes.assert_not_called()
        qwen_bytes.assert_called_once_with(groups)
        default.assert_not_called()

    def test_default_fallback_when_no_model_layout_matches(self):
        builder = _builder()
        groups = self._groups()
        vllm_config = MagicMock()
        with (
            patch.object(builder_module, "_get_glm5_next_cache_layout", return_value=None),
            patch.object(builder_module, "get_qwen4_exp_pool_bytes_per_block", return_value=None),
            patch.object(
                builder_module.DefaultKVCacheConfigBuilder, "_get_pool_bytes_per_block", return_value=777
            ) as default,
        ):
            self.assertEqual(builder._get_pool_bytes_per_block(vllm_config, groups), 777)
        default.assert_called_once_with(vllm_config, groups)

    def test_empty_groups_still_dispatch_through_glm_check_first(self):
        # Even with no groups the GLM layout probe runs first.
        builder = _builder()
        with (
            patch.object(builder_module, "_get_glm5_next_cache_layout", return_value=object()) as layout,
            patch.object(builder_module, "get_glm5_next_pool_bytes_per_block", return_value=2048),
            patch.object(builder_module, "get_qwen4_exp_pool_bytes_per_block", return_value=4096) as qwen_bytes,
        ):
            self.assertEqual(builder._get_pool_bytes_per_block(MagicMock(), []), 2048)
        layout.assert_called_once_with([])
        qwen_bytes.assert_not_called()

    def test_subclass_relationship(self):
        self.assertTrue(issubclass(AscendKVCacheConfigBuilder, builder_module.DefaultKVCacheConfigBuilder))


class TestDispatchPrecedenceMatrix(unittest.TestCase):
    """Exhaustive (glm_layout, qwen_bytes) -> source matrix."""

    def _run(self, glm_layout, qwen_bytes):
        builder = _builder()
        with (
            patch.object(builder_module, "_get_glm5_next_cache_layout", return_value=glm_layout),
            patch.object(builder_module, "get_glm5_next_pool_bytes_per_block", return_value=111),
            patch.object(builder_module, "get_qwen4_exp_pool_bytes_per_block", return_value=qwen_bytes),
            patch.object(builder_module.DefaultKVCacheConfigBuilder, "_get_pool_bytes_per_block", return_value=222),
        ):
            result = builder._get_pool_bytes_per_block(MagicMock(), [MagicMock()])
        # Distinct sentinel values make the source observable: the qwen helper
        # is probed in the default-fallback branch as well, so "was it called"
        # is not a reliable branch marker.
        if result == 111:
            source = "glm"
        elif result == 222:
            source = "default"
        else:
            source = "qwen"
        return result, source

    def test_glm_layout_present(self):
        result, source = self._run(object(), 333)
        self.assertEqual((result, source), (111, "glm"))

    def test_only_qwen_layout(self):
        result, source = self._run(None, 333)
        self.assertEqual((result, source), (333, "qwen"))

    def test_qwen_returns_none_falls_to_default(self):
        result, source = self._run(None, None)
        self.assertEqual((result, source), (222, "default"))
