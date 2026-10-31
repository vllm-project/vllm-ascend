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
"""CPU-only unit tests for ``vllm_ascend.core.kv_cache_config_builder``.

``AscendKVCacheConfigBuilder`` subclasses vLLM's default KV cache config
builder and overrides the physical pool accounting for two model families
whose layouts the default one-backing probe cannot infer:

* GLM-Next (main MLA/KDA plus compressed-indexer/tail pages live in
  separate device tensors) -> ``get_glm5_next_pool_bytes_per_block``
* Qwen4-exp slab layouts -> ``get_qwen4_exp_pool_bytes_per_block``

Everything else falls through to the vLLM default implementation. The
dispatch order (GLM first, Qwen second, default last) is exactly what
these tests pin.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

try:
    import vllm_ascend.core.kv_cache_config_builder as builder_module
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
except ImportError:
    # The builder subclasses ``vllm.v1.core.kv_cache_planning.
    # DefaultKVCacheConfigBuilder``, which the oldest pinned vLLM matrix leg
    # (v0.32.0) does not ship. A module-level import error would abort the
    # whole cpu-ut collection, so skip the module on that leg instead.
    pytest.skip(
        "kv_cache_config_builder requires a vLLM version that ships vllm.v1.core.kv_cache_planning",
        allow_module_level=True,
    )


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
