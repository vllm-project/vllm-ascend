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
# This file is a part of the vllm-ascend project.
#
"""NPUWorker.initialize_from_config must mirror core's Worker contract.

Core's ``Worker.initialize_from_config`` publishes the post-profiling
block count on the cache config
(``self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks``) as
its FIRST statement, before building any KV-transfer connector, so that
the value is available to the warmup stage; connectors read the
attribute in ``register_kv_caches``.

The regression pinned here: the mirror was only ever set on the legacy
``initialize_cache`` path, so a KV connector reading
``cache_config.num_gpu_blocks`` during registration saw ``None`` and
crashed with an ``AssertionError`` on every non-legacy boot.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.worker.worker import NPUWorker


def _make_kv_cache_config(num_blocks: int = 4242) -> SimpleNamespace:
    """The attributes initialize_from_config reads on this path."""
    return SimpleNamespace(
        num_blocks=num_blocks,
        needs_kv_cache_zeroing=False,
        has_mamba_layers=False,
    )


def _make_worker() -> NPUWorker:
    """A worker stub bypassing __init__ (no NPU required)."""
    worker = object.__new__(NPUWorker)
    worker.cache_config = SimpleNamespace(num_gpu_blocks=None)
    # MagicMock attribute lookups never equal plain strings, so the
    # speculative "dspark" branch inside initialize_from_config is not
    # taken.
    worker.vllm_config = MagicMock()
    worker.model_runner = MagicMock()
    worker.use_v2_model_runner = False
    return worker


class TestNumGpuBlocksMirror(unittest.TestCase):
    def test_initialize_from_config_mirrors_num_blocks(self):
        """The mirror attribute is populated from the authoritative
        kv_cache_config, exactly as core's Worker does."""
        worker = _make_worker()
        kv_cache_config = _make_kv_cache_config(num_blocks=4242)
        with (
            patch("vllm_ascend.worker.worker.ensure_kv_transfer_initialized") as ensure,
            patch.object(
                NPUWorker,
                "_maybe_get_memory_pool_context",
                return_value=MagicMock(),
            ),
        ):
            NPUWorker.initialize_from_config(worker, kv_cache_config)
        ensure.assert_called_once()
        self.assertEqual(worker.cache_config.num_gpu_blocks, 4242)
        worker.model_runner.initialize_kv_cache.assert_called_once()

    def test_mirror_visible_to_connector_construction(self):
        """The mirror is set BEFORE the connector is constructed
        (ensure_kv_transfer_initialized) AND before the caches are
        registered (model_runner.initialize_kv_cache, where
        register_kv_caches runs and the original AssertionError fired):
        both observe the authoritative block count."""
        worker = _make_worker()
        kv_cache_config = _make_kv_cache_config(num_blocks=777)
        seen = {}

        def ensure(_vllm_config, _kv_cache_config):
            seen["at_construction"] = worker.cache_config.num_gpu_blocks

        def initialize_kv_cache(_config, **_kwargs):
            seen["at_registration"] = worker.cache_config.num_gpu_blocks

        worker.model_runner.initialize_kv_cache.side_effect = initialize_kv_cache
        with (
            patch(
                "vllm_ascend.worker.worker.ensure_kv_transfer_initialized",
                side_effect=ensure,
            ),
            patch.object(
                NPUWorker,
                "_maybe_get_memory_pool_context",
                return_value=MagicMock(),
            ),
        ):
            NPUWorker.initialize_from_config(worker, kv_cache_config)
        self.assertEqual(seen["at_construction"], 777)
        self.assertEqual(seen["at_registration"], 777)

    def test_legacy_initialize_cache_value_matches(self):
        """Consistency with the legacy path: initialize_cache assigns
        the same attribute, so both paths publish one coherent value."""
        worker = _make_worker()
        NPUWorker.initialize_cache(worker, 99, 7)
        self.assertEqual(worker.cache_config.num_gpu_blocks, 99)
        self.assertEqual(worker.cache_config.num_cpu_blocks, 7)


if __name__ == "__main__":
    unittest.main()
