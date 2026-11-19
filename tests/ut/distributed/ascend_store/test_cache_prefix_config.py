# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real upstream configuration coverage; never substitute config/hash mocks."""

import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.ut.distributed.ascend_store._mock_deps import _MOCK_VLLM_DEPS

if not _MOCK_VLLM_DEPS:
    import torch
    from transformers import OPTConfig
    from vllm.config import CacheConfig, DeviceConfig, KVTransferConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, UniformTypeKVCacheSpecs

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import resolve_cache_prefix
    from vllm_ascend.utils import vllm_version_is

    if vllm_version_is("0.33.0"):
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (  # type: ignore[import-not-found]
            worker as mooncake_worker,
        )
    else:
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import worker as mooncake_worker


def make_config(model_path, prefix="auto", **kwargs):
    return VllmConfig(
        model_config=ModelConfig(model=str(model_path), dtype="float16"),
        device_config=DeviceConfig("cpu"),
        kv_transfer_config=KVTransferConfig(
            kv_connector="AscendStoreConnector",
            # Resolve the real connector through the supported upstream API,
            # without depending on plugin-wide startup registration in CPU UTs.
            kv_connector_module_path=(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector"
            ),
            kv_role="kv_both",
            kv_connector_extra_config={"backend": "mooncake", "cache_prefix": prefix},
        ),
        **kwargs,
    )


@unittest.skipIf(
    _MOCK_VLLM_DEPS, "requires real pinned vLLM/torch/transformers; mock results are not config validation"
)
class TestRealCachePrefixConfig(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.model_path = Path(directory.name)
        OPTConfig(
            hidden_size=64, num_hidden_layers=2, num_attention_heads=4, ffn_dim=128, word_embed_proj_dim=64
        ).save_pretrained(self.model_path)
        env = patch.dict(os.environ, VLLM_IMAGE_TAG="prefix-test-image-v1", VLLM_BUILD_COMMIT="unknown")
        env.start()
        self.addCleanup(env.stop)

    def test_literal_conversion_without_upstream_private_helper(self):
        for value, expected in (
            ("", ""),
            ("custom", "custom"),
            ("AUTO", "AUTO"),
            (None, "None"),
            (False, "False"),
            (0, "0"),
        ):
            with self.subTest(value=value):
                self.assertEqual(resolve_cache_prefix(make_config(self.model_path, value)), expected)

    def test_upstream_parity_for_literal_conversion_and_auto(self):
        upstream_resolver = getattr(mooncake_worker, "_resolve_cache_prefix", None)
        if upstream_resolver is None:
            self.skipTest("this vLLM release has no automatic cache-prefix reference helper")
        values: tuple[object, ...] = ("", "custom", "AUTO", None, False, 0, [], {}, "auto")
        for value in values:
            with self.subTest(value=value):
                config = make_config(self.model_path, value)
                self.assertEqual(resolve_cache_prefix(config), upstream_resolver(config))

    def test_actual_group_geometry_normalizes_without_mutating_configs(self):
        worker = make_config(self.model_path, cache_config=CacheConfig(block_size=32))
        scheduler = deepcopy(worker)
        scheduler.cache_config.block_size = 16
        spec = FullAttentionSpec(block_size=16, num_kv_heads=4, head_size=16, dtype=torch.float16)
        uniform = UniformTypeKVCacheSpecs.from_specs({"model.decoder.layers.1": spec.copy_with_new_block_size(32)})
        plan = SimpleNamespace(
            kv_cache_groups=[
                KVCacheGroupSpec(["model.decoder.layers.0"], spec),
                KVCacheGroupSpec(["model.decoder.layers.1"], uniform),
            ]
        )
        self.assertEqual(resolve_cache_prefix(worker, plan), resolve_cache_prefix(scheduler, plan))
        self.assertEqual((worker.cache_config.block_size, scheduler.cache_config.block_size), (32, 16))

    def test_actual_cache_and_model_factors_and_runtime_capacity(self):
        config = make_config(self.model_path)
        original = resolve_cache_prefix(config)
        other = deepcopy(config)
        other.cache_config.num_gpu_blocks = 100
        other.cache_config.num_cpu_blocks = 10
        other.cache_config.gpu_memory_utilization = 0.5
        self.assertEqual(resolve_cache_prefix(other), original)
        other = make_config(
            self.model_path,
            scheduler_config=SchedulerConfig(
                max_model_len=2048, is_encoder_decoder=False, max_num_seqs=8, max_num_batched_tokens=4096
            ),
        )
        self.assertEqual(resolve_cache_prefix(other), original)
        other_block_size = 32 if config.cache_config.block_size != 32 else 64
        for field, value in (("cache_dtype", "fp8"), ("block_size", other_block_size)):
            other = deepcopy(config)
            setattr(other.cache_config, field, value)
            self.assertNotEqual(resolve_cache_prefix(other), original)
        other = deepcopy(config)
        other.model_config.hf_config._commit_hash = "new-hf-revision"
        self.assertNotEqual(resolve_cache_prefix(other), original)
        other = deepcopy(config)
        other.model_config.dtype = torch.bfloat16
        self.assertNotEqual(resolve_cache_prefix(other), original)
        with patch.dict(os.environ, VLLM_IMAGE_TAG="prefix-test-image-v2"):
            self.assertNotEqual(resolve_cache_prefix(config), original)

    def test_independent_process_uses_identical_namespace(self):
        code = (
            "import sys; "
            # A plain Python child does not load pytest's conftest. Reuse the
            # existing CPU dependency bootstrap before importing _mock_deps.
            "import tests.ut.conftest; "
            "from tests.ut.distributed.ascend_store.test_cache_prefix_config import make_config; "
            "from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import resolve_cache_prefix; "
            "print('PREFIX_RESULT=' + resolve_cache_prefix(make_config(sys.argv[1])))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(self.model_path)],
            cwd=Path(__file__).resolve().parents[4],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        prefixes = [
            line.removeprefix("PREFIX_RESULT=")
            for line in result.stdout.splitlines()
            if line.startswith("PREFIX_RESULT=")
        ]
        self.assertEqual(prefixes, [resolve_cache_prefix(make_config(self.model_path))])
