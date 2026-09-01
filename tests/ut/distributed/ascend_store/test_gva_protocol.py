#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import unittest

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    backend_map,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import (
    Backend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.gva_protocol import (
    GVAKeyFactory,
)

# isort: on

_GVA_STORE_METHODS = (
    "batch_get_key_info",
    "batch_alloc",
    "batch_add_lease",
    "batch_remove_lease",
    "batch_write_finish",
)


class TestGvaMemcacheExclusivity(unittest.TestCase):
    """GVA is a memcache-exclusive protocol.

    The five GVA store calls must be overridden by MemcacheBackend and
    inherited as NotImplementedError stubs by every other backend
    (python's MRO: an override wins over the inherited stub).
    """

    def _backend_classes(self):
        import importlib

        for name, entry in backend_map.items():
            module = importlib.import_module(entry["path"])
            yield name, getattr(module, entry["name"])

    def test_gva_store_methods_only_on_memcache_backend(self):
        for name, backend_class in self._backend_classes():
            with self.subTest(backend=name):
                for method in _GVA_STORE_METHODS:
                    with self.subTest(method=method):
                        owns_override = any(method in vars(cls) for cls in backend_class.__mro__ if cls is not Backend)
                        self.assertEqual(owns_override, name == "memcache")


class TestGVAKeyFactory(unittest.TestCase):
    """Byte-for-byte snapshots of the GVA key formats.

    These strings are wire formats shared with deployed clusters: a single
    character of drift turns hits into misses after an upgrade. The
    expectations are transcribed from the pre-refactor pool_worker /
    pool_scheduler implementations.
    """

    def test_full_key_single_group_keeps_pr_11585_format(self):
        self.assertEqual(
            GVAKeyFactory.full_key("model", 0, "hash0", 3, 1),
            "model@hash0@3",
        )

    def test_full_key_multi_group_includes_group_id(self):
        self.assertEqual(
            GVAKeyFactory.full_key("model", 2, "hash0", 3, 4),
            "model@2@hash0@3",
        )

    def test_partial_key_format(self):
        self.assertEqual(
            GVAKeyFactory.partial_key("model", "r1", 0, 1, 20, 3),
            "model@partial@r1@0@1@20@3",
        )

    def test_hit_check_keys_single_group_one_key_per_rank(self):
        self.assertEqual(
            GVAKeyFactory.hit_check_keys("model", 0, "hash0", 4, 1),
            ["model@hash0@0", "model@hash0@1", "model@hash0@2", "model@hash0@3"],
        )

    def test_hit_check_keys_multi_group_one_key_per_rank(self):
        self.assertEqual(
            GVAKeyFactory.hit_check_keys("model", 1, "hash0", 2, 3),
            ["model@1@hash0@0", "model@1@hash0@1"],
        )

    def test_hit_check_keys_empty_when_no_ranks(self):
        self.assertEqual(GVAKeyFactory.hit_check_keys("model", 0, "hash0", 0, 1), [])

    def test_full_key_and_hit_check_key_share_rank_format(self):
        """The hit-check key of rank r must equal that rank's full_key."""
        for num_groups in (1, 2):
            for rank in range(3):
                with self.subTest(num_groups=num_groups, rank=rank):
                    self.assertEqual(
                        GVAKeyFactory.hit_check_keys("model", 0, "hash0", 3, num_groups)[rank],
                        GVAKeyFactory.full_key("model", 0, "hash0", rank, num_groups),
                    )


if __name__ == "__main__":
    unittest.main()
