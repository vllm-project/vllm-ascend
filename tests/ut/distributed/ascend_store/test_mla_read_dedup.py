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

import ctypes
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVReadGroup,
    _mla_read_dedup_byte_budget,
    _partition_dedup_failure,
    sharded_get_all_gather,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KVCacheByteRangeMatrix,
    KeyMetadata,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


class _AllGatherContext:
    def __init__(self, world_size: int):
        self.payloads: list[torch.Tensor | None] = [None] * world_size
        self.barrier = threading.Barrier(world_size, timeout=10)


class _AllGatherGroup:
    def __init__(self, context: _AllGatherContext, rank: int):
        self._context = context
        self.rank_in_group = rank
        self.world_size = len(context.payloads)

    def all_gather(self, value: torch.Tensor, dim: int = 0) -> torch.Tensor:
        self._context.payloads[self.rank_in_group] = value.clone()
        self._context.barrier.wait()
        payloads = [payload for payload in self._context.payloads if payload is not None]
        assert len(payloads) == self.world_size
        return torch.cat(payloads, dim=dim)


class _ByteStore:
    def __init__(self, payloads: dict[str, bytes], failed: set[str] | None = None):
        self.payloads = payloads
        self.failed = failed or set()
        self.get_calls: list[list[str]] = []
        self.strict_get_results = False

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> list[int]:
        self.get_calls.append(list(keys))
        for key, key_addrs, key_sizes in zip(keys, addrs, sizes, strict=True):
            if key in self.failed:
                continue
            for addr, size in zip(key_addrs, key_sizes, strict=True):
                ctypes.memmove(addr, self.payloads[key][:size], size)
        return [1 if key in self.failed else 0 for key in keys]


def _task(cache: torch.Tensor, payloads: dict[str, bytes], block_ids: list[int], context: _AllGatherContext, rank: int):
    block_size = cache.shape[1]
    database = ChunkedTokenDatabase([KeyMetadata("m", 0, 0, 0)], [block_size], None)
    database.set_group_buffers(
        {0: [cache.data_ptr()]},
        {0: [block_size]},
        {0: [block_size]},
        group_num_layers={0: 1},
        group_cache_storages={0: [(cache.untyped_storage(), 0, cache.device)]},
    )
    addrs, sizes, block_id_list, views = [], [], [], KVCacheByteRangeMatrix()
    for key_index, physical_block_id in enumerate(block_ids):
        addr, size, spans, _ = database.prepare_value_with_ranges(
            0,
            block_size,
            block_ids,
            block_id=physical_block_id,
        )
        addrs.append(addr)
        sizes.append(size)
        block_id_list.append(physical_block_id)
        views.append(spans)
    store = _ByteStore(payloads)
    return store, _AllGatherGroup(context, rank), rank, list(payloads), addrs, sizes, block_id_list, views


class TestMLAReadDedup(unittest.TestCase):
    def test_tp_shards_are_byte_identical(self):
        block_size = 8
        block_ids = [4, 2, 0, 3, 1]
        payloads = {f"k{index}": bytes([index + 1]) * block_size for index in range(len(block_ids))}

        for tp_size in (2, 4, 8):
            with self.subTest(tp_size=tp_size):
                caches = [torch.zeros(len(block_ids), block_size, dtype=torch.uint8) for _ in range(tp_size)]
                context = _AllGatherContext(tp_size)
                tasks = [_task(cache, payloads, block_ids, context, rank) for rank, cache in enumerate(caches)]
                errors: list[BaseException] = []

                def run(task, errors=errors):
                    try:
                        sharded_get_all_gather(*task)
                    except BaseException as error:  # pragma: no cover - asserted below
                        errors.append(error)

                threads = [threading.Thread(target=run, args=(task,)) for task in tasks]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10)

                self.assertFalse(any(thread.is_alive() for thread in threads))
                self.assertEqual(errors, [])
                expected = torch.empty_like(caches[0])
                for key_index, physical_block_id in enumerate(block_ids):
                    expected[physical_block_id].fill_(key_index + 1)
                for cache in caches:
                    torch.testing.assert_close(cache, expected)
                self.assertEqual(sum(len(keys) for store, *_ in tasks for keys in store.get_calls), len(payloads))

    def test_shard_failure_is_propagated_to_every_rank(self):
        block_size = 8
        block_ids = list(range(9))
        payloads = {f"k{index}": bytes([index + 1]) * block_size for index in range(len(block_ids))}
        caches = [torch.zeros(len(block_ids), block_size, dtype=torch.uint8) for _ in range(2)]
        context = _AllGatherContext(2)
        tasks = [_task(cache, payloads, block_ids, context, rank) for rank, cache in enumerate(caches)]
        tasks[0][0].failed = {"k0"}
        tasks[1][0].failed = {"k8"}
        results = []
        errors: list[BaseException] = []

        def run(task, errors=errors):
            try:
                results.append(sharded_get_all_gather(*task))
            except BaseException as error:  # pragma: no cover - asserted below
                errors.append(error)

        threads = [threading.Thread(target=run, args=(task,)) for task in tasks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        for request_failed, missing_block_ids, _ in results:
            self.assertFalse(request_failed)
            self.assertEqual(missing_block_ids, {0, 8})

    def test_fallback_guards(self):
        mib = 1024 * 1024
        self.assertIsNone(_mla_read_dedup_byte_budget(2, local_sdma=True))
        self.assertIsNone(_mla_read_dedup_byte_budget(4, local_sdma=True))
        self.assertEqual(_mla_read_dedup_byte_budget(8, local_sdma=True), 64 * mib)
        self.assertEqual(_mla_read_dedup_byte_budget(2, local_sdma=False), 32 * mib)
        self.assertEqual(_mla_read_dedup_byte_budget(4, local_sdma=False), 16 * mib)
        self.assertEqual(_mla_read_dedup_byte_budget(8, local_sdma=False), 8 * mib)

        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker._transfer_layout = SimpleNamespace(host_memory_group_ids=frozenset({1}))
        self.assertFalse(worker._mla_read_dedup_supported_for_groups([0, 1]))
        self.assertTrue(worker._mla_read_dedup_supported_for_groups([0]))

        self.assertEqual(_partition_dedup_failure(ReqMeta("single", block_ids_by_group=[[7]]), {7}), ({7}, False))
        self.assertEqual(_partition_dedup_failure(ReqMeta("multi", block_ids_by_group=[[7], [8]]), {7}), (set(), True))

    def test_read_group_is_closed(self):
        device_group = object()
        with (
            patch.object(torch.distributed, "new_group", return_value=device_group),
            patch.object(torch.distributed, "get_backend", return_value="hccl"),
            patch.object(torch.distributed, "destroy_process_group") as destroy,
            patch.object(torch.npu, "Stream"),
        ):
            group = KVReadGroup(MagicMock(ranks=[0, 1], world_size=2))
            group.close()
        destroy.assert_called_once_with(device_group)


if __name__ == "__main__":
    unittest.main()
