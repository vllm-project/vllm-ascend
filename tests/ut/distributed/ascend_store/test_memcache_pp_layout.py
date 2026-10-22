# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Round-trip nonuniform PP cache pages through the real GVA builders."""

import ctypes
import unittest
from types import SimpleNamespace

import numpy as np

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import bind_layerwise_keys
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


class TestMemcachePipelineLayout(unittest.TestCase):
    def make_worker(self, stage, dcp_size=1, dcp_rank=0, lengths=None):
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.pp_size = 2
        worker.pp_rank = stage
        worker.num_layers = 42 if stage == 0 else 37  # PP1 includes a draft cache.
        worker.total_layers = 78
        worker.layerwise_key_layers = 42 if stage == 0 else 36
        worker.layerwise_key_layer_offset = 0 if stage == 0 else 42
        worker.use_block_key_layerwise = False
        worker.num_kv_cache_groups = 1
        worker.dcp_size = dcp_size
        worker.dcp_rank = dcp_rank
        worker.put_step = 16
        if lengths is None:
            # A different cache-entry count on some layers exercises indexer
            # pages as well as the last stage's extra physical cache layer.
            layers = [[128 * (1 + layer % 3)] + ([256] if layer % 4 == 0 else []) for layer in range(worker.num_layers)]
        else:
            layers = [[length] for length in lengths]
        offsets = [0]
        for entries in layers:
            offsets.append(offsets[-1] + len(entries))
        flat = [length for entries in layers for length in entries]
        arrays = [
            (np.arange(3 * width, dtype=np.uint32) * 17 + stage * 71 + dcp_rank * 31 + i * 11)
            .astype(np.uint8)
            .reshape(3, width)
            for i, width in enumerate(flat)
        ]
        worker.group_num_layers = {0: len(layers)}
        worker.group_block_len = {0: flat}
        worker.token_database = SimpleNamespace(
            group_block_len=worker.group_block_len,
            group_block_stride={0: flat},
            group_kv_caches_base_addr={0: [array.ctypes.data for array in arrays]},
            group_layer_cache_entry_offsets={0: offsets},
        )
        return worker, arrays

    def test_pp1_real_byte_total_does_not_create_average_layer_offset(self):
        worker, arrays = self.make_worker(1, lengths=[89600] * 36 + [96000])
        total = sum(worker.group_block_len[0])
        self.assertEqual(total, 3321600)
        self.assertEqual((total // 37 * 42) % 128, 56)  # Regression trigger.
        builder = worker._build_group_layer_builders()[0]
        self.assertEqual(builder.layer_byte_offset, 0)
        self.assertEqual(builder.page_size_bytes, total)
        self.assertEqual(worker._global_group_alloc_size(0), total)
        self.assertEqual(len(arrays), 37)

    def test_uneven_pp_stages_and_shared_dcp_shards_roundtrip(self):
        for dcp_size in (1, 2):
            with self.subTest(dcp_size=dcp_size):
                lanes = [self.make_worker(stage, dcp_size, shard) for stage in (0, 1) for shard in range(dcp_size)]
                objects = {}
                descriptors = []
                originals = []
                keys = bind_layerwise_keys(
                    vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=2)),
                    kv_cache_config=None,
                    model_name="model",
                    use_hybrid=False,
                    grouped_block_size=[128],
                )
                wire_keys = set()
                for worker, arrays in lanes:
                    builder = worker._build_group_layer_builders()[0]
                    size = worker._global_group_alloc_size(0)
                    payload = sum(worker.group_block_len[0])
                    stride = size // dcp_size
                    self.assertEqual(builder.layer_byte_offset, worker.dcp_rank * stride)
                    if dcp_size > 1:
                        self.assertEqual(stride % (2 * 1024 * 1024), 0)
                        self.assertGreaterEqual(stride, payload)
                    else:
                        self.assertEqual(size, payload)
                    bases = [(worker.pp_rank * 3 + block + 1) << 32 for block in range(3)]
                    for block, base in enumerate(bases):
                        key = keys.make_full_key(0, str(block), 0, worker.pp_rank)
                        self.assertIn(key, keys.make_hit_check_keys(0, str(block), 1))
                        self.assertIn("@gva2@", key)
                        wire_keys.add(key)
                        objects.setdefault(base, bytearray(size))
                    lane = [
                        builder._build_transfer_arrays(np.arange(3), np.asarray(bases), layer)
                        for layer in range(worker.num_layers)
                    ]
                    descriptors.append(lane)
                    originals.append([array.copy() for array in arrays])
                    for addrs, sizes, gvas in lane:
                        for addr, count, gva in zip(addrs, sizes, gvas):
                            base = int(gva) >> 32 << 32
                            offset = int(gva) - base
                            self.assertEqual(offset % 128, 0)
                            self.assertLessEqual(offset + int(count), len(objects[base]))
                            objects[base][offset : offset + int(count)] = ctypes.string_at(int(addr), int(count))
                self.assertEqual(len(wire_keys), 6)  # Three block keys per stage.
                for _, arrays in lanes:
                    for array in arrays:
                        array.fill(0)
                for (_, arrays), lane, expected in zip(lanes, descriptors, originals):
                    for addrs, sizes, gvas in lane:
                        for addr, count, gva in zip(addrs, sizes, gvas):
                            base = int(gva) >> 32 << 32
                            offset = int(gva) - base
                            ctypes.memmove(int(addr), bytes(objects[base][offset : offset + int(count)]), int(count))
                    for actual, original in zip(arrays, expected):
                        np.testing.assert_array_equal(actual, original)

    def test_distinct_dcp_keys_do_not_add_shared_shard_padding(self):
        worker, arrays = self.make_worker(1, dcp_size=2, dcp_rank=1)
        worker.put_step = 1
        builder = worker._build_group_layer_builders()[0]
        self.assertEqual(builder.layer_byte_offset, 0)
        self.assertEqual(worker._global_group_alloc_size(0), sum(worker.group_block_len[0]))


if __name__ == "__main__":
    unittest.main()
