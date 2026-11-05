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
from types import SimpleNamespace
from unittest.mock import MagicMock

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ced_incremental_save import (
    CED_LAYERWISE_CONFIG_KEY,
    CEDSaveChunks,
    CEDSessionSendThread,
    build_ced_trigger_map,
    chunk_member_transfer,
    is_ced_model,
    resolve_ced_layerwise,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    ReqMeta,
)

# isort: on

V41_SOURCES = (2, 8, 14, 20)


def _vllm_config(extra, architectures=("DeepseekV41ForCausalLM",)):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        model_config=SimpleNamespace(architectures=list(architectures)),
    )


def _v41_layout(
    num_layers=40,
    sources=V41_SOURCES,
    with_state=True,
    with_folded=False,
    with_draft=False,
):
    """Production-shaped V4.1 cache-group layer-name layout."""
    groups = {}
    mla_members = []
    for source in sources:
        mla_members.append(f"model.layers.{source}.self_attn.long_kv_cache")
        mla_members.append(f"model.layers.{source}.self_attn.indexer.k_cache")
        if with_folded:
            mla_members.append(f"model.layers.{source}.self_attn.indexer.k_cache_folded")
    groups[0] = mla_members
    next_group = 1
    if with_state:
        groups[next_group] = [f"model.layers.{source}.self_attn.compressor.state_cache" for source in sources[:-1]]
        next_group += 1
    for start in range(0, num_layers, 4):
        groups[next_group] = [f"model.layers.{layer}.self_attn.swa_cache" for layer in range(start, start + 4)]
        next_group += 1
    if with_draft:
        groups[next_group] = [f"model.mtp.{i}.self_attn.swa_cache" for i in range(3)]
    return groups


class TestResolveCEDLayerwise(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertFalse(resolve_ced_layerwise(_vllm_config({})))
        self.assertFalse(resolve_ced_layerwise(_vllm_config({CED_LAYERWISE_CONFIG_KEY: False})))

    def test_enabled_on_v41(self):
        config = _vllm_config({CED_LAYERWISE_CONFIG_KEY: True})
        self.assertTrue(resolve_ced_layerwise(config))

    def test_rejects_use_layerwise_combination(self):
        config = _vllm_config({CED_LAYERWISE_CONFIG_KEY: True, "use_layerwise": True})
        with self.assertRaises(ValueError):
            resolve_ced_layerwise(config)

    def test_ignored_on_non_v41_model(self):
        config = _vllm_config({CED_LAYERWISE_CONFIG_KEY: True}, architectures=("Qwen3ForCausalLM",))
        self.assertFalse(resolve_ced_layerwise(config))

    def test_is_ced_model_architectures(self):
        self.assertTrue(is_ced_model(_vllm_config({}, architectures=("DeepseekV41ForCausalLM",))))
        self.assertTrue(is_ced_model(_vllm_config({}, architectures=("DeepseekV41ForConditionalGeneration",))))
        self.assertFalse(is_ced_model(_vllm_config({}, architectures=("DeepseekV3ForCausalLM",))))


class TestBuildCEDTriggerMap(unittest.TestCase):
    def test_bounded_off_full_layout(self):
        groups = _v41_layout()
        trigger_map = build_ced_trigger_map(groups, V41_SOURCES)

        mla_slices = trigger_map.slices_by_group[0]
        self.assertEqual(
            [(s.member_start, s.member_end, s.trigger_layer) for s in mla_slices],
            [(0, 2, 2), (2, 4, 8), (4, 6, 14), (6, 8, 20)],
        )
        # The compressor-state group keeps the step-end bulk save.
        self.assertNotIn(1, trigger_map.slices_by_group)
        # Ten SWA groups, one slice each, triggered by the group-tail layer.
        for group_id in range(2, 12):
            tail_layer = (group_id - 2) * 4 + 3
            self.assertEqual(len(trigger_map.slices_by_group[group_id]), 1)
            self.assertEqual(trigger_map.slices_by_group[group_id][0].trigger_layer, tail_layer)
            self.assertEqual(trigger_map.publish_layers[group_id], tail_layer)
        self.assertEqual(trigger_map.publish_layers[0], 20)
        self.assertEqual(
            set(trigger_map.slices_by_trigger),
            {2, 8, 14, 20} | {3, 7, 11, 15, 19, 23, 27, 31, 35, 39},
        )
        # Slices of the MLA group cover the full member list without gaps.
        covered = [member for s in mla_slices for member in range(s.member_start, s.member_end)]
        self.assertEqual(covered, list(range(len(groups[0]))))

    def test_bounded_on_only_mla_group_pooled(self):
        groups = {0: _v41_layout()[0]}
        trigger_map = build_ced_trigger_map(groups, V41_SOURCES)
        self.assertEqual(set(trigger_map.slices_by_group), {0})
        self.assertEqual(trigger_map.trigger_layers(), (2, 8, 14, 20))

    def test_folded_indexer_stays_inside_slot_slice(self):
        groups = _v41_layout(with_folded=True)
        trigger_map = build_ced_trigger_map({0: groups[0]}, V41_SOURCES)
        self.assertEqual(
            [(s.member_start, s.member_end, s.trigger_layer) for s in trigger_map.slices_by_group[0]],
            [(0, 3, 2), (3, 6, 8), (6, 9, 14), (9, 12, 20)],
        )

    def test_draft_group_keeps_bulk_save(self):
        groups = _v41_layout(with_draft=True)
        trigger_map = build_ced_trigger_map(groups, V41_SOURCES)
        self.assertNotIn(12, trigger_map.slices_by_group)
        self.assertEqual(len(trigger_map.slices_by_group), 11)

    def test_non_ced_layout_yields_empty_map(self):
        groups = {
            0: [f"model.layers.{i}.self_attn.kv_cache" for i in range(8)],
            1: [f"model.layers.{i}.self_attn.kv_cache" for i in range(8, 16)],
        }
        trigger_map = build_ced_trigger_map(groups, V41_SOURCES)
        self.assertEqual(trigger_map.slices_by_group, {})
        self.assertEqual(trigger_map.slices_by_trigger, {})
        self.assertEqual(trigger_map.publish_layers, {})

    def test_partial_source_coverage_reports_missing_slots(self):
        groups = {0: _v41_layout()[0][:4]}
        with self.assertLogs(level="WARNING") as captured:
            trigger_map = build_ced_trigger_map(groups, V41_SOURCES)
        self.assertEqual(len(trigger_map.slices_by_group[0]), 2)
        self.assertIn("no members", "; ".join(captured.output))


def _mla_database() -> ChunkedTokenDatabase:
    """Two-slot MLA group database: members (kv, idx) x 2, block size 16."""
    database = ChunkedTokenDatabase([KeyMetadata("m", 0, 0, 0)], [16], None)
    database.set_group_buffers(
        {0: [0x100, 0x200, 0x300, 0x400]},
        {0: [8, 4, 8, 4]},
        {0: [24, 24, 24, 24]},
        group_num_layers={0: 1},
    )
    return database


_MLA_MEMBER_BASES = (0x100, 0x200, 0x300, 0x400)
_MLA_MEMBER_SIZES = (8, 4, 8, 4)


class FakeCEDStore:
    """Session-capable store recording GVA traffic into simulated memory."""

    def __init__(self, device: bytearray):
        self.device = device
        self.pool = bytearray(1 << 20)
        self.next_gva = 0x1000
        self.requires_exists_before_put = False
        self.alloc_base: dict[str, int] = {}
        self.alloc_sizes: dict[str, int] = {}
        self.published: dict[str, bytes] = {}
        self.finish_calls: list[tuple[list[str], list[int]]] = []
        self.put_calls: list[tuple[list[str], list[list[int]], list[list[int]]]] = []
        self.fail_copy = False

    @property
    def store(self):
        return self

    def set_device(self):
        pass

    def batch_is_readable(self, keys):
        return [key in self.published for key in keys]

    def batch_alloc(self, keys, sizes, lease_ttl_ms=0):
        gvas = []
        for key, size in zip(keys, sizes, strict=True):
            if key in self.published or key in self.alloc_sizes:
                gvas.append(-1)
                continue
            base = self.next_gva
            self.next_gva += size
            self.alloc_base[key] = base
            self.alloc_sizes[key] = size
            gvas.append(base)
        return gvas

    def batch_copy(self, gvas, addrs, sizes, direction):
        assert direction == 0, "session writes use the L2G direction"
        if self.fail_copy:
            return -1
        for dst, src, size in zip(gvas, addrs, sizes, strict=True):
            self.pool[dst : dst + size] = self.device[src : src + size]
        return 0

    def batch_write_finish(self, keys, results):
        self.finish_calls.append((list(keys), list(results)))
        for key, result in zip(keys, results, strict=True):
            if int(result) == 0:
                base = self.alloc_base.pop(key)
                self.published[key] = bytes(self.pool[base : base + self.alloc_sizes.pop(key)])
            else:
                self.alloc_base.pop(key, None)
                self.alloc_sizes.pop(key, None)
        return [0] * len(keys)

    def batch_remove_lease(self, keys):
        return 0

    def put(self, keys, addrs, sizes):
        self.put_calls.append((list(keys), list(addrs), list(sizes)))
        return [True] * len(keys)


class TestChunkMemberTransfer(unittest.TestCase):
    def test_full_block_slices_match_atomic_layout(self):
        database = _mla_database()
        atomic_addrs, atomic_sizes, _ = database.prepare_value(0, 16, [7], kv_cache_group_id=0, block_id=7)
        slice_rows = chunk_member_transfer(database, 0, 7, 0, 16, 0, 2) + chunk_member_transfer(
            database, 0, 7, 0, 16, 2, 4
        )
        self.assertEqual([row[0] for row in slice_rows], atomic_addrs)
        self.assertEqual([row[2] for row in slice_rows], atomic_sizes)
        self.assertEqual([row[1] for row in slice_rows], [0, 8, 12, 20])

    def test_partial_chunk_slices_match_atomic_prefix(self):
        database = _mla_database()
        atomic_addrs, atomic_sizes, _ = database.prepare_value(0, 6, [7], kv_cache_group_id=0, block_id=7)
        slice_rows = chunk_member_transfer(database, 0, 7, 0, 6, 0, 2) + chunk_member_transfer(
            database, 0, 7, 0, 6, 2, 4
        )
        self.assertEqual([row[0] for row in slice_rows], atomic_addrs)
        self.assertEqual([row[2] for row in slice_rows], atomic_sizes)
        self.assertEqual([row[1] for row in slice_rows], [0, 3, 4, 7])


class TestCEDSessionFlow(unittest.TestCase):
    def _fixture(self, published=None):
        database = _mla_database()
        trigger_map = build_ced_trigger_map(
            {
                0: [
                    "model.layers.2.self_attn.long_kv_cache",
                    "model.layers.2.self_attn.indexer.k_cache",
                    "model.layers.8.self_attn.long_kv_cache",
                    "model.layers.8.self_attn.indexer.k_cache",
                ]
            },
            (2, 8),
        )
        device = bytearray(0x10000)
        for member, base in enumerate(_MLA_MEMBER_BASES):
            addr = base + 7 * 24
            for i in range(_MLA_MEMBER_SIZES[member]):
                device[addr + i] = (0x11 * (member + 1) + i) & 0xFF
        store = FakeCEDStore(device)
        store.published.update(published or {})
        thread = CEDSessionSendThread(store, database, [16], tp_rank=0, trigger_map=trigger_map)
        return thread, store

    def _device_bytes(self, device):
        chunks = []
        for member, base in enumerate(_MLA_MEMBER_BASES):
            addr = base + 7 * 24
            chunks.append(bytes(device[addr : addr + _MLA_MEMBER_SIZES[member]]))
        return b"".join(chunks)

    def test_session_publishes_byte_identical_object(self):
        thread, store = self._fixture()
        req_meta = ReqMeta("r1", block_ids=[7])
        chunks = CEDSaveChunks(starts=[0], ends=[16], keys=["k0"], block_ids=[7])
        session = thread.open_ced_session(req_meta, 0, chunks, object_size=24)
        thread._handle_ced_open(session)
        self.assertEqual(session.live_keys, ["k0"])
        thread._handle_ced_slice(session, 0, MagicMock())
        self.assertNotIn("k0", store.published)
        thread._handle_ced_slice(session, 1, MagicMock())
        self.assertEqual(store.published["k0"], self._device_bytes(store.device))
        self.assertIn("r1", thread.finished_requests)
        self.assertIsNone(thread.get_stored_request_count("r1"))

    def test_prefiltered_keys_skip_session(self):
        thread, store = self._fixture(published={"k0": b"existing"})
        req_meta = ReqMeta("r1", block_ids=[7])
        chunks = CEDSaveChunks(starts=[0], ends=[16], keys=["k0"], block_ids=[7])
        session = thread.open_ced_session(req_meta, 0, chunks, object_size=24)
        thread._handle_ced_open(session)
        self.assertEqual(session.live_keys, [])
        thread._handle_ced_slice(session, 0, MagicMock())
        self.assertEqual(store.published["k0"], b"existing")
        self.assertEqual(store.finish_calls, [])

    def test_failed_slice_revokes_session(self):
        thread, store = self._fixture()
        store.fail_copy = True
        req_meta = ReqMeta("r1", block_ids=[7])
        chunks = CEDSaveChunks(starts=[0], ends=[16], keys=["k0"], block_ids=[7])
        session = thread.open_ced_session(req_meta, 0, chunks, object_size=24)
        thread._handle_ced_open(session)
        thread._handle_request(_slice_task(session, 0))
        self.assertTrue(session.failed)
        self.assertNotIn("k0", store.published)
        self.assertEqual(store.finish_calls, [(["k0"], [-1])])
        self.assertIsNone(thread.get_stored_request_count("r1"))

    def test_claimed_requests_do_not_advance_saved_offset(self):
        thread, store = self._fixture()
        request = ReqMeta(
            "r1",
            token_len_chunk=16,
            block_ids=[7],
            block_hashes=["h0"],
            can_save=True,
            kv_cache_group_ids=[0],
        )
        self.assertTrue(thread._handle_stored_request(request))
        with thread._ced_lock:
            thread._ced_claim_counts["r1"] = 1
        self.assertFalse(thread._handle_stored_request(request))


def _slice_task(session, slice_index):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ced_incremental_save import (
        CEDSessionSliceTask,
    )

    return CEDSessionSliceTask(session, slice_index, MagicMock())


if __name__ == "__main__":
    unittest.main()
