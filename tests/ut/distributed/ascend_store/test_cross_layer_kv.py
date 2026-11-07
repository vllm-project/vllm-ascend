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

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
import torch
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cross_layer_kv_hooks import (
    dispatch_cross_layer_kv_hook,
    supports_cross_layer_kv_hooks,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cross_layer_kv import (
    CrossLayerKVLoadTask,
    CrossLayerKVLoadUnit,
    CrossLayerKVSaveChunks,
    CrossLayerKVPoolWorker,
    CrossLayerKVSendingThread,
    CrossLayerKVRecvingThread,
    build_cross_layer_kv_trigger_map,
    chunk_member_transfer,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.transfer_mode import (
    KVPoolTransferMode,
    kv_source_layer_ids,
    resolve_kv_pool_transfer_mode,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

# isort: on

V41_SOURCES = (2, 8, 14, 20)


class TestCrossLayerKVHookDispatch(unittest.TestCase):
    def test_nested_multi_connector_reaches_all_children(self):
        wait = MagicMock()
        other_wait = MagicMock()
        connector = SimpleNamespace(
            sub_connectors=[
                SimpleNamespace(),
                SimpleNamespace(sub_connectors=[SimpleNamespace(wait_for_cross_layer_kv_load=wait)]),
                SimpleNamespace(wait_for_cross_layer_kv_load=other_wait),
            ]
        )
        dispatch_cross_layer_kv_hook(connector, "wait_for_cross_layer_kv_load", 8)
        wait.assert_called_once_with(8)
        other_wait.assert_called_once_with(8)

    def test_direct_hook_owns_forwarding(self):
        save = MagicMock()
        child_save = MagicMock()
        connector = SimpleNamespace(save_kv_group=save, sub_connectors=[SimpleNamespace(save_kv_group=child_save)])
        dispatch_cross_layer_kv_hook(connector, "save_kv_group", 14)
        save.assert_called_once_with(14)
        child_save.assert_not_called()


def _vllm_config(extra, architectures=("DeepseekV41ForCausalLM",), text_config=None):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        model_config=SimpleNamespace(
            architectures=list(architectures),
            hf_text_config=text_config or SimpleNamespace(num_hidden_layers=40, kv_source_layer_ids=list(V41_SOURCES)),
        ),
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


class TestResolveKVPoolTransferMode(unittest.TestCase):
    def test_disabled_by_default_even_for_shared_models(self):
        self.assertIs(resolve_kv_pool_transfer_mode(_vllm_config({})), KVPoolTransferMode.BULK)
        self.assertIs(resolve_kv_pool_transfer_mode(_vllm_config({"use_layerwise": False})), KVPoolTransferMode.BULK)

    def test_ds4_compression_keeps_physical_layerwise(self):
        config = _vllm_config(
            {"use_layerwise": True},
            architectures=("DeepseekV4ForCausalLM",),
            text_config=SimpleNamespace(num_hidden_layers=40, compress_ratios=[0, 4, 128] * 14),
        )
        self.assertIs(resolve_kv_pool_transfer_mode(config), KVPoolTransferMode.PHYSICAL_LAYER)

    def test_shared_topology_uses_existing_switch_regardless_of_architecture(self):
        for architecture in (
            "DeepseekV41ForCausalLM",
            "DeepseekV41ForConditionalGeneration",
            "FutureSharedKVForCausalLM",
        ):
            config = _vllm_config(
                {"use_layerwise": True},
                architectures=(architecture,),
                text_config=SimpleNamespace(num_hidden_layers=16, kv_source_layer_ids=[1, 5, 9]),
            )
            self.assertIs(resolve_kv_pool_transfer_mode(config), KVPoolTransferMode.CROSS_LAYER_KV)

    def test_all_layers_own_kv_keeps_physical_layerwise(self):
        config = _vllm_config(
            {"use_layerwise": True}, text_config={"num_hidden_layers": 4, "kv_source_layer_ids": [0, 1, 2, 3]}
        )
        self.assertIs(resolve_kv_pool_transfer_mode(config), KVPoolTransferMode.PHYSICAL_LAYER)

    def test_nested_raw_text_config_and_explicit_text_config_priority(self):
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config={"text_config": {"num_hidden_layers": 16, "kv_source_layer_ids": [1, 9]}}
            )
        )
        self.assertIs(resolve_kv_pool_transfer_mode(config, True), KVPoolTransferMode.CROSS_LAYER_KV)
        config.model_config.hf_text_config = {"num_hidden_layers": 16, "kv_source_layer_ids": []}
        self.assertIs(resolve_kv_pool_transfer_mode(config, True), KVPoolTransferMode.PHYSICAL_LAYER)

    def test_invalid_source_declarations_do_not_silently_enter_old_path(self):
        for sources in ([1, 1], [5, 1], [-1, 5], [1, 16], [True, 5], ["1", 5], "1,5"):
            with self.subTest(sources=sources):
                text = {"num_hidden_layers": 16, "kv_source_layer_ids": sources}
                self.assertFalse(kv_source_layer_ids(text))
                with self.assertRaisesRegex(ValueError, "topology"):
                    resolve_kv_pool_transfer_mode(_vllm_config({"use_layerwise": True}, text_config=text))
                self.assertIs(
                    resolve_kv_pool_transfer_mode(_vllm_config({}, text_config=text)), KVPoolTransferMode.BULK
                )

    def test_every_backbone_consumer_requires_hooks(self):
        context = {
            str(layer): SimpleNamespace(layer_idx=layer, supports_cross_layer_kv_transfer_hooks=True)
            for layer in range(16)
        }
        self.assertTrue(supports_cross_layer_kv_hooks(context, 16))
        context["8"].supports_cross_layer_kv_transfer_hooks = False
        self.assertFalse(supports_cross_layer_kv_hooks(context, 16))
        context["8"].supports_cross_layer_kv_transfer_hooks = True
        del context["9"]
        self.assertFalse(supports_cross_layer_kv_hooks(context, 16))

    def test_changed_source_count_and_positions_derive_matching_units(self):
        sources = (1, 5, 9)
        groups = _v41_layout(num_layers=16, sources=sources)
        offsets = {0: [0, 2, 4, 6], **{group: [0, 1, 2, 3, 4] for group in range(2, 6)}}
        plan = build_cross_layer_kv_trigger_map(groups, sources, offsets)
        self.assertEqual([part.trigger_layer for part in plan.slices_by_group[0]], list(sources))
        self.assertEqual(plan.publish_layers[0], 9)
        worker = _plan_worker(_mla_database(), plan)
        worker.kv_cache_config.kv_cache_groups[0].layer_names = groups[0]
        self.assertEqual([unit.gate_layer for unit in worker._build_cross_layer_kv_load_units({0})], list(sources))
        worker.hf_config = SimpleNamespace(num_hidden_layers=16, kv_source_layer_ids=list(sources))
        worker._cross_layer_kv_save_capable = worker._cross_layer_kv_step_active = True
        worker.on_cross_layer_kv_save_trigger(9)
        worker.kv_send_thread.enqueue_slice_tasks.assert_not_called()
        worker.on_cross_layer_kv_save_trigger(5)
        self.assertEqual(worker.kv_send_thread.enqueue_slice_tasks.call_args.args[0], 5)


class TestBuildCrossLayerKVTriggerMap(unittest.TestCase):
    def test_bounded_off_full_layout(self):
        groups = _v41_layout()
        trigger_map = build_cross_layer_kv_trigger_map(groups, V41_SOURCES)

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
        trigger_map = build_cross_layer_kv_trigger_map(groups, V41_SOURCES)
        self.assertEqual(set(trigger_map.slices_by_group), {0})
        self.assertEqual(trigger_map.trigger_layers(), (2, 8, 14, 20))

    def test_folded_indexer_stays_inside_slot_slice(self):
        groups = _v41_layout(with_folded=True)
        trigger_map = build_cross_layer_kv_trigger_map({0: groups[0]}, V41_SOURCES)
        self.assertEqual(
            [(s.member_start, s.member_end, s.trigger_layer) for s in trigger_map.slices_by_group[0]],
            [(0, 3, 2), (3, 6, 8), (6, 9, 14), (9, 12, 20)],
        )

    def test_draft_group_keeps_bulk_save(self):
        groups = _v41_layout(with_draft=True)
        trigger_map = build_cross_layer_kv_trigger_map(groups, V41_SOURCES)
        self.assertNotIn(12, trigger_map.slices_by_group)
        self.assertEqual(len(trigger_map.slices_by_group), 11)

    def test_non_cross_layer_kv_layout_yields_empty_map(self):
        groups = {
            0: [f"model.layers.{i}.self_attn.kv_cache" for i in range(8)],
            1: [f"model.layers.{i}.self_attn.kv_cache" for i in range(8, 16)],
        }
        trigger_map = build_cross_layer_kv_trigger_map(groups, V41_SOURCES)
        self.assertEqual(trigger_map.slices_by_group, {})
        self.assertEqual(trigger_map.slices_by_trigger, {})
        self.assertEqual(trigger_map.publish_layers, {})

    def test_partial_source_coverage_reports_missing_slots(self):
        groups = {0: _v41_layout()[0][:4]}
        with self.assertLogs(level="WARNING") as captured:
            trigger_map = build_cross_layer_kv_trigger_map(groups, V41_SOURCES)
        self.assertNotIn(0, trigger_map.slices_by_group)
        self.assertIn("configured KV sources", "; ".join(captured.output))


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


class FakeCrossLayerKVStore:
    """Session-capable store recording GVA traffic into simulated memory."""

    supports_exclusive_allocation = True

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
        if self.fail_copy:
            return -1
        for gva, addr, size in zip(gvas, addrs, sizes, strict=True):
            if direction == 0:
                # save: device -> pool
                self.pool[gva : gva + size] = self.device[addr : addr + size]
            else:
                # load: pool -> device
                self.device[addr : addr + size] = self.pool[gva : gva + size]
        return 0

    def batch_write_finish(self, keys, results):
        self.finish_calls.append((list(keys), list(results)))
        for key, result in zip(keys, results, strict=True):
            if int(result) == 0:
                base = self.alloc_base.get(key)
                self.published[key] = bytes(self.pool[base : base + self.alloc_sizes[key]])
                self.alloc_sizes.pop(key, None)
            else:
                self.alloc_base.pop(key, None)
                self.alloc_sizes.pop(key, None)
        return [0] * len(keys)

    def batch_get_key_info(self, keys, *, for_load=False):
        infos = []
        for key in keys:
            base = self.alloc_base.get(key)
            if base is None:
                infos.append(None)
            else:
                size = len(self.published[key]) if key in self.published else self.alloc_sizes[key]
                infos.append(SimpleNamespace(gva_list=lambda b=base: [b], size=lambda s=size: s))
        return infos

    def batch_remove_lease(self, keys):
        return 0

    def batch_add_lease(self, keys, lease_ttl_ms=0):
        return [0 if key in self.published else -1 for key in keys]

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


class TestCrossLayerKVSessionFlow(unittest.TestCase):
    def _fixture(self, published=None):
        database = _mla_database()
        trigger_map = build_cross_layer_kv_trigger_map(
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
        store = FakeCrossLayerKVStore(device)
        store.published.update(published or {})
        thread = CrossLayerKVSendingThread(store, database, [16], tp_rank=0, trigger_map=trigger_map)
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
        chunks = CrossLayerKVSaveChunks(starts=[0], ends=[16], keys=["k0"], block_ids=[7])
        session = thread.open_cross_layer_kv_session(req_meta, 0, chunks, object_size=24)
        thread._handle_cross_layer_kv_open(session)
        self.assertEqual(session.live_keys, ["k0"])
        thread._handle_cross_layer_kv_slice(session, 0, MagicMock())
        self.assertNotIn("k0", store.published)
        thread._handle_cross_layer_kv_slice(session, 1, MagicMock())
        self.assertEqual(store.published["k0"], self._device_bytes(store.device))
        self.assertIn("r1", thread.finished_requests)
        self.assertIsNone(thread.get_stored_request_count("r1"))

    def test_prefiltered_keys_skip_session(self):
        thread, store = self._fixture(published={"k0": b"existing"})
        req_meta = ReqMeta("r1", block_ids=[7])
        chunks = CrossLayerKVSaveChunks(starts=[0], ends=[16], keys=["k0"], block_ids=[7])
        session = thread.open_cross_layer_kv_session(req_meta, 0, chunks, object_size=24)
        thread._handle_cross_layer_kv_open(session)
        self.assertEqual(session.live_keys, [])
        thread._handle_cross_layer_kv_slice(session, 0, MagicMock())
        self.assertEqual(store.published["k0"], b"existing")
        self.assertEqual(store.finish_calls, [])

    def test_failed_slice_revokes_session(self):
        thread, store = self._fixture()
        store.fail_copy = True
        req_meta = ReqMeta("r1", block_ids=[7])
        chunks = CrossLayerKVSaveChunks(starts=[0], ends=[16], keys=["k0"], block_ids=[7])
        session = thread.open_cross_layer_kv_session(req_meta, 0, chunks, object_size=24)
        thread._handle_cross_layer_kv_open(session)
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
        with thread._cross_layer_kv_lock:
            thread._cross_layer_kv_claimed_requests.add("r1")
        self.assertFalse(thread._handle_stored_request(request))


def _slice_task(session, slice_index):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cross_layer_kv import (
        CrossLayerKVSessionSliceTask,
    )

    return CrossLayerKVSessionSliceTask(session, slice_index, MagicMock())


class TestCrossLayerKVPipelinedLoad(unittest.TestCase):
    def _derived_key(self, database, block_hashes=("h0",)):
        for _start, _end, key, _bh, _block_id in database.process_token_key_strings_with_block_ids(
            16,
            list(block_hashes),
            [7],
            0,
            kv_cache_group_id=0,
            skip_null_blocks=False,
            chunk_filter=lambda start: True,
        ):
            return key
        raise AssertionError("chunk enumeration produced no key")

    def _publish_reference_object(self):
        """Publish one block through the session save and return the pieces."""
        database = _mla_database()
        database.load_mask = lambda block_hashes, token_len: None
        database.mask_allows_chunk = lambda masks, group_id, start: True
        trigger_map = build_cross_layer_kv_trigger_map(
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
        store = FakeCrossLayerKVStore(device)
        send_thread = CrossLayerKVSendingThread(store, database, [16], tp_rank=0, trigger_map=trigger_map)
        req_meta = ReqMeta("r1", block_ids=[7])
        chunks = CrossLayerKVSaveChunks(starts=[0], ends=[16], keys=[self._derived_key(database)], block_ids=[7])
        session = send_thread.open_cross_layer_kv_session(req_meta, 0, chunks, object_size=24)
        send_thread._handle_cross_layer_kv_open(session)
        send_thread._handle_cross_layer_kv_slice(session, 0, MagicMock())
        send_thread._handle_cross_layer_kv_slice(session, 1, MagicMock())
        return database, trigger_map, store

    def test_load_roundtrip_restores_device_bytes_and_sets_gates(self):
        database, _trigger_map, store = self._publish_reference_object()
        expected = next(iter(store.published.values()))
        device_target = bytearray(0x10000)
        store.device = device_target
        recv_thread = CrossLayerKVRecvingThread(store, database, [16], tp_rank=0)
        load_spec = SimpleNamespace(vllm_cached_tokens=0)
        req_meta = ReqMeta(
            "r1",
            token_len_chunk=16,
            block_ids=[[7], [7]],
            block_hashes=["h0"],
            load_spec=load_spec,
            kv_cache_group_ids=[0],
        )
        units = [
            CrossLayerKVLoadUnit(0, 0, 2, 2),
            CrossLayerKVLoadUnit(0, 2, 4, 8),
        ]
        gate_events = {2: threading.Event(), 8: threading.Event()}
        task = CrossLayerKVLoadTask(
            req_meta=req_meta,
            token_len=16,
            units=units,
            gate_events=gate_events,
            remaining_by_gate={2: 1, 8: 1},
        )
        recv_thread.request_queue.put(task)
        recv_thread._handle_request(task)
        self.assertTrue(gate_events[2].is_set())
        self.assertTrue(gate_events[8].is_set())
        restored = b"".join(
            bytes(device_target[base + 7 * 24 : base + 7 * 24 + _MLA_MEMBER_SIZES[member]])
            for member, base in enumerate(_MLA_MEMBER_BASES)
        )
        self.assertEqual(restored, expected)

    def test_missing_key_marks_invalid_block_and_fails_request(self):
        database, _trigger_map, store = self._publish_reference_object()
        store.alloc_base.clear()
        device_target = bytearray(0x10000)
        store.device = device_target
        recv_thread = CrossLayerKVRecvingThread(store, database, [16], tp_rank=0)
        load_spec = SimpleNamespace(vllm_cached_tokens=0)
        req_meta = ReqMeta(
            "r1",
            token_len_chunk=16,
            block_ids=[[7], [7]],
            block_hashes=["h0"],
            load_spec=load_spec,
            kv_cache_group_ids=[0],
        )
        task = CrossLayerKVLoadTask(
            req_meta=req_meta,
            token_len=16,
            units=[CrossLayerKVLoadUnit(0, 0, 2, 2)],
            gate_events={2: threading.Event()},
            remaining_by_gate={2: 1},
        )
        recv_thread.request_queue.put(task)
        recv_thread._handle_request(task)
        self.assertTrue(task.gate_events[2].is_set())
        self.assertIn("r1", recv_thread.failed_requests)
        self.assertIn(7, recv_thread._invalid_block_ids)


def _plan_worker(database, trigger_map, *, store=None):
    worker = object.__new__(CrossLayerKVPoolWorker)
    worker.token_database = database
    worker.cross_layer_kv_trigger_map = trigger_map
    worker.cross_layer_kv_trigger_layers = trigger_map.trigger_layers()
    worker.cacheable_group_ids = [0]
    worker.group_uses_align_state = [False]
    worker.group_layer_cache_entry_offsets = {0: [0, 2, 4]}
    worker.cache_transfer_granularity = 16
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[
                    "model.layers.2.self_attn.long_kv_cache",
                    "model.layers.2.self_attn.indexer.k_cache",
                    "model.layers.8.self_attn.long_kv_cache",
                    "model.layers.8.self_attn.indexer.k_cache",
                ]
            )
        ]
    )
    worker._cross_layer_kv_load_tasks = []
    worker._cross_layer_kv_load_gates = {}
    worker._cross_layer_kv_fired_triggers = set()
    worker._cross_layer_kv_fallback_groups = {}
    worker._cross_layer_kv_save_capable = False
    worker._cross_layer_kv_load_capable = True
    worker._cross_layer_kv_step_active = False
    worker._cross_layer_kv_load_started = False
    worker.can_put = True
    worker.m_store = store
    worker.kv_send_thread = MagicMock(spec=CrossLayerKVSendingThread)
    worker.kv_recv_thread = MagicMock(spec=CrossLayerKVRecvingThread)
    worker.hf_config = SimpleNamespace(num_hidden_layers=12, kv_source_layer_ids=[2, 8])
    worker.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(swa_bounded_replay=True))
    return worker


class TestCrossLayerKVRegisteredLayout(unittest.TestCase):
    def test_real_registration_includes_index_scale_and_uses_effective_bytes(self):
        worker = object.__new__(KVPoolWorker)
        worker.num_blocks = 2
        worker.num_layers = 40
        worker.hf_config = SimpleNamespace(num_hidden_layers=40)
        worker._global_to_local_layer = {}
        worker.use_layerwise = False
        worker.use_kvpp = False
        worker._extra_config = {}
        worker.grouped_block_size = [128]
        worker.cache_transfer_granularity = 128
        worker.dcp_size = 1
        worker.group_uses_align_state = [False]
        worker.kv_caches = {}
        for attr in (
            "group_kv_caches_base_addr",
            "group_block_len",
            "group_block_stride",
            "group_layer_cache_entry_offsets",
            "group_cache_entry_segments",
            "group_num_layers",
        ):
            setattr(worker, attr, {})
        backings = []
        for index, layer in enumerate(V41_SOURCES):
            rows = 64 if index < 3 else 128
            stride = 131072 if index < 3 else 147712
            backing = torch.zeros(2 * stride, dtype=torch.uint8)
            backings.append(backing)
            views = []
            for dtype, width, offset in (
                (torch.bfloat16, 512, 0),
                (torch.int8, 128, rows * 1024),
                (torch.float16, 1, rows * (1024 + 128)),
            ):
                item = torch.empty((), dtype=dtype).element_size()
                views.append(
                    torch.as_strided(
                        backing.view(dtype),
                        (2, rows, 1, width),
                        (stride // item, width, width, 1),
                        offset // item,
                    )
                )
            prefix = f"model.layers.{layer}.self_attn"
            worker.kv_caches[prefix + ".long_kv_cache"] = views[0]
            worker.kv_caches[prefix + ".indexer.k_cache"] = tuple(views[1:])
        names = list(reversed(worker.kv_caches))
        worker._infer_cache_group_metadata(0, names)
        database = ChunkedTokenDatabase([KeyMetadata("m", 0, 0, 0)], [128], None)
        database.set_group_buffers(
            worker.group_kv_caches_base_addr,
            worker.group_block_len,
            worker.group_block_stride,
            group_layer_cache_entry_offsets=worker.group_layer_cache_entry_offsets,
            group_cache_entry_segments=worker.group_cache_entry_segments,
        )
        plan = build_cross_layer_kv_trigger_map({0: names}, V41_SOURCES, worker.group_layer_cache_entry_offsets)
        self.assertEqual(
            [(p.member_start, p.member_end) for p in plan.slices_by_group[0]], [(0, 3), (3, 6), (6, 9), (9, 12)]
        )
        addrs, sizes, _ = database.prepare_value(0, 128, [1])
        rows = [
            row
            for part in plan.slices_by_group[0]
            for row in chunk_member_transfer(database, 0, 1, 0, 128, part.member_start, part.member_end)
        ]
        self.assertEqual([row[0] for row in rows], addrs)
        self.assertEqual([row[2] for row in rows], sizes)
        self.assertEqual(sum(sizes), 369280)
        self.assertEqual([sum(sizes[: p.member_start]) for p in plan.slices_by_group[0]], [0, 73856, 147712, 221568])

    def test_unknown_member_rejects_the_whole_group(self):
        names = _v41_layout()[0] + ["model.layers.2.self_attn.unknown_cache"]
        self.assertNotIn(0, build_cross_layer_kv_trigger_map({0: names}, V41_SOURCES).slices_by_group)

    def test_bounded_off_load_has_44_units(self):
        groups = _v41_layout()
        groups.pop(1)
        offsets = {group: list(range(len(names) + 1)) if group else [0, 3, 6, 9, 12] for group, names in groups.items()}
        plan = build_cross_layer_kv_trigger_map(groups, V41_SOURCES, offsets)
        worker = _plan_worker(_mla_database(), plan)
        worker.group_layer_cache_entry_offsets = offsets
        worker.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=groups.get(i, [])) for i in range(12)]
        )
        worker.cacheable_group_ids = list(groups)
        units = worker._build_cross_layer_kv_load_units(set(groups))
        self.assertEqual(len(units), 44)
        self.assertEqual([u.gate_layer for u in units], sorted(u.gate_layer for u in units))


class TestCrossLayerKVSafePublication(unittest.TestCase):
    _fixture = TestCrossLayerKVSessionFlow._fixture
    _device_bytes = TestCrossLayerKVSessionFlow._device_bytes

    def test_uncertain_commit_never_sends_compensating_write_fail(self):
        sender, store = self._fixture()
        original_finish = store.batch_write_finish

        def lose_reply(keys, results):
            original_finish(keys, results)
            raise TimeoutError("response lost after remote commit")

        store.batch_write_finish = lose_reply
        request = ReqMeta("r", block_ids=[7])
        session = sender.open_cross_layer_kv_session(request, 0, CrossLayerKVSaveChunks([0], [16], ["k"], [7]), 24)
        sender._handle_cross_layer_kv_open(session)
        sender._handle_cross_layer_kv_slice(session, 0, MagicMock())
        with self.assertRaises(TimeoutError):
            sender._handle_cross_layer_kv_slice(session, 1, MagicMock())
        sender._fail_session(session)
        sender._fail_session(session)
        self.assertTrue(session.commit_started)
        self.assertTrue(session.finished)
        self.assertIn("k", store.published)
        self.assertEqual(store.finish_calls, [(["k"], [0])])

    def test_shared_allocation_contract_is_rejected_before_allocating(self):
        sender, store = self._fixture()
        store.supports_exclusive_allocation = False
        session = sender.open_cross_layer_kv_session(ReqMeta("r"), 0, CrossLayerKVSaveChunks([0], [16], ["k"], [7]), 24)
        with self.assertRaisesRegex(RuntimeError, "exclusive"):
            sender._handle_cross_layer_kv_open(session)
        self.assertFalse(store.alloc_base)

    def test_duplicate_completed_slice_does_not_publish_or_retire_twice(self):
        sender, store = self._fixture()
        session = sender.open_cross_layer_kv_session(ReqMeta("r"), 0, CrossLayerKVSaveChunks([0], [16], ["k"], [7]), 24)
        sender._handle_cross_layer_kv_open(session)
        sender._handle_cross_layer_kv_slice(session, 0, MagicMock())
        sender._handle_cross_layer_kv_slice(session, 1, MagicMock())
        sender._handle_cross_layer_kv_slice(session, 1, MagicMock())
        self.assertEqual(store.finish_calls, [(["k"], [0])])
        self.assertEqual(sender._cross_layer_kv_claim_counts, {})

    def test_step_end_seal_covers_missing_and_cut_hooks(self):
        sender, store = self._fixture()
        sender.open_cross_layer_kv_session(ReqMeta("r"), 0, CrossLayerKVSaveChunks([0], [16], ["k"], [7]), 24)
        event = MagicMock()
        sender.seal_cross_layer_kv_step(event)
        fence = sender.add_save_batch([])
        while not sender.request_queue.empty():
            sender._handle_request(sender.request_queue.get_nowait())
        self.assertEqual(store.published["k"], self._device_bytes(store.device))
        self.assertTrue(fence.done.is_set())
        self.assertEqual(sender.request_queue.unfinished_tasks, 0)

    def test_bulk_fallback_keeps_all_unclaimed_groups(self):
        sender, _store = self._fixture()
        worker = _plan_worker(sender.token_database, sender.cross_layer_kv_trigger_map)
        request = ReqMeta("r", block_ids=[7], can_save=True, kv_cache_group_ids=[0, 2])
        worker.wait_for_save(SimpleNamespace(requests=[request]))
        requests = worker.kv_send_thread.add_save_batch.call_args.args[0]
        self.assertEqual(requests[0].kv_cache_group_ids, [0, 2])
        self.assertIsNot(requests[0], request)

    def test_bounded_replay_cut_is_deferred_until_step_end(self):
        sender, _store = self._fixture()
        worker = _plan_worker(sender.token_database, sender.cross_layer_kv_trigger_map)
        worker._cross_layer_kv_save_capable = True
        worker._cross_layer_kv_step_active = True
        worker.on_cross_layer_kv_save_trigger(8)
        worker.kv_send_thread.enqueue_slice_tasks.assert_not_called()
        worker.on_cross_layer_kv_save_trigger(2)
        worker.kv_send_thread.enqueue_slice_tasks.assert_called_once()
        worker.on_cross_layer_kv_save_trigger(2)
        self.assertEqual(worker.kv_send_thread.enqueue_slice_tasks.call_count, 1)


class TestCrossLayerKVLoadLifecycle(unittest.TestCase):
    _derived_key = TestCrossLayerKVPipelinedLoad._derived_key
    _publish_reference_object = TestCrossLayerKVPipelinedLoad._publish_reference_object

    def test_lease_then_refresh_and_release_once_after_all_units(self):
        database, _plan, store = self._publish_reference_object()
        key = self._derived_key(database)
        old = store.alloc_base[key]
        expected = bytes(store.pool[old : old + 24])
        refreshed = old + 1024
        store.pool[refreshed : refreshed + 24] = expected
        order = []
        original_get = store.batch_get_key_info

        def key_info(keys, *, for_load=False):
            order.append("rewarm" if for_load else "refresh")
            return original_get(keys, for_load=for_load)

        def add_lease(keys, ttl):
            order.append("lease")
            store.alloc_base[key] = refreshed
            return [0] * len(keys)

        store.batch_get_key_info = key_info
        store.batch_add_lease = add_lease
        store.batch_remove_lease = MagicMock(return_value=0)
        request = ReqMeta("r", block_ids=[7], block_hashes=["h0"], load_spec=SimpleNamespace(vllm_cached_tokens=0))
        task = CrossLayerKVLoadTask(
            request,
            16,
            [CrossLayerKVLoadUnit(0, 0, 2, 2), CrossLayerKVLoadUnit(0, 2, 4, 8)],
            {2: threading.Event(), 8: threading.Event()},
            {2: 1, 8: 1},
        )
        store.device = bytearray(0x10000)
        receiver = CrossLayerKVRecvingThread(store, database, [16], tp_rank=0)
        receiver.request_queue.put(task)
        receiver._handle_request(task)
        self.assertIsNone(task.error)
        self.assertEqual(order, ["rewarm", "lease", "refresh"])
        store.batch_remove_lease.assert_called_once_with([key])
        restored = b"".join(
            bytes(store.device[base + 168 : base + 168 + size])
            for base, size in zip(_MLA_MEMBER_BASES, _MLA_MEMBER_SIZES)
        )
        self.assertEqual(restored, expected)

    def test_no_early_finished_notification_and_interleaved_prefetch(self):
        database, plan, store = self._publish_reference_object()
        worker = _plan_worker(database, plan, store=store)
        requests = [
            ReqMeta(
                req_id,
                block_ids=[7],
                block_hashes=["h0"],
                load_spec=SimpleNamespace(can_load=True, vllm_cached_tokens=0, kvpool_cached_tokens=16),
                kv_cache_group_ids=[0, 1],
            )
            for req_id in ("r1", "r2")
        ]
        handled = worker._prepare_cross_layer_kv_loads(SimpleNamespace(requests=requests))
        self.assertEqual(handled, {"r1", "r2"})
        submitted = [call.args[0] for call in worker.kv_recv_thread.add_request.call_args_list]
        self.assertEqual(
            [(s.task.req_meta.req_id, s.unit.gate_layer) for s in submitted],
            [("r1", 2), ("r2", 2), ("r1", 8), ("r2", 8)],
        )
        worker.kv_recv_thread.set_finished_request.assert_not_called()
        self.assertTrue(all(r.load_spec.can_load for r in requests))
        self.assertIsNot(submitted[0].task.req_meta.block_ids_by_group, requests[0].block_ids_by_group)

    def test_empty_units_settle_gates_without_waiting_forever(self):
        database, _plan, store = self._publish_reference_object()
        request = ReqMeta("r", block_ids=[7], block_hashes=["h0"], load_spec=SimpleNamespace(vllm_cached_tokens=16))
        task = CrossLayerKVLoadTask(request, 16, [CrossLayerKVLoadUnit(0, 0, 2, 2)], {2: threading.Event()}, {2: 1})
        receiver = CrossLayerKVRecvingThread(store, database, [16], tp_rank=0)
        receiver.request_queue.put(task)
        receiver._handle_request(task)
        self.assertTrue(task.done.is_set())
        self.assertTrue(task.gate_events[2].is_set())
        self.assertIsNone(task.error)

    def test_task_error_survives_consumed_failed_request_notifications(self):
        database, plan, _store = self._publish_reference_object()
        worker = _plan_worker(database, plan)
        task = CrossLayerKVLoadTask(ReqMeta("r"), 16, [], {2: threading.Event()})
        task.error = RuntimeError("late unit failure")
        task.gate_events[2].set()
        task.done.set()
        worker._cross_layer_kv_load_tasks = [task]
        worker.kv_recv_thread.failed_requests = set()
        with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
            worker.wait_for_cross_layer_kv_load(2)

    def test_next_step_cannot_clear_failed_inflight_state(self):
        database, plan, _store = self._publish_reference_object()
        worker = _plan_worker(database, plan)
        task = CrossLayerKVLoadTask(ReqMeta("r"), 16, [])
        task.done.set()
        task.error = RuntimeError("cannot resume")
        worker._cross_layer_kv_load_tasks = [task]
        with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
            worker.prepare_cross_layer_kv_step(SimpleNamespace(requests=[]))
        self.assertEqual(worker._cross_layer_kv_load_tasks, [task])

    def test_partial_lease_failure_releases_successful_leases(self):
        database, _plan, store = self._publish_reference_object()
        key = self._derived_key(database)
        store.batch_add_lease = MagicMock(return_value=[0, -1])
        store.batch_remove_lease = MagicMock(return_value=0)
        request = ReqMeta("r", block_ids=[7], block_hashes=["h0"], load_spec=SimpleNamespace(vllm_cached_tokens=0))
        task = CrossLayerKVLoadTask(request, 16, [CrossLayerKVLoadUnit(0, 0, 2, 2)], {2: threading.Event()}, {2: 1})
        receiver = CrossLayerKVRecvingThread(store, database, [16], tp_rank=0)
        receiver.request_queue.put(task)
        receiver._handle_request(task)
        self.assertIsNotNone(task.error)
        self.assertTrue(task.done.is_set())
        store.batch_remove_lease.assert_called_once_with([key])

    def test_first_consumer_runs_while_later_unit_is_still_copying(self):
        database, plan, store = self._publish_reference_object()
        key = self._derived_key(database)
        base = store.alloc_base[key]
        original_copy = store.batch_copy
        later_started, release_later = threading.Event(), threading.Event()

        def copy_with_late_second_source(gvas, addrs, sizes, direction):
            if direction == 1 and gvas[0] == base + 12:
                later_started.set()
                if not release_later.wait(timeout=2):
                    raise TimeoutError("test did not release the later source")
            return original_copy(gvas, addrs, sizes, direction)

        store.batch_copy = copy_with_late_second_source
        receiver = CrossLayerKVRecvingThread(store, database, [16], tp_rank=0)
        worker = _plan_worker(database, plan, store=store)
        worker.kv_recv_thread = receiver
        request = ReqMeta(
            "r",
            block_ids=[7],
            block_hashes=["h0"],
            kv_cache_group_ids=[0],
            load_spec=SimpleNamespace(can_load=True, vllm_cached_tokens=0, kvpool_cached_tokens=16),
        )
        receiver.start()
        self.assertTrue(receiver.ready_event.wait(timeout=2))
        try:
            worker._prepare_cross_layer_kv_loads(SimpleNamespace(requests=[request]))
            worker.wait_for_cross_layer_kv_load(2)
            self.assertTrue(later_started.wait(timeout=2))
            self.assertFalse(worker._cross_layer_kv_load_tasks[0].done.is_set())
            release_later.set()
            worker.wait_for_cross_layer_kv_load(8)
            worker.finish_cross_layer_kv_loads()
            self.assertTrue(worker._cross_layer_kv_load_tasks[0].done.is_set())
            self.assertFalse(receiver.finished_requests)
        finally:
            release_later.set()
            receiver.stop()
            receiver.join(timeout=2)
        self.assertFalse(receiver.is_alive())


class TestCrossLayerKVWorkerCapabilities(unittest.TestCase):
    def test_memcache_shared_alloc_keeps_bulk_save_and_enables_load(self):
        database = _mla_database()
        store = FakeCrossLayerKVStore(bytearray(0x10000))
        store.supports_exclusive_allocation = False
        names = _v41_layout(sources=(2, 8), num_layers=8)[0]
        config = SimpleNamespace(
            compilation_config=SimpleNamespace(
                cudagraph_mode=SimpleNamespace(name="NONE"),
                static_forward_context={
                    str(layer): SimpleNamespace(layer_idx=layer, supports_cross_layer_kv_transfer_hooks=True)
                    for layer in range(12)
                },
            ),
            model_config=SimpleNamespace(enforce_eager=True),
            parallel_config=SimpleNamespace(use_ubatching=False, use_sequence_parallel_moe=False),
            speculative_config=None,
        )

        def initialize(worker, *_args):
            worker.vllm_config = config
            worker._extra_config = {"backend": "memcache", "use_layerwise": True, "load_async": True}
            worker.pp_size = worker.pcp_size = worker.dcp_size = worker.tp_size = 1
            worker.tp_rank = 0
            worker.use_kvpp = worker.load_async = worker.enable_kv_events = False
            worker.load_async = True
            worker.can_put = True
            worker._transfer_threads_started = False
            worker.token_database = database
            worker.grouped_block_size = [16]
            worker.group_layer_cache_entry_offsets = {0: [0, 2, 4]}
            worker.cacheable_group_ids = [0]
            worker.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=names)])
            worker.hf_config = SimpleNamespace(num_hidden_layers=12, kv_source_layer_ids=[2, 8])
            worker.m_store = store
            worker._invalid_block_ids = set()
            worker._invalid_block_ids_lock = threading.Lock()

        def receiver_factory(*args, **kwargs):
            return SimpleNamespace(start=args[6].set)

        module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cross_layer_kv"
        with (
            patch.object(KVPoolWorker, "__init__", autospec=True, side_effect=initialize),
            patch.object(KVPoolWorker, "_start_kv_transfer_threads", autospec=True) as start_bulk,
            patch(module + ".CrossLayerKVRecvingThread", side_effect=receiver_factory),
        ):
            worker = CrossLayerKVPoolWorker(config, True)
            worker._start_kv_transfer_threads()
        self.assertTrue(worker._cross_layer_kv_runtime_supported)
        self.assertTrue(worker._cross_layer_kv_load_capable)
        self.assertFalse(worker._cross_layer_kv_save_capable)
        self.assertFalse(worker.load_async)
        self.assertTrue(worker._extra_config["load_async"])
        start_bulk.assert_called_once_with(worker)
        self.assertFalse(store.alloc_base)

        del config.compilation_config.static_forward_context["8"]
        with (
            patch.object(KVPoolWorker, "__init__", autospec=True, side_effect=initialize),
            patch.object(KVPoolWorker, "_start_kv_transfer_threads", autospec=True) as start_bulk,
            patch(module + ".CrossLayerKVRecvingThread") as create_receiver,
        ):
            worker = CrossLayerKVPoolWorker(config, True)
            worker._start_kv_transfer_threads()
        self.assertFalse(worker._cross_layer_kv_load_capable)
        self.assertFalse(worker._cross_layer_kv_save_capable)
        start_bulk.assert_called_once_with(worker)
        create_receiver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
