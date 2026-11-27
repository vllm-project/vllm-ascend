# SPDX-License-Identifier: Apache-2.0
"""CPU qualification of fixed side-group membership and slot safety."""

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]
MODULE = ROOT / "vllm_ascend/models/deepseek_v41/engram/shared_cann.py"

META = {"tables": {"1": {"code_dtype": "F8_E4M3", "scale_dtype": "F8_E8M0", "rows": 8}}}


class Buffer:
    @staticmethod
    def get_engram_storage_size_hint(*args):
        return 2097152

    def __init__(self, group, *args, **kwargs):
        self.destroyed = False
        self._engram_context_tensor = group
        self._engram_dtype_int = 0

    def engram_write(self, storage, *, owner_only_num_entries):
        assert storage.shape == (1, 132)
        self.rows = owner_only_num_entries

    def engram_fetch(self, indices):
        fetched = torch.ops.cann_ops_transformer.engram_fetch(
            self._engram_context_tensor, indices, 132, self.rows, self._engram_dtype_int
        )
        return lambda: torch.ops.cann_ops_transformer.engram_fetch_wait(self._engram_context_tensor, fetched)

    def destroy(self):
        self.destroyed = True


def load_reader():
    module_names = [
        "cann_ops_transformer",
        "cann_ops_transformer.ops",
        "cann_ops_transformer.ops.mc2",
        "cann_ops_transformer.ops.mc2.common",
        "cann_ops_transformer.ops.mc2.common.elastic_buffer",
        "torch_npu",
        "vllm_ascend",
        "vllm_ascend.ops",
        "fake_engram",
        "fake_engram.shared_rdma",
    ]
    modules = {name: types.ModuleType(name) for name in module_names}
    modules["fake_engram"].__path__ = []
    modules["vllm_ascend"].ops = modules["vllm_ascend.ops"]
    modules["fake_engram.shared_rdma"].checkpoint_metadata = lambda _: META
    modules["cann_ops_transformer.ops.mc2.common.elastic_buffer"].ElasticBuffer = Buffer
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("fake_engram.shared_cann", MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class Store:
    def __init__(self):
        self.values = {
            "reader-count": b"2",
            "metadata": json.dumps(META).encode(),
            "prepared-1-0": "1",
            "prepared-1-1": "1",
        }
        self.claims = {}

    def get(self, key):
        return self.values[key]

    def set(self, key, value):
        self.values[key] = value

    def add(self, key, value):
        self.claims[key] = self.claims.get(key, 0) + value
        return self.claims[key]

    def wait(self, keys, timeout):
        assert all(key in self.values for key in keys)


class SlotTest(unittest.TestCase):
    def setUp(self):
        self.module = load_reader()
        self.store = Store()
        self.groups = []

    def reader(self, base, count=2, capacity=128):
        def group(store, name, side_rank, world, mapping):
            self.groups.append((side_rank, world, mapping))
            return name

        with (
            patch.object(self.module.dist, "get_rank", return_value=0),
            patch.object(self.module.dist, "get_world_size", return_value=1),
            patch.object(self.module.dist, "TCPStore", return_value=self.store),
            patch.object(self.module, "_group", side_effect=group),
        ):
            return self.module.CANNReader("/model", "127.0.0.1:30970", capacity=capacity, readers=count, rank_base=base)

    def test_two_independent_model_worlds_claim_distinct_slots_and_close(self):
        first = self.reader(1)
        second = self.reader(2)
        self.assertEqual(self.groups, [(1, 3, {0: 1}), (2, 3, {0: 2})])
        self.assertEqual(self.store.values["ready-1"], "1")
        self.assertEqual(self.store.values["ready-2"], "1")
        with patch.object(torch, "npu", types.SimpleNamespace(synchronize=lambda: None), create=True):
            first.close()
            second.close()
            first.close()
        self.assertEqual(self.store.values["done-1"], "1")
        self.assertEqual(self.store.values["done-2"], "1")
        self.assertTrue(first.closed and second.closed)

    def test_duplicate_slot_and_out_of_range_rejected_before_group_creation(self):
        self.reader(1)
        with self.assertRaisesRegex(ValueError, "already claimed"):
            self.reader(1)
        with self.assertRaisesRegex(ValueError, "do not fit"):
            self.reader(3)
        self.assertEqual(len(self.groups), 1)

    def test_owner_reader_count_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "membership differ"):
            self.reader(1, count=1)
        self.assertEqual(self.groups, [])

    def test_mismatched_owner_metadata_is_rejected_before_claim_or_group(self):
        remote = json.loads(json.dumps(META))
        remote["tables"]["1"]["rows"] = 9
        self.store.values["metadata"] = json.dumps(remote).encode()
        with self.assertRaisesRegex(ValueError, "checkpoint metadata differ"):
            self.reader(1)
        self.assertEqual(self.groups, [])
        self.assertEqual(self.store.claims, {})

    def test_matching_metadata_sets_same_logical_rows_for_readers(self):
        first = self.reader(1)
        second = self.reader(2)
        self.assertEqual(first.buffers["1"][0][2].rows, META["tables"]["1"]["rows"])
        self.assertEqual(second.buffers["1"][0][2].rows, META["tables"]["1"]["rows"])
        self.assertEqual(len(first.buffers["1"]), 1)
        self.assertEqual(len(second.buffers["1"]), 1)

    def test_reader_declares_device_only_lookup_so_hashes_stay_on_npu(self):
        self.assertTrue(self.module.CANNReader.device_only)

    def test_capacity_is_validated_before_remote_fetch(self):
        with self.assertRaisesRegex(ValueError, "invalid CANN Engram row capacity"):
            self.reader(1, capacity=0)
        reader = self.reader(1, capacity=2)
        triton = types.ModuleType("vllm_ascend.ops.triton")
        lookup = types.ModuleType("vllm_ascend.ops.triton.engram_lookup")
        lookup.dequantize_packed_engram = lambda *args: None
        table = types.SimpleNamespace(
            _checkpoint_key="layers.1.engram.embed.weight",
            num_embeddings=8,
            vocab_start_idx=0,
            vocab_end_idx=8,
            head_start=0,
            part_n_hash_cols=3,
        )
        with (
            patch.dict(
                sys.modules,
                {"vllm_ascend.ops.triton": triton, "vllm_ascend.ops.triton.engram_lookup": lookup},
            ),
            self.assertRaisesRegex(ValueError, "exceeds configured row capacity"),
        ):
            reader.lookup(table, torch.zeros((1, 3), dtype=torch.int64), torch.empty((1, 3, 256)))

    def test_wait_failure_retains_buffers_and_never_signals_done(self):
        reader = self.reader(1, capacity=2)
        triton = types.ModuleType("vllm_ascend.ops.triton")
        lookup = types.ModuleType("vllm_ascend.ops.triton.engram_lookup")
        lookup.dequantize_packed_engram = lambda *args: None
        fetched = torch.zeros((1, 256), dtype=torch.bfloat16)

        def fail_wait(*args):
            raise RuntimeError("injected wait failure")

        fake_ops = types.SimpleNamespace(engram_fetch=lambda *args: fetched, engram_fetch_wait=fail_wait)
        table = types.SimpleNamespace(
            _checkpoint_key="layers.1.engram.embed.weight",
            num_embeddings=8,
            vocab_start_idx=0,
            vocab_end_idx=8,
            head_start=0,
            part_n_hash_cols=1,
        )
        with (
            patch.dict(
                sys.modules,
                {"vllm_ascend.ops.triton": triton, "vllm_ascend.ops.triton.engram_lookup": lookup},
            ),
            patch.object(torch.ops, "cann_ops_transformer", fake_ops),
            self.assertRaisesRegex(RuntimeError, "injected wait failure"),
        ):
            reader.lookup(table, torch.zeros((1, 1), dtype=torch.int64), torch.empty((1, 1, 256)))
        self.assertTrue(reader.failed)
        self.assertTrue(
            any(
                callable(value) and any(cell.cell_contents is fetched for cell in (value.__closure__ or ()))
                for value in reader._uncertain
            )
        )
        self.assertIn(reader, self.module._HELD_READERS)
