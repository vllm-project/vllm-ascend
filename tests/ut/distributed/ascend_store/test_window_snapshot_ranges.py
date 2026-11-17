# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU retention tests, independent of NPU imports and SDK initialization.

Execute the production methods, with only their external dependencies doubled.
The standalone entry point also runs on a host without torch or vLLM installed.
"""

import ast
import logging
import unittest
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


def load_methods(path, class_name, method_names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in method_names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[class_name]


class TestWindowSnapshotRanges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[4]
        base = root / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store"
        cls.window_manager = type("WindowManagerDouble", (), {})
        namespace = {
            "np": np,
            "Sequence": Sequence,
            "cdiv": lambda x, y: (x + y - 1) // y,
            "SlidingWindowManager": cls.window_manager,
            "_get_manager_class": lambda spec: spec.manager,
            "logger": logging.getLogger(__name__),
            "get_block_hashes": lambda hashes, size, hash_size: hashes,
            "block_hash_to_str": str,
            "get_partial_block_index": lambda end, size, hashes, offload: end // size if end % size else None,
        }
        # Preserve the actual plan definition as well as its methods.
        tree = ast.parse((base / "layerwise/pool_worker.py").read_text(encoding="utf-8"))
        plan = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_SaveGroupPlan")
        namespace["dataclass"] = dataclass
        exec(compile(ast.fix_missing_locations(ast.Module(body=[plan], type_ignores=[])), "plan", "exec"), namespace)
        cls.coordinator_type = load_methods(
            base / "coordinator.py", "AscendStoreCoordinator", {"continuation_block_range"}, namespace
        )
        cls.worker_type = load_methods(
            base / "layerwise/pool_worker.py",
            "LayerwiseKVPoolWorker",
            {
                "_mask_allocated_blocks",
                "_continuation_ranges",
                "_window_publication_masks",
                "_compute_reachable_store_masks",
                "_compute_reachable_load_masks",
                "_plan_save_group",
                "_layerwise_save_full_end",
                "_uses_execution_full_key",
                "_layerwise_execution_full_key",
            },
            namespace,
        )

    def make_case(self, end=64, public_end=64, window=8, size=2, extra=0):
        coordinator = self.coordinator_type()
        coordinator.group_effective_specs = [
            SimpleNamespace(
                manager=self.window_manager, sliding_window=window, block_size=size, extra_retained_tokens=extra
            )
        ]
        worker = self.worker_type()
        worker.layerwise_offload = worker.use_layerwise_transfer = True
        worker.block_key_hybrid = False
        worker.cache_coordinator = coordinator
        worker.num_kv_cache_groups = 1
        worker.vllm_config = SimpleNamespace(speculative_config=None)
        worker.dcp_size = worker.pcp_size = 1
        worker.cache_transfer_granularity = 16
        worker.group_uses_align_state = [False]
        worker.grouped_block_size = [size]
        worker.hash_block_size = size
        worker.cacheable_group_ids = {0}
        worker._reused_layers_by_group = {0: {0, 1}}
        worker._snapshot_keys = {}
        worker._is_layerwise_save_owner = lambda group: True
        worker._make_layerwise_full_key = lambda group, key: f"public:{key}"
        worker._make_layerwise_partial_key = lambda req, group, block, stop: f"private:{req.req_id}:{block}:{stop}"
        blocks = list(range(1, (end + size - 1) // size + 1))
        request = SimpleNamespace(
            req_id="a",
            target_token_len=end,
            num_prompt_tokens=128,
            save_start_token=0,
            save_end_token=public_end,
            can_save=True,
            needs_private_snapshot=True,
            block_ids_by_group=[blocks],
            block_ids_by_group_np=[np.asarray(blocks)],
            block_ids_np=np.asarray(blocks),
            block_hashes=list(range(end // size)),
            layerwise_recompute_start_token=0,
            store_masks=None,
            load_spec=SimpleNamespace(layerwise_previous_step=True, can_load=True),
        )
        public = [False] * (public_end // size)
        # One public checkpoint deliberately far behind the private tail.
        if len(public) > 12:
            public[12] = True
        worker.token_database = SimpleNamespace(
            store_mask=Mock(return_value=(public,)),
            load_mask=Mock(return_value=(public,)),
        )
        return worker, request

    @staticmethod
    def selected(mask):
        return [index for index, keep in enumerate(mask) if keep]

    def test_c4_tail_and_public_checkpoint_are_both_saved(self):
        worker, request = self.make_case()
        self.assertEqual(self.selected(worker._compute_reachable_store_masks(request)[0]), [12, 28, 29, 30, 31])
        self.assertEqual(self.selected(worker._compute_reachable_load_masks(request, 64)[0]), [28, 29, 30, 31])
        worker.token_database.load_mask.assert_not_called()

    def test_unaligned_private_endpoint_does_not_round_down_to_prefix_boundary(self):
        worker, request = self.make_case(end=65)
        self.assertEqual(self.selected(worker._compute_reachable_load_masks(request, 65)[0]), [29, 30, 31, 32])

    def test_c128_and_swa_windows_use_their_own_page_size(self):
        for window, size, expected in [(128, 4, list(range(480, 512))), (128, 32, [60, 61, 62, 63])]:
            with self.subTest(size=size):
                worker, request = self.make_case(end=2048, public_end=0, window=window, size=size)
                self.assertEqual(self.selected(worker._compute_reachable_store_masks(request)[0]), expected)
                self.assertEqual(worker._window_publication_masks(request), {0: ()})

    def test_full_history_group_is_not_trimmed(self):
        worker, request = self.make_case()
        worker.cache_coordinator.group_effective_specs[0].manager = object
        self.assertEqual(worker._compute_reachable_load_masks(request, 64), ([True] * 32,))
        self.assertEqual(worker._compute_reachable_store_masks(request), ([True] * 32,))

    def test_reserved_null_pages_remain_unreadable(self):
        worker, request = self.make_case()
        request.block_ids_by_group[0][30] = 0
        self.assertEqual(self.selected(worker._compute_reachable_load_masks(request, 64)[0]), [28, 29, 31])

    def test_external_prefix_uses_coordinator_mask(self):
        worker, request = self.make_case()
        request.load_spec.layerwise_previous_step = False
        self.assertEqual(self.selected(worker._compute_reachable_load_masks(request, 64)[0]), [12])
        worker.token_database.load_mask.assert_called_once_with(request.block_hashes, 64)

    def test_unsupported_replay_and_parallelism_keep_conservative_policy(self):
        for field in ("speculative", "dcp_size", "pcp_size"):
            with self.subTest(field=field):
                worker, request = self.make_case()
                if field == "speculative":
                    worker.vllm_config.speculative_config = object()
                else:
                    setattr(worker, field, 2)
                self.assertEqual(worker._compute_reachable_load_masks(request, 64), ([True] * 32,))
                self.assertEqual(worker._window_publication_masks(request), {})

    def test_explicit_extra_retention_is_preserved(self):
        worker, request = self.make_case(extra=8)
        self.assertEqual(self.selected(worker._compute_reachable_load_masks(request, 64)[0]), list(range(24, 32)))

    def test_window_mask_unknown_does_not_discard_snapshot_pages(self):
        worker, request = self.make_case()
        worker.token_database.store_mask.side_effect = AssertionError("unknown boundary")
        self.assertEqual(worker._compute_reachable_store_masks(request), ([True] * 32,))
        self.assertEqual(worker._window_publication_masks(request), {})

    def test_publication_does_not_duplicate_private_only_pages(self):
        worker, request = self.make_case()
        request.store_masks = worker._compute_reachable_store_masks(request)
        public_mask = worker._window_publication_masks(request)[0]
        plan = worker._plan_save_group(request, 0, public_mask)
        self.assertEqual(set(plan.keys), {12, 28, 29, 30, 31})
        self.assertEqual(plan.publication_keys, {12: "public:12"})
        self.assertTrue(all(key.startswith("private:a:") for key in plan.keys.values()))

    def test_partial_page_retains_private_identity(self):
        worker, request = self.make_case(end=65)
        request.store_masks = worker._compute_reachable_store_masks(request)
        plan = worker._plan_save_group(request, 0, worker._window_publication_masks(request)[0])
        self.assertEqual(plan.partial_key, "private:a:32:65")
        self.assertNotIn(32, plan.keys)
        self.assertEqual(worker._snapshot_keys["a"][0][32], (65, "private:a:32:65"))

    def test_private_keys_stay_request_local(self):
        worker, first = self.make_case()
        _, second = self.make_case()
        second.req_id = "b"
        for request in (first, second):
            request.store_masks = worker._compute_reachable_store_masks(request)
        a = worker._plan_save_group(first, 0, worker._window_publication_masks(first)[0])
        b = worker._plan_save_group(second, 0, worker._window_publication_masks(second)[0])
        self.assertFalse(set(a.keys.values()) & set(b.keys.values()))
        self.assertEqual(a.publication_keys, b.publication_keys)

    def test_byte_preserving_restore_covers_every_next_window_query(self):
        for size, window in [(2, 8), (4, 128), (32, 128)]:
            for end in (1, 7, 128, 2047, 2048, 2049):
                with self.subTest(size=size, window=window, end=end):
                    worker, request = self.make_case(end=end, public_end=0, window=window, size=size)
                    original = np.arange(len(request.block_ids_by_group[0]) * size * 3, dtype=np.int32).reshape(-1, 3)
                    restored = np.full_like(original, -1)
                    mask = worker._compute_reachable_load_masks(request, end)[0]
                    for page in self.selected(mask):
                        restored[page * size : (page + 1) * size] = original[page * size : (page + 1) * size]
                    for query in range(end, end + 256):
                        history_start = max(0, query - window + 1)
                        np.testing.assert_array_equal(restored[history_start:end], original[history_start:end])

    def test_allocated_intersection_matches_existing_contract(self):
        worker, request = self.make_case()
        rng = np.random.default_rng(7)
        for count in (0, 1, 37, 65536):
            ids = rng.integers(0, 100, size=count).tolist()
            request.block_ids_by_group = [ids]
            for mask in (None, [], [True], rng.integers(0, 2, size=count).astype(bool).tolist()):
                expected = [
                    value != 0 and (mask is None or mask[i])
                    for i, value in enumerate(ids)
                    if mask is None or i < len(mask)
                ]
                self.assertEqual(worker._mask_allocated_blocks(request, (mask,)), (expected,))


if __name__ == "__main__":
    unittest.main()
