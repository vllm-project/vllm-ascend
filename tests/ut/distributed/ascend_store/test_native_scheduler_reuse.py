# SPDX-License-Identifier: Apache-2.0
"""CPU-only native lifecycle regressions; stub imports, execute native source.

Lookahead identity and finish-time save semantics belong to upstream #64096
suites. These checks retain only job pin lifetime across request cleanup.

Run directly with Python with the companion vLLM source on PYTHONPATH.
Neither an NPU runtime nor a vLLM install is needed.
"""

import ast
import importlib.util
import logging
import sys
import types
import unittest
from importlib.machinery import PathFinder
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

_vllm_spec = PathFinder.find_spec("vllm")
assert _vllm_spec is not None and _vllm_spec.submodule_search_locations
VLLM = Path(_vllm_spec.submodule_search_locations[0])
STORE = VLLM / "distributed/kv_transfer/kv_connector/v1/mooncake/store"
NATIVE = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store"


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stub_imports(path: Path) -> None:
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ImportFrom) or node.module is None or not node.module.startswith("vllm."):
            continue
        module = sys.modules.get(node.module) or types.ModuleType(node.module)
        for alias in node.names:
            if not hasattr(module, alias.name):
                setattr(module, alias.name, type(alias.name, (), {}))
        sys.modules[node.module] = module


@unittest.skipUnless((STORE / "layout.py").is_file(), "requires vLLM split Mooncake Store implementation")
class NativeSchedulerReuseTest(unittest.TestCase):
    modules = patch.dict(sys.modules)
    data: ClassVar[Any]
    native: ClassVar[Any]

    @classmethod
    def setUpClass(cls):
        cls.modules.start()
        cls.addClassCleanup(cls.modules.stop)
        stub_imports(STORE / "data.py")
        torch_stub = types.ModuleType("torch")
        torch_stub.__dict__.update(cuda=types.SimpleNamespace(Event=object))
        sys.modules["torch"] = torch_stub
        sys.modules["vllm.logger"].__dict__.update(init_logger=logging.getLogger)
        sys.modules["vllm.utils.math_utils"].__dict__.update(cdiv=lambda a, b: (a + b - 1) // b)
        cls.data = load(NATIVE + ".data", STORE / "data.py")
        stub_imports(STORE / "scheduler.py")
        sys.modules["vllm.v1.attention.backends.utils"].__dict__.update(NULL_BLOCK_ID=0)
        cls.native = load(NATIVE + ".scheduler", STORE / "scheduler.py")

    def setUp(self):
        self.scheduler = self.native.MooncakeStoreScheduler.__new__(self.native.MooncakeStoreScheduler)
        s = self.scheduler
        s._hash_block_size = s._block_size = 4
        s._num_workers = 2
        s._next_store_job_id = 0
        s._pinned_saves = {}
        s._finished_partial_tail_metas = {}
        s._finished_lookahead_metas = {}
        s._store_group_ids = [0, 1]
        s._boundary_state_group_ids = frozenset({1})
        s._store_group_id_by_kv_cache_group_id = {0: 0, 1: 1}
        s._store_coord = types.SimpleNamespace(
            kv_cache_groups=[types.SimpleNamespace(kv_cache_spec=types.SimpleNamespace(block_size=4))] * 2,
            mamba_group_ids={1},
            num_hash_lookahead_tokens=0,
        )
        s._gpu_block_pool = MagicMock()
        s._gpu_block_pool.blocks = list(range(20))
        s.kv_role = "kv_both"
        s.save_decode_cache = True
        s._capacity_only = s.enable_partial_hash_hits = s.enable_kv_events = False
        s.lookup_async = False
        s.client = MagicMock()
        s.load_specs = {}
        self.request = types.SimpleNamespace(
            request_id="r",
            block_hashes=[b"a" * 32],
            all_token_ids=[1, 2, 3, 4],
            num_tokens=4,
            num_prompt_tokens=4,
            num_computed_tokens=4,
        )
        s._unfinished_requests = {"r": (self.request, ())}
        s._unfinished_request_ids = {"r"}
        s._request_trackers = {"r": self.data.RequestTracker("r", 4, ([1], [2]), prefill_end_tokens=4)}

    def output(self, *, offload=False, finished=False, preempted=False):
        return types.SimpleNamespace(
            finished_req_ids={"r"} if finished else set(),
            preempted_req_ids={"r"} if preempted else set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=types.SimpleNamespace(req_ids=[]),
            kv_connector_block_state=types.SimpleNamespace(
                boundary_state_offloads={"r": [(1, 2, 4)]} if offload else {},
                get_block_ids=lambda req_id: ([1], [2]),
            ),
        )

    def complete_one_rank(self, job_id):
        self.scheduler.update_connector_output(
            types.SimpleNamespace(kv_connector_worker_meta=self.data.MooncakeStoreWorkerMetadata({job_id: 1}))
        )

    def test_finish_or_preempt_keeps_dispatched_pin_until_all_ranks_complete(self):
        for reason in ("finished", "preempted"):
            with self.subTest(reason=reason):
                self.setUp()
                s = self.scheduler
                (job,) = s.build_connector_meta(self.output(offload=True)).requests
                s._gpu_block_pool.touch.assert_called_once_with([2])
                self.assertTrue(s.has_pending_push_work())
                self.assertEqual(s.build_connector_meta(self.output(**{reason: True})).requests, [])
                s._gpu_block_pool.free_blocks.assert_not_called()
                self.complete_one_rank(job.store_job_id)
                s._gpu_block_pool.free_blocks.assert_not_called()
                self.assertTrue(s.has_pending_push_work())
                self.complete_one_rank(job.store_job_id)
                self.assertEqual(list(s._gpu_block_pool.free_blocks.call_args.args[0]), [2])
                self.assertFalse(s.has_pending_push_work())
                # A replayed completion must not release the job twice.
                self.complete_one_rank(job.store_job_id)
                s._gpu_block_pool.free_blocks.assert_called_once()

    def test_finish_time_handoff_keeps_pin_across_request_cleanup(self):
        s = self.scheduler
        delay_free = s.register_finished_partial_tail(self.request, ([1], [2]), [(1, 2, 4)])
        self.assertFalse(delay_free)
        s._gpu_block_pool.touch.assert_called_once_with([2])
        (job,) = s.build_connector_meta(self.output(finished=True)).requests
        self.assertEqual(job.boundary_puts, [self.data.BoundaryPut(1, 2, 4)])
        self.assertEqual(s.build_connector_meta(self.output()).requests, [])
        s._gpu_block_pool.free_blocks.assert_not_called()
        self.complete_one_rank(job.store_job_id)
        s._gpu_block_pool.free_blocks.assert_not_called()
        self.assertTrue(s.has_pending_push_work())
        self.complete_one_rank(job.store_job_id)
        self.assertEqual(list(s._gpu_block_pool.free_blocks.call_args.args[0]), [2])
        self.assertFalse(s.has_pending_push_work())


if __name__ == "__main__":
    unittest.main()
