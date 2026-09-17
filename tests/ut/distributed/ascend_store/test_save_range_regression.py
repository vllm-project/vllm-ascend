"""CPU regressions using production method bodies without importing NPU deps.

Run directly with Python, or collect with pytest. Only surrounding objects are
stubbed; range generation, allocation, filling and deduplication are real code.
"""

import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np


SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store"


def extract_methods(filename, class_name, names):
    tree = ast.parse((SOURCE / filename).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == names
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future] + methods, type_ignores=[])
    env = dict(np=np, logger=logging.getLogger(__name__), LayerBlockRange=NS,
               LayerTransferTask=NS, SharedBlockData=NS,
               get_block_hashes=lambda hashes, *args: hashes)
    exec(compile(ast.fix_missing_locations(module), str(SOURCE / filename), "exec"), env)
    return env


BUILDER = extract_methods("kv_transfer.py", "LayerBatchBuilder", {
    "_ensure_buf", "_require_request_arrays", "_dedupe_transfer_blocks", "build_shared",
})
WORKER = extract_methods("pool_worker.py", "KVPoolWorker", {
    "_process_save_for_layer_batch", "_get_partial_block_index",
})


def request(index, target, skip=0, hash_count=None):
    count = (target + 127) // 128
    return NS(req_id=f"req-{index}", can_save=True, save_start_token=0,
              save_end_token=target // 128 * 128, target_token_len=target,
              block_hashes=list(range(target // 128 if hash_count is None else hash_count)),
              load_spec=NS(can_load=True, kvpool_store_skip_tokens=skip, kvpool_cached_tokens=skip),
              partial_block_index=None, save_keys=[], load_keys=[], is_last_chunk=False,
              block_ids_by_group_np=[np.arange(count) + index * 1000],
              block_gvas_by_group_np=[np.arange(count) + index * 10000 + 100000],
              gva_block_offset=0, partial_save_gva_per_group=[900000 + index], last_block_gva=None)


def ranges_for(requests):
    worker = NS(tp_rank=0, put_step=1, hash_block_size=128, layerwise_offload=True,
                _get_effective_group_block_size=lambda group: 128, layer_save_tasks=[[]],
                _get_partial_block_index=WORKER["_get_partial_block_index"])
    WORKER["_process_save_for_layer_batch"](worker, requests, 0)
    return worker.layer_save_tasks[0]


def make_builder():
    builder = NS(group_id=0, _block_ids_buf=None, _block_gvas_buf=None)
    builder._ensure_buf = lambda count: BUILDER["_ensure_buf"](builder, count)
    builder._require_request_arrays = lambda block_range, save: BUILDER["_require_request_arrays"](
        builder, block_range, save)
    builder._dedupe_transfer_blocks = BUILDER["_dedupe_transfer_blocks"]
    return builder


def build(task, builder=None):
    return BUILDER["build_shared"](builder or make_builder(), task, True)


class TestSaveRangeRegression(unittest.TestCase):
    def test_incident_six_requests(self):
        # Target and skip lengths from 0917_fail.log, not synthetic ranges.
        specs = [(39475, 39424), (21322, 21248), (23515, 640),
                 (31290, 1024), (28947, 17408), (29659, 29696)]
        reqs = [request(i, target, skip) for i, (target, skip) in enumerate(specs)]
        task = ranges_for(reqs)[0]
        last = task.block_ranges[-1]
        self.assertEqual((last.start_block, last.end_block, last.partial_block_index), (231, 231, 231))
        result = build(task)
        self.assertEqual(len(result.block_ids_arr), 510)
        # Independently specified expected full-block indices + partial index.
        expected_ranges = [(308, 308), (166, 166), (5, 183), (8, 244), (136, 226), (231, 231)]
        ids, gvas = [], []
        for req, (start, end) in zip(reqs, expected_ranges):
            ids.extend(req.block_ids_by_group_np[0][start:end])
            gvas.extend(req.block_gvas_by_group_np[0][start:end])
            ids.append(req.block_ids_by_group_np[0][end])
            gvas.append(req.partial_save_gva_per_group[0])
        np.testing.assert_array_equal(result.block_ids_arr, ids)
        np.testing.assert_array_equal(result.block_gvas_arr, gvas)

    def test_worker_reversed_without_partial_is_skipped(self):
        self.assertEqual(ranges_for([request(0, 256, 640)]), [])

    def test_builder_reversed_range_with_fresh_buffer(self):
        req = request(0, 1025)
        for start, partial in [(3, 2), (5, 2), (5, None)]:
            with self.subTest(start=start, partial=partial):
                r = NS(request=req, start_block=start, end_block=2, partial_block_index=partial)
                result = build(NS(group_id=0, block_ranges=[r]))
                expected = [] if partial is None else [2]
                np.testing.assert_array_equal(result.block_ids_arr, expected)
                np.testing.assert_array_equal(result.block_gvas_arr, [] if partial is None else [900000])

    def test_worker_reversed_with_partial_has_only_partial(self):
        task = ranges_for([request(0, 257, 640)])[0]
        r = task.block_ranges[0]
        self.assertEqual((r.start_block, r.end_block, r.partial_block_index), (2, 2, 2))
        result = build(task)
        np.testing.assert_array_equal(result.block_ids_arr, [2])
        np.testing.assert_array_equal(result.block_gvas_arr, [900000])

    def test_builder_defends_reversed_ranges_and_reused_buffers(self):
        req = request(0, 1025)
        builder = make_builder()
        for start, end, partial, expected in [
            (0, 8, None, list(range(8))),  # grow reusable buffer
            (5, 2, 2, [2]),              # negative length > 1, partial only
            (5, 2, None, []),
            (2, 2, 2, [2]),
            (2, 2, None, []),
            (1, 3, 3, [1, 2, 3]),       # normal range after empty calls
        ]:
            with self.subTest(start=start, end=end, partial=partial):
                r = NS(request=req, start_block=start, end_block=end, partial_block_index=partial)
                result = build(NS(group_id=0, block_ranges=[r]), builder)
                np.testing.assert_array_equal(result.block_ids_arr, expected)
                expected_gvas = [100000 + i for i in expected]
                if partial is not None:
                    expected_gvas[-1] = 900000
                np.testing.assert_array_equal(result.block_gvas_arr, expected_gvas)

    def test_full_boundary_and_partial_boundary(self):
        for target, hash_count, expected in [(256, 2, None), (256, 1, 1), (257, 2, 2), (0, 0, None)]:
            with self.subTest(target=target, hash_count=hash_count):
                self.assertEqual(WORKER["_get_partial_block_index"](target, 128, hash_count, True), expected)
        self.assertIsNone(WORKER["_get_partial_block_index"](257, 128, 2, False))
        req = request(0, 256, hash_count=1)
        req.save_end_token = 128  # from_request_tracker's boundary_without_hash rule
        result = build(ranges_for([req])[0])
        np.testing.assert_array_equal(result.block_ids_arr, [0, 1])
        np.testing.assert_array_equal(result.block_gvas_arr, [100000, 900000])


if __name__ == "__main__":
    unittest.main()
