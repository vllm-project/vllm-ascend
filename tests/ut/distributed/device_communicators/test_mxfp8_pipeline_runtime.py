# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered multi-rank simulation of the production MXFP8 pipeline."""

import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import torch

from tests.ut.distributed.device_communicators.test_mxfp8_all_reduce import _decode, _make
from vllm_ascend.distributed.device_communicators import mxfp8_all_reduce as mx


class _Work:
    def __init__(self, engine, index, rank):
        self.engine = engine
        self.index = index
        self.rank = rank
        self.waited = False

    def wait(self):
        with self.engine.condition:
            self.engine.events[self.rank].append(("wait", self.index))
            ready = self.engine.condition.wait_for(
                lambda: self.engine.completed > self.index or self.engine.error is not None, timeout=20
            )
            if self.engine.error is not None:
                raise self.engine.error
            if not ready:
                raise AssertionError(f"Collective {self.index}, rank {self.rank} did not complete")
            self.waited = True


class _OrderedCollectives:
    """One FIFO HCCL stream. Weak references expose prematurely freed buffers."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.condition = threading.Condition()
        self.local = threading.local()
        self.calls = {}
        self.sequences = [0] * world_size
        self.events: list[list[tuple[str, int]]] = [[] for _ in range(world_size)]
        self.completed = 0
        self.error = None

    def abort(self, error):
        with self.condition:
            self.error = error
            self.condition.notify_all()

    def _collective(self, kind, output, source, group=None, async_op=False):
        rank = self.local.rank
        with self.condition:
            index = self.sequences[rank]
            self.sequences[rank] += 1
            self.events[rank].append((kind, index))
            work = _Work(self, index, rank)
            calls = self.calls.setdefault(index, {})
            calls[rank] = (kind, weakref.ref(output), weakref.ref(source), work)
            while len(self.calls.get(self.completed, {})) == self.world_size:
                batch = self.calls[self.completed]
                ordered = [batch[owner] for owner in range(self.world_size)]
                if len({entry[0] for entry in ordered}) != 1:
                    raise AssertionError("Collective ordering differs across ranks")
                outputs = [entry[1]() for entry in ordered]
                sources = [entry[2]() for entry in ordered]
                if any(tensor is None for tensor in outputs + sources):
                    raise AssertionError("Async collective buffer was released before completion")
                for owner, target in enumerate(outputs):
                    if ordered[0][0] == "a2a":
                        target.copy_(torch.cat([tensor.chunk(self.world_size)[owner] for tensor in sources]))
                    else:
                        target.copy_(torch.cat(sources))
                self.completed += 1
                self.condition.notify_all()
        if not async_op:
            work.wait()
            return None
        return work

    def all_to_all(self, output, source, **kwargs):
        return self._collective("a2a", output, source, **kwargs)

    def all_gather(self, output, source, **kwargs):
        return self._collective("gather", output, source, **kwargs)


class TestMxfp8PipelineRuntime(unittest.TestCase):
    def test_chunk_ranges_preserve_rank_splits_and_cover_uneven_tail(self):
        for rows, size, expected in (
            (72, 8, [(0, 40), (40, 72)]),
            (64, 8, [(0, 32), (32, 64)]),
            (16, 8, [(0, 8), (8, 16)]),
        ):
            with self.subTest(rows=rows, size=size):
                self.assertEqual(mx._chunk_ranges(rows, size), expected)
        for rows, size in ((71, 8), (8, 8), (0, 2)):
            with self.subTest(rows=rows, size=size), self.assertRaises(ValueError):
                mx._chunk_ranges(rows, size)

    def test_multirank_uneven_chunks_repeated_calls_and_noncontiguous_inputs(self):
        for world_size in (2, 4, 8):
            for noncontiguous in (False, True):
                with self.subTest(tp=world_size, noncontiguous=noncontiguous):
                    self._run_ranks(world_size, noncontiguous)

    def _run_ranks(self, world_size, noncontiguous):
        rows, hidden = world_size * 9, 128
        engine = _OrderedCollectives(world_size)
        fixtures = []
        for rank in range(world_size):
            indices = torch.arange(rows * hidden).reshape(rows, hidden)
            token_indices = torch.arange(rows).unsqueeze(1)
            raw_values = ((indices + token_indices + rank) % 8 + 0x30 + (0x80 if rank % 2 else 0)).to(torch.uint8)
            scales = ((torch.arange(rows * hidden // 32).reshape(rows, -1) + rank) % 7 + 124).to(torch.uint8)
            values = raw_values.view(torch.float8_e4m3fn)
            fixtures.append((values, scales, _decode(values, scales).to(torch.bfloat16)))
        expected = torch.stack([fixture[2].float() for fixture in fixtures]).sum(0).to(torch.bfloat16)

        def quantize(source, **kwargs):
            rank = engine.local.rank
            start = source.storage_offset() // hidden
            end = start + source.shape[0]
            values, scales, original = fixtures[rank]
            torch.testing.assert_close(source, original[start:end], rtol=0, atol=0)
            self.assertEqual(kwargs["block_size"], 32)
            self.assertEqual(kwargs["scale_alg"], 1)
            return values[start:end], scales[start:end]

        def dequantize(values, scales, **kwargs):
            self.assertEqual(kwargs["dst_type"], 15)
            self.assertEqual(kwargs["src_type"], 292)
            return _decode(values, scales).to(torch.bfloat16)

        wrappers = [
            _make(world_size=world_size, enable_pipeline=True, quantize=quantize, dequantize=dequantize)
            for _ in range(world_size)
        ]

        def collective(kind, output, source, **kwargs):
            self.assertIs(kwargs["group"], wrappers[engine.local.rank].group)
            self.assertTrue(kwargs["async_op"])
            return engine._collective(kind, output, source, **kwargs)

        def worker(rank):
            engine.local.rank = rank
            original = fixtures[rank][2]
            try:
                for _ in range(3):
                    # Shape [3, 3*TP, H] exercises flattening and shape restoration.
                    source = original.clone().reshape(3, 3 * world_size, hidden)
                    if noncontiguous:
                        source = source.transpose(0, 1).contiguous().transpose(0, 1)
                        self.assertFalse(source.is_contiguous())
                    output = wrappers[rank].all_reduce(source)
                    self.assertIsNot(output, source)
                    self.assertNotEqual(output.data_ptr(), source.data_ptr())
                    self.assertEqual(output.shape, source.shape)
                    self.assertEqual(output.dtype, torch.bfloat16)
                    self.assertTrue(output.is_contiguous())
                    torch.testing.assert_close(source.reshape(rows, hidden), original, rtol=0, atol=0)
                    torch.testing.assert_close(output.reshape(rows, hidden), expected, rtol=0, atol=0)
                    with engine.condition:
                        for batch in engine.calls.values():
                            if rank in batch:
                                self.assertTrue(batch[rank][3].waited)
            except BaseException as error:
                engine.abort(error)
                raise

        with (
            patch.object(torch.npu, "is_current_stream_capturing", return_value=False),
            patch.object(torch.npu, "synchronize", side_effect=AssertionError("No device synchronization in pipeline")),
            patch.object(mx.dist, "all_reduce", side_effect=AssertionError("Unexpected BF16 fallback")),
            patch.object(mx.dist, "new_group", side_effect=AssertionError("Must reuse the existing group")),
            patch.object(mx.dist, "all_to_all_single", side_effect=lambda *a, **kw: collective("a2a", *a, **kw)),
            patch.object(
                mx.dist, "all_gather_into_tensor", side_effect=lambda *a, **kw: collective("gather", *a, **kw)
            ),
            ThreadPoolExecutor(max_workers=world_size) as pool,
        ):
            futures = [pool.submit(worker, rank) for rank in range(world_size)]
            for future in futures:
                future.result(timeout=30)
        self.assertEqual(engine.completed, 12)
        for events in engine.events:
            self.assertEqual(events[:3], [("a2a", 0), ("a2a", 1), ("wait", 0)])
        expected_order = ["a2a", "a2a", "gather", "gather"]
        self.assertEqual([engine.calls[index][0][0] for index in range(4)], expected_order)


if __name__ == "__main__":
    unittest.main()
