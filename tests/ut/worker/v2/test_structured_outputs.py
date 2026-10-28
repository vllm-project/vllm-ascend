# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch

from vllm_ascend.worker.v2 import structured_outputs as worker_module
from vllm_ascend.worker.v2.structured_outputs import AscendStructuredOutputsWorker


class FakeStream:
    def wait_stream(self, stream):
        pass


class FakeEvent:
    def __init__(self):
        self.synchronize_count = 0

    def record(self, stream):
        pass

    def synchronize(self):
        self.synchronize_count += 1


class FakeKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(logits, logits_stride, indices, offsets, bitmask, bitmask_stride, vocab_size, **kwargs):
            self.calls.append((indices.clone(), bitmask.clone(), kwargs))

        return launch


@contextmanager
def fake_stream_context(stream):
    yield


def make_worker(monkeypatch, mask_stride=1):
    stream = FakeStream()
    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", fake_stream_context)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: FakeStream())
    kernel = FakeKernel()
    monkeypatch.setattr(worker_module.structured_outputs, "_apply_grammar_bitmask_kernel", kernel)
    worker = AscendStructuredOutputsWorker(8, 64, torch.device("cpu"), mask_stride)
    return worker, kernel


def count_mapping_copies(monkeypatch, worker):
    original_copy = torch.Tensor.copy_
    count = 0

    def counted_copy(self, source, non_blocking=False):
        nonlocal count
        if self.ndim == 1:
            count += 1
        return original_copy(self, source, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "copy_", counted_copy)
    return lambda: count


def test_decode_reuses_mapping_but_refreshes_bitmask(monkeypatch):
    worker, kernel = make_worker(monkeypatch)
    mapping_copies = count_mapping_copies(monkeypatch, worker)
    input_batch = SimpleNamespace(
        req_ids=["a", "b", "c"],
        cu_num_logits=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    logits = torch.zeros(3, 64)
    first_mask = np.full((2, 2), -1, dtype=np.int32)
    second_mask = np.zeros((2, 2), dtype=np.int32)

    worker.apply_grammar_bitmask(logits, input_batch, ["c", "a"], first_mask)
    worker.apply_grammar_bitmask(logits, input_batch, ["c", "a"], second_mask)

    assert mapping_copies() == 1
    assert kernel.calls[0][0].tolist() == [2, 0]
    assert kernel.calls[1][0].tolist() == [2, 0]
    assert kernel.calls[0][1].tolist() == first_mask.tolist()
    assert kernel.calls[1][1].tolist() == second_mask.tolist()
    assert worker._copy_event.synchronize_count == 1

    worker.apply_grammar_bitmask(logits, input_batch, ["b", "c"], first_mask)
    assert mapping_copies() == 2
    assert kernel.calls[-1][0].tolist() == [1, 2]


def test_speculative_mapping_is_rebuilt_after_request_changes(monkeypatch):
    worker, kernel = make_worker(monkeypatch, mask_stride=2)
    mapping_copies = count_mapping_copies(monkeypatch, worker)
    input_batch = SimpleNamespace(
        req_ids=["a", "b"],
        cu_num_logits=torch.tensor([0, 2, 4], dtype=torch.int32),
        cu_num_logits_np=np.array([0, 2, 4], dtype=np.int32),
        num_draft_tokens_per_req=None,
    )
    bitmask = np.full((2, 2), -1, dtype=np.int32)

    worker.apply_grammar_bitmask(torch.zeros(4, 64), input_batch, ["a"], bitmask)
    worker.apply_grammar_bitmask(torch.zeros(4, 64), input_batch, ["b"], bitmask)

    assert mapping_copies() == 2
    assert kernel.calls[0][0].tolist() == [0, 1]
    assert kernel.calls[1][0].tolist() == [2, 3]


def test_no_grammar_requests_skip_host_and_device_work(monkeypatch):
    worker, kernel = make_worker(monkeypatch)
    input_batch = SimpleNamespace(req_ids=[], cu_num_logits=torch.tensor([0]))

    worker.apply_grammar_bitmask(torch.zeros(0, 64), input_batch, [], np.empty((0, 2), dtype=np.int32))

    assert not kernel.calls
    assert not worker._copies_pending
