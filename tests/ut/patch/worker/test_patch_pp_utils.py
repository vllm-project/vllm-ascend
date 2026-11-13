# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from vllm.v1.worker.gpu import pp_utils

from vllm_ascend.patch.worker.patch_v2.patch_pp_utils import compute_need_sampled_mask


class _Tensor:
    """CPU transport boundary, preserving row order and tensor shapes."""

    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    def __getitem__(self, index):
        if isinstance(index, _Tensor):
            index = index.values
        return _Tensor(self.values[index])

    def contiguous(self):
        return self

    def unbind(self, dim=0):
        assert dim == 0
        return tuple(_Tensor(row) for row in self.values)

    def record_stream(self, stream):
        pass


@pytest.fixture
def transport(monkeypatch):
    sent = []
    empty = Mock(side_effect=lambda *shape, dtype, device: _Tensor(np.zeros(shape, dtype=dtype)))

    def pad(tensor, padding):
        before, after = padding
        return _Tensor(np.pad(tensor.values, ((0, 0), (before, after))))

    fake_torch = SimpleNamespace(
        empty=empty,
        int64=np.int64,
        int32=np.int32,
        cuda=SimpleNamespace(stream=lambda _stream: nullcontext()),
        distributed=SimpleNamespace(broadcast=lambda tensor, *, src, group: sent.append(tensor.values.copy())),
        nn=SimpleNamespace(functional=SimpleNamespace(pad=pad)),
        stack=lambda tensors, dim: _Tensor(np.stack([tensor.values for tensor in tensors], axis=dim)),
    )
    monkeypatch.setattr(pp_utils, "torch", fake_torch)
    monkeypatch.setattr(pp_utils, "current_platform", SimpleNamespace(is_xpu=lambda: False))
    monkeypatch.setattr(pp_utils, "async_tensor_h2d", lambda values, *, device: _Tensor(values))
    return SimpleNamespace(sent=sent, empty=empty)


def _batch(computed, scheduled, prefill, *, is_prefilling=None, caps=None, indices=None):
    num_reqs = len(computed)
    indices = np.arange(num_reqs, dtype=np.int32) if indices is None else np.asarray(indices, dtype=np.int32)
    batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_computed_tokens_np=np.asarray(computed, dtype=np.int32),
        num_scheduled_tokens=np.asarray(scheduled, dtype=np.int32),
        prefill_len_np=np.asarray(prefill, dtype=np.int32),
        is_prefilling_np=np.asarray(
            np.asarray(computed) < np.asarray(prefill) if is_prefilling is None else is_prefilling,
            dtype=np.bool_,
        ),
        idx_mapping_np=indices,
        idx_mapping=_Tensor(indices),
    )
    if caps is not None:
        batch.max_seq_len_np = np.asarray(caps, dtype=np.int32)
    return batch


def _handler(*, last_rank, num_speculative_steps=1):
    handler = pp_utils.PPHandler.__new__(pp_utils.PPHandler)
    handler.is_last_rank = last_rank
    handler.last_rank = 1
    handler.device = "cpu"
    handler.max_sample_len = num_speculative_steps + 1
    handler.num_speculative_steps = num_speculative_steps
    handler.main_stream = Mock()
    handler.broadcast_stream = Mock()
    handler.broadcast_group = object()
    handler.req_idx_gen_np = np.zeros(8, dtype=np.int32)
    handler.queue = deque([None])
    return handler


def _invoke(method, handler, batch):
    if method == "receive":
        return handler.receive(batch)
    if method == "broadcast":
        return handler.broadcast(
            _Tensor(np.arange(batch.num_reqs, dtype=np.int64).reshape(-1, 1)),
            _Tensor(np.ones(batch.num_reqs, dtype=np.int32)),
            _Tensor(np.zeros(batch.num_reqs, dtype=np.int32)),
            batch,
        )
    return handler.broadcast_drafts(
        _Tensor(np.arange(8 * handler.num_speculative_steps, dtype=np.int64).reshape(8, -1)), batch
    )


def test_replaces_native_module_global_gate():
    assert pp_utils.compute_need_sampled_mask is compute_need_sampled_mask


@pytest.mark.parametrize("method", ["receive", "broadcast", "broadcast_drafts"])
@pytest.mark.parametrize("num_speculative_steps", [1, 5])
def test_all_finishing_prefills_skip_native_transport(transport, method, num_speculative_steps):
    # APC can leave just one prompt token; both rows finish after one output.
    batch = _batch([131071, 1020], [1, 4], [131072, 1024], caps=[131073, 1025])
    handler = _handler(last_rank=method != "receive", num_speculative_steps=num_speculative_steps)

    assert compute_need_sampled_mask(batch) is None
    result = _invoke(method, handler, batch)

    assert result is False if method == "receive" else result is None
    assert transport.sent == []
    transport.empty.assert_not_called()
    handler.broadcast_stream.wait_stream.assert_not_called()
    assert handler.queue[-1] is None


@pytest.mark.parametrize("method", ["receive", "broadcast", "broadcast_drafts"])
@pytest.mark.parametrize("num_speculative_steps", [1, 5])
def test_native_transport_keeps_decode_past_request_cap(transport, method, num_speculative_steps):
    batch = _batch([14176], [1 + num_speculative_steps], [12175], caps=[12176], is_prefilling=[False])
    handler = _handler(last_rank=method != "receive", num_speculative_steps=num_speculative_steps)

    np.testing.assert_array_equal(compute_need_sampled_mask(batch), [True])
    result = _invoke(method, handler, batch)

    assert result is True if method == "receive" else result is None
    expected_transfers = {"receive": 3, "broadcast": 2, "broadcast_drafts": 1}
    assert len(transport.sent) == expected_transfers[method]


@pytest.mark.parametrize("method", ["receive", "broadcast", "broadcast_drafts"])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_mixed_batch_preserves_siblings_and_original_row_alignment(transport, method, reverse_order):
    # Finishing APC tail, decode past cap, middle chunk, final prefill that
    # still has output budget. Keep decode and the non-finishing sibling.
    order = np.arange(4)[::-1] if reverse_order else np.arange(4)
    batch = _batch(
        np.array([131071, 64, 16, 30])[order],
        np.array([1, 2, 8, 2])[order],
        np.array([131072, 32, 128, 32])[order],
        caps=np.array([131073, 33, 129, 40])[order],
        is_prefilling=np.array([True, False, True, True])[order],
        indices=np.array([3, 0, 2, 1])[order],
    )
    expected = np.array([False, True, False, True])[order]
    handler = _handler(last_rank=method != "receive")

    np.testing.assert_array_equal(compute_need_sampled_mask(batch), expected)
    result = _invoke(method, handler, batch)

    assert transport.sent[0].shape[0] == batch.num_reqs
    if method == "receive":
        assert result is False  # Some rows need the ordinary prefill path.
        np.testing.assert_array_equal(handler.queue[-1].need_sampled_mask, expected)
        previous = handler.get_prev_sampled_outputs()
        np.testing.assert_array_equal(previous["idx_mapping"].values, np.where(expected, batch.idx_mapping_np, -1))
        handler.main_stream.wait_event.assert_called_once()
    elif method == "broadcast_drafts":
        np.testing.assert_array_equal(transport.sent[0][:, 0], batch.idx_mapping_np)


@pytest.mark.parametrize("metadata", ["missing", "none"])
def test_missing_length_caps_conservatively_keep_final_prefill(transport, metadata):
    batch = _batch([131071, 16], [1, 8], [131072, 128])
    if metadata == "none":
        batch.max_seq_len_np = None
    handler = _handler(last_rank=False)

    np.testing.assert_array_equal(compute_need_sampled_mask(batch), [True, False])
    assert handler.receive(batch) is False
    assert len(transport.sent) == 3
    np.testing.assert_array_equal(handler.queue[-1].need_sampled_mask, [True, False])


def test_decode_without_speculation_is_not_finished_prefill():
    batch = _batch([64], [1], [32], caps=[33], is_prefilling=[False])
    np.testing.assert_array_equal(compute_need_sampled_mask(batch), [True])


def test_all_nonfinal_prefill_chunks_still_skip():
    batch = _batch([0, 16], [8, 8], [128, 128], caps=[129, 140])
    assert compute_need_sampled_mask(batch) is None
