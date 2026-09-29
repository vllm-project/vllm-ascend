# SPDX-License-Identifier: Apache-2.0

from collections import deque
from dataclasses import dataclass
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu import pp_utils
from vllm.v1.worker.gpu.pp_utils import PendingRecv, PPHandler

from vllm_ascend.patch.worker.patch_v2 import patch_pp
from vllm_ascend.patch.worker.patch_v2.patch_pp import (
    compute_need_sampled_mask,
    install_pp_token_transport,
)


@dataclass
class _InputBatch:
    num_computed_tokens_np: np.ndarray
    num_scheduled_tokens: np.ndarray
    prefill_len_np: np.ndarray
    # The participation gate must ignore rank-local progress estimates.
    max_seq_len_np: np.ndarray | None = None
    is_prefilling_np: np.ndarray | None = None


class TestPureParticipationGate:
    def test_matches_upstream_gate_for_regular_decode(self):
        batch = _InputBatch(
            num_computed_tokens_np=np.array([56, 0], dtype=np.int32),
            num_scheduled_tokens=np.array([8, 25], dtype=np.int32),
            prefill_len_np=np.array([25, 50], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            compute_need_sampled_mask(batch),
            np.array([True, False]),
        )

    def test_ignores_rank_local_max_seq_len_bound(self):
        """Ranks must agree despite different local progress estimates."""
        base = dict(
            num_computed_tokens_np=np.array([56], dtype=np.int32),
            num_scheduled_tokens=np.array([8], dtype=np.int32),
            prefill_len_np=np.array([25], dtype=np.int32),
        )
        tight = _InputBatch(max_seq_len_np=np.array([57]), **base)
        loose = _InputBatch(max_seq_len_np=np.array([98304]), **base)
        np.testing.assert_array_equal(
            compute_need_sampled_mask(tight),
            compute_need_sampled_mask(loose),
        )
        assert compute_need_sampled_mask(tight) is not None

    def test_all_prefill_chunks_only_returns_none(self):
        batch = _InputBatch(
            num_computed_tokens_np=np.array([0, 8], dtype=np.int32),
            num_scheduled_tokens=np.array([8, 8], dtype=np.int32),
            prefill_len_np=np.array([1024, 512], dtype=np.int32),
        )
        assert compute_need_sampled_mask(batch) is None


class _StreamStub:
    def wait_stream(self, other):
        pass

    def record_event(self):
        return object()

    def wait_event(self, event):
        pass


class _StreamCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class TestProtocolInstall:
    def _handler(self, num_speculative_steps=7):
        handler = SimpleNamespace(
            is_last_rank=False,
            device="cpu",
            max_sample_len=num_speculative_steps + 1,
            num_speculative_steps=num_speculative_steps,
            last_rank=1,
            broadcast_group=object(),
            broadcast_stream=_StreamStub(),
            main_stream=_StreamStub(),
            req_idx_gen_np=np.zeros(4, dtype=np.int64),
            queue=deque([None, None]),
        )
        handler.get_prev_sampled_outputs = MethodType(PPHandler.get_prev_sampled_outputs, handler)
        return handler

    def _patch_transport(self, monkeypatch, sent):
        """Swap the module-global ``torch`` for a stub namespace.  Patching
        ``torch.distributed`` attributes directly is unreliable once
        vllm-ascend has rebinded collectives onto torch_npu."""
        from types import SimpleNamespace as NS

        def fake_broadcast(tensor, src=None, group=None):
            sent.append(tensor.clone())

        fake_torch = NS(
            distributed=NS(broadcast=fake_broadcast),
            cuda=NS(stream=lambda _s: _StreamCtx()),
            empty=torch.empty,
            int64=torch.int64,
            int32=torch.int32,
            nn=torch.nn,
            stack=torch.stack,
            as_tensor=torch.as_tensor,
        )
        monkeypatch.setattr(patch_pp, "torch", fake_torch)
        monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)

    def test_installs_and_is_idempotent(self, monkeypatch):
        handler = self._handler()
        req_states = SimpleNamespace()
        upstream_consumer = handler.get_prev_sampled_outputs
        install_pp_token_transport(handler, req_states)
        installed = (handler.receive, handler.broadcast, handler.broadcast_drafts)
        install_pp_token_transport(handler, req_states)
        assert (handler.receive, handler.broadcast, handler.broadcast_drafts) == installed
        assert handler.get_prev_sampled_outputs is upstream_consumer
        assert not hasattr(handler, "broadcast_draft_tokens")

    def test_receive_issues_three_broadcasts_and_reserves_slot(self, monkeypatch):
        handler = self._handler()
        req_states = SimpleNamespace(max_seq_len=np.full(4, 1024))
        install_pp_token_transport(handler, req_states)

        batch = _InputBatch(
            num_computed_tokens_np=np.array([56], dtype=np.int32),
            num_scheduled_tokens=np.array([8], dtype=np.int32),
            prefill_len_np=np.array([25], dtype=np.int32),
            is_prefilling_np=np.array([False]),
        )
        batch.num_reqs = 1  # type: ignore[attr-defined]
        batch.idx_mapping_np = np.array([2], dtype=np.int64)  # type: ignore[attr-defined]
        batch.idx_mapping = object()  # type: ignore[attr-defined]

        sent = []
        self._patch_transport(monkeypatch, sent)
        gather_all = handler.receive(batch)

        # sampled tokens, combined and drafts: three broadcasts, in order.
        assert [tuple(t.shape) for t in sent] == [(1, 8), (2, 1), (1, 7)]
        assert gather_all is True
        slot = handler.queue[-1]
        assert isinstance(slot, PendingRecv)
        assert slot.draft_tokens is not None

    def test_receive_without_sample_needs_skips_transport(self, monkeypatch):
        handler = self._handler()
        req_states = SimpleNamespace()
        install_pp_token_transport(handler, req_states)

        batch = _InputBatch(
            num_computed_tokens_np=np.array([0], dtype=np.int32),
            num_scheduled_tokens=np.array([8], dtype=np.int32),
            prefill_len_np=np.array([1024], dtype=np.int32),
        )
        sent = []
        self._patch_transport(monkeypatch, sent)
        assert handler.receive(batch) is False
        assert sent == []
        assert handler.queue[-1] is None


def _make_batch(*, num_computed, num_scheduled, prefill_len, is_prefilling, idx_mapping):
    return SimpleNamespace(
        num_reqs=len(idx_mapping),
        num_computed_tokens_np=np.asarray(num_computed, dtype=np.int32),
        num_scheduled_tokens=np.asarray(num_scheduled, dtype=np.int32),
        prefill_len_np=np.asarray(prefill_len, dtype=np.int32),
        is_prefilling_np=np.asarray(is_prefilling, dtype=np.bool_),
        idx_mapping_np=np.asarray(idx_mapping, dtype=np.int64),
        idx_mapping=torch.tensor(idx_mapping, dtype=torch.int64),
    )


def _make_req_states(max_seq_len):
    return SimpleNamespace(max_seq_len=np.asarray(max_seq_len, dtype=np.int32))


def test_skip_when_final_prefill_chunk_reaches_length_cap():
    # Disaggregated prefill: prompt 8192, max_tokens=1, final 4096-token chunk.
    batch = _make_batch(
        num_computed=[4096], num_scheduled=[4096], prefill_len=[8192], is_prefilling=[True], idx_mapping=[3]
    )
    assert compute_need_sampled_mask(batch, _make_req_states([0, 0, 0, 8193])) is None


def test_keep_broadcast_when_tokens_remain():
    batch = _make_batch(
        num_computed=[4096], num_scheduled=[4096], prefill_len=[8192], is_prefilling=[True], idx_mapping=[0]
    )
    np.testing.assert_array_equal(compute_need_sampled_mask(batch, _make_req_states([8192 + 64])), [True])


def test_skip_when_final_chunk_alongside_non_final_chunk():
    # A finishing max_tokens=1 request beside another request still mid-prefill.
    batch = _make_batch(
        num_computed=[4096, 0],
        num_scheduled=[4096, 4096],
        prefill_len=[8192, 16384],
        is_prefilling=[True, True],
        idx_mapping=[0, 1],
    )
    assert compute_need_sampled_mask(batch, _make_req_states([8193, 16384 + 256])) is None


def test_keep_broadcast_for_non_final_prefill_chunk():
    batch = _make_batch(
        num_computed=[0], num_scheduled=[4096], prefill_len=[8192], is_prefilling=[True], idx_mapping=[0]
    )
    assert compute_need_sampled_mask(batch, _make_req_states([8193])) is None


def test_keep_broadcast_when_decode_request_in_batch():
    batch = _make_batch(
        num_computed=[4096, 17],
        num_scheduled=[4096, 1],
        prefill_len=[8192, 16],
        is_prefilling=[True, False],
        idx_mapping=[0, 1],
    )
    np.testing.assert_array_equal(compute_need_sampled_mask(batch, _make_req_states([8193, 1024])), [True, True])


def test_keep_broadcast_without_request_limits():
    batch = _make_batch(
        num_computed=[4096], num_scheduled=[4096], prefill_len=[8192], is_prefilling=[True], idx_mapping=[0]
    )
    np.testing.assert_array_equal(compute_need_sampled_mask(batch), [True])


@pytest.mark.parametrize("num_speculative_steps", [0, 3])
@pytest.mark.parametrize("max_tokens", [1, 64])
def test_send_receive_agree_on_skip(monkeypatch, num_speculative_steps, max_tokens):
    fixture = TestProtocolInstall()
    handler = fixture._handler(num_speculative_steps)
    req_states = _make_req_states([8192 + max_tokens])
    install_pp_token_transport(handler, req_states)
    sent = []
    fixture._patch_transport(monkeypatch, sent)
    batch = _make_batch(
        num_computed=[4096], num_scheduled=[4096], prefill_len=[8192], is_prefilling=[True], idx_mapping=[0]
    )
    assert handler.receive(batch) is (max_tokens > 1)
    received_shapes = [tuple(t.shape) for t in sent]
    sent.clear()
    handler.is_last_rank = True
    handler.broadcast(torch.tensor([[42]]), torch.tensor([1]), torch.tensor([0]), batch)
    if num_speculative_steps:
        handler.broadcast_drafts(torch.ones(1, num_speculative_steps, dtype=torch.int64), batch)
    assert [tuple(t.shape) for t in sent] == received_shapes
    if max_tokens == 1:
        assert sent == []
        # Skipping leaves the existing FIFO placeholders intact.
        assert list(handler.queue) == [None, None]
    else:
        assert len(sent) == (3 if num_speculative_steps else 2)
        assert isinstance(handler.queue[-1], PendingRecv)


def test_draft_broadcast_gathers_request_slots(monkeypatch):
    fixture = TestProtocolInstall()
    handler = fixture._handler(3)
    handler.is_last_rank = True
    req_states = _make_req_states([1024] * 4)
    req_states.draft_tokens = torch.arange(12, dtype=torch.int64).reshape(4, 3)
    install_pp_token_transport(handler, req_states)
    sent = []
    fixture._patch_transport(monkeypatch, sent)
    batch = _make_batch(
        num_computed=[20, 30],
        num_scheduled=[4, 4],
        prefill_len=[16, 16],
        is_prefilling=[False, False],
        idx_mapping=[3, 1],
    )
    handler.broadcast_drafts(req_states.draft_tokens, batch)
    assert len(sent) == 1
    torch.testing.assert_close(sent[0], req_states.draft_tokens[[3, 1]])


def test_draft_broadcast_uses_current_batch(monkeypatch):
    fixture = TestProtocolInstall()
    handler = fixture._handler(3)
    handler.is_last_rank = True
    req_states = _make_req_states([8193, 1024])
    req_states.draft_tokens = torch.ones(2, 3, dtype=torch.int64)
    install_pp_token_transport(handler, req_states)
    sent = []
    fixture._patch_transport(monkeypatch, sent)
    decode = _make_batch(num_computed=[20], num_scheduled=[4], prefill_len=[16], is_prefilling=[False], idx_mapping=[1])
    finishing = _make_batch(
        num_computed=[4096], num_scheduled=[4096], prefill_len=[8192], is_prefilling=[True], idx_mapping=[0]
    )
    handler.broadcast(torch.ones(1, 4, dtype=torch.int64), torch.ones(1), torch.zeros(1), decode)
    sent.clear()
    handler.broadcast(None, None, None, finishing)
    handler.broadcast_drafts(req_states.draft_tokens, finishing)
    assert sent == []


def test_deferred_consumer_filters_freed_requests(monkeypatch):
    fixture = TestProtocolInstall()
    handler = fixture._handler(3)
    req_states = _make_req_states([1024] * 4)
    req_states.draft_tokens = torch.zeros(4, 3, dtype=torch.int64)
    install_pp_token_transport(handler, req_states)
    fixture._patch_transport(monkeypatch, [])
    monkeypatch.setattr(pp_utils, "async_tensor_h2d", lambda data, device: torch.as_tensor(data))
    slot = PendingRecv(
        event=object(),
        sampled_tokens=torch.tensor([[10], [20]]),
        num_sampled=torch.tensor([1, 1]),
        num_rejected=torch.tensor([2, 2]),
        idx_mapping=torch.tensor([3, 1]),
        idx_mapping_np=np.array([3, 1]),
        need_sampled_mask=np.array([True, True]),
        gen_at_receive_np=np.array([0, 0]),
        draft_tokens=torch.tensor([[11, 12, 13], [21, 22, 23]]),
    )
    handler.queue = deque([None, slot])
    handler.req_idx_gen_np[1] = 1
    assert handler.get_prev_sampled_outputs(req_states.draft_tokens) is None
    output = handler.get_prev_sampled_outputs(req_states.draft_tokens)
    assert output["idx_mapping"].tolist() == [3, -1]
    assert req_states.draft_tokens[3].tolist() == [11, 12, 13]
    assert req_states.draft_tokens[1].tolist() == [0, 0, 0]
    assert list(handler.queue) == [None, None]
