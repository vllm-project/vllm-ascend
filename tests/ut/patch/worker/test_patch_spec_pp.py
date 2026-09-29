# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest

from vllm_ascend.patch.worker.patch_v2 import patch_spec_pp
from vllm_ascend.patch.worker.patch_v2.patch_spec_pp import (
    SpecPPPendingRecv,
    compute_need_sampled_mask,
    install_upstream_spec_pp_protocol,
)


@dataclass
class _InputBatch:
    num_computed_tokens_np: np.ndarray
    num_scheduled_tokens: np.ndarray
    prefill_len_np: np.ndarray
    # Rank-local bound consulted by the release gate; the vendored 0.30
    # gate must ignore it entirely.
    max_seq_len_np: np.ndarray | None = None


class TestPureParticipationGate:
    def test_matches_release_gate_for_regular_decode(self):
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
        """The release gate drops requests whose next token may hit
        max_seq_len; two ranks with different views of that bound must
        still compute the same mask here."""
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


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def unbind(self, dim=0):
        return _FakeTensor((self.shape[1],)), _FakeTensor((self.shape[1],))

    def record_stream(self, stream):
        pass


class _StreamStub:
    def wait_stream(self, other):
        pass

    def record_event(self):
        return object()


class _StreamCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class TestProtocolInstall:
    def _handler(self):
        return SimpleNamespace(
            is_last_rank=False,
            device="npu:0",
            max_sample_len=8,
            last_rank=1,
            broadcast_group=object(),
            broadcast_stream=_StreamStub(),
            main_stream=_StreamStub(),
            req_idx_gen_np=np.zeros(4, dtype=np.int64),
            queue=[None],
        )

    def _patch_transport(self, monkeypatch, sent):
        """Swap the module-global ``torch`` for a stub namespace.  Patching
        ``torch.distributed`` attributes directly is unreliable once
        vllm-ascend has rebinded collectives onto torch_npu."""
        from types import SimpleNamespace as NS

        import vllm_ascend.patch.worker.patch_v2.patch_spec_pp as mod

        def fake_broadcast(tensor, src=None, group=None):
            sent.append(tuple(tensor.shape))

        fake_torch = NS(
            distributed=NS(broadcast=fake_broadcast),
            cuda=NS(stream=lambda _s: _StreamCtx()),
            empty=lambda *a, **k: _FakeTensor((a[0], a[1])),
            int64=object(),
            int32=object(),
        )
        monkeypatch.setattr(mod, "torch", fake_torch)

    def test_installs_and_is_idempotent(self, monkeypatch):
        handler = self._handler()
        req_states = SimpleNamespace()
        install_upstream_spec_pp_protocol(handler, req_states, num_speculative_steps=7)
        installed = (handler.receive, handler.broadcast, handler.broadcast_drafts)
        install_upstream_spec_pp_protocol(handler, req_states, num_speculative_steps=7)
        assert (handler.receive, handler.broadcast, handler.broadcast_drafts) == installed
        assert handler.broadcast_draft_tokens is handler.broadcast_drafts

    def test_receive_issues_three_broadcasts_and_reserves_slot(self, monkeypatch):
        handler = self._handler()
        req_states = SimpleNamespace()
        install_upstream_spec_pp_protocol(handler, req_states, num_speculative_steps=7)

        batch = _InputBatch(
            num_computed_tokens_np=np.array([56], dtype=np.int32),
            num_scheduled_tokens=np.array([8], dtype=np.int32),
            prefill_len_np=np.array([25], dtype=np.int32),
        )
        batch.num_reqs = 1  # type: ignore[attr-defined]
        batch.idx_mapping_np = np.array([2], dtype=np.int64)  # type: ignore[attr-defined]
        batch.idx_mapping = object()  # type: ignore[attr-defined]

        sent: list[tuple[int, ...]] = []
        self._patch_transport(monkeypatch, sent)
        gather_all = handler.receive(batch)

        # sampled tokens, combined and drafts: three broadcasts, in order.
        assert sent == [(1, 8), (2, 1), (1, 7)]
        assert gather_all is True
        slot = handler.queue[-1]
        assert isinstance(slot, SpecPPPendingRecv)
        assert slot.draft_tokens is not None

    def test_receive_without_sample_needs_skips_transport(self, monkeypatch):
        handler = self._handler()
        req_states = SimpleNamespace()
        install_upstream_spec_pp_protocol(handler, req_states, num_speculative_steps=7)

        batch = _InputBatch(
            num_computed_tokens_np=np.array([0], dtype=np.int32),
            num_scheduled_tokens=np.array([8], dtype=np.int32),
            prefill_len_np=np.array([1024], dtype=np.int32),
        )
        sent: list[tuple[int, ...]] = []
        self._patch_transport(monkeypatch, sent)
        assert handler.receive(batch) is False
        assert sent == []
        assert handler.queue[-1] is None

    def test_pending_recv_extends_release_fields_with_drafts(self):
        fields = SpecPPPendingRecv.__dataclass_fields__
        release_fields = (
            "event",
            "sampled_tokens",
            "num_sampled",
            "num_rejected",
            "idx_mapping",
            "idx_mapping_np",
            "need_sampled_mask",
            "gen_at_receive_np",
        )
        for name in release_fields:
            assert name in fields
        assert fields["draft_tokens"].default is None


def _make_batch(*, num_computed, num_scheduled, prefill_len, is_prefilling, idx_mapping):
    return SimpleNamespace(
        num_computed_tokens_np=np.asarray(num_computed, dtype=np.int32),
        num_scheduled_tokens=np.asarray(num_scheduled, dtype=np.int32),
        prefill_len_np=np.asarray(prefill_len, dtype=np.int32),
        is_prefilling_np=np.asarray(is_prefilling, dtype=np.bool_),
        idx_mapping_np=np.asarray(idx_mapping, dtype=np.int64),
    )


def _make_handler(max_seq_len):
    handler = SimpleNamespace()
    if max_seq_len is not None:
        handler.ascend_request_states = SimpleNamespace(max_seq_len=np.asarray(max_seq_len, dtype=np.int32))
    return handler


def test_skip_when_final_prefill_chunk_reaches_length_cap():
    # Disaggregated prefill: prompt 8192, max_tokens=1, final 4096-token chunk.
    batch = _make_batch(
        num_computed=[4096], num_scheduled=[4096], prefill_len=[8192], is_prefilling=[True], idx_mapping=[3]
    )
    assert patch_spec_pp.broadcast_has_no_consumer(_make_handler([0, 0, 0, 8193]), batch)


def test_keep_broadcast_when_tokens_remain():
    batch = _make_batch(
        num_computed=[4096], num_scheduled=[4096], prefill_len=[8192], is_prefilling=[True], idx_mapping=[0]
    )
    assert not patch_spec_pp.broadcast_has_no_consumer(_make_handler([8192 + 64]), batch)


def test_skip_when_final_chunk_alongside_non_final_chunk():
    # A finishing max_tokens=1 request beside another request still mid-prefill.
    batch = _make_batch(
        num_computed=[4096, 0],
        num_scheduled=[4096, 4096],
        prefill_len=[8192, 16384],
        is_prefilling=[True, True],
        idx_mapping=[0, 1],
    )
    assert patch_spec_pp.broadcast_has_no_consumer(_make_handler([8193, 16384 + 256]), batch)


def test_keep_broadcast_for_non_final_prefill_chunk():
    batch = _make_batch(
        num_computed=[0], num_scheduled=[4096], prefill_len=[8192], is_prefilling=[True], idx_mapping=[0]
    )
    # Upstream already skips this step because it produces no sample.
    assert not patch_spec_pp.broadcast_has_no_consumer(_make_handler([8193]), batch)


def test_keep_broadcast_when_decode_request_in_batch():
    batch = _make_batch(
        num_computed=[4096, 17],
        num_scheduled=[4096, 1],
        prefill_len=[8192, 16],
        is_prefilling=[True, False],
        idx_mapping=[0, 1],
    )
    assert not patch_spec_pp.broadcast_has_no_consumer(_make_handler([8193, 1024]), batch)


def test_keep_broadcast_without_attached_request_states():
    batch = _make_batch(
        num_computed=[4096], num_scheduled=[4096], prefill_len=[8192], is_prefilling=[True], idx_mapping=[0]
    )
    assert not patch_spec_pp.broadcast_has_no_consumer(_make_handler(None), batch)


@pytest.fixture
def patched_pp_handler(monkeypatch):
    # Stub transport only; run the real patch and participation predicate on CPU.
    handler_cls = type(
        "PPHandlerStub",
        (),
        {
            name: create_autospec(lambda self, *args: None, return_value="original")
            for name in ("receive", "broadcast", "broadcast_drafts")
        },
    )
    monkeypatch.setattr(patch_spec_pp.pp_utils, "PPHandler", handler_cls)
    patch_spec_pp._patch_pp_handler()
    return handler_cls


@pytest.mark.parametrize("method,num_args", [("receive", 0), ("broadcast", 3), ("broadcast_drafts", 1)])
@pytest.mark.parametrize("max_tokens", [1, 64])
def test_patched_transport_skips_only_without_consumer(patched_pp_handler, method, num_args, max_tokens):
    handler = patched_pp_handler()
    handler.ascend_request_states = _make_handler([8192 + max_tokens]).ascend_request_states
    handler.queue = [None]
    batch = _make_batch(
        num_computed=[4096], num_scheduled=[4096], prefill_len=[8192], is_prefilling=[True], idx_mapping=[0]
    )
    args = [object() for _ in range(num_args)]
    patched = getattr(handler, method)
    result = patched(*args, batch)
    if max_tokens == 1:
        assert result is (False if method == "receive" else None)
        patched.__wrapped__.assert_not_called()
    else:
        assert result == "original"
        patched.__wrapped__.assert_called_once_with(handler, *args, batch)
    assert handler.queue == [None]


def test_patch_is_idempotent(patched_pp_handler):
    methods = ("receive", "broadcast", "broadcast_drafts")
    before = [getattr(patched_pp_handler, name) for name in methods]
    patch_spec_pp._patch_pp_handler()
    assert [getattr(patched_pp_handler, name) for name in methods] == before
    assert all(getattr(method, patch_spec_pp._PATCHED_ATTR, False) for method in before)


def test_native_pp_patch_is_registered():
    assert getattr(patch_spec_pp.pp_utils.PPHandler.receive, patch_spec_pp._PATCHED_ATTR, False)
