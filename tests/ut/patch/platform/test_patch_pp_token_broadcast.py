# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the no-consumer PP token broadcast skip.

``patch_pp_token_broadcast`` skips ``PPHandler.receive``/``broadcast``/
``broadcast_drafts`` when every sample-producing request is a final prefill
chunk that reaches its length cap with its single sampled token
(``max_tokens=1`` requests, as disaggregated-prefill routers submit on
pure prefill nodes and hybrid producer-consumer instances). These tests
exercise the skip verdict and the patched entry points on CPU with stub
batches.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_ascend.patch.platform import patch_pp_token_broadcast


def _make_batch(
    *,
    num_computed: list[int],
    num_scheduled: list[int],
    prefill_len: list[int],
    is_prefilling: list[bool],
    idx_mapping: list[int],
) -> SimpleNamespace:
    return SimpleNamespace(
        num_computed_tokens_np=np.asarray(num_computed, dtype=np.int32),
        num_scheduled_tokens=np.asarray(num_scheduled, dtype=np.int32),
        prefill_len_np=np.asarray(prefill_len, dtype=np.int32),
        is_prefilling_np=np.asarray(is_prefilling, dtype=np.bool_),
        idx_mapping_np=np.asarray(idx_mapping, dtype=np.int64),
    )


def _make_handler(max_seq_len: list[int] | None) -> SimpleNamespace:
    handler = SimpleNamespace()
    if max_seq_len is not None:
        handler.ascend_request_states = SimpleNamespace(max_seq_len=np.asarray(max_seq_len, dtype=np.int32))
    return handler


def test_skip_when_final_prefill_chunk_reaches_length_cap():
    # Disaggregated prefill: prompt 8192, max_tokens=1, final 4096-token chunk.
    batch = _make_batch(
        num_computed=[4096],
        num_scheduled=[4096],
        prefill_len=[8192],
        is_prefilling=[True],
        idx_mapping=[3],
    )
    handler = _make_handler(max_seq_len=[0, 0, 0, 8193])
    assert patch_pp_token_broadcast.broadcast_has_no_consumer(handler, batch)


def test_keep_broadcast_when_tokens_remain():
    # Same final chunk, but the request must still generate more tokens.
    batch = _make_batch(
        num_computed=[4096],
        num_scheduled=[4096],
        prefill_len=[8192],
        is_prefilling=[True],
        idx_mapping=[0],
    )
    handler = _make_handler(max_seq_len=[8192 + 64])
    assert not patch_pp_token_broadcast.broadcast_has_no_consumer(handler, batch)


def test_skip_when_final_chunk_alongside_non_final_chunk():
    # Hybrid instance: row 0 is a max_tokens=1 request finishing prefill,
    # row 1 is another request still mid-prefill (no sample this step).
    batch = _make_batch(
        num_computed=[4096, 0],
        num_scheduled=[4096, 4096],
        prefill_len=[8192, 16384],
        is_prefilling=[True, True],
        idx_mapping=[0, 1],
    )
    handler = _make_handler(max_seq_len=[8193, 16384 + 256])
    assert patch_pp_token_broadcast.broadcast_has_no_consumer(handler, batch)


def test_keep_broadcast_for_non_final_prefill_chunk():
    batch = _make_batch(
        num_computed=[0],
        num_scheduled=[4096],
        prefill_len=[8192],
        is_prefilling=[True],
        idx_mapping=[0],
    )
    handler = _make_handler(max_seq_len=[8193])
    # No request produces a sample, so the upstream mask is None.
    assert not patch_pp_token_broadcast.broadcast_has_no_consumer(handler, batch)


def test_keep_broadcast_when_decode_request_in_batch():
    # Row 0 is a finishing prefill request; row 1 is a decode request that
    # continues on this engine and needs the broadcast tokens.
    batch = _make_batch(
        num_computed=[4096, 17],
        num_scheduled=[4096, 1],
        prefill_len=[8192, 16],
        is_prefilling=[True, False],
        idx_mapping=[0, 1],
    )
    handler = _make_handler(max_seq_len=[8193, 1024])
    assert not patch_pp_token_broadcast.broadcast_has_no_consumer(handler, batch)


def test_keep_broadcast_without_attached_request_states():
    batch = _make_batch(
        num_computed=[4096],
        num_scheduled=[4096],
        prefill_len=[8192],
        is_prefilling=[True],
        idx_mapping=[0],
    )
    handler = _make_handler(max_seq_len=None)
    assert not patch_pp_token_broadcast.broadcast_has_no_consumer(handler, batch)


def _pp_handler_patch_applied() -> bool:
    from vllm.v1.worker.gpu.pp_utils import PPHandler

    return getattr(PPHandler.receive, patch_pp_token_broadcast._PATCHED_ATTR, False)


@pytest.mark.skipif(
    not _pp_handler_patch_applied(),
    reason="vLLM without the deferred PP token broadcast",
)
def test_patched_receive_skips_broadcast_without_consumer(monkeypatch):
    from vllm.v1.worker.gpu.pp_utils import PPHandler

    calls = []
    monkeypatch.setattr(
        patch_pp_token_broadcast,
        "_original_receive",
        lambda self, input_batch: calls.append(input_batch) or "original",
    )
    handler = object.__new__(PPHandler)

    # Finishing request (prompt 8192, max_tokens=1): the collective is
    # skipped entirely.
    handler.ascend_request_states = _make_handler([8193]).ascend_request_states
    finishing_batch = _make_batch(
        num_computed=[4096],
        num_scheduled=[4096],
        prefill_len=[8192],
        is_prefilling=[True],
        idx_mapping=[0],
    )
    assert PPHandler.receive(handler, finishing_batch) is False
    assert calls == []

    # Continuing request (max_tokens=65): the original receive path runs.
    handler.ascend_request_states = _make_handler([8192 + 64]).ascend_request_states
    continuing_batch = _make_batch(
        num_computed=[4096],
        num_scheduled=[4096],
        prefill_len=[8192],
        is_prefilling=[True],
        idx_mapping=[0],
    )
    assert PPHandler.receive(handler, continuing_batch) == "original"
    assert calls == [continuing_batch]


@pytest.mark.skipif(
    not _pp_handler_patch_applied(),
    reason="vLLM without the deferred PP token broadcast",
)
def test_patch_is_idempotent():
    from vllm.v1.worker.gpu.pp_utils import PPHandler

    patch_pp_token_broadcast._apply_patch()
    marker = patch_pp_token_broadcast._PATCHED_ATTR
    for method in (PPHandler.receive, PPHandler.broadcast, PPHandler.broadcast_drafts):
        assert getattr(method, marker, False)
        assert getattr(method, "__wrapped__", None) is not None
    assert patch_pp_token_broadcast._original_receive is PPHandler.receive.__wrapped__
