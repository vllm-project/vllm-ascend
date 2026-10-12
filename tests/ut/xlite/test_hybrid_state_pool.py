# Copyright (c) 2026 ouyangqing. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the xlite decode-only hybrid state pool (CPU-only).

Covers the slot-major GDN state management that bridges the vllm mamba
pools and the xlite persistent pool: sparse block-ID to dense slot
remapping, batch/persistent copy semantics, row zeroing for fresh
requests, and the gather/scatter skip accounting.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# The real xlite._C probes the NPU runtime at import time and *aborts the
# process* (not an ImportError) on CPU-only runners, so it must only be
# imported when a real NPU is present. Otherwise install the stub below.
_HAVE_XLITE = False
try:
    import subprocess as _sp

    _sp.run(["npu-smi", "info"], capture_output=True, check=True)
    import xlite._C  # noqa: F401

    _HAVE_XLITE = True
except Exception:  # pragma: no cover - CPU-only runners take the stub path
    _HAVE_XLITE = False
    import enum

    class _AttnKind(enum.IntEnum):
        MHA = 0
        MLA = 1
        HYBRID = 2
        DSA = 3

    xlite_mod = types.ModuleType("xlite")
    xlite_c = types.ModuleType("xlite._C")
    # setattr keeps the dynamic ModuleType attributes invisible to mypy.
    for _name, _value in {
        "AttnMHA": _AttnKind.MHA,
        "AttnMLA": _AttnKind.MLA,
        "AttnHybrid": _AttnKind.HYBRID,
        "AttnDSA": _AttnKind.DSA,
        "AttnMeta": type("AttnMeta", (), {}),
        "AttnMetaV2": type("AttnMetaV2", (), {}),
        # required by vllm_ascend/xlite/utils.py
        "Model": type("Model", (), {}),
        "ModelConfig": type("ModelConfig", (), {}),
        "Runtime": MagicMock,
        "ScoringFuncSoftmax": 0,
        "ScoringFuncSigmoid": 1,
    }.items():
        setattr(xlite_c, _name, _value)
    for _name in (
        "graph_capture_begin",
        "graph_capture_end",
        "graph_destroy",
        "graph_exec",
        "set_gdn_chunk_callback",
    ):
        setattr(xlite_c, _name, MagicMock())
    sys.modules.setdefault("xlite", xlite_mod)
    sys.modules.setdefault("xlite._C", xlite_c)

from vllm_ascend.xlite.xlite import XliteWrapper  # noqa: E402

CONV_TAIL = (6, 4)
SSM_TAIL = (2, 4, 4)


def _wrapper(n_layers=2, max_batch=8):
    """Real pool geometry: persistent holds max_batch+1 rows (slot 0 is NULL)."""
    w = object.__new__(XliteWrapper)
    w.device = "cpu"
    w._xlite_hybrid_block_to_slot = {}
    w._xlite_hybrid_free_slots = list(range(1, max_batch + 1))
    w._xlite_hybrid_persistent = [
        (
            torch.full((max_batch + 1, *CONV_TAIL), float(layer_idx + 1)),
            torch.full((max_batch + 1, *SSM_TAIL), float(layer_idx + 1)),
        )
        for layer_idx in range(n_layers)
    ]
    w._xlite_hybrid_batch = [
        (torch.zeros((max_batch, *CONV_TAIL)), torch.zeros((max_batch, *SSM_TAIL))) for _ in range(n_layers)
    ]
    w._xlite_hybrid_last_ids = None
    w._xlite_hybrid_last_dense = None
    w._xlite_hybrid_last_idx = None
    w._xlite_hybrid_dirty = False
    w._xlite_hybrid_skip = False
    return w


def test_hybrid_slots_assigns_dense_slots_and_skips_null():
    w = _wrapper()
    slots = XliteWrapper._hybrid_slots(w, (101, 0, 102))
    # free_slots.pop() hands out the highest free slot first (NULL -> 0).
    assert slots == (8, 0, 7)
    assert w._xlite_hybrid_block_to_slot == {101: 8, 102: 7}
    # Recomputing the same IDs reuses the slots.
    assert XliteWrapper._hybrid_slots(w, (101, 0, 102)) == (8, 0, 7)


def test_hybrid_slots_new_request_gets_next_slot():
    w = _wrapper()
    XliteWrapper._hybrid_slots(w, (101, 102))
    # 101 retires from the batch; 103 is new and gets the next free slot.
    slots = XliteWrapper._hybrid_slots(w, (102, 103))
    assert slots[0] == w._xlite_hybrid_block_to_slot[102]
    assert slots[1] == slots[0] - 1
    assert 103 in w._xlite_hybrid_block_to_slot


def test_hybrid_slots_exhaustion_raises():
    w = _wrapper(max_batch=2)
    XliteWrapper._hybrid_slots(w, (1, 2))
    with pytest.raises(RuntimeError, match="no free GDN slots"):
        XliteWrapper._hybrid_slots(w, (1, 2, 3))


def test_hybrid_slots_zeroes_new_slot_state():
    w = _wrapper()
    XliteWrapper._hybrid_slots(w, (7,))
    slot = w._xlite_hybrid_block_to_slot[7]
    for conv_p, ssm_p in w._xlite_hybrid_persistent:
        assert torch.count_nonzero(conv_p[slot]) == 0
        assert torch.count_nonzero(ssm_p[slot]) == 0


def test_hybrid_copy_roundtrip_between_persistent_and_batch():
    w = _wrapper()
    idx = torch.tensor([1, 3], dtype=torch.long)
    # Seed persistent row 1 and 3 with distinct values, then pull to batch.
    for conv_p, ssm_p in w._xlite_hybrid_persistent:
        conv_p[1] = 10.0
        conv_p[3] = 30.0
    XliteWrapper._hybrid_copy(w, idx, to_batch=True)
    for conv_b, _ssm_b in w._xlite_hybrid_batch:
        assert torch.equal(conv_b[0], torch.full(CONV_TAIL, 10.0))
        assert torch.equal(conv_b[1], torch.full(CONV_TAIL, 30.0))
    # Push back and verify persistent rows were overwritten.
    for conv_b, _ssm_b in w._xlite_hybrid_batch:
        conv_b[0] = 99.0
    XliteWrapper._hybrid_copy(w, idx, to_batch=False)
    for conv_p, _ssm_p in w._xlite_hybrid_persistent:
        assert torch.equal(conv_p[1], torch.full(CONV_TAIL, 99.0))
        assert torch.equal(conv_p[3], torch.full(CONV_TAIL, 30.0))


def test_hybrid_zero_rows_clears_batch_and_persistent_rows():
    w = _wrapper()
    XliteWrapper._hybrid_slots(w, (5,))
    dense = XliteWrapper._hybrid_slots(w, (5, 6))
    # Simulate an earlier scatter: the persistent slots carry state.
    for conv_p, _ssm_p in w._xlite_hybrid_persistent:
        conv_p[dense[0]] = 20.0
        conv_p[dense[1]] = 40.0
    XliteWrapper._hybrid_copy(w, torch.tensor(dense, dtype=torch.long), to_batch=True)
    for conv_b, _ssm_b in w._xlite_hybrid_batch:
        assert torch.count_nonzero(conv_b).item() > 0
    XliteWrapper._hybrid_zero_rows(w, [0, 1], dense)
    for conv_b, ssm_b in w._xlite_hybrid_batch:
        assert torch.count_nonzero(conv_b).item() == 0
        assert torch.count_nonzero(ssm_b).item() == 0
    for conv_p, ssm_p in w._xlite_hybrid_persistent:
        assert torch.count_nonzero(conv_p[dense[0]]).item() == 0
        assert torch.count_nonzero(conv_p[dense[1]]).item() == 0
        assert torch.count_nonzero(ssm_p[dense[1]]).item() == 0


def test_hybrid_gather_copies_state_to_batch_rows():
    w = _wrapper()
    idx = torch.tensor([2, 4], dtype=torch.long)
    slot_a, slot_b = XliteWrapper._hybrid_slots(w, (2, 4))
    for conv_p, _ssm_p in w._xlite_hybrid_persistent:
        conv_p[slot_a] = 20.0
        conv_p[slot_b] = 40.0

    XliteWrapper._hybrid_gather(w, idx, cached_lens=[3, 3], num_reqs=2)

    assert w._xlite_hybrid_skip is False
    for conv_b, _ssm_b in w._xlite_hybrid_batch:
        assert torch.equal(conv_b[0], torch.full(CONV_TAIL, 20.0))
        assert torch.equal(conv_b[1], torch.full(CONV_TAIL, 40.0))


def test_hybrid_gather_skips_when_request_order_unchanged():
    w = _wrapper()
    idx = torch.tensor([2, 4], dtype=torch.long)
    XliteWrapper._hybrid_slots(w, (2, 4))
    XliteWrapper._hybrid_gather(w, idx, cached_lens=[3, 3], num_reqs=2)

    # Same ids again with no fresh rows -> no copy, skip recorded.
    for conv_b, _ssm_b in w._xlite_hybrid_batch:
        conv_b.zero_()
    XliteWrapper._hybrid_gather(w, idx, cached_lens=[3, 3], num_reqs=2)
    assert w._xlite_hybrid_skip is True
    for conv_b, _ssm_b in w._xlite_hybrid_batch:
        assert torch.count_nonzero(conv_b).item() == 0


def test_hybrid_gather_zeroes_fresh_requests_on_unchanged_order():
    w = _wrapper()
    idx = torch.tensor([2, 4], dtype=torch.long)
    slot_a, slot_b = XliteWrapper._hybrid_slots(w, (2, 4))
    # Scatter real state first, then a full gather fills both batch rows.
    for conv_p, _ssm_p in w._xlite_hybrid_persistent:
        conv_p[slot_a] = 20.0
        conv_p[slot_b] = 40.0
    XliteWrapper._hybrid_gather(w, idx, cached_lens=[3, 3], num_reqs=2)
    for conv_b, _ssm_b in w._xlite_hybrid_batch:
        assert torch.count_nonzero(conv_b).item() > 0
    # cached_len == 0 -> fresh request: its batch row must be cleared.
    XliteWrapper._hybrid_gather(w, idx, cached_lens=[3, 0], num_reqs=2)
    for conv_b, _ssm_b in w._xlite_hybrid_batch:
        assert torch.count_nonzero(conv_b[1]).item() == 0
        assert torch.count_nonzero(conv_b[0]).item() > 0


def test_hybrid_scatter_marks_dirty_after_skip():
    w = _wrapper()
    w._xlite_hybrid_skip = True
    XliteWrapper._hybrid_scatter(w)
    assert w._xlite_hybrid_dirty is True

    idx = torch.tensor([1], dtype=torch.long)
    w._xlite_hybrid_last_idx = idx
    w._xlite_hybrid_skip = False
    for conv_b, _ssm_b in w._xlite_hybrid_batch:
        conv_b[0] = 7.0
    XliteWrapper._hybrid_scatter(w)
    assert w._xlite_hybrid_dirty is False
    for conv_p, _ssm_p in w._xlite_hybrid_persistent:
        assert torch.equal(conv_p[1], torch.full(CONV_TAIL, 7.0))


def test_gdn_state_ids_prefers_non_spec_indices(monkeypatch):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    def meta(non_spec, spec):
        m = GDNAttentionMetadata.__new__(GDNAttentionMetadata)
        object.__setattr__(m, "non_spec_state_indices_tensor", non_spec)
        object.__setattr__(m, "spec_state_indices_tensor", spec)
        return m

    non_spec = torch.tensor([9, 8, 7])
    spec = torch.tensor([[5], [4], [3]])
    gdn_meta = meta(non_spec=non_spec, spec=spec)
    other = SimpleNamespace()  # non-GDN entry must be skipped

    ids = XliteWrapper._gdn_state_ids({"layer0": other, "layer1": gdn_meta}, num_reqs=3)
    assert ids is not None and torch.equal(ids, non_spec)

    # Fall back to the first column of the spec layout when non-spec is absent.
    ids = XliteWrapper._gdn_state_ids([meta(non_spec=None, spec=spec)], num_reqs=3)
    assert ids is not None and torch.equal(ids, torch.tensor([5, 4, 3]))

    # Too few indices -> None; zero requests -> None.
    assert XliteWrapper._gdn_state_ids([gdn_meta], num_reqs=5) is None
    assert XliteWrapper._gdn_state_ids([gdn_meta], num_reqs=0) is None


class _Runnable:
    """Records forward calls and returns a sentinel output."""

    def __init__(self, out="native-out"):
        self.out = out
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.out


class _Wrapped:
    """ACLGraph-wrapper stand-in exposing unwrap()."""

    def __init__(self, inner):
        self._inner = inner

    def unwrap(self):
        return self._inner


def _fallback_wrapper():
    w = object.__new__(XliteWrapper)
    w._pending_native_ids = []
    w._xlite_native_dirty = set()
    w._last_decode_keys = None
    w._hybrid_export_to_native = MagicMock()
    return w


def test_native_step_fallback_records_ids_exports_and_unwraps():
    w = _fallback_wrapper()
    idx = torch.tensor([3, 4], dtype=torch.int32)
    gdn_meta = SimpleNamespace(non_spec_state_indices_tensor=idx)
    w._last_decode_keys = torch.tensor([5, 7], dtype=torch.int32)
    inner = _Runnable()
    w.npu_runnable = _Wrapped(inner)

    out = XliteWrapper._hybrid_native_step_fallback(
        w,
        {"gdn0": gdn_meta},
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([0, 1]),
        intermediate_tensors=None,
        inputs_embeds=None,
        model_kwargs={},
    )

    assert out == "native-out"
    # The state-index tensor is recorded by identity (no sync/copy).
    assert len(w._pending_native_ids) == 1
    assert w._pending_native_ids[0] is idx
    # The last decode batch's keys are materialized exactly once and cleared.
    w._hybrid_export_to_native.assert_called_once_with([5, 7])
    assert w._last_decode_keys is None
    # The UNWRAPPED runnable handled the call, not the ACLGraph wrapper.
    assert len(inner.calls) == 1


def test_native_step_fallback_skips_export_without_decode_keys():
    w = _fallback_wrapper()
    w.npu_runnable = _Runnable()

    out = XliteWrapper._hybrid_native_step_fallback(
        w,
        {},
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=None,
        model_kwargs={},
    )

    assert out == "native-out"
    w._hybrid_export_to_native.assert_called_once_with(None)
    assert w._pending_native_ids == []


def test_native_step_fallback_safety_valve_flushes_pending_ids():
    w = _fallback_wrapper()
    w._pending_native_ids = [torch.tensor([i], dtype=torch.int32) for i in range(31)]
    w.npu_runnable = _Runnable()
    idx = torch.tensor([99], dtype=torch.int32)
    gdn_meta = SimpleNamespace(non_spec_state_indices_tensor=idx)

    XliteWrapper._hybrid_native_step_fallback(
        w,
        gdn_meta,
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=None,
        model_kwargs={},
    )

    # 31 pending + 1 new hits the limit: all ids materialize into the dirty
    # set (including the new one) and the list is cleared.
    assert w._pending_native_ids == []
    assert w._xlite_native_dirty == set(range(31)) | {99}


def test_native_step_fallback_ignores_empty_index_tensors():
    w = _fallback_wrapper()
    w.npu_runnable = _Runnable()
    empty = SimpleNamespace(non_spec_state_indices_tensor=torch.empty(0, dtype=torch.int32))
    valid = SimpleNamespace(non_spec_state_indices_tensor=torch.tensor([2]))

    XliteWrapper._hybrid_native_step_fallback(
        w,
        [empty, SimpleNamespace(), valid],
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=None,
        model_kwargs={},
    )

    assert len(w._pending_native_ids) == 1
    assert int(w._pending_native_ids[0][0]) == 2


def test_decode_only_routing_routes_prefill_to_native():
    """Regression: the hybrid decode-only routing (native prefill, xlite
    decode) must survive — the merge once reduced it to `always graph`,
    which broke state bridging and double-counted prefill states."""
    import inspect

    src = inspect.getsource(XliteWrapper.__call__)
    assert "if self._hybrid_decode_only:" in src
    assert "use_xlite_graph = not with_prefill" in src
    # The prefill fallback must go through the bridging helper (records
    # pending ids + exports states + unwraps the ACLGraph wrapper).
    assert "_hybrid_native_step_fallback(" in src
    fallback = inspect.getsource(XliteWrapper._hybrid_native_step_fallback)
    assert "_pending_native_ids.append(" in fallback
    assert "_hybrid_export_to_native(" in fallback
