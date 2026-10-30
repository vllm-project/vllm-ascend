# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quarantine actual private copies that return before a Host exception."""

from types import SimpleNamespace

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.worker.v2.spec_decode.draft_window_dense import DenseDraftRead, prepare_dense_draft_read  # noqa: E402
from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import DraftWindowOwnerPool  # noqa: E402
from vllm_ascend.worker.v2.spec_decode.draft_window_read import DirectPrefixSource  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")


@pytest.mark.parametrize("stage", ["exact", "phase", "cohort"])
def test_real_private_copy_failure_retains_its_actual_operands(monkeypatch, stage):
    table = torch.arange(32, dtype=torch.int32, device="npu").reshape(2, 16)
    before = table.clone()
    source = DirectPrefixSource(table, (1, 0), (16, 16), 128)
    exact = torch.tensor([1151, 1408], dtype=torch.int32, device="npu")
    rows = torch.tensor([1, 0], device="npu")
    pool = DraftWindowOwnerPool(1)
    events, submitted = [], []
    originals = {name: getattr(torch.Tensor, name) for name in ("clone", "to", "copy_")}

    def selected(tensor):
        if stage == "exact":
            return tensor.is_npu and tensor.data_ptr() == exact.data_ptr()
        dtype = torch.bool if stage == "phase" else torch.int64
        return tensor.device.type == "cpu" and tensor.dtype == dtype

    def after_copy(input_tensor, output_tensor):
        assert output_tensor.is_npu
        event = torch.npu.Event()
        event.record(torch.npu.current_stream())
        events.append(event)
        submitted.append((input_tensor, output_tensor))
        assert any(item is input_tensor for item in owner.retained), "copy source missing from active owner"
        assert any(item is output_tensor for item in owner.retained), "copy destination missing from active owner"
        raise RuntimeError("deliberate Host failure after real NPU copy")

    def clone(tensor, *args, **kwargs):
        output = originals["clone"](tensor, *args, **kwargs)
        if stage == "exact" and selected(tensor):
            after_copy(tensor, output)
        return output

    def to(tensor, *args, **kwargs):
        output = originals["to"](tensor, *args, **kwargs)
        if kwargs.get("non_blocking") and selected(tensor):
            after_copy(tensor, output)
        return output

    def copy(destination, tensor, *args, **kwargs):
        output = originals["copy_"](destination, tensor, *args, **kwargs)
        if selected(tensor):
            after_copy(tensor, destination)
        return output

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    full_rows = None if stage == "exact" else torch.tensor([True, True] if stage == "phase" else [False, True])
    try:
        with pytest.raises(RuntimeError, match="after real NPU copy"), pool.operation(lambda: None) as owner:
            pool.begin_native(owner)
            prepare_dense_draft_read(
                source,
                exact_device=exact,
                row_indices=rows,
                upper_cpu=torch.tensor([1154, 1411], dtype=torch.int32),
                window_size=1024,
                anchor_span=128,
                query_width=4,
                causal=False,
                full_rows_cpu=full_rows,
                retain=lambda *items: pool.retain(owner, *items),
            )
        assert len(submitted) == 1
        assert pool.poisoned and owner.state.name == "QUARANTINED"
        assert all(any(item is tensor for item in owner.retained) for tensor in submitted[0])
        with pytest.raises(RuntimeError, match="quarantined"):
            pool.acquire()
    finally:
        # Diagnostic cleanup only. The product never receives these events
        # and must keep its unknown-completion owner quarantined.
        for event in events:
            event.synchronize()
    torch.testing.assert_close(table, before, atol=0, rtol=0)


def test_real_metadata_fill_keeps_destination_quarantined(monkeypatch):
    pool = DraftWindowOwnerPool(1)
    read = DenseDraftRead(
        torch.zeros(2, 4, dtype=torch.int64, device="npu"),
        torch.tensor([3, 4], dtype=torch.int32, device="npu"),
        torch.zeros(2, dtype=torch.int32, device="npu"),
        torch.zeros(2, 1, 1, 4, dtype=torch.bool, device="npu"),
        4,
        1,
        lambda *items: pool.retain(owner, *items),
    )
    original = torch.Tensor.__setitem__
    events, submitted = [], []

    def filled(destination, index, values):
        result = original(destination, index, values)
        if values is read.seq_lens:
            event = torch.npu.Event()
            event.record(torch.npu.current_stream())
            events.append(event)
            submitted.append(destination)
            assert any(item is destination for item in owner.retained), "metadata destination is not owned"
            raise RuntimeError("deliberate Host failure after real metadata fill")
        return result

    monkeypatch.setattr(torch.Tensor, "__setitem__", filled)
    try:
        with pytest.raises(RuntimeError, match="after real metadata fill"), pool.operation(lambda: None) as owner:
            pool.begin_native(owner)
            pool.retain(owner, read)
            read.apply(SimpleNamespace(num_reqs=3))
        assert len(submitted) == 1
        assert pool.poisoned and owner.state.name == "QUARANTINED"
        assert any(item is submitted[0] for item in owner.retained)
        with pytest.raises(RuntimeError, match="quarantined"):
            pool.acquire()
    finally:
        for event in events:
            event.synchronize()
