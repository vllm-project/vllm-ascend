# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proposal ownership must precede a possibly partially submitted copy.

CPU dispatch substitutes expose the ownership boundary only. Actual NPU
enqueue, completion and mixed engine regression need separate device tests.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_dense import DenseDraftRead, prepare_dense_draft_read
from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import DraftWindowOwnerPool
from vllm_ascend.worker.v2.spec_decode.draft_window_read import DirectPrefixSource


@pytest.mark.parametrize("stage", ["exact", "phase", "cohort"])
def test_copy_failure_keeps_actual_input_and_output_owned(monkeypatch, stage):
    table = torch.arange(32, dtype=torch.int32).reshape(2, 16)
    source = DirectPrefixSource(table, (1, 0), (16, 16), 128)
    exact = torch.tensor([1151, 1408], dtype=torch.int32)
    pool = DraftWindowOwnerPool(1)
    submitted = []
    originals = {name: getattr(torch.Tensor, name) for name in ("clone", "to", "copy_")}

    def selected(tensor):
        if stage == "exact":
            return tensor.data_ptr() == exact.data_ptr()
        return tensor.dtype == (torch.bool if stage == "phase" else torch.int64)

    def after_copy(input_tensor, output_tensor):
        submitted.append((input_tensor, output_tensor))
        assert any(item is input_tensor for item in owner.retained), "copy source not owned before submission"
        assert any(item is output_tensor for item in owner.retained), "copy destination not owned before submission"
        raise RuntimeError("deliberate error after copy submission")

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
    with pytest.raises(RuntimeError, match="deliberate error"), pool.operation(lambda: None) as owner:
        pool.begin_native(owner)
        prepare_dense_draft_read(
            source,
            exact_device=exact,
            row_indices=torch.tensor([1, 0]),
            upper_cpu=exact + 3,
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


def test_metadata_length_copy_is_owned_before_partial_fill(monkeypatch):
    pool = DraftWindowOwnerPool(1)
    original = torch.Tensor.__setitem__
    read = DenseDraftRead(
        torch.zeros(2, 4, dtype=torch.int64),
        torch.tensor([3, 4], dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, 1, 1, 4, dtype=torch.bool),
        4,
        1,
        lambda *items: pool.retain(owner, *items),
    )
    submitted = []

    def filled(destination, index, values):
        result = original(destination, index, values)
        if values is read.seq_lens:
            submitted.append(destination)
            assert any(item is destination for item in owner.retained), "metadata copy destination is not owned"
            raise RuntimeError("deliberate error after metadata fill")
        return result

    monkeypatch.setattr(torch.Tensor, "__setitem__", filled)
    with pytest.raises(RuntimeError, match="after metadata fill"), pool.operation(lambda: None) as owner:
        pool.begin_native(owner)
        pool.retain(owner, read)
        read.apply(SimpleNamespace(num_reqs=3))
    assert len(submitted) == 1
    assert pool.poisoned and owner.state.name == "QUARANTINED"
    assert any(item is submitted[0] for item in owner.retained)
