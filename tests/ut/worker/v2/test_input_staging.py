# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.input_staging import BatchInputStaging


@pytest.fixture
def staging(monkeypatch):
    stream = object()
    monkeypatch.setattr(torch.npu, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.npu, "Event", lambda: MagicMock(query=MagicMock(return_value=False)))
    return BatchInputStaging(4, torch.device("cpu"))


def inputs(count, offset=0):
    return (
        np.arange(count, dtype=np.intp) + offset,
        np.arange(count + 1, dtype=np.int32) * 8,
        np.array([0, 8, 16, 16, 16, 16], dtype=np.int32),
        torch.zeros(6, dtype=torch.int32),
    )


def test_busy_pinned_banks_fall_back_without_overwriting_pending_inputs(staging):
    first = inputs(2)
    idx, logits = staging.copy(*first)
    assert idx.tolist() == first[0].tolist()
    assert idx.numpy().dtype == first[0].dtype
    assert logits.dtype == torch.int32
    assert logits.tolist() == first[1].tolist()
    assert first[3].tolist() == first[2].tolist()
    first_host = staging.banks[0][0].clone()
    staging.copy(*inputs(4, 10))
    second_host = staging.banks[1][0].clone()
    assert staging.copy(*inputs(1, 20)) is None
    torch.testing.assert_close(staging.banks[0][0], first_host)
    torch.testing.assert_close(staging.banks[1][0], second_host)
    assert len(staging.banks) == 2
    for _, _, event in staging.banks:
        event.synchronize.assert_not_called()


def test_completed_bank_reuses_storage_and_keeps_cpu_batch_arrays_independent(staging):
    old = inputs(4, 10)
    idx, logits = staging.copy(*old)
    host, device, event = staging.banks[0]
    pointer = device.data_ptr()
    event.query.return_value = True
    new = inputs(1, 20)
    new_query_pointer = new[3].data_ptr()
    idx, logits = staging.copy(*new)
    assert idx.data_ptr() == pointer
    assert idx.tolist() == [20]
    assert logits.tolist() == [0, 8]
    assert new[3].data_ptr() == new_query_pointer
    assert new[3].tolist() == new[2].tolist()
    assert old[0].tolist() == [10, 11, 12, 13]
    assert old[1].tolist() == [0, 8, 16, 24, 32]
    assert staging.banks[0][0] is host
    assert len(staging.banks) == 1


def test_different_execution_stream_uses_original_path(staging, monkeypatch):
    staging.copy(*inputs(2))
    before = staging.banks[0][0].clone()
    monkeypatch.setattr(torch.npu, "current_stream", lambda: object())
    assert staging.copy(*inputs(2, 20)) is None
    torch.testing.assert_close(staging.banks[0][0], before)


@pytest.mark.parametrize("oversized,bad_dtype", [(True, False), (False, True)])
def test_unsupported_layout_falls_back_before_allocating(staging, oversized, bad_dtype):
    args = list(inputs(5 if oversized else 2))
    if bad_dtype:
        args[0] = args[0].astype(np.float64)
    assert staging.copy(*args) is None
    assert not staging.banks
