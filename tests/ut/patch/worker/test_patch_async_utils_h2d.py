# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.utils import torch_utils as vllm_torch_utils

import vllm_ascend.patch.worker.patch_async_utils  # noqa: F401
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _make_delayed_staging_harness(monkeypatch):
    """Simulate pinned H2D staging where every non-blocking DMA executes lazily on flush()."""
    orig_empty = torch.empty
    orig_tensor = torch.tensor
    orig_to = torch.Tensor.to
    orig_copy_ = torch.Tensor.copy_

    staged_ptrs: list[int] = []
    deferred_transfers: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _fake_empty(*args, **kwargs):
        pinned = kwargs.pop("pin_memory", False)
        t = orig_empty(*args, **kwargs)
        if pinned:
            staged_ptrs.append(t.data_ptr())
        return t

    def _fake_tensor(*args, **kwargs):
        pinned = kwargs.pop("pin_memory", False)
        t = orig_tensor(*args, **kwargs)
        if pinned:
            staged_ptrs.append(t.data_ptr())
        return t

    def _delayed_to(self, *args, **kwargs):
        if kwargs.get("non_blocking", False):
            dst = orig_empty(self.shape, dtype=kwargs.get("dtype") or self.dtype, device="cpu")
            deferred_transfers.append((dst, self))
            return dst
        return orig_to(self, *args, **kwargs)

    def _delayed_copy_(self, src, non_blocking=False):
        if non_blocking:
            deferred_transfers.append((self, src))
            return self
        return orig_copy_(self, src, non_blocking=False)

    monkeypatch.setattr(vllm_torch_utils, "is_pin_memory_available", lambda: True)
    monkeypatch.setattr(torch, "empty", _fake_empty)
    monkeypatch.setattr(torch, "tensor", _fake_tensor)
    monkeypatch.setattr(torch.Tensor, "to", _delayed_to)
    monkeypatch.setattr(torch.Tensor, "copy_", _delayed_copy_)

    def flush() -> None:
        while deferred_transfers:
            dst, src = deferred_transfers.pop(0)
            orig_copy_(dst, src, non_blocking=False)

    return staged_ptrs, deferred_transfers, flush


@pytest.mark.parametrize("input_kind", ["tensor", "list", "numpy"])
def test_async_tensor_h2d_wraparound_delayed_transfer_preserves_payloads(monkeypatch, input_kind):
    """Submitting >16 same-dtype H2D copies with delayed DMA must not overwrite earlier payloads."""
    staged_ptrs, deferred, flush = _make_delayed_staging_harness(monkeypatch)
    num_calls = 24
    expected = []
    outputs = []
    reusable_np = np.zeros(8, dtype=np.int32)

    for i in range(num_calls):
        vals = [i * 100 + j for j in range(8)]
        expected.append(vals)
        if input_kind == "tensor":
            payload = torch.tensor(vals, dtype=torch.int32)
            out = vllm_torch_utils.async_tensor_h2d(payload, device="cpu", dtype=torch.int32)
            payload.fill_(-1)
        elif input_kind == "list":
            out = vllm_torch_utils.async_tensor_h2d(vals, device="cpu", dtype=torch.int32)
        else:
            reusable_np[:] = vals
            out = vllm_torch_utils.async_tensor_h2d(reusable_np, device="cpu", dtype=torch.int32)
            reusable_np.fill(-1)
        outputs.append(out)

    assert len(deferred) == num_calls
    assert len(staged_ptrs) == num_calls
    flush()

    for i, out in enumerate(outputs):
        assert out.tolist() == expected[i], f"call {i} payload corrupted after wraparound"


def test_model_runner_async_h2d_1d_wraparound_delayed_transfer_out_none(monkeypatch):
    """Submitting >4 same-key _async_h2d_1d calls (out=None) with delayed DMA preserves every payload."""
    staged_ptrs, deferred, flush = _make_delayed_staging_harness(monkeypatch)
    runner = SimpleNamespace(device=torch.device("cpu"), max_num_reqs=32)

    num_calls = 10
    reusable_np = np.zeros(6, dtype=np.int32)
    expected = []
    outputs = []

    for i in range(num_calls):
        vals = [i * 10 + j for j in range(6)]
        expected.append(vals)
        reusable_np[:] = vals
        out = NPUModelRunner._async_h2d_1d(runner, "idx_mapping", reusable_np)
        reusable_np.fill(-999)
        outputs.append(out)

    assert len(deferred) == num_calls
    assert len(staged_ptrs) == num_calls
    assert len({t.data_ptr() for t in outputs}) == num_calls
    flush()

    for i, out in enumerate(outputs):
        assert out.tolist() == expected[i], f"call {i} payload corrupted after >4 wraparound"


def test_model_runner_async_h2d_1d_wraparound_delayed_transfer_with_preallocated_out(monkeypatch):
    """Submitting >4 same-key _async_h2d_1d calls with preallocated prefix out buffers preserves payloads."""
    staged_ptrs, deferred, flush = _make_delayed_staging_harness(monkeypatch)
    runner = SimpleNamespace(device=torch.device("cpu"), max_num_reqs=32)

    num_calls = 10
    reusable_np = np.zeros(5, dtype=np.int32)
    expected = []
    out_buffers = [torch.full((12,), -7, dtype=torch.int32) for _ in range(num_calls)]
    returned = []

    for i in range(num_calls):
        vals = [i * 50 + j for j in range(5)]
        expected.append(vals)
        reusable_np[:] = vals
        ret = NPUModelRunner._async_h2d_1d(runner, "query_start_loc", reusable_np, out=out_buffers[i])
        reusable_np.fill(-555)
        assert ret is out_buffers[i]
        returned.append(ret)

    assert len(deferred) == num_calls
    assert len(staged_ptrs) == num_calls
    flush()

    for i, out in enumerate(returned):
        assert out[:5].tolist() == expected[i], f"call {i} prefix corrupted after >4 wraparound"
        assert out[5:].tolist() == [-7] * 7
