# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The NPU stream patch changes exactly three CUDA entry points."""

from types import SimpleNamespace

import torch

from vllm_ascend.patch.torch_streams import patch_cuda_streams


def test_stream_patch_is_minimal_and_reapplicable(monkeypatch):
    names = ("Stream", "stream", "current_stream")
    for name in names:
        monkeypatch.setattr(torch.cuda, name, object())
    before = dict(vars(torch.cuda))
    event = torch.Event
    accelerator = torch.accelerator.current_stream
    npu = SimpleNamespace(**{name: object() for name in names})
    monkeypatch.setattr(torch, "npu", npu, raising=False)

    patch_cuda_streams()
    changed = {name for name, value in vars(torch.cuda).items() if before.get(name) is not value}
    assert changed == set(names)
    for name in names:
        assert getattr(torch.cuda, name) is getattr(npu, name)
    assert torch.Event is event
    assert torch.accelerator.current_stream is accelerator

    patch_cuda_streams()
    for name in names:
        assert getattr(torch.cuda, name) is getattr(npu, name)

    # A later NPU entry point must refresh aliases after another wrapper has
    # changed them, rather than treating import-time installation as sufficient.
    npu.stream = object()
    patch_cuda_streams()
    assert torch.cuda.stream is npu.stream
