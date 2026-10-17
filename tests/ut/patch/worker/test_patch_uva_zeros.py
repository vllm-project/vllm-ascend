# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.patch.worker.patch_v2.patch_uva import _int64_storage, _zeros_on_npu


def test_uint64_npu_zeros_view_int64_storage(monkeypatch):
    viewed = {}

    class _FakeStorage:
        def view(self, dtype):
            viewed["dtype"] = dtype
            return "uint64-view"

    def _zeros(*args, **kwargs):
        viewed["zeros"] = (args, kwargs)
        return _FakeStorage()

    monkeypatch.setattr(torch, "zeros", _zeros)
    cpu = torch.empty(4, dtype=torch.uint64)

    assert _zeros_on_npu(cpu) == "uint64-view"
    assert viewed["zeros"] == ((cpu.shape,), {"dtype": torch.int64, "device": "npu"})
    assert viewed["dtype"] == torch.uint64


def test_supported_dtype_uses_zeros_like(monkeypatch):
    monkeypatch.setattr(torch, "zeros_like", lambda cpu, device: ("zeros_like", device, cpu.dtype))
    cpu = torch.empty(2, dtype=torch.int32)

    assert _zeros_on_npu(cpu) == ("zeros_like", "npu", torch.int32)


def test_uint64_int64_storage_keeps_pointer_bits():
    high_bit = torch.tensor([0, 2**63 + 3], dtype=torch.uint64)
    storage = torch.zeros(2, dtype=torch.int64)
    viewed = storage.view(torch.uint64)

    _int64_storage(viewed).copy_(_int64_storage(high_bit))

    assert viewed.tolist() == high_bit.tolist()
    assert _int64_storage(torch.zeros(2, dtype=torch.int32)).dtype == torch.int32
