# SPDX-License-Identifier: Apache-2.0

import ctypes
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp.ple_bm_layout import mapping_pointers, publish_rows
from vllm_ascend.models.qwen4_exp.ple_layout import PLELayout


def test_padded_publication_and_mapping_pointers():
    layout = PLELayout(rows=5, row_bytes=6, mapping_bytes=16)
    memory = ctypes.create_string_buffer(layout.allocation_bytes)
    base = ctypes.addressof(memory)
    calls = []

    def device_pointer(offset, size):
        calls.append((offset, size))
        return base + offset

    buffer = SimpleNamespace(
        owner=True,
        closed=False,
        _ready=True,
        size=layout.allocation_bytes,
        dtype=torch.bfloat16,
        width=3,
        gva=0,
        device="cpu",
        runtime=SimpleNamespace(bm=SimpleNamespace(BmMemType=SimpleNamespace(LOCAL_HOST=0))),
        handle=SimpleNamespace(gva_to_va=lambda address, _: base + address),
        device_pointer=device_pointer,
    )
    rows = torch.arange(15, dtype=torch.float32).reshape(5, 3).to(torch.bfloat16)
    publish_rows(buffer, layout, 0, rows)
    raw = bytes(memory)
    for i in range(5):
        assert raw[layout.offset(i) : layout.offset(i) + 6] == rows[i].view(torch.uint8).numpy().tobytes()
    assert raw[12:16] == bytes(4)
    assert mapping_pointers(buffer, layout).tolist() == [base, base + 16, base + 32]
    assert calls == [(0, 12), (16, 12), (32, 6)]
    with pytest.raises(ValueError, match="outside the table"):
        publish_rows(buffer, layout, 2, rows)
