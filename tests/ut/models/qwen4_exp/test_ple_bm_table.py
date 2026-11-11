# SPDX-License-Identifier: Apache-2.0

import ctypes
from types import SimpleNamespace

import torch

from vllm_ascend.models.qwen4_exp import ple_bm_table
from vllm_ascend.models.qwen4_exp.ple_layout import PLELayout


def test_load_and_attach_padded_table(monkeypatch):
    original_layout = PLELayout
    monkeypatch.setattr(ple_bm_table, "PLELayout", lambda rows, row_bytes: original_layout(rows, row_bytes, 16))
    values = torch.arange(15, dtype=torch.float32).reshape(5, 3).to(torch.bfloat16)
    checkpoint = SimpleNamespace(
        rows=5,
        width=3,
        copy_rows=lambda out, start: out.copy_(values[start : start + len(out)]),
    )
    memory = ctypes.create_string_buffer(48)
    base = ctypes.addressof(memory)
    buffer = SimpleNamespace(
        owner=True,
        closed=False,
        _ready=True,
        size=48,
        dtype=torch.bfloat16,
        width=3,
        gva=0,
        device="cpu",
        runtime=SimpleNamespace(bm=SimpleNamespace(BmMemType=SimpleNamespace(LOCAL_HOST=0))),
        handle=SimpleNamespace(gva_to_va=lambda address, _: base + address),
        device_pointer=lambda offset, size: base + offset,
    )
    calls = []
    owner = SimpleNamespace(add=lambda key, rows, width, dtype: calls.append((key, rows, width, dtype)) or buffer)
    layout = ple_bm_table.load_table(owner, checkpoint, "weights", chunk_rows=2)
    assert calls == [("weights", 8, 3, torch.bfloat16)]
    raw = bytes(memory)
    for i in range(5):
        assert raw[layout.offset(i) : layout.offset(i) + 6] == values[i].view(torch.uint8).numpy().tobytes()
    attached = []
    client = SimpleNamespace(view=lambda *args, **kwargs: attached.append((args, kwargs)), buffers={"weights": buffer})
    pointers = ple_bm_table.attach_table(client, "weights", layout, 3)
    assert pointers.tolist() == [base, base + 16, base + 32]
    assert attached[0][1]["rows"] == 8
