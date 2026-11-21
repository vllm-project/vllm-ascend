# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys

import pytest
import torch

from vllm_ascend.ops import packaged_attention as ops


@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("tokens", [0, 2])
def test_native_mla_retires_same_stream_use_after_device_wait(monkeypatch, capturing, tokens):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    q = torch.ones((tokens, 4, 8), dtype=torch.bfloat16)
    order = []
    stream = MagicMock()
    event = object()

    def record_event():
        order.append("record")
        return event

    stream.record_event.side_effect = record_event
    stream.wait_event.side_effect = lambda e: order.append("wait") if e is event else None
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            is_current_stream_capturing=lambda: capturing,
            current_stream=lambda device: stream,
        ),
    )

    def native(q, **kwargs):
        order.append("native")
        kwargs["out"].copy_(q)

    erased = []

    def erase(tensor, owner):
        assert owner is stream
        order.append("erase")
        erased.append(tensor)

    monkeypatch.setattr(ops, "_get_dsl_ops", lambda name: (native, None))
    monkeypatch.setitem(sys.modules, "torch_npu.npu.utils", SimpleNamespace(_erase_stream=erase))
    out, _ = ops._mixed_quant_sparse_flash_mla_impl(q, quant_mode=0)
    assert torch.equal(out, q)
    assert order == (["native"] if capturing else ["native", "record", "wait", "erase", "erase"])
    if not capturing:
        assert erased[0] is q and erased[1] is out
    stream.synchronize.assert_not_called()
