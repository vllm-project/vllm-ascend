# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU parity and replay checks for the fused DSpark sampler."""

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.ops.triton.sample.fused_add_argmax import TwoStageAddArgmax  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Requires an NPU")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,vocab", [(0, 1), (1, 1), (3, 17), (4, 4096), (2, 4097), (16, 131072)])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_add_argmax_parity_and_replay(dtype, rows, vocab, use_graph):
    capacity = max(rows, 4) + 3
    # Reproduce base_logits[:, step]: non-contiguous rows, unit vocab stride.
    base_owner = torch.randn(capacity, 5, vocab, device="npu", dtype=dtype)
    bias_owner = torch.randn(capacity, vocab + 13, device="npu", dtype=dtype)
    a, b = base_owner[:rows, 2], bias_owner[:rows, :vocab]
    output_owner = torch.full((capacity, 5), -7, device="npu", dtype=torch.int32)
    next_owner = torch.full((5, capacity), -7, device="npu", dtype=torch.int64)
    out, next_ids = output_owner[:rows, 2], next_owner[2, :rows]
    ws = TwoStageAddArgmax(capacity, vocab, a.device)
    pointers = ws.values.data_ptr(), ws.indices.data_ptr(), next_ids.data_ptr()

    expected = (a + b).argmax(-1)
    ws.run(a, b, out)
    ws.run(a, b, next_ids)
    torch.testing.assert_close(out.long(), expected, rtol=0, atol=0)
    torch.testing.assert_close(next_ids, expected, rtol=0, atol=0)

    graph = None
    if use_graph and rows:
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            ws.run(a, b, out)
            ws.run(a, b, next_ids)

    # New contents at the same addresses must replace stale partial reductions.
    a.zero_()
    b.zero_()
    if rows:
        a[:, -1] = 1
    if graph is None:
        ws.run(a, b, out)
        ws.run(a, b, next_ids)
    else:
        graph.replay()
    expected = (a + b).argmax(-1)
    torch.testing.assert_close(out.long(), expected, rtol=0, atol=0)
    torch.testing.assert_close(next_ids, expected, rtol=0, atol=0)
    assert pointers == (ws.values.data_ptr(), ws.indices.data_ptr(), next_ids.data_ptr())
    assert torch.all(output_owner[rows:] == -7)
    assert torch.all(output_owner[:, :2] == -7)
    assert torch.all(output_owner[:, 3:] == -7)
    assert torch.all(next_owner[:2] == -7)
    assert torch.all(next_owner[3:] == -7)
    assert torch.all(next_owner[2, rows:] == -7)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("block", [2048, 4096, 8192])
@torch.inference_mode()
def test_add_argmax_special_values_and_rounding(dtype, block):
    vocab = 8193
    a = torch.full((6, vocab), -float("inf"), device="npu", dtype=dtype)
    b = torch.zeros_like(a)
    # Across-block ties, NaN priority/first NaN, infinities, and an all -inf row.
    a[0, 3] = a[0, 8192] = 5
    a[1, 7] = a[1, 8192] = float("nan")
    a[1, 4] = float("inf")
    a[2, 9] = a[2, 8192] = float("inf")
    a[4, 12] = float("inf")
    b[4, 12] = -float("inf")  # Source addition produces NaN.
    a[5, 0:2] = 1
    if dtype == torch.float16:
        b[5, 0], b[5, 1] = 2**-12, 2**-11
    elif dtype == torch.bfloat16:
        b[5, 0], b[5, 1] = 2**-10, 2**-9
    else:
        b[5, 0], b[5, 1] = 0.25, 0.5
    expected = (a + b).argmax(-1)
    out = torch.empty(6, device="npu", dtype=torch.int32)
    next_ids = torch.empty(6, device="npu", dtype=torch.int64)
    # Keep the original two-dimensional program grid.
    ws = TwoStageAddArgmax(6, vocab, a.device, block=block)
    ws.run(a, b, out)
    ws.run(a, b, next_ids)
    torch.testing.assert_close(out.long(), expected, rtol=0, atol=0)
    torch.testing.assert_close(next_ids, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_add_argmax_unsupported_layout_returns_without_writing():
    ws = TwoStageAddArgmax(2, 16, torch.device("npu"))
    a = torch.ones(2, 32, device="npu")[:, ::2]
    b = torch.ones_like(a)
    out = torch.full((2,), -7, device="npu", dtype=torch.int32)
    next_ids = torch.full((2,), -7, device="npu", dtype=torch.int64)
    assert not ws.try_run(a, b, out)
    assert not ws.try_run(a, b, next_ids)
    torch.testing.assert_close(out, torch.full_like(out, -7), rtol=0, atol=0)
    torch.testing.assert_close(next_ids, torch.full_like(next_ids, -7), rtol=0, atol=0)
