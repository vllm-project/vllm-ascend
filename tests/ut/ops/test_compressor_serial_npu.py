# SPDX-License-Identifier: Apache-2.0
"""Small edge cases for the installed A5 serial extension, including replay."""

import pytest
import torch

pytest.importorskip("torch_npu")

from benchmarks.benchmark_compressor_serial import load_serial_module  # noqa: E402
from tests.ut.ops.compressor_serial_reference import compressor_reference  # noqa: E402


@pytest.mark.parametrize(
    "layout,cache_mode,dtype,rotary,optional",
    [
        ("bsh", 2, torch.float16, 1, False),
        ("bsh", 1, torch.bfloat16, 2, True),
        ("tnd", 2, torch.bfloat16, 1, False),
        ("tnd", 1, torch.float16, 2, False),
    ],
)
def test_serial_edges(layout, cache_mode, dtype, rotary, optional):
    torch.npu.set_device(0)
    torch.manual_seed(20261026)
    batch, sequence, hidden, dim, ratio, coff = 2, 5, 1024, 128, 4, 2
    width = coff * dim

    def random(shape, kind=torch.float32):
        return (torch.randn(shape) * 0.1).to(kind).npu()

    shape = (batch, sequence, hidden) if layout == "bsh" else (batch * sequence, hidden)
    x = random(shape, dtype)
    wk, wg = random((width, hidden), dtype), random((width, hidden), dtype)
    backing = random((8, 32 * 2 * width + 16))
    cache = backing[:, : 32 * 2 * width].view(8, 32, 2 * width)
    padding = backing[:, -16:].clone()
    ape, gamma = random((ratio, width)), random((dim,))
    rope_shape = (batch, 2, 64) if layout == "bsh" else (4, 64)
    sin, cos = random(rope_shape), random(rope_shape)
    table = None
    if not optional:
        table = torch.tensor([1, 3] if cache_mode == 2 else [[1], [3]], dtype=torch.int32, device="npu")
    cu = None if layout == "bsh" else torch.tensor([0, sequence, 2 * sequence], dtype=torch.int32, device="npu")
    used = None if optional else torch.tensor([5, 4], dtype=torch.int32, device="npu")
    starts = None if optional else torch.tensor([3, 2], dtype=torch.int32, device="npu")
    args = (x, wk, wg, cache, ape, gamma, sin, cos, table, cu, used, starts)
    cpu_args = tuple(None if tensor is None else tensor.cpu() for tensor in args)
    expected, state = compressor_reference(cpu_args, ratio=ratio, coff=coff, rotary=rotary, cache_mode=cache_mode)
    initial = cache.clone()
    plan = load_serial_module().CompressorSerialPlan.prepare(
        *args, cmp_ratio=ratio, coff=coff, rotary_mode=rotary, cache_mode=cache_mode
    )
    actual = plan(x)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(cache.cpu(), state, rtol=1e-4, atol=3e-4)
    torch.testing.assert_close(backing[:, -16:], padding, rtol=0, atol=0)
    cache.copy_(initial)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = plan(x)
    cache.copy_(initial)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(cache.cpu(), state, rtol=1e-4, atol=3e-4)
    torch.testing.assert_close(backing[:, -16:], padding, rtol=0, atol=0)
