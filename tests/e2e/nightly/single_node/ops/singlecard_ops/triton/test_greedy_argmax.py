# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.sample.greedy_argmax import greedy_argmax
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.fixture(scope="module", autouse=True)
def npu_environment():
    init_device_properties_triton()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [4095, 4096, 4097, 129280])
def test_edges(dtype, vocab):
    # Nonzero storage offset and row stride resembling logits[:, step, :].
    x = torch.randn(8, 5, vocab + 1, device="npu", dtype=dtype)[:, 2, 1:]
    x[0].zero_()
    x[1].fill_(-float("inf"))
    x[2].fill_(-float("inf"))
    x[2, -1] = 1
    x[3, [5, -1]] = float("inf")
    x[4, [5, -1]] = float("nan")
    x[5].fill_(float("nan"))
    x[6].fill_(-float("inf"))
    x[6, [vocab // 2, -1]] = 10
    torch.testing.assert_close(greedy_argmax(x), x.argmax(-1), rtol=0, atol=0)


@pytest.mark.parametrize("rows", [0, 1, 2, 4, 8, 16, 32, 64, 128])
def test_rows(rows):
    x = torch.randn(rows, 129280, device="npu", dtype=torch.float32)
    result = greedy_argmax(x)
    assert result.dtype == torch.int64
    torch.testing.assert_close(result, x.argmax(-1), rtol=0, atol=0)


def test_column_stride_fallback():
    x = torch.randn(16, 16384, device="npu")[:, ::2]
    torch.testing.assert_close(greedy_argmax(x), x.argmax(-1), rtol=0, atol=0)


def test_graph_replay_reads_new_values():
    x = torch.randn(16, 129280, device="npu", dtype=torch.bfloat16)
    # Compile and initialize device metadata before capturing.
    for _ in range(3):
        greedy_argmax(x)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = greedy_argmax(x)
    for winner in [0, 4096, 129279]:
        x.fill_(-float("inf"))
        x[:, winner] = 1
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(result, x.argmax(-1), rtol=0, atol=0)
