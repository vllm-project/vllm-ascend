# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.fla.fused_gdn_prepare import fused_gdn_prepare


def check_outputs(outputs, x, ba, width, heads):
    qkv, z, b, a, core = outputs
    for actual, expected in (
        (qkv, x[:, :width]),
        (z, x[:, width:].reshape(x.shape[0], heads, 128)),
        (b, ba[:, :heads]),
        (a, ba[:, heads:]),
        (core, torch.zeros_like(core)),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert z.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert all(t.is_contiguous() for t in (qkv, b, a, core))
    assert b.dtype == a.dtype == ba.dtype


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens,padding", [(0, 0), (1, 0), (17, 17), (1025, 17)])
def test_generic_kernel_exact_copy_zero(tp, dtype, tokens, padding):
    width, heads = 8192 // tp, 32 // tp
    x = torch.randn(tokens + 1, width + heads * 128 + padding, dtype=dtype, device="npu")[1:, : width + heads * 128]
    ba = torch.randn(tokens + 1, heads * 2 + padding, dtype=torch.float32, device="npu")[1:, : heads * 2]
    before_x, before_ba = x.clone(), ba.clone()
    outputs = fused_gdn_prepare(x, ba, width, heads, 128)
    check_outputs(outputs, x, ba, width, heads)
    torch.testing.assert_close(x, before_x, rtol=0, atol=0)
    torch.testing.assert_close(ba, before_ba, rtol=0, atol=0)


def test_generic_aclgraph_replay_refreshes_outputs():
    x = torch.randn(1025, 12288, dtype=torch.bfloat16, device="npu")
    ba = torch.randn(1025, 64, dtype=torch.float32, device="npu")
    # Compile the Triton specialization before capture.
    fused_gdn_prepare(x, ba, 8192, 32, 128)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = fused_gdn_prepare(x, ba, 8192, 32, 128)
    for _ in range(3):
        x.add_(1)
        ba.add_(1)
        outputs[-1].fill_(7)
        graph.replay()
        torch.npu.synchronize()
        check_outputs(outputs, x, ba, 8192, 32)


@pytest.mark.parametrize("width", [1023, 1024, 1025, 16385])
def test_column_tail_and_wide_projection(width):
    x = torch.randn(17, width + 4096, dtype=torch.bfloat16, device="npu")
    ba = torch.randn(17, 64, dtype=torch.float32, device="npu")
    check_outputs(fused_gdn_prepare(x, ba, width, 32, 128), x, ba, width, 32)
