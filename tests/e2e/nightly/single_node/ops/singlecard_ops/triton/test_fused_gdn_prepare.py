# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.fla.fused_gdn_prepare import fused_gdn_prepare, fused_gdn_prepare_prefill_kernel


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
@pytest.mark.parametrize("tokens,padding", [(0, 0), (1, 0), (17, 17), (1023, 0), (1024, 0), (1025, 17)])
def test_kernels_exact_copy_zero(tp, dtype, tokens, padding):
    width, heads = 8192 // tp, 32 // tp
    x = torch.randn(tokens + 1, width + heads * 128 + padding, dtype=dtype, device="npu")[1:, : width + heads * 128]
    ba = torch.randn(tokens + 1, heads * 2 + padding, dtype=torch.float32, device="npu")[1:, : heads * 2]
    before_x, before_ba = x.clone(), ba.clone()
    outputs = fused_gdn_prepare(x, ba, width, heads, 128)
    check_outputs(outputs, x, ba, width, heads)
    torch.testing.assert_close(x, before_x, rtol=0, atol=0)
    torch.testing.assert_close(ba, before_ba, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [17, 1025])
def test_aclgraph_replay_refreshes_outputs(tokens):
    x = torch.randn(tokens, 12288, dtype=torch.bfloat16, device="npu")
    ba = torch.randn(tokens, 64, dtype=torch.float32, device="npu")
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


@pytest.mark.parametrize("tokens", [17, 1025])
@pytest.mark.parametrize(
    "dtype,ba_dtype", [(torch.float32, torch.float32), (torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16)]
)
def test_dtype_and_prefill_column_tail(tokens, dtype, ba_dtype):
    width, heads = 6143, 32
    x = torch.randn(tokens, width + heads * 128, dtype=dtype, device="npu")
    ba = torch.randn(tokens, 2 * heads, dtype=ba_dtype, device="npu")
    check_outputs(fused_gdn_prepare(x, ba, width, heads, 128), x, ba, width, heads)


def test_prefill_token_counts_reuse_compiled_kernel():
    """Changing only the row mask must reuse the compiled prefill artifact."""
    # Reuse storage to hold pointer alignment, dtype and strides constant.
    x = torch.randn(1040, 12288, dtype=torch.bfloat16, device="npu")
    ba = torch.randn(1040, 64, dtype=torch.float32, device="npu")
    qkv = torch.empty(1040, 8192, dtype=x.dtype, device=x.device)
    b = torch.empty(1040, 32, dtype=ba.dtype, device=ba.device)
    a = torch.empty_like(b)
    core = torch.empty(1040, 32, 128, dtype=x.dtype, device=x.device)
    compiled = []
    for tokens in (1024, 1025, 1031, 1040):
        compiled.append(
            fused_gdn_prepare_prefill_kernel.warmup(
                qkv,
                b,
                a,
                core,
                x,
                ba,
                tokens,
                x.stride(0),
                ba.stride(0),
                8192,
                32,
                4096,
                BLOCK_ROWS=4,
                BLOCK_WIDTH=8192,
                BLOCK_HEADS=32,
                grid=((tokens + 3) // 4,),
            )
        )
        check_outputs(fused_gdn_prepare(x[:tokens], ba[:tokens], 8192, 32, 128), x[:tokens], ba[:tokens], 8192, 32)
    assert compiled[0] is not None
    assert all(kernel is compiled[0] for kernel in compiled[1:])
