# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import enable_custom_op


@pytest.mark.skipif(
    not get_current_hardware_profile().supports(HardwareCapability.HC_POST_PRE_NORM),
    reason="HcPostPreNorm is configured for Ascend A2/A3",
)
@pytest.mark.parametrize(("num_tokens", "capture_graph"), [(1, False), (33, False), (36, True), (129, True)])
def test_hc_post_pre_norm_matches_split_path(num_tokens, capture_graph):
    assert enable_custom_op()
    torch.manual_seed(13)
    hidden_size = 5120
    streams = 4
    hc_eps = 1e-6
    norm_eps = 1e-20
    kwargs = dict(hc_sinkhorn_iters=20, hc_eps=hc_eps)
    attn_out = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="npu")
    residual = torch.randn(num_tokens, streams, hidden_size, dtype=torch.bfloat16, device="npu")
    attn_post = torch.rand(num_tokens, streams, device="npu")
    attn_comb = torch.rand(num_tokens, streams, streams, device="npu").softmax(-1)
    attn_pre = torch.rand(num_tokens, streams, device="npu")
    fn = torch.randn(24, streams * hidden_size, device="npu") * 0.01
    scale = torch.rand(3, device="npu")
    base = torch.rand(24, device="npu")
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device="npu")
    ops = torch.ops._C_ascend
    args = (attn_out, residual, attn_post, attn_comb, attn_pre, fn, scale, base, weight)

    def fused(**output_options):
        return ops.npu_hc_post_pre_norm(
            *args,
            hc_norm_eps=norm_eps,
            ffn_norm_eps=norm_eps,
            **kwargs,
            **output_options,
        )

    def check(actual):
        updated = ops.npu_hc_post(
            attn_out.unsqueeze(0), residual.unsqueeze(0), attn_post.unsqueeze(0), attn_comb.unsqueeze(0)
        ).squeeze(0)
        mixed, post, comb, pre = ops.npu_hc_pre_v3(updated, fn, scale, base, attn_pre, norm_eps=norm_eps, **kwargs)
        y, y_fp32 = ops.npu_rms_norm_cast(mixed, weight, norm_eps)
        torch.testing.assert_close(actual[2], updated, rtol=0, atol=0)
        torch.testing.assert_close(actual[0], y, rtol=0.016, atol=0.002)
        torch.testing.assert_close(actual[1], y_fp32, rtol=0.016, atol=0.002)
        torch.testing.assert_close(actual[1], actual[0].float(), rtol=0, atol=0)
        for result, expected in zip(actual[3:], (post, comb, pre), strict=True):
            torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def check_bf16_only(actual, expected):
        assert actual[1] is None
        for result, reference in zip(actual[:1] + actual[2:], expected[:1] + expected[2:], strict=True):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)

    full = fused()
    check(full)
    check_bf16_only(fused(return_fp32=False), full)

    meta_args = tuple(torch.empty_like(arg, device="meta") for arg in args)
    full_meta = ops.npu_hc_post_pre_norm(*meta_args)
    bf16_meta = ops.npu_hc_post_pre_norm(*meta_args, return_fp32=False)
    assert full_meta[1].shape == (num_tokens, hidden_size)
    assert full_meta[1].dtype == torch.float32
    assert bf16_meta[1] is None
    for actual, expected in zip(bf16_meta[:1] + bf16_meta[2:], full_meta[:1] + full_meta[2:], strict=True):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype

    if capture_graph:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            full = fused()
            bf16_only = fused(return_fp32=False)
        for _ in range(2):
            attn_out.mul_(-1)
            graph.replay()
            check(full)
            check_bf16_only(bf16_only, full)
