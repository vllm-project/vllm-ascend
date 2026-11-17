# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.deepseek_v41 import mhc
from vllm_ascend.ops.triton.mhc_post_collapse_norm import MHC_FUSION_HIDDEN_SIZE
from vllm_ascend.utils import enable_custom_op

_HC_MULT = 4
_SINKHORN_ITERS = 20
_EPS = 1e-6
_ACCURACY = {
    torch.bfloat16: (2**-6, 2**-6, 1.0),
    torch.float32: (2**-10, 2**-16, 0.01),
}


def _native_boundary(x, residual, post, comb, pre, weight, fn, scale, base, *, to_moe):
    ops = torch.ops._C_ascend
    updated = ops.npu_hc_post(x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)).squeeze(0)
    collapsed, next_post, next_comb, next_pre = ops.npu_hc_pre_v3(
        updated, fn, scale, base, pre, hc_mult=_HC_MULT, hc_sinkhorn_iters=_SINKHORN_ITERS, norm_eps=_EPS, hc_eps=_EPS
    )
    if to_moe:
        normalized, _ = ops.npu_rms_norm_cast(collapsed, weight, _EPS)
    else:
        normalized, _ = torch_npu.npu_rms_norm(collapsed, weight, _EPS)
    return updated, normalized, next_post, next_comb, next_pre


def _check_boundary(actual, expected, label, record_property):
    names = ("residual", "normalized", "post", "comb", "pre")
    for name, result, reference in zip(names, actual, expected, strict=True):
        assert (result.shape, result.dtype) == (reference.shape, reference.dtype), (label, name)
        rtol, atol, max_error = _ACCURACY[reference.dtype]
        golden = reference.cpu().float()
        error = (result.cpu().float() - golden).abs()
        pass_ratio = (error <= atol + rtol * golden.abs()).float().mean().item()
        max_abs = error.max().item()
        record_property(f"{label}/{name}", f"pass_ratio={pass_ratio:.6f}, max_abs_error={max_abs:.8g}")
        assert pass_ratio >= 0.99 and max_abs <= max_error, (label, name, pass_ratio, max_abs)


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A2, reason="mHC boundary fusion is enabled on A2")
@torch.inference_mode()
def test_dsv41_mhc_boundary_full_graph_replay(record_property):
    assert enable_custom_op()
    torch.manual_seed(1024)
    rows, hidden_size = mhc.MHC_MAX_ROWS, MHC_FUSION_HIDDEN_SIZE
    shape = (rows, hidden_size)
    attn_out = torch.randn(shape, device="npu", dtype=torch.bfloat16)
    moe_out = torch.randn_like(attn_out)
    residual = torch.randn((rows, _HC_MULT, hidden_size), device="npu", dtype=torch.bfloat16)
    post = torch.rand((rows, _HC_MULT), device="npu", dtype=torch.float32) * 2
    comb = torch.rand((rows, _HC_MULT, _HC_MULT), device="npu", dtype=torch.float32).softmax(-1)
    pre = torch.rand_like(post)
    weight = torch.rand(hidden_size, device="npu", dtype=torch.bfloat16) + 0.5
    fan_in = _HC_MULT * hidden_size
    fn = torch.randn((2 * _HC_MULT + _HC_MULT**2, fan_in), device="npu", dtype=torch.float32) / fan_in**0.5
    scale = torch.rand(3, device="npu", dtype=torch.float32)
    base = torch.randn(fn.shape[0], device="npu", dtype=torch.float32) * 0.1
    params = (weight, fn, scale, base)
    fused_params = (*params, _SINKHORN_ITERS, _EPS, _EPS)
    ops = torch.ops.vllm

    def run_boundaries():
        # Enter with prepared Attention gates. The first producer joins the
        # auxiliary stream to this graph before any coefficient wait.
        moe = ops.dsv41_mhc_moe_input(attn_out, residual, post, comb, pre, *fused_params, True, False)
        updated, _, _, next_post, next_comb, next_pre = moe
        attention = ops.dsv41_mhc_attention_input(moe_out, updated, next_post, next_comb, next_pre, *fused_params, True)
        # Exercise the terminal join used when no following boundary is fused.
        updated, normalized, next_post, next_comb, _ = attention
        tail = ops.dsv41_mhc_finish_overlap(normalized, updated, next_post, next_comb)
        return moe, attention, tail

    run_boundaries()  # Compile the two Triton variants before capture.
    mhc._coefficient_stream(residual.device)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with (
        torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True),
        mhc.record_mhc_overlap(),
    ):
        moe, attention, _tail = run_boundaries()

    for replay in range(3):
        for tensor in (attn_out, moe_out, residual):
            tensor.normal_()
        graph.replay()
        torch.npu.synchronize()

        updated, normalized, routing, next_post, next_comb, next_pre = moe
        expected_moe = _native_boundary(attn_out, residual, post, comb, pre, *params, to_moe=True)
        _check_boundary(
            (updated, normalized, next_post, next_comb, next_pre), expected_moe, f"moe/{replay}", record_property
        )
        torch.testing.assert_close(routing, normalized.float(), rtol=0, atol=0)

        # Use the same incoming state as the captured Attention boundary to
        # measure this boundary's error without compounding earlier rounding.
        expected_attention = _native_boundary(moe_out, updated, next_post, next_comb, next_pre, *params, to_moe=False)
        _check_boundary(attention, expected_attention, f"attention/{replay}", record_property)
