# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.hc_post import fused_hc_post_add
from vllm_ascend.utils import enable_custom_op


def _inputs(tokens, dim, strided):
    torch.manual_seed(1024)
    shared = torch.randn(tokens, dim, dtype=torch.bfloat16, device="npu")
    routed = torch.randn_like(shared)
    residual = torch.randn(tokens, 4, dim, dtype=torch.bfloat16, device="npu")
    post = torch.randn(tokens, 4, dtype=torch.float32, device="npu")
    comb = torch.randn(tokens, 4, 4, dtype=torch.float32, device="npu")
    if strided:
        shared = torch.stack((shared, shared), dim=1)[:, 0]
        routed = torch.stack((routed, routed), dim=1)[:, 0]
        residual = torch.stack((residual, residual), dim=1)[:, 0]
    return shared, routed, residual, post, comb


def _native(shared, routed, residual, post, comb):
    x = torch._foreach_add([shared], [routed])[0]
    return torch.ops._C_ascend.npu_hc_post(
        x.contiguous().unsqueeze(0),
        residual.contiguous().unsqueeze(0),
        post.contiguous().unsqueeze(0),
        comb.contiguous().unsqueeze(0),
    ).squeeze(0)


@torch.inference_mode()
@pytest.mark.parametrize("tokens", [1, 6, 10, 12, 48, 96])
@pytest.mark.parametrize("dim", [256, 4096, 7168])
@pytest.mark.parametrize("strided", [False, True])
def test_hc_post_add_matches_native_a3(tokens, dim, strided):
    if not get_current_hardware_profile().supports(HardwareCapability.HC_POST_ADD_SEPARATE_ROUNDING):
        pytest.skip("requires the A3 HcPost separate-rounding contract")
    enable_custom_op()
    inputs = _inputs(tokens, dim, strided)
    originals = [t.clone() for t in inputs]
    actual = fused_hc_post_add(*inputs)
    expected = _native(*inputs)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for value, original in zip(inputs, originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("dim", [1, 129, 255, 257])
def test_hc_post_add_tail_matches_ordered_reference(dim):
    if not get_current_hardware_profile().supports(HardwareCapability.HC_POST_ADD_SEPARATE_ROUNDING):
        pytest.skip("requires A3")
    # The A3 native HcPost does not support arbitrary nonaligned widths;
    # calling it at dim=129 faults the device context. Check masked tails
    # against the same ordered arithmetic on CPU, without that reference op.
    inputs = _inputs(3, dim, False)
    shared, routed, residual, post, comb = [value.cpu() for value in inputs]
    x = (shared + routed).float()
    expected = x.unsqueeze(1) * post.unsqueeze(-1)
    for source in range(4):
        expected = expected + residual[:, source, :].float().unsqueeze(1) * comb[:, source, :].unsqueeze(-1)
    actual = fused_hc_post_add(*inputs)
    torch.testing.assert_close(actual.cpu(), expected.bfloat16(), rtol=0, atol=0)


@torch.inference_mode()
def test_hc_post_add_graph_replay_observes_new_inputs():
    if not get_current_hardware_profile().supports(HardwareCapability.HC_POST_ADD_SEPARATE_ROUNDING):
        pytest.skip("requires A3")
    enable_custom_op()
    inputs = _inputs(12, 4096, False)
    for _ in range(3):
        fused_hc_post_add(*inputs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        output = fused_hc_post_add(*inputs)
    for _ in range(3):
        inputs[0].add_(0.25)
        inputs[2].mul_(0.5)
        graph.replay()
        torch.testing.assert_close(output, _native(*inputs), rtol=0, atol=0)


@torch.inference_mode()
def test_hc_post_add_preserves_bf16_tie_rounding():
    if not get_current_hardware_profile().supports(HardwareCapability.HC_POST_ADD_SEPARATE_ROUNDING):
        pytest.skip("requires A3")
    enable_custom_op()
    inputs = _inputs(1, 4096, False)
    inputs[0].fill_(1)
    inputs[1].fill_(2**-8)
    inputs[2].zero_()
    inputs[3].fill_(1.5)
    inputs[4].zero_()
    torch.testing.assert_close(fused_hc_post_add(*inputs), _native(*inputs), rtol=0, atol=0)
