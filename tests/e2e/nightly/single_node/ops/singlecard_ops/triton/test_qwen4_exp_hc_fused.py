# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from torch import nn

from vllm_ascend.models.qwen4_exp.hc_fused import AscendTextGatedResidual
from vllm_ascend.ops.triton.qwen4_exp.hc import combine_final

pytest.importorskip("cann_ops_transformer.ops.mhc.text_gated_residual.text_gated_residual")


def make_hc(use_combine, hc_per_branch_norm=True):
    # Exercise the checkpoint-facing weight preparation without distributed
    # Linear initialization; the operator consumes these ordinary ND tensors.
    module = AscendTextGatedResidual.__new__(AscendTextGatedResidual)
    nn.Module.__init__(module)
    module.config = SimpleNamespace(hc_lowrank=320, rms_norm_eps=1e-6, hc_per_branch_norm=hc_per_branch_norm)
    module.hc_count = 4
    module.hidden_size = 2560
    module.use_combine = use_combine
    module._textgr_enabled = hc_per_branch_norm
    for name in ("_textgr_gamma", "_textgr_down", "_textgr_up", "_textgr_gate"):
        module.register_buffer(name, None, persistent=False)
    module.hc_norm = nn.Linear(10240, 1, bias=False, dtype=torch.bfloat16, device="npu")
    module.hc_norm.weight.data = torch.randn(10240, device="npu", dtype=torch.bfloat16) * 0.1
    down = nn.Linear(10240, 336 if use_combine else 320, bias=False, dtype=torch.bfloat16, device="npu")
    if use_combine:
        module.input_mix_weight_down_block_inject = down
    else:
        module.input_mix_weight_down = down
    module.input_mix_weight_up = nn.Linear(320, 10240, bias=False, dtype=torch.bfloat16, device="npu")
    module.prepare_textgr_weights()
    return module


def reference_mix(module, x):
    groups = x.float().reshape(-1, 4, 2560)
    norm = groups * torch.rsqrt(groups.square().mean(-1, keepdim=True) + module.config.rms_norm_eps)
    norm = (norm.flatten(1) * (1 + module.hc_norm.weight.float())).to(x.dtype)
    lowrank = F.linear(norm, module._textgr_down)
    lowrank = F.silu(lowrank.float() / 4).to(x.dtype)
    gates = F.linear(lowrank, module._textgr_up)
    mixed = (norm.float() * gates.float().sigmoid()).reshape(-1, 4, 2560).mean(1).to(x.dtype)
    injection = None
    if module.use_combine:
        injection = (2 * torch.sigmoid(F.linear(norm, module._textgr_gate).float() / 4)).to(x.dtype)
    return mixed, injection


@pytest.mark.parametrize("tokens", [1, 4, 512, 1024, 4096])
@pytest.mark.parametrize("use_combine", [False, True])
@torch.inference_mode()
def test_mix_and_pending_combine(tokens, use_combine):
    torch.manual_seed(21794)
    module = make_hc(use_combine)
    x = torch.randn(tokens, 10240, device="npu", dtype=torch.bfloat16)
    expected, expected_injection = reference_mix(module, x)
    residual, mixed, injection = module.mix(x)
    assert residual is x
    torch.testing.assert_close(mixed, expected, atol=0.02, rtol=0.02)
    if use_combine:
        torch.testing.assert_close(injection, expected_injection, atol=0.02, rtol=0.02)
    else:
        assert injection is None

    block = torch.randn(tokens, 2560, device="npu", dtype=torch.bfloat16)
    final_injection = torch.rand(tokens, 4, device="npu", dtype=torch.bfloat16) * 2
    combined = x.float().reshape(-1, 4, 2560) + block.float()[:, None] * final_injection.float()[:, :, None]
    combined = combined.flatten(1).to(x.dtype)
    actual_residual, actual_mixed, _ = module.combine_and_mix(x, block, final_injection)
    torch.testing.assert_close(actual_residual, combined, atol=0, rtol=0)
    torch.testing.assert_close(actual_mixed, reference_mix(module, combined)[0], atol=0.02, rtol=0.02)


@pytest.mark.parametrize("tokens", [0, 1, 7])
@pytest.mark.parametrize("with_injection", [False, True])
@torch.inference_mode()
def test_combine_uses_final_weights(tokens, with_injection):
    x = torch.randn(tokens, 10240, device="npu", dtype=torch.bfloat16)
    block = torch.randn(tokens, 2560, device="npu", dtype=torch.bfloat16)
    # Zero weights distinguish final weights from logits (sigmoid(0) != 0).
    injection = torch.zeros(tokens, 4, device="npu", dtype=x.dtype) if with_injection else None
    expected = x if with_injection else (x.float().reshape(-1, 4, 2560) + block.float()[:, None]).flatten(1).to(x.dtype)
    torch.testing.assert_close(combine_final(x, block, injection, 4), expected, atol=0, rtol=0)


def test_shared_hc_norm_uses_upstream_implementation():
    module = make_hc(True, hc_per_branch_norm=False)
    module.prepare_textgr_weights()
    assert module._textgr_enabled is False
    assert module._textgr_gamma is None


@torch.inference_mode()
def test_reject_incompatible_inputs():
    module = make_hc(True)
    x = torch.zeros(1, 10240, device="npu", dtype=torch.bfloat16)
    block = torch.zeros(1, 2560, device="npu", dtype=x.dtype)
    with pytest.raises(ValueError, match="model input"):
        module.mix(x.unsqueeze(0))
    with pytest.raises(ValueError, match="block output"):
        combine_final(x, block[:, :-1], None, 4)
    with pytest.raises(ValueError, match="final injection"):
        combine_final(x, block, torch.zeros(1, 3, device="npu"), 4)
    with pytest.raises(ValueError, match="divisible"):
        combine_final(torch.zeros(1, 10242, device="npu"), block, None, 4)
    with pytest.raises(ValueError, match="inner strides"):
        combine_final(x, block, torch.zeros(1, 8, device="npu")[:, ::2], 4)
    with pytest.raises(ValueError, match="inner strides"):
        combine_final(torch.zeros(1, 20480, device="npu")[:, ::2], block, None, 4)
    cpu_residual = torch.zeros(1, 10240)
    cpu_block = torch.ones(1, 2560)
    expected = cpu_residual.reshape(1, 4, 2560) + cpu_block[:, None]
    torch.testing.assert_close(combine_final(cpu_residual, cpu_block, None, 4), expected.flatten(1))
