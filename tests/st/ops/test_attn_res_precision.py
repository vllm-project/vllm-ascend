# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 AttnRes regression against the pinned CPU FP32/BF16 contract."""

import math

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend.vllm_ascend_C")


@pytest.fixture(scope="module", autouse=True)
def require_a5():
    if not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    if "950" not in torch.npu.get_device_name():
        pytest.skip("this regression targets A5")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, previous_threads))
    yield
    torch.set_num_threads(previous_threads)


def check_case(optimize_prefill, tokens, hidden, blocks, seed):
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(shape, generator=generator).bfloat16()

    prefix = rand(tokens, hidden)
    addend = rand(tokens, hidden)
    bank = rand(tokens, 8, hidden)
    proj = (rand(1, hidden).float() / math.sqrt(hidden)).bfloat16()
    norm = rand(hidden)
    output_norm = rand(hidden)
    for value in (prefix, addend, bank):
        if value is not None:
            value.mul_(1000)
    bank[:, blocks:] = float("nan")
    raw = (prefix.float() + addend.float()).bfloat16()
    values = torch.cat((bank[:, :blocks], raw[:, None]), dim=1).float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-5)
    scores = (normalized * (proj.float() * norm.float())).sum(-1)
    materialized = (scores.softmax(-1)[:, :, None] * values).sum(1).bfloat16()
    expected = materialized
    if output_norm is not None:
        value = materialized.float()
        expected = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5) * output_norm.float()).bfloat16()

    p, a, b, w, g, og = [None if x is None else x.npu() for x in (prefix, addend, bank, proj, norm, output_norm)]
    outputs = torch.ops._C_ascend.attn_res_fwd.fused(
        p, a, b, w, g, 1e-5, blocks, og, 1e-5, -1, True, True, optimize_prefill
    )
    torch.npu.synchronize()
    pairs = [(outputs[0], expected), (outputs[2], materialized)]
    for actual, golden in pairs:
        actual = actual.cpu().float()
        golden = golden.float()
        error = (actual - golden).abs()
        assert torch.isfinite(actual).all()
        assert (error <= (1 + golden.abs()) / 64).float().mean() >= 0.99
        assert error.max() <= 1, (optimize_prefill, tokens, hidden, blocks, seed, error.max().item())
    assert torch.equal(outputs[1].cpu(), raw)


@pytest.mark.parametrize("optimize_prefill", [False, True])
@pytest.mark.parametrize("hidden", [4096, 7168])
@pytest.mark.parametrize("tokens", [1, 16, 128])
@pytest.mark.parametrize("blocks", [1, 4, 8])
def test_large_values_shapes(optimize_prefill, hidden, tokens, blocks):
    check_case(optimize_prefill, tokens, hidden, blocks, 27177)
