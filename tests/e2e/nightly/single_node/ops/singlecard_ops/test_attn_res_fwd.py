# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.ops.kimi_attn_res import AttnResBackend, AttnResQueryPlan, load_attn_res_fused_ops


def cpu_reference(prefix, addend, blocks, projection, gamma, valid, output_gamma, epsilon, mix):
    raw = (prefix.cpu() + addend.cpu()) if addend is not None else prefix.cpu()
    if mix and valid:
        values = torch.cat((blocks[:, :valid].cpu(), raw.unsqueeze(1)), dim=1).float()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
        logits = (normalized * gamma.cpu().float() * projection.cpu().float()).sum(-1)
        materialized = (logits.softmax(-1).unsqueeze(-1) * values).sum(1).bfloat16()
    else:
        materialized = raw

    output = materialized
    if output_gamma is not None:
        value = materialized.float()
        output = (
            value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon) * output_gamma.cpu().float()
        ).bfloat16()
    return output, raw, materialized


def assert_precision(actual, expected):
    error = (actual.float().cpu() - expected.float()).abs()
    assert torch.isfinite(actual).all()
    assert (error <= (1 + expected.float().abs()) / 64).float().mean() >= 0.99
    assert error.max() <= 1


@pytest.mark.parametrize(
    "tokens,valid,with_add,with_output_norm,optimize_prefill,mix",
    [
        (1, 0, True, False, False, False),
        (1, 1, False, False, False, True),
        (16, 4, True, True, False, True),
        (128, 8, True, True, False, True),
        (625, 8, True, True, True, True),
        (2048, 8, False, True, True, True),
    ],
)
@torch.inference_mode()
def test_attn_res_fwd_precision(tokens, valid, with_add, with_output_norm, optimize_prefill, mix):
    hidden = 7168
    torch.manual_seed(42 + tokens + valid)
    prefix = torch.randn(tokens, hidden, device="npu", dtype=torch.bfloat16)
    addend = torch.randn_like(prefix) if with_add else None
    bank = torch.randn(tokens, 8, hidden, device="npu", dtype=torch.bfloat16)
    bank[:, valid:] = float("nan")
    projection = (torch.randn(1, hidden, device="npu") / hidden**0.5).bfloat16()
    gamma = torch.randn(hidden, device="npu", dtype=torch.bfloat16)
    output_gamma = torch.randn_like(gamma) if with_output_norm else None
    expected_output, expected_raw, expected_mix = cpu_reference(
        prefix, addend, bank, projection, gamma, valid, output_gamma, 1e-5, mix
    )
    write_idx = valid if valid < 8 else -1
    output, raw, materialized = torch.ops._C_ascend.attn_res_fwd(
        prefix,
        addend,
        bank,
        projection,
        gamma,
        1e-5,
        valid,
        output_gamma,
        1e-5,
        write_idx,
        True,
        mix,
        optimize_prefill=optimize_prefill,
    )
    torch_npu.npu.synchronize()

    torch.testing.assert_close(raw.cpu(), expected_raw, rtol=0, atol=0)
    assert_precision(materialized, expected_mix)
    assert_precision(output, expected_output)
    if write_idx >= 0:
        torch.testing.assert_close(bank[:, write_idx].cpu(), expected_raw, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["two_phase", "original"])
@pytest.mark.parametrize("tokens,valid", [(0, 0), (1, 0), (16, 4), (625, 8)])
@torch.inference_mode()
def test_imported_attn_res_backends_match_native(mode, tokens, valid):
    hidden = 7168
    torch.manual_seed(51)
    prefix = torch.randn(tokens, hidden, device="npu", dtype=torch.bfloat16)
    delta = torch.randn_like(prefix)
    bank = torch.randn(tokens, 9, hidden, device="npu", dtype=torch.bfloat16)
    bank[:, valid:] = float("nan")
    projection = SimpleNamespace(weight=(torch.randn(1, hidden, device="npu") / hidden**0.5).bfloat16())
    norm = SimpleNamespace(weight=torch.randn_like(prefix.new_empty(hidden)), variance_epsilon=1e-6)
    output_gamma = torch.randn_like(norm.weight)
    op = AttnResBackend(mode, AttnResQueryPlan([(projection, norm, valid)]))
    args = (prefix, delta, bank, projection.weight, norm.weight, norm.variance_epsilon, valid)
    kwargs = dict(
        output_norm_weight=output_gamma,
        output_norm_eps=1e-4,
        block_write_idx=valid,
        return_materialized=True,
    )
    expected = torch.ops._C_ascend.attn_res_fwd(*args, **kwargs)
    actual = op(*args, **kwargs)
    torch_npu.npu.synchronize()
    torch.testing.assert_close(actual[1].cpu(), expected[1].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(bank[:, valid].cpu(), actual[1].cpu(), rtol=0, atol=0)
    if tokens:
        assert_precision(actual[0], expected[0].cpu())
        assert_precision(actual[2], expected[2].cpu())
    else:
        assert actual[0].shape == expected[0].shape


@torch.inference_mode()
def test_two_phase_graph_replay_observes_reloaded_queries():
    hidden = 7168
    prefix = -torch.ones(4, hidden, device="npu", dtype=torch.bfloat16)
    bank = torch.ones(4, 1, hidden, device="npu", dtype=torch.bfloat16)
    projection = SimpleNamespace(weight=torch.full((1, hidden), 1 / hidden, device="npu", dtype=torch.bfloat16))
    norm = SimpleNamespace(weight=torch.ones_like(prefix[0]), variance_epsilon=1e-6)
    specs = [(projection, norm, 1)]
    plan = AttnResQueryPlan(specs)
    query = plan.get_queries((1, norm.variance_epsilon))
    pointer = query.data_ptr()

    def forward():
        op = AttnResBackend("two_phase", plan)
        return op(prefix, None, bank, projection.weight, norm.weight, norm.variance_epsilon, 1)[0]

    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = forward()
    graph.replay()
    before = output.cpu()
    for _ in range(2):
        projection.weight.neg_()
        plan.refresh(specs)
        assert plan.get_queries((1, norm.variance_epsilon)).data_ptr() == pointer
        torch.npu.synchronize()
        graph.replay()
        expected = torch.ops._C_ascend.attn_res_fwd(
            prefix, None, bank, projection.weight, norm.weight, norm.variance_epsilon, 1
        )[0]
        actual = output.cpu()
        assert not torch.allclose(actual, before)
        assert_precision(actual, expected.cpu())
        before = actual


@pytest.mark.parametrize("capture_materialized", [False, True])
@torch.inference_mode()
def test_fused_two_phase_operators_and_graph_replay(capture_materialized):
    fused_ops = load_attn_res_fused_ops()
    if fused_ops is None:
        pytest.skip("Install transformer prepare/update operators to validate the fused backend")
    hidden, tokens = 7168, 4
    torch.manual_seed(63)
    prefix = torch.randn(tokens, hidden, device="npu", dtype=torch.bfloat16)
    delta = torch.randn_like(prefix)
    bank = torch.randn(tokens, 2, hidden, device="npu", dtype=torch.bfloat16)
    projection = SimpleNamespace(weight=(torch.randn(1, hidden, device="npu") / hidden**0.5).bfloat16())
    norm = SimpleNamespace(weight=torch.ones(hidden, device="npu", dtype=torch.bfloat16), variance_epsilon=1e-6)
    plan = AttnResQueryPlan([(projection, norm, 2)])

    def forward():
        op = AttnResBackend("fused", plan, fused_ops)
        return op(
            prefix,
            delta,
            bank,
            projection.weight,
            norm.weight,
            norm.variance_epsilon,
            2,
            output_norm_weight=norm.weight,
            output_norm_eps=norm.variance_epsilon,
            return_materialized=capture_materialized,
        )

    # The fused source accumulates in FP32 before scoring. Use that partial
    # in the independent reference, then round the mixture before RMSNorm.
    values = torch.cat((bank.float().cpu(), (prefix.float() + delta.float()).cpu().unsqueeze(1)), dim=1)
    query = projection.weight.float().cpu() * norm.weight.float().cpu()
    logits = (values * query).sum(-1) * torch.rsqrt(values.square().mean(-1) + norm.variance_epsilon)
    materialized = (logits.softmax(-1).unsqueeze(-1) * values).sum(1).bfloat16()
    value = materialized.float()
    expected = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + norm.variance_epsilon)).bfloat16()
    eager, eager_raw, eager_materialized = forward()
    assert_precision(eager, expected)
    torch.testing.assert_close(eager_raw.cpu(), (prefix.float() + delta.float()).bfloat16().cpu(), atol=0, rtol=0)
    if capture_materialized:
        assert_precision(eager_materialized, materialized)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        replay_output = forward()[0]
    for _ in range(2):
        graph.replay()
        assert_precision(replay_output, expected)
