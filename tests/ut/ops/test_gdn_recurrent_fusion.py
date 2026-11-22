# SPDX-License-Identifier: Apache-2.0
"""Exercise GDN dispatch and tensor adaptation with mocked native operators."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops import gdn


@pytest.fixture
def runtime(monkeypatch):
    ns = gdn
    monkeypatch.setattr(gdn, "GDNAttentionMetadata", SimpleNamespace)
    monkeypatch.setattr(ns, "_supports_recurrent_gdn_fusion", lambda: True)
    monkeypatch.setattr(ns, "recurrent_gated_delta_rule", Mock(side_effect=lambda q, k, v, state, **kw: v.clone()))
    monkeypatch.setattr(
        ns, "fused_recurrent_gated_delta_rule", Mock(side_effect=lambda **kw: (kw["v"].clone(), kw["initial_state"]))
    )
    monkeypatch.setattr(ns, "l2norm_fwd", Mock(side_effect=lambda x: x))
    gate = Mock(side_effect=lambda A, a, b, bias: (-a.unsqueeze(0), b.sigmoid().unsqueeze(0)))
    monkeypatch.setattr(ns, "DeviceOperator", SimpleNamespace(fused_gdn_gating=gate))
    return ns


def test_probe_only_accepts_explicit_fused_abi(monkeypatch):
    ns = gdn

    def new(
        *args,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        A_log=None,
        dt_bias=None,
    ):
        raise AssertionError("Capability detection must not execute the operator")

    for op, expected in [(new, True), (lambda *a, **kw: None, False), (lambda q, k, v: None, False)]:
        monkeypatch.setattr(gdn, "recurrent_gated_delta_rule", op)
        ns._supports_recurrent_gdn_fusion.cache_clear()
        assert ns._supports_recurrent_gdn_fusion() == expected
    ns._supports_recurrent_gdn_fusion.cache_clear()


def test_recurrent_parameters_are_materialized_once_in_fp32():
    layer = torch.nn.Module()
    layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
    layer.conv1d = torch.nn.Module()
    layer.conv1d.weight = torch.nn.Parameter(torch.empty(6, 1, 4))
    layer.A_log = torch.nn.Parameter(torch.tensor([1.5], dtype=torch.bfloat16))
    layer.dt_bias = torch.nn.Parameter(torch.tensor([-0.25], dtype=torch.bfloat16))
    a_loader, dt_loader = object(), object()
    layer.A_log.weight_loader = a_loader
    layer.dt_bias.weight_loader = dt_loader

    gdn.initialize_packed_conv_weight(layer)

    assert layer.A_log.dtype == layer.dt_bias.dtype == torch.float32
    torch.testing.assert_close(layer.A_log, torch.tensor([1.5]))
    torch.testing.assert_close(layer.dt_bias, torch.tensor([-0.25]))
    assert layer.A_log.weight_loader is a_loader
    assert layer.dt_bias.weight_loader is dt_loader


@pytest.mark.parametrize(
    "indices,lengths,accepted",
    [
        ([[10, 11, 12, 13], [20, 21, 22, 23]], [0, 1, 2], [4, 2]),
        ([[10], [20], [0], [0]], [0, 1, 1, 0, 0], None),
    ],
)
def test_fused_retains_checkpoint_width_and_zero_length_rows(runtime, indices, lengths, accepted):
    ns = runtime
    lengths = torch.tensor(lengths, dtype=torch.int32)
    table = torch.tensor(indices, dtype=torch.int32)
    count = int(lengths.sum())
    q = torch.randn(1, count, 1, 128, dtype=torch.bfloat16)
    a, b = torch.randn(count, 1), torch.randn(count, 1)
    state = torch.zeros(24, 1, 128, 128)
    accepted = None if accepted is None else torch.tensor(accepted, dtype=torch.int32)
    A_log = torch.zeros(1)
    dt_bias = torch.zeros(1)
    result = ns._run_recurrent_gdn(
        q,
        q,
        q,
        state,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        actual_seq_lengths=lengths,
        state_indices=table,
        num_accepted_tokens=accepted,
    )
    args, kw = ns.recurrent_gated_delta_rule.call_args
    assert args[3] is state
    assert kw["g"] is a and kw["beta"] is b
    assert kw["ssm_state_indices"].data_ptr() == table.data_ptr()
    torch.testing.assert_close(kw["ssm_state_indices"], table)
    assert kw["actual_seq_lengths"] is lengths
    assert kw["use_qk_l2norm_in_kernel"] and kw["use_gate_in_kernel"] and kw["use_beta_sigmoid_in_kernel"]
    assert not kw["allow_neg_eigval"]
    assert kw["scale"] == 128**-0.5
    assert kw["A_log"].dtype == kw["dt_bias"].dtype == torch.float32
    assert kw["A_log"] is A_log and kw["dt_bias"] is dt_bias
    if accepted is not None:
        assert kw["num_accepted_tokens"].dtype == torch.int32
        assert kw["num_accepted_tokens"].data_ptr() == accepted.data_ptr()
    ns.DeviceOperator.fused_gdn_gating.assert_not_called()
    ns.l2norm_fwd.assert_not_called()
    assert result.shape == q.shape


def test_missing_request_rows_fails_before_state_mutation(runtime):
    q = torch.zeros(1, 2, 1, 128)
    with pytest.raises(ValueError, match="checkpoint row"):
        runtime._run_recurrent_gdn(
            q,
            q,
            q,
            q,
            a=q,
            b=q,
            A_log=q,
            dt_bias=q,
            actual_seq_lengths=torch.tensor([0, 1, 1]),
            state_indices=torch.tensor([7]),
        )
    runtime.recurrent_gated_delta_rule.assert_not_called()


@pytest.mark.parametrize("upstream", [False, True])
def test_legacy_and_upstream_keep_external_preprocessing(runtime, upstream):
    ns = runtime
    ns._supports_recurrent_gdn_fusion = lambda: False
    q = torch.zeros(1, 3, 1, 128)
    a = torch.tensor([[1.0], [2.0], [3.0]])
    table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    ns._run_recurrent_gdn(
        q,
        q,
        q,
        q,
        a=a,
        b=a,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        actual_seq_lengths=torch.tensor([0, 2, 1]),
        state_indices=table,
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 3]),
        use_upstream_recurrent=upstream,
    )
    assert ns.l2norm_fwd.call_count == 2
    ns.DeviceOperator.fused_gdn_gating.assert_called_once()
    if upstream:
        ns.recurrent_gated_delta_rule.assert_not_called()
        assert ns.fused_recurrent_gated_delta_rule.call_args.kwargs["ssm_state_indices"] is table
    else:
        kw = ns.recurrent_gated_delta_rule.call_args.kwargs
        assert "use_gate_in_kernel" not in kw
        torch.testing.assert_close(kw["g"], -a)
        torch.testing.assert_close(kw["ssm_state_indices"], torch.tensor([10, 11, 20], dtype=torch.int32))
