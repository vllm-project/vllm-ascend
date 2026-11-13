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
    monkeypatch.setattr(ns, "l2norm_fwd", Mock(side_effect=lambda x: x))
    gate = Mock(side_effect=lambda A, a, b, bias: (-a.unsqueeze(0), b.sigmoid().unsqueeze(0)))
    prefill = Mock(side_effect=lambda **kw: (kw["v"].clone(), kw["initial_state"]))
    monkeypatch.setattr(ns, "DeviceOperator", SimpleNamespace(fused_gdn_gating=gate, fla_gdn_prefill=prefill))
    monkeypatch.setattr(ns, "get_pcp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(ns, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda cap: True))
    monkeypatch.setattr(ns, "HardwareCapability", SimpleNamespace(FLA_GDN_PREFILL=1))
    monkeypatch.setattr(ns, "_get_fla_gdn_prefill_op", lambda: object())
    monkeypatch.setattr(ns, "_get_packed_conv_weights", lambda layer: None)
    monkeypatch.setattr(ns, "_normalize_causal_cache_indices", lambda x: x)
    monkeypatch.setattr(ns, "causal_conv1d_fn", lambda x, *a, **kw: x.clone())
    monkeypatch.setattr(ns, "causal_conv1d_update", lambda x, *a, **kw: x.clone())
    monkeypatch.setattr(ns, "gather_ssm_states", lambda state, *a: state[:1].clone())
    monkeypatch.setattr(ns, "scatter_ssm_states_", Mock())
    monkeypatch.setattr(ns, "PAD_SLOT_ID", -1)
    for name in (
        "wait_for_kv_layer_from_connector",
        "record_attention_compute_start",
        "maybe_save_kv_layer_to_connector",
    ):
        monkeypatch.setattr(ns, name, Mock())
    monkeypatch.setattr(gdn, "get_forward_context", gdn.get_forward_context)
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


@pytest.mark.parametrize(
    "indices,lengths,accepted",
    [
        ([[10, 11, 12, 13], [20, 21, 22, 23]], [0, 1, 2], [4, 2]),
        ([[10, 11], [0, 0], [20, 21]], [0, 2, 0, 1], [1, 0, 2]),
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
    accepted = None if accepted is None else torch.tensor(accepted, dtype=torch.int64)
    result = ns._run_recurrent_gdn(
        q,
        q,
        q,
        state,
        a=a,
        b=b,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
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
    if accepted is not None:
        torch.testing.assert_close(kw["num_accepted_tokens"], accepted.int())
    ns.DeviceOperator.fused_gdn_gating.assert_not_called()
    ns.l2norm_fwd.assert_not_called()
    assert result.shape == q.shape


def test_decode_rows_follow_metadata_not_token_count(runtime):
    ns = runtime
    q = torch.zeros(1, 2, 1, 128, dtype=torch.bfloat16)
    lengths = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32)
    indices = torch.tensor([7, 0, 9, 0, 99], dtype=torch.int64)
    ns._run_recurrent_gdn(
        q,
        q,
        q,
        q,
        a=torch.zeros(2, 1),
        b=torch.zeros(2, 1),
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        actual_seq_lengths=lengths,
        state_indices=indices,
    )
    kw = ns.recurrent_gated_delta_rule.call_args.kwargs
    torch.testing.assert_close(kw["ssm_state_indices"], torch.tensor([[7], [0], [9], [0]], dtype=torch.int32))


def test_missing_request_rows_fails_before_state_mutation(runtime):
    q = torch.zeros(1, 2, 1, 128)
    with pytest.raises(ValueError, match="checkpoint row"):
        runtime["_run_recurrent_gdn"](q, q, q, q, q, q, q, q, torch.tensor([0, 1, 1]), torch.tensor([7]))
    runtime["recurrent_gated_delta_rule"].assert_not_called()


def test_legacy_keeps_external_preprocessing(runtime):
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
    )
    assert ns.l2norm_fwd.call_count == 2
    ns.DeviceOperator.fused_gdn_gating.assert_called_once()
    kw = ns.recurrent_gated_delta_rule.call_args.kwargs
    assert "use_gate_in_kernel" not in kw
    torch.testing.assert_close(kw["g"], -a)
    torch.testing.assert_close(kw["ssm_state_indices"], torch.tensor([10, 11, 20], dtype=torch.int32))


@pytest.mark.parametrize("mode", ["decode", "spec", "prefill", "decode_prefill", "spec_decode", "spec_prefill"])
def test_forward_routes_raw_gate_rows_and_merges_outputs(runtime, mode):
    ns = runtime
    spec = "spec" in mode
    prefill = "prefill" in mode
    decode = "decode" in mode
    spec_ids = [1, 3] if spec and (prefill or decode) else ([0, 1] if spec else [])
    non_ids = [0, 2] if spec and (prefill or decode) else ([] if spec else list(range(3 if prefill and decode else 2)))
    count = len(spec_ids) + len(non_ids)
    nd = (1 if prefill else len(non_ids)) if decode else 0
    npref = 1 if prefill else 0
    prefix = "gdn"
    qsl = torch.tensor([0, 2], dtype=torch.int32)
    conv = SimpleNamespace(query_start_loc=qsl, cache_indices=torch.tensor([0]), initial_state_mode=None)
    non_lengths = [0] + [1] * nd
    non_states = torch.arange(nd + npref, dtype=torch.int32) + 10
    meta = SimpleNamespace(
        num_prefills=npref,
        num_decodes=nd,
        num_decode_tokens=nd,
        num_actual_tokens=count,
        spec_sequence_masks=torch.tensor([True]) if spec else None,
        spec_token_indx=torch.tensor(spec_ids, dtype=torch.int64),
        non_spec_token_indx=torch.tensor(non_ids, dtype=torch.int64),
        spec_state_indices_tensor=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
        non_spec_state_indices_tensor=non_states,
        spec_decode_metadata=SimpleNamespace(
            actual_seq_lengths=torch.tensor([0, 2], dtype=torch.int32),
            spec_causal_conv1d=SimpleNamespace(
                query_start_loc=qsl,
                cache_indices=torch.tensor([1]),
                num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
            ),
        ),
        non_spec_decode_metadata=SimpleNamespace(
            causal_conv1d=conv, actual_seq_lengths=torch.tensor(non_lengths, dtype=torch.int32)
        ),
        non_spec_prefill_metadata=SimpleNamespace(causal_conv1d=conv, chunk=None),
        prefill_query_start_loc=torch.tensor([0, len(non_ids) - nd], dtype=torch.int32),
        prefill_state_indices=torch.tensor([10 + nd]),
        prefill_has_initial_state=torch.tensor([True]),
    )
    layer = SimpleNamespace(
        prefix=prefix,
        kv_cache=(torch.zeros(20, 2, 6), torch.zeros(20, 1, 2, 2)),
        head_k_dim=2,
        head_v_dim=2,
        num_k_heads=1,
        num_v_heads=1,
        tp_size=1,
        key_dim=2,
        value_dim=2,
        num_spec=3,
        conv1d=SimpleNamespace(bias=None),
        activation=None,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
    )
    layer.rearrange_mixed_qkv = lambda x: (None, None, None) if x is None else ns._rearrange_decode_qkv(x, 2, 2, 2, 2)
    ns.get_forward_context = lambda: SimpleNamespace(attn_metadata={prefix: meta})
    tokens = torch.arange(count, dtype=torch.bfloat16)
    packed = tokens[:, None].expand(-1, 6).contiguous()
    a, b = tokens[:, None].float() + 10, tokens[:, None].float() + 20
    output = torch.empty(count, 1, 2, dtype=torch.bfloat16)
    ns.AscendGatedDeltaNetAttention._forward_core(layer, packed, b, a, output)
    torch.testing.assert_close(output[:, 0, 0], tokens)
    expected_groups = ([spec_ids] if spec else []) + ([non_ids[:nd]] if decode else [])
    for call, ids in zip(ns.recurrent_gated_delta_rule.call_args_list, expected_groups, strict=True):
        torch.testing.assert_close(call.kwargs["g"], a[ids])
        torch.testing.assert_close(call.kwargs["beta"], b[ids])
        assert call.kwargs["ssm_state_indices"].ndim == 2
    if prefill:
        gate_args = ns.DeviceOperator.fused_gdn_gating.call_args.args
        torch.testing.assert_close(gate_args[1], a[non_ids[nd:]])
        torch.testing.assert_close(gate_args[2], b[non_ids[nd:]])
        ns.DeviceOperator.fused_gdn_gating.assert_called_once()
    else:
        ns.DeviceOperator.fused_gdn_gating.assert_not_called()
    ns.l2norm_fwd.assert_not_called()
