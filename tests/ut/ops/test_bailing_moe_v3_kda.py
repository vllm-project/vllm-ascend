# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from vllm.model_executor.models.bailing_moe_v3 import (
    BailingMoeV3KimiDeltaAttention,
    bailing_v3_kda_attention,
)

from vllm_ascend.ops.bailing_moe_v3_kda import AscendBailingMoeV3KimiDeltaAttention


class _FixedLinear(nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(self, _input: torch.Tensor):
        return self.value, None


class _RecordingNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        self.inputs = (value, gate)
        return value


def _new_attention() -> AscendBailingMoeV3KimiDeltaAttention:
    attention = AscendBailingMoeV3KimiDeltaAttention.__new__(AscendBailingMoeV3KimiDeltaAttention)
    nn.Module.__init__(attention)
    return attention


@pytest.mark.parametrize("num_speculative_tokens", [0, 1, 7, 8])
def test_init_enforces_recurrent_sequence_length_limit(num_speculative_tokens):
    def fake_base_init(instance, *_args, **_kwargs):
        nn.Module.__init__(instance)
        instance.num_speculative_tokens = num_speculative_tokens

    with (
        patch.object(BailingMoeV3KimiDeltaAttention, "__init__", fake_base_init),
        patch("vllm_ascend.ops.bailing_moe_v3_kda.is_conv_state_dim_first", return_value=False),
    ):
        if num_speculative_tokens <= 7:
            AscendBailingMoeV3KimiDeltaAttention(None)
        else:
            with pytest.raises(ValueError, match="at most 7 speculative tokens"):
                AscendBailingMoeV3KimiDeltaAttention(None)


def test_upstream_custom_op_calls_ascend_forward_with_g1_keyword():
    attention = _new_attention()
    attention.prefix = "model.layers.0.self_attn"
    output = torch.full((1, 2, 1, 2), torch.nan)
    qkv = torch.empty(2, 2)
    gate = torch.empty(1, 2, 1, 2)
    beta = torch.empty(1, 2, 1)

    with (
        patch(
            "vllm.model_executor.models.bailing_moe_v3.get_forward_context",
            return_value=SimpleNamespace(no_compile_layers={attention.prefix: attention}),
        ),
        patch(
            "vllm_ascend.ops.bailing_moe_v3_kda.get_forward_context",
            return_value=SimpleNamespace(attn_metadata=None),
        ),
    ):
        bailing_v3_kda_attention(qkv, qkv, qkv, gate, beta, output, attention.prefix)

    assert torch.equal(output, torch.zeros_like(output))


@pytest.mark.parametrize("separate_b_proj", [False, True])
def test_forward_passes_raw_gate_and_fp32_sigmoid_beta(separate_b_proj: bool):
    attention = _new_attention()
    num_tokens, num_heads, head_dim = 4, 2, 3
    projection_size = num_heads * head_dim
    qkvb = torch.arange(
        num_tokens * (3 * projection_size + num_heads),
        dtype=torch.bfloat16,
    ).reshape(num_tokens, -1)
    expected_q, expected_k, expected_v, beta_logits = qkvb.split(
        [projection_size, projection_size, projection_size, num_heads],
        dim=-1,
    )
    raw_gate = torch.linspace(-3, 3, num_tokens * projection_size, dtype=torch.float32).reshape(
        num_tokens, projection_size
    )
    output_gate = torch.randn(num_tokens, projection_size)

    attention.separate_b_proj = separate_b_proj
    attention.qkvb_proj = None if separate_b_proj else _FixedLinear(qkvb)
    attention.qkv_proj = (
        _FixedLinear(torch.cat((expected_q, expected_k, expected_v), dim=-1)) if separate_b_proj else None
    )
    attention.b_proj = _FixedLinear(beta_logits) if separate_b_proj else None
    attention.f_proj = _FixedLinear(raw_gate)
    attention.g_proj = _FixedLinear(output_gate)
    attention.o_norm = _RecordingNorm()
    attention.o_proj = _FixedLinear(torch.randn(num_tokens, projection_size))
    attention.projection_size_per_partition = projection_size
    attention.local_num_heads = num_heads
    attention.head_dim = head_dim
    attention.prefix = "model.layers.0.linear_attn"

    hidden_states = torch.randn(num_tokens, 7, dtype=torch.bfloat16)
    output = torch.empty(num_tokens, projection_size)
    with patch.object(
        torch.ops.vllm,
        "bailing_v3_kda_attention",
        create=True,
    ) as custom_op:
        result = attention.forward(hidden_states, torch.arange(num_tokens), output)

    assert result is None
    custom_op.assert_called_once()
    q_arg, k_arg, v_arg, gate_arg, beta_arg, core_arg, prefix_arg = custom_op.call_args.args
    torch.testing.assert_close(q_arg, expected_q)
    torch.testing.assert_close(k_arg, expected_k)
    torch.testing.assert_close(v_arg, expected_v)
    torch.testing.assert_close(gate_arg, raw_gate.reshape(1, num_tokens, num_heads, head_dim))
    assert beta_arg.dtype == torch.float32
    torch.testing.assert_close(beta_arg, beta_logits.float().sigmoid().unsqueeze(0))
    assert tuple(core_arg.shape) == (1, num_tokens, num_heads, head_dim)
    assert prefix_arg == attention.prefix
    assert attention.o_norm.inputs is not None
    torch.testing.assert_close(attention.o_norm.inputs[1], output_gate.reshape(num_tokens, num_heads, head_dim))


@pytest.mark.parametrize(
    ("safe_gate", "expected_lower_bound"),
    [(False, None), (True, -4.0)],
)
def test_recurrent_uses_current_kda_contract(safe_gate: bool, expected_lower_bound: float | None):
    attention = _new_attention()
    attention.safe_gate = safe_gate
    attention.lower_bound = -4.0
    attention.A_log = torch.randn(1)
    attention.dt_bias = torch.randn(2)
    q = torch.randn(1, 3, 1, 2)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_gate = torch.randn_like(q)
    beta = torch.rand(1, 3, 1)
    recurrent_state = torch.zeros(4, 1, 2, 2)
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    state_indices = torch.tensor([2], dtype=torch.int32)
    accepted = torch.tensor([2], dtype=torch.int32)

    with patch(
        "vllm_ascend.ops.bailing_moe_v3_kda.run_recurrent_kda",
        return_value=v,
    ) as run_recurrent:
        actual = attention._run_recurrent(
            q,
            k,
            v,
            raw_gate,
            beta,
            recurrent_state,
            cu_seqlens,
            state_indices,
            num_accepted_tokens=accepted,
        )

    assert actual is v
    expected_args = (
        q,
        k,
        v,
        raw_gate,
        beta,
        recurrent_state,
        cu_seqlens,
        state_indices,
        attention.A_log,
        attention.dt_bias,
    )
    assert len(run_recurrent.call_args.args) == len(expected_args)
    for actual_arg, expected_arg in zip(run_recurrent.call_args.args, expected_args):
        assert actual_arg is expected_arg
    assert run_recurrent.call_args.kwargs["lower_bound"] == expected_lower_bound
    assert run_recurrent.call_args.kwargs["num_accepted_tokens"] is accepted


@pytest.mark.parametrize(
    ("safe_gate", "expected_lower_bound"),
    [(False, None), (True, -4.0)],
)
def test_prefill_uses_current_chunk_contract_and_writes_back_state(
    safe_gate: bool,
    expected_lower_bound: float | None,
):
    attention = _new_attention()
    attention.safe_gate = safe_gate
    attention.lower_bound = -4.0
    attention.A_log = torch.randn(1)
    attention.dt_bias = torch.randn(2)
    q = torch.randn(1, 3, 1, 2)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_gate = torch.randn_like(q)
    beta = torch.rand(1, 3, 1)
    recurrent_state = torch.zeros(4, 1, 2, 2, dtype=torch.float64)
    state_indices = torch.tensor([1, 3], dtype=torch.int64)
    has_initial_state = torch.tensor([True, False])
    metadata = SimpleNamespace(
        cu_seqlens_host=(0, 1, 3),
        cu_seqlens_kern=None,
        keep_meta=None,
        chunk_indices_chunk64_host=(0, 0),
    )
    output = torch.randn_like(v)
    final_state = torch.randn(2, 1, 2, 2, dtype=torch.float32)

    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.clear_ssm_states") as clear_states,
        patch(
            "vllm_ascend.ops.bailing_moe_v3_kda.run_chunk_kda",
            return_value=(output, final_state),
        ) as run_chunk,
    ):
        actual = attention._run_prefill(
            q,
            k,
            v,
            raw_gate,
            beta,
            recurrent_state,
            state_indices,
            has_initial_state,
            metadata,
        )

    assert actual is output
    initial_state = run_chunk.call_args.args[5]
    assert initial_state.is_contiguous()
    clear_states.assert_called_once()
    assert clear_states.call_args.args[0] is initial_state
    assert clear_states.call_args.args[1] is has_initial_state
    expected_args = (
        q,
        k,
        v,
        raw_gate,
        beta,
        initial_state,
        metadata.cu_seqlens_host,
        metadata.chunk_indices_chunk64_host,
        attention.A_log,
        attention.dt_bias,
    )
    assert len(run_chunk.call_args.args) == len(expected_args)
    for actual_arg, expected_arg in zip(run_chunk.call_args.args, expected_args):
        assert actual_arg is expected_arg
    assert run_chunk.call_args.kwargs == {"lower_bound": expected_lower_bound}
    assert recurrent_state.dtype == torch.float64
    torch.testing.assert_close(recurrent_state[state_indices], final_state.double())


def test_causal_conv_consumes_output_alias_and_writes_ds_dtype_cache_back():
    attention = _new_attention()
    attention._conv_state_dim_first = True
    mixed_qkv = torch.randn(4, 6, dtype=torch.float32)
    conv_weights_t = torch.randn(3, 6)
    # Runtime cache is DS and may use a different storage dtype from activation.
    conv_state_storage = torch.zeros(2, 6, 3, dtype=torch.float64)
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    cache_indices = torch.tensor([1], dtype=torch.int32)
    initial_state_mode = torch.tensor([0], dtype=torch.int32)
    returned_alias = torch.full_like(mixed_qkv, 11)
    seen_state: list[torch.Tensor] = []

    def fake_conv(output, input_, weight, **kwargs):
        assert tuple(output.shape) == tuple(mixed_qkv.shape)
        assert input_ is mixed_qkv
        assert weight is conv_weights_t
        seen_state.append(kwargs["conv_state"])
        kwargs["conv_state"].fill_(7)
        return returned_alias

    with patch.object(
        torch.ops._C_ascend,
        "npu_causal_conv1d_custom",
        side_effect=fake_conv,
        create=True,
    ) as conv_op:
        actual = attention._run_causal_conv1d(
            mixed_qkv,
            conv_weights_t,
            conv_state_storage,
            query_start_loc,
            cache_indices,
            initial_state_mode,
            run_mode=0,
        )

    assert actual is returned_alias
    assert len(seen_state) == 1
    assert seen_state[0].shape == (2, 3, 6)
    assert seen_state[0].dtype == mixed_qkv.dtype
    assert seen_state[0].is_contiguous()
    assert conv_state_storage.dtype == torch.float64
    torch.testing.assert_close(conv_state_storage, torch.full_like(conv_state_storage, 7))
    assert conv_op.call_args.kwargs["query_start_loc_opt"] is query_start_loc
    assert conv_op.call_args.kwargs["cache_indices_opt"] is cache_indices
    assert conv_op.call_args.kwargs["initial_state_mode_opt"] is initial_state_mode
    assert conv_op.call_args.kwargs["run_mode"] == 0


def test_profile_clears_preallocated_output():
    attention = _new_attention()
    attention.prefix = "model.layers.0.linear_attn"
    core_attn_out = torch.full((1, 5, 1, 2), torch.nan)

    with patch(
        "vllm_ascend.ops.bailing_moe_v3_kda.get_forward_context",
        return_value=SimpleNamespace(attn_metadata=None),
    ):
        attention._forward(
            torch.empty(5, 2),
            torch.empty(5, 2),
            torch.empty(5, 2),
            torch.empty(1, 5, 1, 2),
            torch.empty(1, 5, 1),
            core_attn_out,
        )

    assert torch.equal(core_attn_out, torch.zeros_like(core_attn_out))


@pytest.mark.parametrize("mode", ["spec", "decode", "mixed"])
def test_forward_routes_spec_decode_and_mixed_rows_and_clears_tail(mode: str):
    num_allocated_tokens = 8
    num_actual_tokens = 6
    head_dim = 2
    q = torch.arange(num_allocated_tokens * head_dim, dtype=torch.float32).reshape(-1, head_dim)
    k = q + 100
    v = q + 200
    raw_gate = torch.zeros(1, num_allocated_tokens, 1, head_dim)
    beta = torch.full((1, num_allocated_tokens, 1), 0.5)
    accepted = torch.tensor([2], dtype=torch.int32)
    conv_meta = SimpleNamespace(
        query_start_loc=torch.tensor([0, num_actual_tokens], dtype=torch.int32),
        cache_indices=torch.tensor([0], dtype=torch.int32),
        initial_state_mode=None,
        num_accepted_tokens=accepted,
    )
    spec_token_indices = torch.tensor([0, 2, 4], dtype=torch.int64)
    non_spec_token_indices = torch.tensor([1, 3, 5], dtype=torch.int64)
    spec_query_end = 2 if mode == "mixed" else 4
    metadata = SimpleNamespace(
        num_actual_tokens=num_actual_tokens,
        num_prefills=0,
        num_decodes=0 if mode == "spec" else 1,
        num_decode_tokens=3 if mode == "mixed" else num_actual_tokens,
        spec_sequence_masks=None if mode == "decode" else torch.tensor([True]),
        spec_token_indx=spec_token_indices,
        non_spec_token_indx=non_spec_token_indices,
        spec_decode_metadata=SimpleNamespace(spec_causal_conv1d=conv_meta),
        non_spec_decode_metadata=SimpleNamespace(causal_conv1d=conv_meta),
        non_spec_prefill_metadata=None,
        spec_query_start_loc=torch.tensor([0, spec_query_end], dtype=torch.int32),
        non_spec_query_start_loc=torch.tensor(
            [0, 3 if mode == "mixed" else num_actual_tokens],
            dtype=torch.int32,
        ),
        spec_state_indices_tensor=torch.tensor([0], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([0], dtype=torch.int32),
        prefill_state_indices=None,
        prefill_has_initial_state=None,
    )
    conv = MagicMock(side_effect=lambda mixed, *_args, **_kwargs: mixed)
    recurrent = MagicMock(side_effect=lambda _q, _k, value, *_args, **_kwargs: value.clone())
    attention = SimpleNamespace(
        prefix="model.layers.0.linear_attn",
        head_dim=head_dim,
        kv_cache=(torch.zeros(1), torch.zeros(1)),
        _conv_weights_t=lambda _dtype: torch.empty(1),
        _run_causal_conv1d=conv,
        _run_recurrent=recurrent,
    )
    core_attn_out = torch.full((1, num_allocated_tokens, 1, head_dim), torch.nan)

    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.GDNAttentionMetadata", SimpleNamespace),
        patch(
            "vllm_ascend.ops.bailing_moe_v3_kda.get_forward_context",
            return_value=SimpleNamespace(attn_metadata={attention.prefix: metadata}),
        ),
    ):
        AscendBailingMoeV3KimiDeltaAttention._forward(
            attention,
            q,
            k,
            v,
            raw_gate,
            beta,
            core_attn_out,
        )

    expected = torch.zeros_like(core_attn_out)
    if mode == "spec":
        expected[0, :4, 0] = v[:4]
        assert recurrent.call_count == 1
        assert conv.call_count == 1
        assert recurrent.call_args.kwargs["num_accepted_tokens"] is accepted
    elif mode == "decode":
        expected[0, :num_actual_tokens, 0] = v[:num_actual_tokens]
        assert recurrent.call_count == 1
        assert conv.call_count == 1
        assert "num_accepted_tokens" not in recurrent.call_args.kwargs
    else:
        expected[0, spec_token_indices[:2], 0] = v[spec_token_indices[:2]]
        expected[0, non_spec_token_indices, 0] = v[non_spec_token_indices]
        assert recurrent.call_count == 2
        assert conv.call_count == 2
        assert recurrent.call_args_list[0].kwargs["num_accepted_tokens"] is accepted

    torch.testing.assert_close(core_attn_out, expected)
    torch.testing.assert_close(
        core_attn_out[:, num_actual_tokens:],
        torch.zeros_like(core_attn_out[:, num_actual_tokens:]),
    )
