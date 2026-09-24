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
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

from tests.ut.ops.test_gdn_attn_builder import (
    BatchSpec,
    _build_attn_metadata,
    _patch_missing_runtime_cdiv,
)
from vllm_ascend.ops.bailing_moe_v3_kda import (
    AscendBailingMoeV3KimiDeltaAttention,
    _zero_padded_spec_output,
)


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
@pytest.mark.parametrize("has_empty_sequence", [False, True])
def test_prefill_uses_current_chunk_contract_and_writes_back_state(
    safe_gate: bool,
    expected_lower_bound: float | None,
    has_empty_sequence: bool,
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
    recurrent_state = torch.arange(16, dtype=torch.float32).reshape(4, 1, 2, 2)
    original_state = recurrent_state.clone()
    state_indices = torch.tensor([1, 2, 3] if has_empty_sequence else [1, 3], dtype=torch.int64)
    has_initial_state = torch.tensor([True, True, False] if has_empty_sequence else [True, False])
    metadata = SimpleNamespace(
        cu_seqlens_host=(0, 1, 1, 3) if has_empty_sequence else (0, 1, 3),
        cu_seqlens_kern=(0, 1, 3) if has_empty_sequence else None,
        keep_meta=torch.tensor([True, False, True]) if has_empty_sequence else None,
        chunk_indices_chunk64_host=(0, 0),
    )
    output = torch.randn_like(v)
    final_state = torch.randn(2, 1, 2, 2, dtype=torch.float32)

    with (
        patch(
            "vllm_ascend.ops.bailing_moe_v3_kda.clear_ssm_states",
            side_effect=lambda state, flags: state.masked_fill_(~flags[:, None, None, None], 0),
        ) as clear_states,
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
    torch.testing.assert_close(clear_states.call_args.args[1], torch.tensor([True, False]))
    torch.testing.assert_close(initial_state[0], original_state[1])
    assert torch.count_nonzero(initial_state[1]) == 0
    expected_args = (
        q,
        k,
        v,
        raw_gate,
        beta,
        initial_state,
        metadata.cu_seqlens_kern if has_empty_sequence else metadata.cu_seqlens_host,
        metadata.chunk_indices_chunk64_host,
        attention.A_log,
        attention.dt_bias,
    )
    assert len(run_chunk.call_args.args) == len(expected_args)
    for actual_arg, expected_arg in zip(run_chunk.call_args.args, expected_args):
        assert actual_arg is expected_arg
    assert run_chunk.call_args.kwargs == {"lower_bound": expected_lower_bound}
    torch.testing.assert_close(recurrent_state[[1, 3]], final_state)
    torch.testing.assert_close(recurrent_state[[0, 2]], original_state[[0, 2]])


@pytest.mark.parametrize("layout", ["sd", "ds", "page_strided"])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("run_mode", [0, 1])
def test_causal_conv_updates_only_active_cache_rows(layout, cache_dtype, run_mode):
    """Only explicit pad/null sentinels may bypass FLA cache bounds."""
    attention = _new_attention()
    attention._conv_state_dim_first = layout == "ds"
    mixed_qkv = torch.randn(6, 6, dtype=torch.bfloat16)
    conv_weights_t = torch.randn(3, 6, dtype=torch.bfloat16)
    shape = (9, 6, 3) if layout == "ds" else (9, 3, 6)
    backing = torch.arange(18 * 18, dtype=cache_dtype).reshape(18, *shape[1:]) % 13
    storage = backing[::2] if layout == "page_strided" else backing[:9]
    original_backing = backing.clone()
    state_sd = storage.transpose(-1, -2) if layout == "ds" else storage
    # Include a reserved null block, an empty request and a padding index.
    query_start_loc = torch.tensor([0, 2, 3, 3, 4, 6], dtype=torch.int32)
    cache_indices = torch.tensor([[2, 8], [NULL_BLOCK_ID, 8], [5, 8], [PAD_SLOT_ID, 8], [4, 8]], dtype=torch.int32)
    initial_state_mode = torch.tensor([True, False, True, False, True])
    accepted = torch.ones(5, dtype=torch.int32)
    result = torch.full_like(mixed_qkv, 11)
    staged = layout != "sd" or cache_dtype != mixed_qkv.dtype

    def copy_state(cache, packed, indices, starts, packed_indices, *, write_back):
        assert packed.shape == (5, 3, 6)  # Requests, never the nine-slot pool.
        assert packed.dtype == mixed_qkv.dtype
        assert packed.is_contiguous()
        for request, slot in enumerate(indices.tolist()):
            active = 0 <= slot < cache.shape[0] and starts[request + 1] > starts[request]
            if write_back:
                if active:
                    cache[slot].copy_(packed[request])
            else:
                packed[request].copy_(cache[slot] if active else torch.zeros_like(packed[request]))
                packed_indices[request] = request if active else PAD_SLOT_ID

    def apply_conv(input_, weight, state, indices, kwargs):
        assert input_ is mixed_qkv and weight is conv_weights_t
        assert state.dtype == mixed_qkv.dtype
        assert state.is_contiguous()
        assert indices.dim() == 1 and indices.is_contiguous()
        assert kwargs["query_start_loc"] is query_start_loc
        assert kwargs["activation"] == "silu"
        expected_null_block_id = cache_indices.shape[0] if staged else NULL_BLOCK_ID
        assert kwargs["null_block_id"] == expected_null_block_id
        if staged:
            assert indices.tolist() == [0, expected_null_block_id, expected_null_block_id, expected_null_block_id, 4]
        else:
            assert state is storage
            assert indices.tolist() == [2, NULL_BLOCK_ID, 5, NULL_BLOCK_ID, 4]
        pad_slot_id = kwargs.get("pad_slot_id")
        for request, slot in enumerate(indices.tolist()):
            if slot == kwargs["null_block_id"] or slot == pad_slot_id:
                continue
            assert 0 <= slot < state.shape[0]
            if query_start_loc[request + 1] == query_start_loc[request]:
                continue
            state[slot].add_(7)
        return result

    def prefill(input_, weight, bias, **kwargs):
        assert bias is None
        assert kwargs["has_initial_state"] is initial_state_mode
        assert kwargs["pad_slot_id"] == PAD_SLOT_ID
        return apply_conv(input_, weight, kwargs["conv_states"], kwargs["cache_indices"], kwargs)

    def decode(input_, state, weight, **kwargs):
        assert kwargs["num_accepted_tokens"] is accepted
        assert kwargs["max_query_len"] == 2
        assert torch.count_nonzero(kwargs["out"]) == 0
        return apply_conv(input_, weight, state, kwargs["conv_state_indices"], kwargs)

    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.copy_conv_state", side_effect=copy_state) as copy_op,
        patch("vllm_ascend.ops.bailing_moe_v3_kda.causal_conv1d_fn", side_effect=prefill) as prefill_op,
        patch("vllm_ascend.ops.bailing_moe_v3_kda.causal_conv1d_update", side_effect=decode) as decode_op,
    ):
        actual = attention._run_causal_conv1d(
            mixed_qkv,
            conv_weights_t,
            storage,
            query_start_loc,
            cache_indices,
            initial_state_mode,
            run_mode=run_mode,
            max_query_len=2,
            num_accepted_tokens=accepted if run_mode else None,
        )

    assert actual is result
    assert prefill_op.call_count == (run_mode == 0)
    assert decode_op.call_count == (run_mode == 1)
    assert copy_op.call_count == (2 if staged else 0)
    expected = original_backing
    for slot in (2, 4):
        expected[slot * (2 if layout == "page_strided" else 1)].add_(7)
    torch.testing.assert_close(backing, expected)
    assert state_sd.shape == (9, 3, 6)


def test_causal_conv_packed_all_null_mrv2_uses_nonnegative_sentinel():
    """FLA update filters only a configured nonnegative null-block ID."""
    attention = _new_attention()
    attention._conv_state_dim_first = False
    requests, tokens_per_request = 32, 4
    state_len, dim = 6, 6
    mixed_qkv = torch.randn(requests * tokens_per_request, dim, dtype=torch.bfloat16)
    conv_weights_t = torch.randn(4, dim, dtype=torch.bfloat16)
    conv_state = torch.randn(8, state_len, dim, dtype=torch.float32)
    original_state = conv_state.clone()
    query_start_loc = torch.arange(
        0,
        requests * tokens_per_request + 1,
        tokens_per_request,
        dtype=torch.int32,
    )
    cache_indices = torch.full((requests, tokens_per_request), NULL_BLOCK_ID, dtype=torch.int32)
    accepted = torch.ones(requests, dtype=torch.int32)

    def copy_state(cache, packed, indices, starts, packed_indices, *, write_back):
        assert cache is conv_state
        assert packed.shape == (requests, state_len, dim)
        assert packed.dtype == mixed_qkv.dtype
        assert torch.equal(indices, torch.full_like(indices, PAD_SLOT_ID))
        assert starts is query_start_loc
        if not write_back:
            packed.zero_()
            packed_indices.fill_(PAD_SLOT_ID)

    def decode(input_, state, weight, **kwargs):
        assert input_ is mixed_qkv and weight is conv_weights_t
        assert state.shape == (requests, state_len, dim)
        assert state.dtype == mixed_qkv.dtype
        assert kwargs["query_start_loc"] is query_start_loc
        assert kwargs["num_accepted_tokens"] is accepted
        assert kwargs["max_query_len"] == tokens_per_request
        assert kwargs["null_block_id"] == requests
        indices = kwargs["conv_state_indices"]
        assert indices.tolist() == [requests] * requests
        assert torch.all(indices >= 0)
        assert torch.count_nonzero(kwargs["out"]) == 0
        return kwargs["out"]

    with (
        patch("vllm_ascend.ops.bailing_moe_v3_kda.copy_conv_state", side_effect=copy_state) as copy_op,
        patch("vllm_ascend.ops.bailing_moe_v3_kda.causal_conv1d_update", side_effect=decode) as decode_op,
    ):
        actual = attention._run_causal_conv1d(
            mixed_qkv,
            conv_weights_t,
            conv_state,
            query_start_loc,
            cache_indices,
            None,
            run_mode=1,
            max_query_len=tokens_per_request,
            num_accepted_tokens=accepted,
        )

    assert torch.equal(actual, torch.zeros_like(actual))
    torch.testing.assert_close(conv_state, original_state)
    assert copy_op.call_count == 2
    decode_op.assert_called_once()


@pytest.mark.parametrize("metadata", [None, {}, {"other_layer": object()}])
def test_profile_clears_preallocated_output(metadata):
    attention = _new_attention()
    attention.prefix = "model.layers.0.linear_attn"
    core_attn_out = torch.full((1, 5, 1, 2), torch.nan)

    with patch(
        "vllm_ascend.ops.bailing_moe_v3_kda.get_forward_context",
        return_value=SimpleNamespace(attn_metadata=metadata),
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


def test_spec_padding_discards_unwritten_nan_output():
    output = torch.full((1, 6, 1, 2), torch.nan)
    output[:, :4] = 3
    query_start_loc = torch.tensor([0, 2, 4, 4], dtype=torch.int32)
    actual = _zero_padded_spec_output(output, query_start_loc)
    torch.testing.assert_close(actual[:, :4], output[:, :4])
    assert torch.equal(actual[:, 4:], torch.zeros_like(actual[:, 4:]))


@pytest.mark.parametrize(
    ("seq_lens", "query_lens", "draft_tokens"),
    [
        pytest.param([3, 4], [3, 2], None, id="prefill"),
        pytest.param([10, 20], [1, 1], None, id="decode"),
        pytest.param([10, 8, 7], [1, 4, 2], None, id="decode-prefill"),
        pytest.param([10, 20], [3, 3], [2, 2], id="spec"),
        pytest.param([10, 7, 20], [3, 4, 1], [2, -1, -1], id="spec-prefill-decode"),
    ],
)
def test_forward_routes_real_builder_metadata_and_clears_tail(monkeypatch, seq_lens, query_lens, draft_tokens):
    _patch_missing_runtime_cdiv(monkeypatch)
    num_speculative_tokens = 2 if draft_tokens is not None else 0
    _, _, metadata = _build_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        num_speculative_tokens=num_speculative_tokens,
        num_decode_draft_tokens_cpu=None if draft_tokens is None else torch.tensor(draft_tokens, dtype=torch.int32),
    )
    num_actual_tokens = sum(query_lens)
    num_allocated_tokens = num_actual_tokens + 2
    head_dim = 2
    q = torch.arange(num_allocated_tokens * head_dim, dtype=torch.float32).reshape(-1, head_dim)
    k = q + 100
    v = q + 200
    raw_gate = torch.zeros(1, num_allocated_tokens, 1, head_dim)
    beta = torch.full((1, num_allocated_tokens, 1), 0.5)

    def conv_forward(mixed, _weight, _state, starts, *_args, **_kwargs):
        assert starts[-1] == mixed.shape[0]
        return mixed

    def recurrent_forward(_q, _k, value, _gate, _beta, _state, starts, *_args, **_kwargs):
        assert starts[-1] == value.shape[1]
        return value + 10

    def prefill_forward(_q, _k, value, _gate, _beta, _state, state_indices, has_state, chunk_metadata):
        assert chunk_metadata.cu_seqlens_host[-1] == value.shape[1]
        assert state_indices is metadata.prefill_state_indices
        assert has_state is metadata.prefill_has_initial_state
        return value + 20

    conv = MagicMock(side_effect=conv_forward)
    recurrent = MagicMock(side_effect=recurrent_forward)
    prefill = MagicMock(side_effect=prefill_forward)
    attention = SimpleNamespace(
        prefix="model.layers.0.linear_attn",
        head_dim=head_dim,
        num_speculative_tokens=num_speculative_tokens,
        kv_cache=(torch.zeros(1), torch.zeros(1)),
        _conv_weights_t=lambda _dtype: torch.empty(1),
        _run_causal_conv1d=conv,
        _run_recurrent=recurrent,
        _run_prefill=prefill,
    )
    core_attn_out = torch.full((1, num_allocated_tokens, 1, head_dim), torch.nan)

    with patch(
        "vllm_ascend.ops.bailing_moe_v3_kda.get_forward_context",
        return_value=SimpleNamespace(attn_metadata={attention.prefix: metadata}),
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
    if metadata.spec_sequence_masks is not None:
        assert metadata.num_decodes == 0
        spec_indices = metadata.spec_token_indx
        expected[0, spec_indices, 0] = v[spec_indices] + 10
        if metadata.num_prefills:
            non_spec_indices = metadata.non_spec_token_indx
            expected[0, non_spec_indices, 0] = v[non_spec_indices] + 20
        assert (
            recurrent.call_args.kwargs["num_accepted_tokens"]
            is metadata.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens
        )
        assert conv.call_args_list[0].kwargs["max_query_len"] == num_speculative_tokens + 1
    else:
        split = metadata.num_decode_tokens
        expected[0, :split, 0] = v[:split] + 10
        expected[0, split:num_actual_tokens, 0] = v[split:num_actual_tokens] + 20
    assert prefill.call_count == (metadata.num_prefills > 0)
    assert recurrent.call_count == (metadata.num_decodes > 0 or metadata.spec_sequence_masks is not None)
    assert conv.call_count == (2 if metadata.num_prefills and metadata.spec_sequence_masks is not None else 1)

    torch.testing.assert_close(core_attn_out, expected)
    torch.testing.assert_close(
        core_attn_out[:, num_actual_tokens:],
        torch.zeros_like(core_attn_out[:, num_actual_tokens:]),
    )
