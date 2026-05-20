from types import SimpleNamespace

import pytest
import torch

from vllm_ascend._310p.ops.fla import gdn_310


class FakeGDNAttentionMetadata(SimpleNamespace):
    pass


def _patch_gdn_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gdn_310, "GDNAttentionMetadata", FakeGDNAttentionMetadata)
    monkeypatch.setattr(gdn_310, "is_conv_state_dim_first", lambda: False)
    monkeypatch.setattr(gdn_310, "maybe_save_kv_layer_to_connector", lambda *args, **kwargs: None)


def _make_attention() -> gdn_310.AscendGatedDeltaNetAttention310:
    attn = gdn_310.AscendGatedDeltaNetAttention310.__new__(gdn_310.AscendGatedDeltaNetAttention310)
    attn.prefix = "model.layers.0.linear_attn"
    attn.activation = None
    attn.conv1d = SimpleNamespace(
        weight=torch.arange(18, dtype=torch.float32).reshape(6, 1, 3),
        bias=torch.arange(6, dtype=torch.float32),
    )
    attn.kv_cache = [
        torch.zeros(8, 6, 3, dtype=torch.float16),
        torch.arange(8 * 2 * 4 * 3, dtype=torch.float16).reshape(8, 2, 4, 3),
    ]
    attn.A_log = torch.zeros(2, dtype=torch.float32)
    attn.dt_bias = torch.zeros(2, dtype=torch.float32)

    def rearrange_mixed_qkv(mixed_qkv: torch.Tensor | None):
        if mixed_qkv is None:
            return None, None, None

        total_tokens = mixed_qkv.shape[0]
        q = torch.ones(1, total_tokens, 1, 3, dtype=torch.float16)
        k = 2 * torch.ones(1, total_tokens, 1, 3, dtype=torch.float16)
        v = 3 * torch.ones(1, total_tokens, 2, 4, dtype=torch.float16)
        return q, k, v

    attn.rearrange_mixed_qkv = rearrange_mixed_qkv
    return attn


def _make_metadata(**overrides) -> FakeGDNAttentionMetadata:
    values = dict(
        has_initial_state=torch.zeros(0, dtype=torch.bool),
        spec_query_start_loc=torch.tensor([0], dtype=torch.int32),
        non_spec_query_start_loc=torch.tensor([0], dtype=torch.int32),
        spec_sequence_masks=None,
        spec_token_indx=torch.empty(0, dtype=torch.int64),
        non_spec_token_indx=torch.empty(0, dtype=torch.int64),
        spec_state_indices_tensor=torch.empty((0, 0), dtype=torch.int32),
        non_spec_state_indices_tensor=torch.empty(0, dtype=torch.int32),
        num_actual_tokens=0,
        num_accepted_tokens=None,
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=0,
    )
    values.update(overrides)
    return FakeGDNAttentionMetadata(**values)


def _set_forward_context(monkeypatch: pytest.MonkeyPatch, attn, metadata) -> None:
    forward_context = SimpleNamespace(attn_metadata={attn.prefix: metadata})
    monkeypatch.setattr(gdn_310, "get_forward_context", lambda: forward_context)


def _patch_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_gating(A_log, a, b, dt_bias):
        del A_log, b, dt_bias
        total_tokens = a.shape[0]
        g = -0.25 * torch.ones(1, total_tokens, 2, dtype=torch.float32)
        beta = 0.5 * torch.ones(1, total_tokens, 2, dtype=a.dtype)
        return g, beta

    monkeypatch.setattr(gdn_310, "fused_gdn_gating_pytorch", fake_gating)


def test_flatten_state_indices_width_one_fast_path():
    actual_seq_lengths = torch.tensor([1, 1, 1], dtype=torch.int32)
    state_indices = torch.tensor([[5], [6], [7]], dtype=torch.int64)

    out = gdn_310._flatten_state_indices(
        ssm_state_indices=state_indices,
        actual_seq_lengths=actual_seq_lengths,
        total_tokens=3,
    )

    assert out.dtype == torch.int32
    assert torch.equal(out, torch.tensor([5, 6, 7], dtype=torch.int32))


def test_npu_recurrent_gated_delta_rule_310_flattens_state_indices(monkeypatch):
    op_calls = []

    def fake_recurrent_op(**kwargs):
        op_calls.append(kwargs)
        value = kwargs["value"]
        return torch.zeros(value.shape[0], value.shape[1], value.shape[2], dtype=value.dtype)

    monkeypatch.setattr(
        gdn_310.torch.ops._C_ascend,
        "npu_recurrent_gated_delta_rule_310",
        fake_recurrent_op,
        raising=False,
    )

    q = torch.ones(1, 5, 1, 3, dtype=torch.float16)
    k = torch.ones(1, 5, 1, 3, dtype=torch.float16)
    v = torch.ones(1, 5, 2, 4, dtype=torch.float16)
    g = torch.zeros(1, 5, 2, dtype=torch.float32)
    beta = torch.ones(1, 5, 2, dtype=torch.float16)
    state = torch.zeros(10, 2, 4, 3, dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int64)
    state_indices = torch.tensor([[4, 5, 6], [7, 8, 9]], dtype=torch.int64)
    num_accepted_tokens = torch.tensor([1, 2, 99], dtype=torch.int64)

    out = gdn_310.npu_recurrent_gated_delta_rule_310(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        state=state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=False,
    )

    assert out.shape == (1, 5, 2, 4)
    call = op_calls[0]
    assert call["query"].shape == (5, 1, 3)
    assert call["actual_seq_lengths"].dtype == torch.int32
    assert torch.equal(call["actual_seq_lengths"], torch.tensor([2, 3], dtype=torch.int32))
    assert call["ssm_state_indices"].dtype == torch.int32
    assert torch.equal(call["ssm_state_indices"], torch.tensor([4, 5, 7, 8, 9], dtype=torch.int32))
    assert call["num_accepted_tokens"].dtype == torch.int32
    assert torch.equal(call["num_accepted_tokens"], torch.tensor([1, 2], dtype=torch.int32))
    assert call["scale_value"] == pytest.approx(3**-0.5)


def test_gdn_310_forward_core_decode_uses_recurrent_op(monkeypatch):
    _patch_gdn_runtime(monkeypatch)
    _patch_gating(monkeypatch)

    attn = _make_attention()
    metadata = _make_metadata(
        non_spec_query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([5, 6, 7, 99], dtype=torch.int32),
        num_actual_tokens=3,
        num_decodes=3,
    )
    _set_forward_context(monkeypatch, attn, metadata)

    conv_calls = []

    def fake_causal_conv1d(input_tensor, conv_weights, **kwargs):
        conv_calls.append((input_tensor, conv_weights, kwargs))
        return input_tensor + 1

    recurrent_calls = []
    recurrent_out = torch.arange(3 * 2 * 4, dtype=torch.float16).reshape(1, 3, 2, 4)

    def fake_recurrent(**kwargs):
        recurrent_calls.append(kwargs)
        return recurrent_out

    monkeypatch.setattr(
        gdn_310.torch.ops._C_ascend,
        "npu_causal_conv1d_310",
        fake_causal_conv1d,
        raising=False,
    )
    monkeypatch.setattr(gdn_310, "npu_recurrent_gated_delta_rule_310", fake_recurrent)

    mixed_qkv = torch.zeros(5, 6, dtype=torch.float16)
    b = torch.zeros(5, 2, dtype=torch.float16)
    a = torch.zeros(5, 2, dtype=torch.float16)
    core_attn_out = torch.zeros(5, 2, 4, dtype=torch.float16)

    attn._forward_core(mixed_qkv, b, a, core_attn_out)

    conv_input, _, conv_kwargs = conv_calls[0]
    assert conv_input.shape[0] == 3
    assert conv_kwargs["run_mode"] == 1
    assert torch.equal(conv_kwargs["cache_indices"], torch.tensor([5, 6, 7], dtype=torch.int64))

    recurrent_call = recurrent_calls[0]
    assert recurrent_call["cu_seqlens"] is metadata.non_spec_query_start_loc
    assert recurrent_call["ssm_state_indices"] is metadata.non_spec_state_indices_tensor
    assert recurrent_call["g"].shape == (1, 3, 2)
    assert recurrent_call["beta"].shape == (1, 3, 2)
    torch.testing.assert_close(core_attn_out[:3], recurrent_out.squeeze(0))
    torch.testing.assert_close(core_attn_out[3:], torch.zeros_like(core_attn_out[3:]))


def test_gdn_310_forward_core_prefill_uses_chunk_fallback_and_updates_state(monkeypatch):
    _patch_gdn_runtime(monkeypatch)
    _patch_gating(monkeypatch)

    attn = _make_attention()
    original_ssm_state = attn.kv_cache[1].clone()
    metadata = _make_metadata(
        has_initial_state=torch.tensor([True, False], dtype=torch.bool),
        non_spec_query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([1, 3], dtype=torch.int32),
        num_actual_tokens=5,
        num_prefills=2,
    )
    _set_forward_context(monkeypatch, attn, metadata)

    conv_calls = []

    def fake_causal_conv1d(input_tensor, conv_weights, **kwargs):
        conv_calls.append((input_tensor, conv_weights, kwargs))
        return input_tensor

    chunk_calls = []
    chunk_out = 11 * torch.ones(1, 5, 2, 4, dtype=torch.float16)
    last_state = 7 * torch.ones(2, 2, 4, 3, dtype=torch.float32)

    def fake_chunk_gated_delta_rule_pytorch(**kwargs):
        chunk_calls.append(kwargs)
        initial_state = kwargs["initial_state"]
        torch.testing.assert_close(initial_state[0], original_ssm_state[1])
        torch.testing.assert_close(initial_state[1], torch.zeros_like(initial_state[1]))
        return chunk_out, last_state

    monkeypatch.setattr(
        gdn_310.torch.ops._C_ascend,
        "npu_causal_conv1d_310_host",
        fake_causal_conv1d,
        raising=False,
    )
    monkeypatch.setattr(
        gdn_310.torch.ops._C_ascend,
        "npu_causal_conv1d_310",
        lambda *args, **kwargs: pytest.fail("prefill should use host metadata op"),
        raising=False,
    )
    monkeypatch.setattr(gdn_310, "chunk_gated_delta_rule_pytorch", fake_chunk_gated_delta_rule_pytorch)

    mixed_qkv = torch.zeros(5, 6, dtype=torch.float16)
    b = torch.zeros(5, 2, dtype=torch.float16)
    a = torch.zeros(5, 2, dtype=torch.float16)
    core_attn_out = torch.zeros(5, 2, 4, dtype=torch.float16)

    attn._forward_core(mixed_qkv, b, a, core_attn_out)

    _, _, conv_kwargs = conv_calls[0]
    assert conv_kwargs["run_mode"] == 0
    assert conv_kwargs["query_start_loc"] == (0, 2, 5)
    assert conv_kwargs["cache_indices"] == (1, 3)
    assert conv_kwargs["initial_state_mode"] == (1, 0)
    assert conv_kwargs["num_accepted_tokens"] == ()

    chunk_call = chunk_calls[0]
    assert chunk_call["cu_seqlens"] is metadata.non_spec_query_start_loc
    assert chunk_call["head_first"] is False
    assert chunk_call["use_qk_l2norm_in_kernel"] is True
    torch.testing.assert_close(core_attn_out, chunk_out.squeeze(0))
    torch.testing.assert_close(attn.kv_cache[1][1], last_state[0].to(torch.float16))
    torch.testing.assert_close(attn.kv_cache[1][3], last_state[1].to(torch.float16))
