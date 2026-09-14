# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.models.glm5next.kda as model_kda


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("dim_first", [False, True])
def test_decode_and_prefill_use_their_own_metadata_and_merge_outputs(monkeypatch, speculative, dim_first):
    layer = model_kda.Glm5NextLinearAttention.__new__(model_kda.Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer"
    layer.local_num_heads = 1
    layer.head_dim = layer.local_projection_size = 128
    layer.kda_lower_bound = -4.0
    state_shape = (8, 384, 6) if dim_first else (8, 6, 384)
    layer.kv_cache = (torch.zeros(state_shape, dtype=torch.bfloat16), torch.zeros(8, 1, 128, 128))
    layer._conv_state_dim_first = dim_first
    layer._merged_conv_weight = None
    # Checkpoints retain FP32 convolution weights; AscendC needs the activation
    # dtype and [width, q|k|v] ordering in the cached packed weight.
    for index, name in enumerate(("q_conv1d", "k_conv1d", "v_conv1d"), start=1):
        setattr(layer, name, SimpleNamespace(bias=None, weight=torch.full((128, 1, 4), float(index))))
    layer.A_log = torch.zeros(1)
    layer.dt_bias = torch.zeros(128)
    tokens = 5 if speculative else 4
    metadata = object.__new__(model_kda.GDNAttentionMetadata)
    values = dict(
        has_initial_state=torch.tensor([True, False]),
        non_spec_query_start_loc=torch.tensor([0, 1, 4]),
        non_spec_state_indices_tensor=torch.tensor([2, 5]),
        num_actual_tokens=tokens,
        spec_sequence_masks=None,
        spec_query_start_loc=None,
        spec_state_indices_tensor=None,
        spec_token_indx=None,
        non_spec_token_indx=None,
        num_accepted_tokens=None,
        num_spec_decodes=0,
        num_prefills=1,
        num_decodes=1,
        num_decode_tokens=1,
        prefill_state_indices=torch.tensor([5]),
        prefill_has_initial_state=torch.tensor([False]),
        non_spec_prefill_metadata=SimpleNamespace(chunk=object()),
    )
    if speculative:
        values.update(
            has_initial_state=torch.tensor([False]),
            non_spec_query_start_loc=torch.tensor([0, 3]),
            non_spec_state_indices_tensor=torch.tensor([5]),
            spec_sequence_masks=torch.tensor([True, False]),
            spec_query_start_loc=torch.tensor([0, 2]),
            spec_state_indices_tensor=torch.tensor([[2, 3]]),
            spec_token_indx=torch.tensor([0, 1]),
            non_spec_token_indx=torch.tensor([2, 3, 4]),
            num_accepted_tokens=torch.tensor([1]),
            num_spec_decodes=1,
            num_decodes=0,
            num_decode_tokens=0,
        )
    for name, value in values.items():
        setattr(metadata, name, value)
    prefill_conv = SimpleNamespace(
        query_start_loc=metadata.non_spec_query_start_loc,
        cache_indices=metadata.non_spec_state_indices_tensor[:, None],
        initial_state_mode=metadata.has_initial_state,
    )
    metadata.non_spec_prefill_metadata.causal_conv1d = prefill_conv
    if speculative:
        metadata.spec_decode_metadata = SimpleNamespace(
            spec_causal_conv1d=SimpleNamespace(
                query_start_loc=metadata.spec_query_start_loc,
                cache_indices=metadata.spec_state_indices_tensor,
                num_accepted_tokens=metadata.num_accepted_tokens,
            )
        )
    monkeypatch.setattr(model_kda, "get_forward_context", lambda: SimpleNamespace(attn_metadata={"layer": metadata}))
    conv_calls = []
    conv_entry = model_kda.causal_conv1d

    def cpu_conv_entry(x, weight, state, *args, **kwargs):
        # This UT checks model dispatch and layout; the NPU tests exercise the
        # non-contiguous state's device-side gather/scatter and original alias.
        assert state.data_ptr() == layer.kv_cache[0].data_ptr()
        assert state.shape == (8, 6, 384)
        return conv_entry(x, weight, state.contiguous(), *args, **kwargs)

    monkeypatch.setattr(model_kda, "causal_conv1d", cpu_conv_entry)

    def conv(output, x, weight, **kwargs):
        conv_calls.append(kwargs["run_mode"])
        state = kwargs["conv_state"]
        assert state.shape == (8, 6, 384)
        assert weight.shape == (4, 384)
        assert weight.dtype == x.dtype == state.dtype == torch.bfloat16
        expected_weight = torch.arange(1, 4, dtype=torch.bfloat16).repeat_interleave(128).expand(4, 384)
        torch.testing.assert_close(weight, expected_weight)
        assert weight is layer._merged_conv_weight
        assert torch.count_nonzero(output) == 0
        if kwargs["run_mode"] == 0:
            assert kwargs["cache_indices_opt"] is prefill_conv.cache_indices
            assert kwargs["initial_state_mode_opt"] is prefill_conv.initial_state_mode
            assert kwargs["num_accepted_tokens_opt"] is None
        else:
            spec_conv = metadata.spec_decode_metadata.spec_causal_conv1d
            assert kwargs["cache_indices_opt"] is spec_conv.cache_indices
            assert kwargs["num_accepted_tokens_opt"] is spec_conv.num_accepted_tokens
            assert kwargs["initial_state_mode_opt"] is None
        # A distinct return catches accidentally ignoring the declared op result.
        return x.clone()

    monkeypatch.setattr(torch.ops._C_ascend, "npu_causal_conv1d_custom", conv, raising=False)
    calls = []

    def recurrent(q, k, v, gate, beta, state, starts, indices, *args):
        calls.append("recurrent")
        assert q.shape[1] == (2 if speculative else 1)
        torch.testing.assert_close(starts, torch.tensor([0, q.shape[1]]))
        return q * 2

    def prefill(q, k, v, gate, beta, state, indices, initial, chunk, *args):
        calls.append("prefill")
        assert q.shape[1] == 3
        assert chunk is metadata.non_spec_prefill_metadata.chunk
        torch.testing.assert_close(indices, metadata.prefill_state_indices)
        return q * 3

    monkeypatch.setattr(model_kda, "recurrent_kda", recurrent)
    monkeypatch.setattr(model_kda, "chunk_kda", prefill)
    qkv = torch.arange(1, tokens + 1, dtype=torch.bfloat16)[:, None].expand(tokens, 384).clone()
    out = torch.full((1, tokens + 1, 1, 128), float("nan"), dtype=torch.bfloat16)
    layer._forward(qkv, torch.zeros(1, tokens, 1, 128), torch.zeros(1, tokens, 1), out)
    expected = torch.arange(1, tokens + 1, dtype=torch.bfloat16) * 3
    expected[: 2 if speculative else 1] = torch.arange(1, 3 if speculative else 2) * 2
    torch.testing.assert_close(out[0, :tokens, 0, 0], expected)
    assert torch.count_nonzero(out[:, tokens:]) == 0
    assert calls == ["recurrent", "prefill"]
    assert conv_calls == ([1, 0] if speculative else [0])


@pytest.mark.parametrize(("width", "num_spec"), [(1, 0), (5, 0), (3, 3)])
def test_unsupported_conv_width_is_rejected_before_execution(monkeypatch, width, num_spec):
    monkeypatch.setattr(
        model_kda.GatedDeltaNetAttention, "__init__", lambda self, *a, **kw: torch.nn.Module.__init__(self)
    )
    config = SimpleNamespace(linear_head_dim=128, linear_conv_kernel_dim=width)
    vllm_config = SimpleNamespace(
        quant_config=None,
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        speculative_config=SimpleNamespace(num_speculative_tokens=num_spec),
    )
    with pytest.raises(ValueError, match="causal-conv requires"):
        model_kda.Glm5NextLinearAttention(config, vllm_config)
