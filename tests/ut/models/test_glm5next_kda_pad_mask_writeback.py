# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contract tests for the KDA padding-mask writeback on the spec-verify path."""

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.models.glm5next.kda as model_kda


def _build_layer() -> "model_kda.Glm5NextLinearAttention":
    layer = model_kda.Glm5NextLinearAttention.__new__(model_kda.Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer"
    layer.local_num_heads = 1
    layer.head_dim = layer.local_projection_size = 128
    layer.kda_lower_bound = -4.0
    layer.kv_cache = (torch.zeros(8, 3, 384), torch.zeros(8, 1, 128, 128))
    layer._conv_state_dim_first = False
    # Skip the merged-weight build: the dispatch itself is under test.
    layer._merged_conv_weight = torch.ones(4, 384)
    layer.A_log = torch.zeros(1)
    layer.dt_bias = torch.zeros(128)
    return layer


def _build_metadata(tokens: int, spec_token_indx, non_spec_token_indx) -> "model_kda.GDNAttentionMetadata":
    metadata = object.__new__(model_kda.GDNAttentionMetadata)
    for key, value in dict(
        non_spec_query_start_loc=torch.tensor([0, tokens]),
        non_spec_state_indices_tensor=torch.tensor([5]),
        num_actual_tokens=tokens,
        spec_sequence_masks=torch.tensor([True]),
        spec_query_start_loc=torch.tensor([0, tokens]),
        spec_state_indices_tensor=torch.tensor([[2, 3]]),
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        num_accepted_tokens=torch.tensor([1]),
        num_spec_decodes=1,
        # No non-spec conv runs below; the spec branch owns the whole step.
        num_prefills=0,
        num_decodes=0,
        num_decode_tokens=0,
    ).items():
        setattr(metadata, key, value)
    metadata.spec_decode_metadata = SimpleNamespace(
        spec_causal_conv1d=SimpleNamespace(
            query_start_loc=metadata.spec_query_start_loc,
            cache_indices=metadata.spec_state_indices_tensor,
            num_accepted_tokens=metadata.num_accepted_tokens,
            output_buffer=None,
        )
    )
    return metadata


def _run(monkeypatch, metadata, layer, qkv, target, seen):
    monkeypatch.setattr(model_kda, "get_forward_context", lambda: SimpleNamespace(attn_metadata={"layer": metadata}))
    monkeypatch.setattr(model_kda, "wait_for_kv_layer_from_connector", lambda prefix: None)
    monkeypatch.setattr(model_kda, "record_attention_compute_start", lambda: None)
    monkeypatch.setattr(model_kda, "maybe_save_kv_layer_to_connector", lambda *args: None)
    monkeypatch.setattr(model_kda, "causal_conv1d", lambda x, *args, **kwargs: x)

    def recurrent(q, k, v, gate, beta, state, starts, indices, *args, **kwargs):
        seen.append(kwargs.get("output_buffer"))
        if kwargs.get("output_buffer") is not None:
            # Reference writeback: copy the valid rows, clear the capacity rows.
            kwargs["output_buffer"].zero_()
            kwargs["output_buffer"][:, : q.shape[1]].copy_(q * 2)
            return kwargs["output_buffer"]
        return q * 2

    monkeypatch.setattr(model_kda, "recurrent_kda", recurrent)
    layer._forward(qkv, torch.zeros(1, qkv.shape[0], 1, 128), torch.zeros(1, qkv.shape[0], 1), target)


def test_pure_spec_step_writes_the_captured_buffer_and_skips_the_scatter(monkeypatch):
    tokens, capacity = 2, 4
    layer = _build_layer()
    # The captured buffer is longer than the scheduled tokens; index 4 does not
    # exist in it, so a surviving identity scatter would raise.
    metadata = _build_metadata(tokens, torch.tensor([0, 1, 2, 3, 4]), None)
    qkv = torch.arange(1, tokens * 384 + 1, dtype=torch.bfloat16).reshape(tokens, 384)
    target = torch.full((1, capacity, 1, 128), float("nan"), dtype=torch.bfloat16)
    seen: list[torch.Tensor | None] = []

    _run(monkeypatch, metadata, layer, qkv, target, seen)

    assert seen == [target]
    torch.testing.assert_close(target[0, :tokens, 0, 0], torch.arange(1, tokens + 1, dtype=torch.bfloat16) * 2)
    # Padding rows are cleared by the writeback instead of by a mask kernel.
    assert torch.count_nonzero(target[:, tokens:]) == 0


def test_mixed_step_keeps_the_scatter_into_the_layer_buffer(monkeypatch):
    tokens = 3
    layer = _build_layer()
    metadata = _build_metadata(tokens, torch.tensor([0, 1]), torch.tensor([2]))
    qkv = torch.arange(1, tokens * 384 + 1, dtype=torch.bfloat16).reshape(tokens, 384)
    target = torch.full((1, tokens, 1, 128), float("nan"), dtype=torch.bfloat16)
    seen: list[torch.Tensor | None] = []

    _run(monkeypatch, metadata, layer, qkv, target, seen)

    # Both token groups exist, so the recurrent kernel still returns a
    # temporary that the dispatch scatters into the layer buffer.
    assert seen == [None]
    torch.testing.assert_close(target[0, :2, 0, 0], torch.arange(1, 3, dtype=torch.bfloat16) * 2)
    # The non-spec row is written by the non-spec path, not by the scatter.
    assert torch.isnan(target[0, 2]).all()


@pytest.mark.parametrize("non_spec", [None, torch.tensor([], dtype=torch.int64)])
def test_step_without_non_spec_tokens_uses_the_writeback(monkeypatch, non_spec):
    tokens = 2
    layer = _build_layer()
    metadata = _build_metadata(tokens, torch.tensor([0, 1]), non_spec)
    qkv = torch.arange(1, tokens * 384 + 1, dtype=torch.bfloat16).reshape(tokens, 384)
    target = torch.full((1, tokens, 1, 128), float("nan"), dtype=torch.bfloat16)
    seen: list[torch.Tensor | None] = []

    _run(monkeypatch, metadata, layer, qkv, target, seen)

    assert seen == [target]
