# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionMetadataBuilder,
    AscendGDNHostMetadataBuilder,
    GDNCausalConv1dMetadata,
    GDNDecodeMetadata,
    GDNSpecCausalConv1dMetadata,
    GDNSpecDecodeMetadata,
)

conv_ops = importlib.import_module("vllm_ascend.models.glm5next.ops.causal_conv1d")


@pytest.fixture(params=[0, 3])
def output_builder(monkeypatch, request):
    # Isolate the output-buffer lifetime from request classification in the
    # parent builder; the existing builder tests cover that classification.
    builder = object.__new__(AscendGDNHostMetadataBuilder)
    builder._conv_layer_name = "layer"
    builder._spec_conv_indices = builder._non_spec_conv_indices = None
    builder._conv_output_buffer = None
    builder.spec_sequence_masks = torch.empty(8, dtype=torch.bool)
    builder.decode_cudagraph_max_bs = 8
    builder.num_spec = request.param
    builder.compilation_config = SimpleNamespace(
        static_forward_context={"layer": SimpleNamespace(use_shared_conv_output=True, local_projection_size=8)}
    )
    builder.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=request.param) if request.param else None,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    monkeypatch.setattr(
        AscendGDNAttentionMetadataBuilder,
        "build",
        lambda self, prefix, metadata, **kwargs: metadata,
    )
    return builder


def _metadata(speculative, *, empty=False):
    starts = torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32)
    indices = torch.tensor([1, 2, 0, 0], dtype=torch.int32)
    metadata = SimpleNamespace(
        spec_decode_metadata=None,
        non_spec_decode_metadata=None,
        non_spec_prefill_metadata=None,
    )
    if empty:
        return metadata
    if speculative:
        metadata.spec_decode_metadata = GDNSpecDecodeMetadata(
            spec_causal_conv1d=GDNSpecCausalConv1dMetadata(starts, indices, torch.ones(4, dtype=torch.int32)),
            actual_seq_lengths=None,
        )
    else:
        metadata.non_spec_decode_metadata = GDNDecodeMetadata(
            causal_conv1d=GDNCausalConv1dMetadata(starts, indices, None),
            actual_seq_lengths=starts,
        )
    return metadata


def _conv_metadata(metadata):
    if metadata.spec_decode_metadata is not None:
        return metadata.spec_decode_metadata.spec_causal_conv1d
    return metadata.non_spec_decode_metadata.causal_conv1d


def test_shared_conv_output_is_reset_without_replacing_captured_storage(output_builder):
    speculative = output_builder.vllm_config.speculative_config is not None
    first = output_builder.build(0, _metadata(speculative))
    captured = _conv_metadata(first).output_buffer
    assert captured.shape[0] >= 4 * (4 if speculative else 1)
    assert captured.shape[1] == 24
    assert captured.dtype == torch.bfloat16

    # A captured graph keeps its original view even as the logical batch
    # shrinks or a captured decode branch becomes entirely inactive.
    captured_view = captured[:4]
    for live in (4, 2, 0, 4):
        captured.fill_(float("nan"))
        metadata = output_builder.build(0, _metadata(speculative, empty=live == 0))
        assert output_builder._conv_output_buffer is captured
        assert torch.count_nonzero(captured_view) == 0
        if live:
            assert _conv_metadata(metadata).output_buffer is captured
            captured[:live].fill_(live)
            assert torch.count_nonzero(captured[live:]) == 0


def test_unmarked_model_does_not_allocate_conv_output(output_builder):
    output_builder.compilation_config.static_forward_context["layer"].use_shared_conv_output = False
    metadata = output_builder.build(0, _metadata(False))
    assert output_builder._conv_output_buffer is None
    assert _conv_metadata(metadata).output_buffer is None


def test_layers_reuse_conv_output_without_clearing_active_rows(output_builder, monkeypatch):
    speculative = output_builder.vllm_config.speculative_config is not None
    metadata = output_builder.build(0, _metadata(speculative))
    conv = _conv_metadata(metadata)
    x = torch.arange(4 * 24, dtype=torch.bfloat16).reshape(4, 24)
    state = torch.zeros(6, 3, 24, dtype=x.dtype)
    weight = torch.ones(4, 24, dtype=x.dtype)
    calls = []

    def update(x, state, weight, **kwargs):
        output = kwargs["out"]
        assert output.data_ptr() == conv.output_buffer.data_ptr()
        # The preceding layer has already written active rows. Its unused
        # rows must stay zero without another full destination clear.
        if calls:
            torch.testing.assert_close(output[:2], x[:2] + calls[-1])
        calls.append(len(calls) + 1)
        output[:2].copy_(x[:2] + calls[-1])
        return output

    monkeypatch.setattr(conv_ops, "causal_conv1d_update", update)
    for layer in (1, 2):
        with patch.object(torch, "zeros_like", side_effect=AssertionError("Unexpected per-layer output clear")):
            result = conv_ops.causal_conv1d(
                x,
                weight,
                state,
                conv.query_start_loc,
                conv.cache_indices,
                run_mode=1,
                output_buffer=conv.output_buffer,
            )
        torch.testing.assert_close(result[:2], x[:2] + layer)
        assert torch.count_nonzero(result[2:]) == 0


def test_conv_output_capacity_is_checked(output_builder):
    metadata = output_builder.build(0, _metadata(False))
    conv = _conv_metadata(metadata)
    with pytest.raises(ValueError, match="output buffer must cover"):
        conv_ops.causal_conv1d(
            torch.ones(4, 24, dtype=torch.bfloat16),
            torch.ones(4, 24, dtype=torch.bfloat16),
            torch.zeros(6, 3, 24, dtype=torch.bfloat16),
            conv.query_start_loc,
            conv.cache_indices,
            run_mode=1,
            output_buffer=conv.output_buffer[:3],
        )
