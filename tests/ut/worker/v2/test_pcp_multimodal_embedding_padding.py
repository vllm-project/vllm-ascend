from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from vllm_ascend.worker.v2.model_states.default import AscendModelState


@pytest.mark.parametrize(
    ("actual_tokens", "padded_tokens", "mm_positions"),
    [(10, 12, ()), (8, 10, (2,))],
)
def test_pcp_multimodal_embedding_keeps_common_padded_span(actual_tokens, padded_tokens, mm_positions):
    state = AscendModelState.__new__(AscendModelState)
    state.pcp_manager = object()
    state.supports_mm_inputs = True
    state.mm_pruner = None
    state.prompt_embeds_state = None
    state.execute_mm_encoder = Mock()
    embeddings = torch.arange(padded_tokens * 4).reshape(padded_tokens, 4)
    state.encoder_runner = SimpleNamespace(get_inputs_embeds=Mock(return_value=embeddings))

    input_ids = torch.arange(1, actual_tokens + 1)
    input_ids = torch.nn.functional.pad(input_ids, (0, padded_tokens - actual_tokens))
    input_batch = SimpleNamespace(
        input_ids=input_ids,
        num_tokens=actual_tokens,
        num_tokens_after_padding=padded_tokens,
    )
    is_mm_embed = torch.zeros(actual_tokens, dtype=torch.bool)
    is_mm_embed[list(mm_positions)] = True
    mm_embeds = [torch.ones((len(mm_positions), 4))] if mm_positions else []

    with patch.object(DefaultModelState, "gather_mm_embeddings", return_value=(mm_embeds, is_mm_embed)):
        result = state.prepare_inputs_embeds({}, input_batch, req_states=None)

    state.execute_mm_encoder.assert_called_once_with({})
    passed_ids, passed_mm_embeds, passed_mask = state.encoder_runner.get_inputs_embeds.call_args.args
    assert torch.equal(passed_ids, input_ids)
    assert passed_mm_embeds is mm_embeds
    assert torch.equal(passed_mask[:actual_tokens], is_mm_embed)
    assert not passed_mask[actual_tokens:].any()
    assert passed_mask.shape == (padded_tokens,)
    assert torch.equal(result, embeddings)


def test_non_pcp_embedding_keeps_upstream_behavior():
    state = AscendModelState.__new__(AscendModelState)
    state.pcp_manager = None
    sentinel = object()
    with patch.object(DefaultModelState, "prepare_inputs_embeds", return_value=sentinel) as upstream:
        result = state.prepare_inputs_embeds({}, object(), req_states=None)

    assert result is sentinel
    upstream.assert_called_once()


def test_pcp_embedding_without_padding_keeps_upstream_behavior():
    state = AscendModelState.__new__(AscendModelState)
    state.pcp_manager = object()
    state.supports_mm_inputs = True
    input_batch = SimpleNamespace(num_tokens=12, num_tokens_after_padding=12)
    sentinel = object()
    with patch.object(DefaultModelState, "prepare_inputs_embeds", return_value=sentinel) as upstream:
        result = state.prepare_inputs_embeds({}, input_batch, req_states=None)

    assert result is sentinel
    upstream.assert_called_once()
