from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelState

from vllm_ascend.worker.v2.model_states.default import AscendModelState


@pytest.mark.parametrize(
    ("actual_tokens", "padded_tokens", "mm_positions"),
    [(10, 12, ()), (8, 10, (2,))],
)
def test_pcp_multimodal_embedding_keeps_common_padded_span(actual_tokens, padded_tokens, mm_positions):
    state = AscendModelState.__new__(AscendModelState)
    state.model = SimpleNamespace(set_embedding_num_tokens_padded=Mock())
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

    with patch.object(ModelState, "gather_mm_embeddings", return_value=(mm_embeds, is_mm_embed)):
        result = state.prepare_inputs_embeds({}, input_batch, req_states=None)

    state.execute_mm_encoder.assert_called_once_with({})
    passed_ids, passed_mm_embeds, passed_mask = state.encoder_runner.get_inputs_embeds.call_args.args
    assert torch.equal(passed_ids, input_ids)
    assert passed_mm_embeds is mm_embeds
    assert torch.equal(passed_mask[:actual_tokens], is_mm_embed)
    assert not passed_mask[actual_tokens:].any()
    assert passed_mask.shape == (padded_tokens,)
    assert torch.equal(result, embeddings)
    state.model.set_embedding_num_tokens_padded.assert_called_once_with(padded_tokens)


@pytest.mark.parametrize(
    ("actual_tokens", "padded_tokens", "mm_positions"),
    [(3, 4, (1,)), (5, 8, (1, 3)), (3, 4, ())],
)
def test_pcp_padding_preserves_evs_position_channels_until_recompute(actual_tokens, padded_tokens, mm_positions):
    hidden_size = 8
    num_position_channels = 5
    num_mm_tokens = len(mm_positions)
    clean_embeddings = torch.arange(num_mm_tokens * hidden_size, dtype=torch.float32).reshape(-1, hidden_size)
    positions = torch.arange(num_mm_tokens * num_position_channels, dtype=torch.float32).reshape(
        -1, num_position_channels
    )
    raw_embeddings = [torch.cat((clean_embeddings, positions), dim=-1)] if mm_positions else []
    is_mm_embed = torch.zeros(actual_tokens, dtype=torch.bool)
    is_mm_embed[list(mm_positions)] = True
    input_batch = SimpleNamespace(
        input_ids=torch.arange(padded_tokens), num_tokens=actual_tokens, num_tokens_after_padding=padded_tokens
    )
    req_states = object()
    events: list[str] = []

    def recompute(embeddings, batch, requests):
        events.append("recompute")
        assert batch is input_batch and requests is req_states
        assert embeddings[0].shape[-1] == hidden_size + num_position_channels
        torch.testing.assert_close(embeddings[0][:, -num_position_channels:], positions)
        return [embeddings[0][:, :-num_position_channels]]

    def merge(input_ids, embeddings, mask):
        events.append("merge")
        torch.testing.assert_close(input_ids, input_batch.input_ids)
        torch.testing.assert_close(mask[:actual_tokens], is_mm_embed)
        assert mask.shape == (padded_tokens,)
        assert not mask[actual_tokens:].any()
        output = torch.zeros(padded_tokens, hidden_size)
        if embeddings:
            output[mask] = torch.cat(embeddings)
        return output

    state = AscendModelState.__new__(AscendModelState)
    state.model = SimpleNamespace(set_embedding_num_tokens_padded=Mock())
    state.pcp_manager = object()
    state.supports_mm_inputs = True
    state.prompt_embeds_state = None
    state.execute_mm_encoder = Mock()
    state.apply_staged_writes = Mock(side_effect=lambda: events.append("flush"))
    state.mm_pruner = SimpleNamespace(
        strip=Mock(side_effect=lambda embeddings: [mm[:, :hidden_size] for mm in embeddings]),
        recompute=Mock(side_effect=recompute),
    )
    state.encoder_runner = SimpleNamespace(get_inputs_embeds=Mock(side_effect=merge))

    with patch.object(ModelState, "gather_mm_embeddings", return_value=(raw_embeddings, is_mm_embed)):
        result = state.prepare_inputs_embeds({}, input_batch, req_states)

    expected = torch.zeros(padded_tokens, hidden_size)
    expected[list(mm_positions)] = clean_embeddings
    torch.testing.assert_close(result, expected)
    assert events == (["recompute", "flush", "merge"] if mm_positions else ["merge"])
    state.mm_pruner.strip.assert_not_called()
    assert state.mm_pruner.recompute.call_count == int(bool(mm_positions))
    assert state.apply_staged_writes.call_count == int(bool(mm_positions))
    state.execute_mm_encoder.assert_called_once_with({})
    state.model.set_embedding_num_tokens_padded.assert_called_once_with(padded_tokens)


def test_non_pcp_embedding_keeps_upstream_behavior():
    state = AscendModelState.__new__(AscendModelState)
    state.model = SimpleNamespace()
    state.pcp_manager = None
    sentinel = object()
    with patch.object(DefaultModelState, "prepare_inputs_embeds", return_value=sentinel) as upstream:
        result = state.prepare_inputs_embeds({}, object(), req_states=None)

    assert result is sentinel
    upstream.assert_called_once()


def test_pcp_embedding_without_padding_keeps_upstream_behavior():
    state = AscendModelState.__new__(AscendModelState)
    state.model = SimpleNamespace()
    state.pcp_manager = object()
    state.supports_mm_inputs = True
    input_batch = SimpleNamespace(num_tokens=12, num_tokens_after_padding=12)
    sentinel = object()
    with patch.object(DefaultModelState, "prepare_inputs_embeds", return_value=sentinel) as upstream:
        result = state.prepare_inputs_embeds({}, input_batch, req_states=None)

    assert result is sentinel
    upstream.assert_called_once()
