# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""GLM wrappers must publish the PCP batch span before embedding lookup."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.model_states.interface import ModelState

import vllm_ascend.ops.vocab_parallel_embedding as vocab_embedding
from vllm_ascend.models.glm5next.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)
from vllm_ascend.worker.v2.model_states.default import AscendModelState


def _model_with_embedding(embedding, multimodal):
    core = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(core)
    core.embed_tokens = embedding
    model = Glm5NextForCausalLM.__new__(Glm5NextForCausalLM)
    nn.Module.__init__(model)
    model.model = core
    if multimodal:
        wrapper = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
        nn.Module.__init__(wrapper)
        wrapper.language_model = model
        wrapper._language_model_names = ["language_model"]
        return wrapper
    return model


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_pcp_embedding_uses_current_batch_span(monkeypatch, multimodal, rank):
    # Keep real runner/model/embedding calls, mocking only collectives and the
    # external encoder result. A missing model hook pads to 16384 and fails.
    weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    layer = vocab_embedding.AscendVocabParallelEmbedding.__new__(vocab_embedding.AscendVocabParallelEmbedding)
    nn.Module.__init__(layer)
    layer.parallel_mode = vocab_embedding.VocabParallelMode.PCP_X_TP
    layer.token_exchange_group = SimpleNamespace(world_size=2)
    layer.output_reduce_group = None
    layer.embedding_tp_capacity = 16384
    layer._embedding_num_tokens_padded = None
    layer.embedding_dim = 4
    layer.shard_indices = SimpleNamespace(
        org_vocab_start_index=rank * 4,
        org_vocab_end_index=(rank + 1) * 4,
        num_org_vocab_padding=0,
        added_vocab_start_index=8,
        added_vocab_end_index=8,
    )
    local_weight = weight[rank * 4 : (rank + 1) * 4]
    layer.quant_method = SimpleNamespace(embedding=lambda _, ids: local_weight[ids])
    monkeypatch.setattr(vocab_embedding, "is_forward_context_available", lambda: False)

    state = AscendModelState.__new__(AscendModelState)
    state.model = _model_with_embedding(layer, multimodal)
    state.supports_mm_inputs = multimodal
    state.prompt_embeds_state = None
    state.mm_pruner = None
    state.execute_mm_encoder = Mock()
    state.inputs_embeds = torch.zeros(4, 4)
    encoder = EncoderRunner.__new__(EncoderRunner)
    encoder.model = state.model
    encoder.inputs_embeds = state.inputs_embeds
    state.encoder_runner = encoder

    # Reuse one model across shrinking batches, an idle rank, and an empty
    # batch. The published span must be refreshed and identical on both ranks.
    batches = [([1, 5], [2, 6, 7], 4), ([7], [1, 6], 2), ([], [2, 5], 2), ([], [], 0)]
    for first_ids, second_ids, padded_tokens in batches:
        peer_ids = [torch.tensor(ids, dtype=torch.long) for ids in (first_ids, second_ids)]
        span = max(1, padded_tokens)
        gathered_ids = torch.zeros(2, span, dtype=torch.long)
        for peer, ids in enumerate(peer_ids):
            gathered_ids[peer, : ids.numel()] = ids
        events: list[str] = []

        def all_gather(input_, dim, *, gathered_ids=gathered_ids, events=events):
            events.append("gather")
            assert dim == 0
            torch.testing.assert_close(input_, gathered_ids[rank])
            return gathered_ids.flatten()

        def reduce_scatter(input_, dim, *, gathered_ids=gathered_ids, events=events):
            events.append("scatter")
            assert dim == 0
            local_mask = (gathered_ids.flatten() // 4 == rank).unsqueeze(-1)
            torch.testing.assert_close(input_, weight[gathered_ids.flatten()] * local_mask)
            return weight[gathered_ids[rank]]

        monkeypatch.setattr(layer.token_exchange_group, "all_gather", all_gather, raising=False)
        monkeypatch.setattr(layer.token_exchange_group, "reduce_scatter", reduce_scatter, raising=False)
        ids = peer_ids[rank]
        batch = SimpleNamespace(
            input_ids=torch.nn.functional.pad(ids, (0, padded_tokens - ids.numel())),
            num_tokens=ids.numel(),
            num_tokens_after_padding=padded_tokens,
        )
        mm_mask = torch.zeros(ids.numel(), dtype=torch.bool)
        if multimodal and ids.numel():
            mm_mask[0] = True
        mm_embeds = [torch.full((1, 4), 99.0)] if mm_mask.any() else []
        with patch.object(ModelState, "gather_mm_embeddings", return_value=(mm_embeds, mm_mask)):
            output = state.prepare_inputs_embeds({}, batch, req_states=None)
        expected = weight[ids].clone()
        expected[mm_mask] = 99.0
        torch.testing.assert_close(output[: ids.numel()], expected)
        assert output.shape == (padded_tokens, 4)
        assert events == ["gather", "scatter"]


@pytest.mark.parametrize("multimodal", [False, True])
def test_publish_span_skips_pp_stage_without_embedding(multimodal):
    model = _model_with_embedding(PPMissingLayer(), multimodal)
    model.set_embedding_num_tokens_padded(4)
