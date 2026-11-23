"""Idle inputs-embeds batches must join the target's embedding exchange."""

import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.worker.gpu.model_states.interface import ModelState

from vllm_ascend.ops.vocab_parallel_embedding import AscendVocabParallelEmbedding, VocabParallelMode
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.model_states import default


def _make_state(supports_mm_inputs, embedding):
    state = default.AscendModelState.__new__(default.AscendModelState)
    state.device = torch.device("cpu")
    state.supports_mm_inputs = supports_mm_inputs
    state.model = torch.nn.Module()
    state.model.add_module("embed_tokens", embedding)
    state.model.embed_input_ids = embedding
    state.model.modules = Mock(wraps=state.model.modules)
    state.prompt_embeds_state = None
    state.mm_pruner = None
    state.execute_mm_encoder = Mock()
    # Both APIs must retain their upstream persistent buffer for graph replay.
    state.inputs_embeds = torch.full((8, 4), -1.0)
    state.encoder_runner = SimpleNamespace(inputs_embeds=state.inputs_embeds)
    return state


def _make_embedding(rank, events):
    # The real DP has tokens from both vocabulary shards. The other DP is idle.
    weight = torch.arange(32, dtype=torch.float32).view(8, 4)
    gathered_ids = torch.zeros((2, 8), dtype=torch.int32)
    gathered_ids[0, :3] = torch.tensor([1, 5, 7])
    layer = AscendVocabParallelEmbedding.__new__(AscendVocabParallelEmbedding)
    torch.nn.Module.__init__(layer)
    layer.parallel_mode = VocabParallelMode.FINE_GRAINED
    layer.embedding_tp_capacity = 8
    layer.embedding_dim = 4
    layer.output_reduce_group = None
    layer.shard_indices = SimpleNamespace(
        org_vocab_start_index=rank * 4,
        org_vocab_end_index=(rank + 1) * 4,
        num_org_vocab_padding=0,
        added_vocab_start_index=8,
        added_vocab_end_index=8,
    )
    local_weight = weight[rank * 4 : (rank + 1) * 4]
    layer.quant_method = SimpleNamespace(embedding=lambda _, ids: local_weight[ids])

    def gather(ids, dim):
        assert dim == 0
        torch.testing.assert_close(ids, gathered_ids[rank])
        events.append((rank, "gather", tuple(ids.shape)))
        return gathered_ids.flatten()

    def scatter(contribution, dim):
        assert dim == 0
        mask = (gathered_ids.flatten() // 4 == rank).unsqueeze(-1)
        torch.testing.assert_close(contribution, weight[gathered_ids.flatten()] * mask)
        events.append((rank, "scatter", tuple(contribution.shape)))
        return weight[gathered_ids[rank]]

    layer.token_exchange_group = SimpleNamespace(world_size=2, all_gather=gather, reduce_scatter=scatter)
    return layer, weight


@pytest.mark.parametrize("supports_mm_inputs", [False, True], ids=["prompt-embeds", "multimodal"])
def test_real_and_idle_inputs_embeds_join_matching_embedding_collectives(monkeypatch, supports_mm_inputs):
    """Run real lookup/masking on CPU, mocking only the distributed exchange."""
    monkeypatch.setattr(default, "embedding_tp_enable", lambda: True, raising=False)
    events: list[tuple[int, str, tuple[int, ...]]] = []
    real_layer, weight = _make_embedding(0, events)
    idle_layer, _ = _make_embedding(1, events)
    real = _make_state(supports_mm_inputs, real_layer)
    idle = _make_state(supports_mm_inputs, idle_layer)

    def embed(ids, multimodal_embeddings, is_mm_embed):
        assert not multimodal_embeddings
        assert not is_mm_embed.any()
        real.inputs_embeds[: ids.numel()].copy_(real_layer(ids))
        return real.inputs_embeds

    real.encoder_runner.get_inputs_embeds = embed
    batch = SimpleNamespace(
        input_ids=torch.tensor([1, 5, 7, 0], dtype=torch.int32), num_tokens=3, num_tokens_after_padding=4
    )
    with patch.object(ModelState, "gather_mm_embeddings", return_value=([], torch.zeros(3, dtype=torch.bool))):
        result = real.prepare_inputs_embeds({}, batch, req_states=None)
    torch.testing.assert_close(result[:3], weight[batch.input_ids[:3]])
    real.model.modules.assert_not_called()  # No extra runner-side exchange on a real batch.

    dummy = idle.dummy_inputs_embeds(4)
    assert dummy.data_ptr() == idle.inputs_embeds.data_ptr()
    torch.testing.assert_close(dummy, torch.full((4, 4), -1.0))
    assert events == [
        (0, "gather", (8,)),
        (0, "scatter", (16, 4)),
        (1, "gather", (8,)),
        (1, "scatter", (16, 4)),
    ]


@pytest.mark.parametrize("supports_mm_inputs", [False, True])
@pytest.mark.parametrize("num_tokens", [0, 1, 8])
def test_dummy_embedding_reuses_ids_and_output_buffer(monkeypatch, supports_mm_inputs, num_tokens):
    monkeypatch.setattr(default, "embedding_tp_enable", lambda: True)
    events: list[tuple[int, str, tuple[int, ...]]] = []
    layer, _ = _make_embedding(1, events)
    state = _make_state(supports_mm_inputs, layer)
    first = state.dummy_inputs_embeds(num_tokens)
    _, token_ids = state._embedding_tp_dummy_join
    second = state.dummy_inputs_embeds(num_tokens)
    assert state._embedding_tp_dummy_join[1] is token_ids
    assert token_ids.dtype == torch.int32
    torch.testing.assert_close(token_ids, torch.zeros(8, dtype=torch.int32))
    assert first.shape == second.shape == (num_tokens, 4)
    if num_tokens:
        assert first.data_ptr() == second.data_ptr() == state.inputs_embeds.data_ptr()
    state.model.modules.assert_called_once()
    assert events == [(1, "gather", (8,)), (1, "scatter", (16, 4))] * 2


@pytest.mark.parametrize("supports_mm_inputs", [False, True], ids=["prompt-embeds", "multimodal"])
@pytest.mark.parametrize("embedding_tp", [False, True], ids=["tp-disabled", "tp-enabled"])
def test_dummy_inputs_embeds_accepts_explicit_none(monkeypatch, supports_mm_inputs, embedding_tp):
    monkeypatch.setattr(default, "embedding_tp_enable", lambda: embedding_tp)
    events: list[tuple[int, str, tuple[int, ...]]] = []
    layer, _ = _make_embedding(1, events)
    state = _make_state(supports_mm_inputs, layer)

    result = state.dummy_inputs_embeds(3, None)

    assert result.data_ptr() == state.inputs_embeds.data_ptr()
    torch.testing.assert_close(result, torch.full((3, 4), -1.0))
    assert events == ([(1, "gather", (8,)), (1, "scatter", (16, 4))] if embedding_tp else [])
    if not embedding_tp:
        state.model.modules.assert_not_called()


@pytest.mark.skipif(
    "input_ids" not in inspect.signature(default.DefaultModelState.dummy_inputs_embeds).parameters,
    reason="Randomized dummy embeddings require the newer upstream model state",
)
@pytest.mark.parametrize("supports_mm_inputs", [False, True], ids=["prompt-embeds", "multimodal"])
@pytest.mark.parametrize("embedding_tp", [False, True], ids=["tp-disabled", "tp-enabled"])
def test_dummy_input_ids_use_upstream_lookup_once(monkeypatch, supports_mm_inputs, embedding_tp):
    monkeypatch.setattr(default, "embedding_tp_enable", lambda: embedding_tp)
    events: list[tuple[int, str, tuple[int, ...]]] = []
    layer, weight = _make_embedding(0, events)
    if not embedding_tp:
        layer = torch.nn.Embedding.from_pretrained(weight)
    state = _make_state(supports_mm_inputs, layer)
    input_ids = torch.tensor([1, 5, 7], dtype=torch.int32)

    with patch.object(layer, "forward", wraps=layer.forward) as lookup:
        result = state.dummy_inputs_embeds(4, input_ids)
    lookup.assert_called_once()
    torch.testing.assert_close(lookup.call_args.args[0], input_ids)

    assert result.shape == (4, 4)
    assert result.data_ptr() == state.inputs_embeds.data_ptr()
    torch.testing.assert_close(result[:3], weight[input_ids])
    torch.testing.assert_close(result[3:], torch.full((1, 4), -1.0))
    torch.testing.assert_close(state.inputs_embeds[4:], torch.full((4, 4), -1.0))
    assert events == ([(0, "gather", (8,)), (0, "scatter", (16, 4))] if embedding_tp else [])
    state.model.modules.assert_not_called()
    assert not hasattr(state, "_embedding_tp_dummy_join")


@pytest.mark.parametrize("supports_mm_inputs", [False, True])
def test_embedding_tp_disabled_preserves_dummy_buffer(monkeypatch, supports_mm_inputs):
    monkeypatch.setattr(default, "embedding_tp_enable", lambda: False)
    state = _make_state(supports_mm_inputs, torch.nn.Identity())
    result = state.dummy_inputs_embeds(3, None)
    torch.testing.assert_close(result, state.inputs_embeds[:3])
    assert result.data_ptr() == state.inputs_embeds.data_ptr()
    state.model.modules.assert_not_called()


@pytest.mark.skipif(vllm_version_is("0.33.0"), reason="Profile input_ids are only passed by vLLM main")
def test_dummy_embedding_uses_profile_input_ids(monkeypatch):
    monkeypatch.setattr(default, "embedding_tp_enable", lambda: True)
    embedding = torch.nn.Embedding(8, 4)
    state = _make_state(True, embedding)
    input_ids = torch.tensor([1, 5, 7])

    result = state.dummy_inputs_embeds(3, input_ids)

    torch.testing.assert_close(result, embedding(input_ids))
    state.model.modules.assert_not_called()


@pytest.mark.parametrize("mode", [VocabParallelMode.STANDARD, VocabParallelMode.PCP_X_TP])
def test_other_embedding_plans_do_not_join(monkeypatch, mode):
    monkeypatch.setattr(default, "embedding_tp_enable", lambda: True)
    events: list[tuple[int, str, tuple[int, ...]]] = []
    layer, _ = _make_embedding(1, events)
    layer.parallel_mode = mode
    state = _make_state(True, layer)
    state.dummy_inputs_embeds(3)
    state.dummy_inputs_embeds(3)
    assert events == []
    assert state._embedding_tp_dummy_join is None
    state.model.modules.assert_called_once()
