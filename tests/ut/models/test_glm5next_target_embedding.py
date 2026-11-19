# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Target embedding provisioning for GLM-5.3-Flash speculative decoding."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from vllm.config import CompilationMode
from vllm.distributed import parallel_state
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

from vllm_ascend.models.glm5next import model as glm5next_model
from vllm_ascend.models.glm5next import mtp as mtp_module
from vllm_ascend.utils import vllm_version_is


class _Embedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, *, prefix: str):
        super().__init__(num_embeddings, embedding_dim)
        self.prefix = prefix


def _pp_group(*, pp_size: int, pp_rank: int):
    return lambda: SimpleNamespace(
        is_first_rank=pp_rank == 0,
        is_last_rank=pp_rank == pp_size - 1,
        world_size=pp_size,
    )


def _draft_pp_context(pp_size: int):
    if vllm_version_is("0.33.0"):
        return patch.object(eagle_utils, "get_pp_group", return_value=SimpleNamespace(world_size=pp_size))
    return patch.object(eagle_utils, "in_single_pp_stage", return_value=pp_size > 1)


def _vllm_config(method: str | None):
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=8,
        index_topk=None,
        num_hidden_layers=0,
        rms_norm_eps=1e-5,
        num_attention_heads=8,
    )
    spec_config = (
        None
        if method is None
        else SimpleNamespace(
            method=method,
            uses_target_hidden_states=True,
            draft_model_config=SimpleNamespace(
                may_use_target_embed=hasattr(mtp_module.Glm5NextMTP, "has_own_embed_tokens"),
            ),
        )
    )
    return SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        model_config=SimpleNamespace(hf_text_config=config),
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
        speculative_config=spec_config,
    )


@pytest.mark.parametrize(
    ("pp_size", "pp_rank", "method", "expected_embedding"),
    [
        (2, 0, None, True),
        (2, 1, "mtp", True),
        (3, 1, "mtp", False),
        (2, 1, None, False),
        (1, 0, "mtp", True),
    ],
)
def test_embedding_is_provisioned_only_where_needed(
    monkeypatch,
    pp_size,
    pp_rank,
    method,
    expected_embedding,
):
    created = []

    def make_embedding(*args, **kwargs):
        embedding = _Embedding(*args, **kwargs)
        created.append(embedding)
        return embedding

    pp_group = _pp_group(pp_size=pp_size, pp_rank=pp_rank)
    monkeypatch.setattr(glm5next_model, "VocabParallelEmbedding", make_embedding)
    monkeypatch.setattr(glm5next_model, "RMSNorm", lambda *_args, **_kwargs: nn.Identity())
    monkeypatch.setattr(glm5next_model, "make_layers", lambda *_args, **_kwargs: (0, 0, nn.ModuleList()))
    monkeypatch.setattr(glm5next_model, "get_pp_group", pp_group)
    monkeypatch.setattr(parallel_state, "get_pp_group", pp_group)
    monkeypatch.setattr(glm5next_model, "get_tensor_model_parallel_world_size", lambda: 1)

    model = glm5next_model.Glm5NextModel(vllm_config=_vllm_config(method), prefix="target")

    if expected_embedding:
        assert isinstance(model.embed_tokens, _Embedding)
        assert model.embed_tokens.prefix == "target.embed_tokens"
        assert len(created) == 1
    else:
        assert isinstance(model.embed_tokens, PPMissingLayer)
        assert not created


def test_last_stage_loader_populates_and_shares_target_embedding():
    target = nn.Module()
    target.language_model = nn.Module()
    target.language_model.model = nn.Module()
    target_embedding = _Embedding(32, 8, prefix="language_model.model.embed_tokens")
    target.language_model.model.embed_tokens = target_embedding

    embedding = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
    loaded = AutoWeightsLoader(target).load_weights(
        [("model.language_model.embed_tokens.weight", embedding)],
        mapper=glm5next_model.Glm5NextForConditionalGeneration.hf_to_vllm_mapper,
    )
    assert loaded == {"language_model.model.embed_tokens.weight"}
    torch.testing.assert_close(target_embedding.weight, embedding)

    draft = nn.Module()
    draft.has_own_embed_tokens = False
    draft.model = nn.Module()
    draft.model.embed_tokens = _Embedding(32, 8, prefix="draft.model.embed_tokens")
    with _draft_pp_context(2):
        eagle_utils.maybe_share_target_embed(draft, draft.model, target.language_model.model)

    assert draft.model.embed_tokens is target_embedding
