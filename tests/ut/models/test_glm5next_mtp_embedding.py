# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""GLM5Next MTP shares target embeddings unless the checkpoint is MTP-specific."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from vllm.distributed import parallel_state
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

from vllm_ascend.models.glm5next import mtp as mtp_module
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator


def _draft_pp_context(pp_size):
    # vLLM main gates on in_single_pp_stage(); v0.34.0 reads the PP group size.
    if hasattr(eagle_utils, "in_single_pp_stage"):
        return patch.object(eagle_utils, "in_single_pp_stage", return_value=pp_size > 1)
    return patch.object(eagle_utils, "get_pp_group", return_value=SimpleNamespace(world_size=pp_size))


@pytest.fixture
def mtp():
    model = mtp_module.Glm5NextMTP.__new__(mtp_module.Glm5NextMTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=45, num_nextn_predict_layers=1, n_routed_experts=0)
    model.is_rot_weight_used = False
    model.model = nn.Module()
    model.model.mtp_start_layer_idx = 45
    model.model.num_mtp_layers = 1
    model.model.embed_tokens = nn.Embedding(8, 4)
    model.model.embed_tokens.weight.data.fill_(float("nan"))
    layer = nn.Module()
    layer.enorm = nn.LayerNorm(4, bias=False)
    model.model.layers = nn.ModuleDict({"45": layer})
    with patch.object(mtp_module, "fused_moe_make_expert_params_mapping", return_value=[]):
        yield model


def _mtp_weight():
    return "model.layers.45.enorm.weight", torch.ones(4)


def test_no_pp_reuses_equal_target_embedding(mtp):
    embedding = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    mtp.load_weights([("model.embed_tokens.weight", embedding), _mtp_weight()])
    target = SimpleNamespace(embed_tokens=nn.Embedding.from_pretrained(embedding))
    with _draft_pp_context(1):
        eagle_utils.maybe_share_target_embed(mtp, mtp.model, target)
    assert mtp.model.embed_tokens is target.embed_tokens


def test_missing_checkpoint_embedding_is_rejected_without_target_copy(mtp):
    mtp.load_weights([_mtp_weight()])
    assert not mtp.has_own_embed_tokens
    target = SimpleNamespace(embed_tokens=PPMissingLayer())
    with (
        _draft_pp_context(2),
        pytest.raises(RuntimeError, match="needs the target input embedding"),
    ):
        eagle_utils.maybe_share_target_embed(mtp, mtp.model, target)


@pytest.mark.parametrize("shared_first", [False, True])
def test_mtp_specific_embedding_takes_precedence(mtp, shared_first, monkeypatch):
    shared = "model.embed_tokens.weight", torch.ones(8, 4)
    own = "model.layers.45.embed_tokens.weight", torch.full((8, 4), 2.0)
    weights = [shared, own] if shared_first else [own, shared]
    mtp.load_weights([*weights, _mtp_weight()])
    torch.testing.assert_close(mtp.model.embed_tokens.weight, own[1])
    assert mtp.has_own_embed_tokens

    # Under PP=1 the target owns its embedding; cleanup must keep it.
    target_embed = nn.Embedding.from_pretrained(shared[1])
    target_inner = SimpleNamespace(embed_tokens=target_embed)
    target = SimpleNamespace(model=target_inner)
    monkeypatch.setattr(parallel_state, "_PP", SimpleNamespace(is_first_rank=True, is_last_rank=True, world_size=1))
    with _draft_pp_context(1):
        eagle_utils.maybe_share_target_embed(mtp, mtp.model, target_inner)
    speculator = AscendMTPSpeculator.__new__(AscendMTPSpeculator)
    speculator.model = mtp
    speculator._target_pp_group = parallel_state.get_pp_group()
    speculator.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(tie_word_embeddings=False)),
    )
    speculator.free_unused_weights(target)
    assert target_inner.embed_tokens is target_embed
    assert mtp.has_own_embed_tokens
    torch.testing.assert_close(mtp.model.embed_tokens.weight, own[1])


def test_embedding_does_not_hide_missing_mtp_layer(mtp):
    with pytest.raises(ValueError, match="layer 45 weights missing"):
        mtp.load_weights([("model.embed_tokens.weight", torch.ones(8, 4))])
