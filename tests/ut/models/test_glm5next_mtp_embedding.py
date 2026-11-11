# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""MTP must load a local embedding when the target's PP stage has none."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

from vllm_ascend.models.glm5next import mtp as mtp_module


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


@pytest.mark.parametrize("prefix", ["model.", "model.language_model."])
def test_loads_checkpoint_embedding_on_last_pp_stage(mtp, prefix):
    embedding = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    loaded = mtp.load_weights([(prefix + "embed_tokens.weight", embedding), _mtp_weight()])
    assert "model.embed_tokens.weight" in loaded
    assert mtp.has_own_embed_tokens
    torch.testing.assert_close(mtp.model.embed_tokens.weight, embedding)
    local_embedding = mtp.model.embed_tokens
    target = SimpleNamespace(embed_tokens=PPMissingLayer())
    with patch.object(eagle_utils, "get_pp_group", return_value=SimpleNamespace(world_size=2)):
        eagle_utils.maybe_share_target_embed(mtp, mtp.model, target)
    assert mtp.model.embed_tokens is local_embedding


def test_no_pp_reuses_equal_target_embedding(mtp):
    embedding = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    mtp.load_weights([("model.embed_tokens.weight", embedding), _mtp_weight()])
    target = SimpleNamespace(embed_tokens=nn.Embedding.from_pretrained(embedding))
    with patch.object(eagle_utils, "get_pp_group", return_value=SimpleNamespace(world_size=1)):
        eagle_utils.maybe_share_target_embed(mtp, mtp.model, target)
    assert mtp.model.embed_tokens is target.embed_tokens


def test_missing_checkpoint_embedding_is_rejected_without_target_copy(mtp):
    mtp.load_weights([_mtp_weight()])
    assert not mtp.has_own_embed_tokens
    target = SimpleNamespace(embed_tokens=PPMissingLayer())
    with (
        patch.object(eagle_utils, "get_pp_group", return_value=SimpleNamespace(world_size=2)),
        pytest.raises(RuntimeError, match="needs the target input embedding"),
    ):
        eagle_utils.maybe_share_target_embed(mtp, mtp.model, target)


@pytest.mark.parametrize("shared_first", [False, True])
def test_mtp_specific_embedding_takes_precedence(mtp, shared_first):
    shared = "model.embed_tokens.weight", torch.ones(8, 4)
    own = "model.layers.45.embed_tokens.weight", torch.full((8, 4), 2.0)
    weights = [shared, own] if shared_first else [own, shared]
    mtp.load_weights([*weights, _mtp_weight()])
    torch.testing.assert_close(mtp.model.embed_tokens.weight, own[1])
    assert mtp.has_own_embed_tokens


def test_embedding_does_not_hide_missing_mtp_layer(mtp):
    with pytest.raises(ValueError, match="layer 45 weights missing"):
        mtp.load_weights([("model.embed_tokens.weight", torch.ones(8, 4))])
