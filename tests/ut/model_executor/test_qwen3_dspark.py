#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend embed_tokensect.
#
"""CPU-only tests for Qwen3 DSpark weight loading."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from vllm.config import set_current_vllm_config

import vllm_ascend.models.quarot_utils as quarot_utils
import vllm_ascend.models.qwen3_dspark as qwen3_dspark
from vllm_ascend.models import qwen3_dflash2
from vllm_ascend.models.kimi_k3 import kimi_k3_dspark
from vllm_ascend.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM
from vllm_ascend.ops import vocab_parallel_embedding as vocab_ops


class TestQwen3DSparkWeightLoading:
    """Tests for Qwen3 DSpark weight loading."""

    def test_rotates_only_fc_weights(self) -> None:
        """Post-processing rotates FC weights while preserving draft-owned vocab weights."""
        model_cls = qwen3_dspark.AscendQwen3DSparkForCausalLM
        model = model_cls.__new__(model_cls)
        nn.Module.__init__(model)
        model.model = nn.Module()
        model.model.fc = nn.Linear(2, 2, bias=False)
        model.model.embed_tokens = nn.Embedding(2, 2)
        model.lm_head = nn.Linear(2, 2, bias=False)
        model.has_own_embed_tokens = model.has_own_lm_head = True
        rotation_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        fc_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        model.model.fc.weight.data.copy_(fc_weight)
        embed_before = model.model.embed_tokens.weight.detach().clone()
        head_before = model.lm_head.weight.detach().clone()
        config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace()))
        with (
            patch.object(quarot_utils, "lmhead_tp_enable", return_value=True),
            patch.object(quarot_utils, "get_rotation_path", return_value="quarot.safetensors"),
            patch.object(quarot_utils, "get_rotation_matrix", return_value=rotation_matrix) as rotation_loader,
        ):
            model.post_process(config)
        rotation_loader.assert_called_once_with("quarot.safetensors")
        torch.testing.assert_close(model.model.fc.weight, fc_weight @ rotation_matrix)
        torch.testing.assert_close(model.model.embed_tokens.weight, embed_before)
        torch.testing.assert_close(model.lm_head.weight, head_before)


@pytest.mark.parametrize(
    "embed_name",
    [
        "language_model.model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
    ],
)
@pytest.mark.parametrize("model_cls", [qwen3_dspark.AscendQwen3DSparkForCausalLM, DFlash2Qwen3ForCausalLM])
@pytest.mark.parametrize("lmhead_tp", [False, True])
def test_quarot_loads_missing_target_vocab_shards(tmp_path, model_cls, embed_name, lmhead_tp) -> None:
    head_name = "language_model.lm_head.weight"
    shard_name = "model-00001-of-00001.safetensors"
    target_weight = torch.arange(8, dtype=torch.float32).view(4, 2)
    save_file({embed_name: target_weight, head_name: target_weight + 10}, tmp_path / shard_name)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {embed_name: shard_name, head_name: shard_name}}),
        encoding="utf-8",
    )
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    rotation_path = tmp_path / "rotation.safetensors"
    save_file({"global_rotation": rotation}, rotation_path)

    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace()
    model.rotation_path = rotation_path
    model.target_model_path = tmp_path
    model.enable_confidence_head = False
    model.model = SimpleNamespace(embed_tokens=nn.Linear(2, 3, bias=False), fc=nn.Linear(2, 2, bias=False))
    model.lm_head = nn.Linear(2, 3, bias=False)
    for layer in (model.model.embed_tokens, model.lm_head):
        layer.weight.data.fill_(99)
        layer.shard_indices = SimpleNamespace(org_vocab_start_index=1, org_vocab_end_index=3)

    original_embed, original_head = model.model.embed_tokens, model.lm_head

    prefixes = []

    def vocab_layer(vocab_size, hidden_size, params_dtype, prefix):
        prefixes.append(prefix)
        layer = nn.Linear(hidden_size, 3, bias=False, dtype=params_dtype)
        layer.shard_indices = SimpleNamespace(org_vocab_start_index=1, org_vocab_end_index=3)
        layer.quant_method = SimpleNamespace(process_weights_after_loading=lambda layer: None)
        return layer

    config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path), hf_text_config=SimpleNamespace(vocab_size=4, hidden_size=2))
    )
    with (
        patch.object(quarot_utils, "lmhead_tp_enable", return_value=lmhead_tp),
        patch.object(quarot_utils, "get_rotation_path", return_value=rotation_path),
        patch.object(qwen3_dflash2, "get_rotation_path", return_value=rotation_path),
        patch.object(quarot_utils, "VocabParallelEmbedding", side_effect=vocab_layer),
        patch.object(quarot_utils, "ParallelLMHead", side_effect=vocab_layer),
    ):
        model.post_process(config)
    expected_head_prefix = "lm_head" if lmhead_tp else ""
    assert prefixes == ["", expected_head_prefix]
    assert model.model.embed_tokens is not original_embed
    assert model.lm_head is not original_head
    torch.testing.assert_close(original_embed.weight, torch.full((3, 2), 99.0))
    torch.testing.assert_close(original_head.weight, torch.full((3, 2), 99.0))

    for layer, weight in ((model.model.embed_tokens, target_weight), (model.lm_head, target_weight + 10)):
        expected = torch.cat((weight[1:3] @ rotation.T, torch.zeros(1, 2)))
        torch.testing.assert_close(layer.weight, expected)
    assert model.has_own_embed_tokens
    assert model.has_own_lm_head


@pytest.mark.parametrize(
    "model_cls,projection_name",
    [
        (qwen3_dspark.AscendQwen3DSparkForCausalLM, "fc"),
        (kimi_k3_dspark.AscendK3DSparkForCausalLM, "context_proj"),
        (DFlash2Qwen3ForCausalLM, "fc"),
    ],
)
@pytest.mark.parametrize("lmhead_tp", [False, True])
def test_quarot_draft_rebuild_preserves_vocab_layout(tmp_path, monkeypatch, model_cls, projection_name, lmhead_tp):
    """Rebuild fine-grained heads and keep PCP-only heads on standard TP."""
    target_weight = torch.arange(128, dtype=torch.float32).reshape(64, 2)
    save_file({"lm_head.weight": target_weight}, tmp_path / "model.safetensors")
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    config = MagicMock()
    config.compilation_config.custom_ops = ["all"]
    config.parallel_config.prefill_context_parallel_size = 1 if lmhead_tp else 8
    config.model_config = SimpleNamespace(
        model=str(tmp_path), hf_text_config=SimpleNamespace(vocab_size=64, hidden_size=2)
    )
    group = SimpleNamespace(world_size=8, rank_in_group=0)
    monkeypatch.setattr(quarot_utils, "get_rotation_path", lambda config: "rotation")
    monkeypatch.setattr(qwen3_dflash2, "get_rotation_path", lambda config: "rotation")
    monkeypatch.setattr(quarot_utils, "get_rotation_matrix", lambda path: rotation)
    monkeypatch.setattr(quarot_utils, "ParallelLMHead", vocab_ops.AscendParallelLMHead)
    monkeypatch.setattr(quarot_utils, "lmhead_tp_enable", lambda: lmhead_tp)
    monkeypatch.setattr(vocab_ops, "lmhead_tp_enable", lambda: lmhead_tp)
    monkeypatch.setattr(vocab_ops, "get_lmhead_tp_group", lambda: group)
    monkeypatch.setattr(vocab_ops, "get_pcp_group", lambda: group)
    monkeypatch.setattr(vocab_ops, "get_tp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0))
    monkeypatch.setattr(vocab_ops, "enable_pcp_embedding_lmhead_weight_sharding", lambda: not lmhead_tp)
    hidden = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    sharded = lmhead_tp
    expected_size = group.world_size if sharded else 1
    expected_mode = vocab_ops.VocabParallelMode.FINE_GRAINED if lmhead_tp else vocab_ops.VocabParallelMode.STANDARD
    logits = []
    with set_current_vllm_config(config):
        for rank in range(group.world_size):
            group.rank_in_group = rank
            draft = model_cls.__new__(model_cls)
            nn.Module.__init__(draft)
            draft.model = nn.Module()
            setattr(draft.model, projection_name, nn.Linear(2, 2, bias=False))
            draft.has_own_embed_tokens = True
            draft.has_own_lm_head = False
            draft.post_process(config)
            head = draft.lm_head
            assert head.parallel_mode is expected_mode
            assert head.weight.shape == (64 // expected_size, 2)
            assert head.tp_size == expected_size
            assert head.tp_rank == (rank if sharded else 0)
            expected_weight = target_weight[rank * 8 : (rank + 1) * 8] if sharded else target_weight
            torch.testing.assert_close(head.weight, expected_weight @ rotation.T)
            logits.append(head.quant_method.apply(head, hidden))
    reference = hidden @ (target_weight @ rotation.T).T
    if sharded:
        torch.testing.assert_close(torch.cat(logits, dim=-1), reference)
    else:
        for replica in logits:
            torch.testing.assert_close(replica, reference)
