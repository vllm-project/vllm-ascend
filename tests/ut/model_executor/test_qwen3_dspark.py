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
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from vllm.model_executor.models.qwen3_dflash import _resolve_layer_attention

import vllm_ascend.models.qwen3_dspark as qwen3_dspark


class TestQwen3DSparkDraftWindow:
    """Tests for scheduler-visible DSpark draft KV window configuration."""

    @staticmethod
    def _make_config(window_size, dflash_config=None):
        draft_hf_config = SimpleNamespace(dflash_config=dflash_config)
        return SimpleNamespace(
            additional_config={"draft_window_size": window_size},
            speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=draft_hf_config)),
        ), draft_hf_config

    def test_enables_physical_swa_cache_for_draft_layers(self) -> None:
        vllm_config, draft_hf_config = self._make_config(1024)

        qwen3_dspark._configure_dspark_draft_window(vllm_config)

        assert draft_hf_config.dflash_config == {
            "use_swa": True,
            "swa_window_size": 1024,
        }
        assert _resolve_layer_attention(draft_hf_config, 0) == (1024, False)

    def test_updates_existing_dflash_config_in_place(self) -> None:
        dflash_config = {
            "causal": False,
            "use_swa": False,
            "swa_window_size": 4096,
        }
        vllm_config, draft_hf_config = self._make_config(
            2048,
            dflash_config,
        )

        qwen3_dspark._configure_dspark_draft_window(vllm_config)

        assert draft_hf_config.dflash_config is dflash_config
        assert draft_hf_config.dflash_config == {
            "causal": False,
            "use_swa": True,
            "swa_window_size": 2048,
        }

    def test_absent_window_keeps_draft_cache_spec_unchanged(self) -> None:
        draft_hf_config = SimpleNamespace(dflash_config={"causal": False})
        vllm_config = SimpleNamespace(
            additional_config={},
            speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=draft_hf_config)),
        )

        qwen3_dspark._configure_dspark_draft_window(vllm_config)

        assert draft_hf_config.dflash_config == {"causal": False}

    @pytest.mark.parametrize(
        "speculative_config",
        [
            None,
            SimpleNamespace(draft_model_config=None),
            SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=None)),
        ],
    )
    def test_missing_draft_config_is_noop(self, speculative_config) -> None:
        vllm_config = SimpleNamespace(
            additional_config={"draft_window_size": 1024},
            speculative_config=speculative_config,
        )

        qwen3_dspark._configure_dspark_draft_window(vllm_config)

    @pytest.mark.parametrize("window_size", [0, -1, 1024.0, True])
    def test_rejects_invalid_window(self, window_size) -> None:
        vllm_config, _ = self._make_config(window_size)

        with pytest.raises(ValueError, match="positive integer"):
            qwen3_dspark._configure_dspark_draft_window(vllm_config)


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
            patch.object(qwen3_dspark, "get_rotation_path", return_value="quarot.safetensors"),
            patch.object(qwen3_dspark, "get_rotation_matrix", return_value=rotation_matrix) as rotation_loader,
        ):
            model.post_process(config)
        rotation_loader.assert_called_once_with("quarot.safetensors")
        torch.testing.assert_close(model.model.fc.weight, fc_weight @ rotation_matrix)
        torch.testing.assert_close(model.model.embed_tokens.weight, embed_before)
        torch.testing.assert_close(model.lm_head.weight, head_before)


def test_quarot_loads_missing_target_vocab_shards(tmp_path) -> None:
    embed_name = "language_model.model.embed_tokens.weight"
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

    model_cls = qwen3_dspark.AscendQwen3DSparkForCausalLM
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

    def vocab_layer(vocab_size, hidden_size, params_dtype):
        layer = nn.Linear(hidden_size, 3, bias=False, dtype=params_dtype)
        layer.shard_indices = SimpleNamespace(org_vocab_start_index=1, org_vocab_end_index=3)
        layer.quant_method = SimpleNamespace(process_weights_after_loading=lambda layer: None)
        return layer

    config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path), hf_text_config=SimpleNamespace(vocab_size=4, hidden_size=2))
    )
    with (
        patch.object(qwen3_dspark, "get_rotation_path", return_value=rotation_path),
        patch.object(qwen3_dspark, "VocabParallelEmbedding", side_effect=vocab_layer),
        patch.object(qwen3_dspark, "ParallelLMHead", side_effect=vocab_layer),
    ):
        model.post_process(config)
    assert model.model.embed_tokens is not original_embed
    assert model.lm_head is not original_head
    torch.testing.assert_close(original_embed.weight, torch.full((3, 2), 99.0))
    torch.testing.assert_close(original_head.weight, torch.full((3, 2), 99.0))

    for layer, weight in ((model.model.embed_tokens, target_weight), (model.lm_head, target_weight + 10)):
        expected = torch.cat((weight[1:3] @ rotation.T, torch.zeros(1, 2)))
        torch.testing.assert_close(layer.weight, expected)
    assert model.has_own_embed_tokens
    assert model.has_own_lm_head
