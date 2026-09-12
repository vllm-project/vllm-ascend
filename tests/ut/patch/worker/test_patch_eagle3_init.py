#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from transformers import LlamaConfig
from vllm.model_executor.models import deepseek_eagle3, llama_eagle3

import vllm_ascend.patch.worker.patch_eagle3_init  # noqa: F401


def _make_llama_config(eagle_config=...):
    kwargs = {
        "draft_vocab_size": 1000,
        "hidden_size": 256,
        "vocab_size": 32000,
        "logit_scale": 1.0,
    }
    if eagle_config is not ...:
        kwargs["eagle_config"] = eagle_config
    return LlamaConfig(**kwargs)


def _make_mock_hf_config(eagle_config=...):
    config = Mock()
    config.draft_vocab_size = 1000
    config.hidden_size = 256
    config.vocab_size = 32000
    config.logit_scale = 1.0
    if eagle_config is not ...:
        config.eagle_config = eagle_config
    return config


def _make_vllm_config(hf_config, *, parallel_drafting=False):
    model_config = Mock()
    model_config.get_total_num_hidden_layers.return_value = 32
    return SimpleNamespace(
        model_config=model_config,
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_config),
            parallel_drafting=parallel_drafting,
        ),
    )


def _patch_llama_dependencies(monkeypatch, *, fc_input_size=768):
    inner_model = SimpleNamespace(
        fc_input_size=fc_input_size,
        use_aux_hidden_state=True,
    )
    model_factory = Mock(return_value=inner_model)
    lm_head_factory = Mock(return_value=Mock())
    logits_processor_factory = Mock(return_value=Mock())
    quant_config = Mock()
    quant_config_getter = Mock(return_value=quant_config)

    monkeypatch.setattr(llama_eagle3, "LlamaModel", model_factory)
    monkeypatch.setattr(llama_eagle3, "ParallelLMHead", lm_head_factory)
    monkeypatch.setattr(llama_eagle3, "LogitsProcessor", logits_processor_factory)
    monkeypatch.setattr(llama_eagle3, "get_draft_quant_config", quant_config_getter)

    return SimpleNamespace(
        inner_model=inner_model,
        model_factory=model_factory,
        lm_head_factory=lm_head_factory,
        quant_config=quant_config,
        quant_config_getter=quant_config_getter,
    )


@pytest.mark.parametrize(
    "hf_config",
    [
        pytest.param(_make_llama_config(), id="real-config-missing"),
        pytest.param(_make_llama_config({}), id="real-config-empty"),
        pytest.param(
            _make_llama_config({"use_aux_hidden_state": False}),
            id="real-config-use-aux-hidden-state",
        ),
        pytest.param(
            SimpleNamespace(
                draft_vocab_size=1000,
                hidden_size=256,
                vocab_size=32000,
                logit_scale=1.0,
                eagle_config={"use_aux_hidden_state": True},
            ),
            id="simulated-config-object",
        ),
        pytest.param(_make_mock_hf_config(), id="upstream-mock-fixture"),
    ],
)
def test_llama_init_uses_upstream_patch_points(monkeypatch, hf_config):
    dependencies = _patch_llama_dependencies(monkeypatch)
    vllm_config = _make_vllm_config(hf_config)
    original_eagle_config = getattr(hf_config, "eagle_config", None)

    model = llama_eagle3.Eagle3LlamaForCausalLM(
        vllm_config=vllm_config,
        prefix="draft",
    )

    assert model.config is hf_config
    assert getattr(model.config, "eagle_config", None) is original_eagle_config
    assert model.config.target_layer_count == 32
    vllm_config.model_config.get_total_num_hidden_layers.assert_called_once_with()
    dependencies.model_factory.assert_called_once_with(
        vllm_config=vllm_config,
        prefix="draft.model",
        start_layer_id=32,
    )
    dependencies.quant_config_getter.assert_called_once_with(vllm_config)
    dependencies.lm_head_factory.assert_called_once()
    assert dependencies.lm_head_factory.call_args.kwargs["quant_config"] is dependencies.quant_config
    assert dependencies.lm_head_factory.call_args.kwargs["prefix"] == "draft.lm_head"


def test_llama_parallel_drafting_uses_model_fc_input_size(monkeypatch):
    dependencies = _patch_llama_dependencies(monkeypatch, fc_input_size=513)
    vllm_config = _make_vllm_config(
        _make_llama_config({"use_aux_hidden_state": False}),
        parallel_drafting=True,
    )

    model = llama_eagle3.Eagle3LlamaForCausalLM(vllm_config=vllm_config)

    assert model.model is dependencies.inner_model
    assert model.mask_hidden.shape == (1, 513)


def test_deepseek_init_uses_upstream_patch_points(monkeypatch):
    hf_config = SimpleNamespace(
        draft_vocab_size=1000,
        hidden_size=256,
        vocab_size=32000,
        logit_scale=1.0,
        eagle_config={"use_aux_hidden_state": True},
    )
    vllm_config = _make_vllm_config(hf_config)
    inner_model = Mock()
    model_factory = Mock(return_value=inner_model)
    lm_head_factory = Mock(return_value=Mock())
    logits_processor_factory = Mock(return_value=Mock())

    monkeypatch.setattr(deepseek_eagle3, "DeepseekV2Eagle3Model", model_factory)
    monkeypatch.setattr(deepseek_eagle3, "ParallelLMHead", lm_head_factory)
    monkeypatch.setattr(
        deepseek_eagle3,
        "LogitsProcessor",
        logits_processor_factory,
    )

    model = deepseek_eagle3.Eagle3DeepseekV2ForCausalLM(
        vllm_config=vllm_config,
        prefix="draft",
    )

    assert model.config is hf_config
    assert model.config.eagle_config == {"use_aux_hidden_state": True}
    assert model.config.target_layer_count == 32
    vllm_config.model_config.get_total_num_hidden_layers.assert_called_once_with()
    model_factory.assert_called_once_with(
        vllm_config=vllm_config,
        prefix="draft.model",
        start_layer_id=32,
    )
    lm_head_factory.assert_called_once()
    assert lm_head_factory.call_args.kwargs["prefix"] == "draft.lm_head"
    assert "quant_config" not in lm_head_factory.call_args.kwargs
