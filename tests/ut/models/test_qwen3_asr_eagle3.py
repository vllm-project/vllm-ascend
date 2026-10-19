# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm import ModelRegistry
from vllm.model_executor.model_loader.utils import get_model_cls
from vllm.model_executor.models.interfaces import EagleModelMixin, supports_eagle3
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration

from vllm_ascend.models import register_model
from vllm_ascend.models.qwen3_asr import AscendQwen3ASRForConditionalGeneration


class _Decoder(nn.Module, EagleModelMixin):
    def __init__(self, result):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity() for _ in range(12)])
        self.result = result

    def forward(self, *args, **kwargs):
        return self.result


def _make_target(result):
    target = AscendQwen3ASRForConditionalGeneration.__new__(AscendQwen3ASRForConditionalGeneration)
    nn.Module.__init__(target)
    language_model = Qwen3ForCausalLM.__new__(Qwen3ForCausalLM)
    nn.Module.__init__(language_model)
    language_model.model = _Decoder(result)
    target.language_model = language_model
    target._language_model_names = ["language_model"]
    return target


def test_qwen3_asr_registry_exposes_eagle3(monkeypatch):
    monkeypatch.setattr(ModelRegistry, "models", ModelRegistry.models.copy())
    register_model()
    config = SimpleNamespace(
        model="/test/qwen3-asr",
        convert_type="none",
        runner_type="generate",
        trust_remote_code=False,
        model_impl="vllm",
        hf_config=SimpleNamespace(architectures=["Qwen3ASRForConditionalGeneration"]),
        registry=ModelRegistry,
        _get_transformers_backend_cls=lambda: "TransformersForCausalLM",
    )
    model_cls = get_model_cls(config)
    assert model_cls is AscendQwen3ASRForConditionalGeneration
    assert supports_eagle3(model_cls)
    assert model_cls._processor_factory is Qwen3ASRForConditionalGeneration._processor_factory


@pytest.mark.parametrize("layers", [(2, 6, 9), ()])
def test_qwen3_asr_delegates_auxiliary_layers(monkeypatch, layers):
    monkeypatch.setattr("vllm.distributed.parallel_state.model_parallel_is_initialized", lambda: False)
    target = _make_target(torch.zeros(1, 3))

    assert supports_eagle3(target)
    assert target.get_language_model() is target.language_model
    assert target.get_eagle3_default_aux_hidden_state_layers() == (2, 6, 9)
    target.set_aux_hidden_state_layers(layers)
    assert target.language_model.model.aux_hidden_state_layers == layers


@pytest.mark.parametrize("with_aux", [False, True])
def test_qwen3_asr_preserves_decoder_output(with_aux):
    hidden = torch.ones(2, 3)
    result = (hidden, [hidden * 2]) if with_aux else hidden
    target = _make_target(result)

    output = target(torch.tensor([1, 2]), torch.arange(2), inputs_embeds=torch.zeros(2, 3))

    assert output is result
