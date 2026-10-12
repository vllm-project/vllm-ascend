# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.compilation import decorators
from vllm.config.compilation import DynamicShapesType

from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model


@pytest.mark.parametrize("num_tokens", [384, 1024])
def test_backbone_marks_embedding_token_dimension_dynamic(monkeypatch, num_tokens):
    """Profile and prefill must not specialize the embedding row count."""
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.do_not_compile = False
    model.compiled = False
    model.compilation_config = SimpleNamespace(dynamic_shapes_config=SimpleNamespace(type=DynamicShapesType.BACKED))
    monkeypatch.setattr(decorators.envs, "VLLM_USE_AOT_COMPILE", False)
    monkeypatch.setattr(decorators, "is_forward_context_available", lambda: False)

    class ReadyToCompile(Exception):
        pass

    def stop_before_compilation():
        # The real decorator has marked inputs at this point. Avoid requiring
        # model weights or a device compiler for this input-contract test.
        raise ReadyToCompile

    monkeypatch.setattr(model, "original_code_object", stop_before_compilation)
    ids = torch.zeros(num_tokens, dtype=torch.long)
    positions = torch.arange(num_tokens)
    embeddings = torch.zeros(num_tokens, 8)
    with pytest.raises(ReadyToCompile):
        model(ids, positions, None, inputs_embeds=embeddings)

    for tensor in (ids, positions, embeddings):
        assert tensor._dynamo_dynamic_indices == {0}
