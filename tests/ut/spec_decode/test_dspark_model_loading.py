# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.models import qwen3_dspark
from vllm_ascend.models.kimi_k3_dspark import AscendK3DSparkForCausalLM
from vllm_ascend.models.qwen3_dspark import AscendQwen3DSparkForCausalLM
from vllm_ascend.worker.v2.spec_decode.dspark import speculator as shared


@pytest.mark.parametrize("fail", [False, True])
def test_post_process_receives_target_config_after_loading(monkeypatch, fail):
    events: list[tuple[str, object]] = []
    config = SimpleNamespace(
        quant_config=object(),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        additional_config={"scheduler_config": {"profiling_chunk_config": {"enabled": True}}},
    )
    spec = shared.AscendDSparkSpeculator.__new__(shared.AscendDSparkSpeculator)
    spec.vllm_config = config
    target = object()
    draft = SimpleNamespace(
        post_process=lambda received: events.append(("post_process", received)),
        configure_target_aux_hidden_capture=lambda received: events.append(("capture", received)),
    )

    def fake_replace(received, **changes):
        values = vars(received).copy()
        values.update(changes)
        return SimpleNamespace(**values)

    def load(self, received, names):
        assert received is target
        # The upstream loader must see the temporary CPP-disabled copy while
        # the target config keeps profiling chunk enabled.
        loader_config = self.vllm_config
        assert loader_config is not config
        assert loader_config.additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"] is False
        assert config.additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"] is True
        events.append(("load", config))
        if fail:
            raise ValueError("load failed")
        return draft

    monkeypatch.setattr(DSparkSpeculator, "load_draft_model", load)
    monkeypatch.setattr(shared, "replace", fake_replace)
    monkeypatch.setattr(shared, "set_current_vllm_config", lambda _: nullcontext())
    with pytest.raises(ValueError, match="load failed") if fail else nullcontext():
        assert spec.load_draft_model(target, set()) is draft
    # The target config object is restored on both the success and failure paths.
    assert spec.vllm_config is config
    expected: list[tuple[str, object]] = [("load", config)]
    if not fail:
        expected.extend([("post_process", config), ("capture", target)])
    assert events == expected


@pytest.mark.parametrize(
    "model_cls,projection_name",
    [
        (AscendQwen3DSparkForCausalLM, "fc"),
        (AscendK3DSparkForCausalLM, "context_proj"),
    ],
)
@pytest.mark.parametrize("own_embed,own_head", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("rotated", [False, True])
def test_post_process_aligns_weights_without_modifying_target(
    monkeypatch, model_cls, projection_name, own_embed, own_head, rotated
):
    target_embed = torch.nn.Embedding(4, 2, dtype=torch.float64)
    target_head = torch.nn.Linear(2, 4, bias=False, dtype=torch.float64)
    target_embed_before = target_embed.weight.detach().clone()
    target_head_before = target_head.weight.detach().clone()
    draft = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(draft)
    draft.model = torch.nn.Module()
    projection = torch.nn.Linear(4, 2, bias=False, dtype=torch.float64)
    setattr(draft.model, projection_name, projection)
    draft.model.embed_tokens = torch.nn.Embedding(4, 2, dtype=torch.float64) if own_embed else target_embed
    draft.lm_head = torch.nn.Linear(2, 4, bias=False, dtype=torch.float64) if own_head else target_head
    draft.has_own_embed_tokens, draft.has_own_lm_head = own_embed, own_head
    original_embed, original_head = draft.model.embed_tokens, draft.lm_head
    original_embed_weight = original_embed.weight.detach().clone()
    original_head_weight = original_head.weight.detach().clone()
    original_parameter = projection.weight
    original_projection = projection.weight.detach().clone()
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="/target", hf_text_config=SimpleNamespace(vocab_size=4, hidden_size=2))
    )
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    monkeypatch.setattr(qwen3_dspark, "get_rotation_path", lambda received: "/rotation" if rotated else None)
    loader = MagicMock(return_value=rotation)
    monkeypatch.setattr(qwen3_dspark, "get_rotation_matrix", loader)
    created = []

    def make_layer(vocab, hidden, params_dtype):
        layer = torch.nn.Embedding(vocab, hidden, dtype=params_dtype)
        layer.quant_method = SimpleNamespace(process_weights_after_loading=MagicMock())
        created.append(layer)
        return layer

    def load_layer(layer, path, names, matrix, label):
        source = target_embed_before if "embed_tokens" in label else target_head_before
        layer.weight.copy_(source @ matrix.T)

    monkeypatch.setattr(qwen3_dspark, "VocabParallelEmbedding", make_layer)
    monkeypatch.setattr(qwen3_dspark, "ParallelLMHead", make_layer)
    monkeypatch.setattr(qwen3_dspark, "load_quarot_target_layer", load_layer)
    draft.post_process(config)
    assert projection.weight is original_parameter

    torch.testing.assert_close(target_embed.weight, target_embed_before)
    torch.testing.assert_close(target_head.weight, target_head_before)
    if rotated:
        loader.assert_called_once_with("/rotation")
        inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
        rotated_inputs = (inputs.view(1, 2, 2) @ rotation).view(1, 4)
        torch.testing.assert_close(rotated_inputs @ projection.weight.T, inputs @ original_projection.T)
        assert len(created) == int(not own_embed) + int(not own_head)
        for layer in created:
            layer.quant_method.process_weights_after_loading.assert_called_once_with(layer)
    else:
        loader.assert_not_called()
        assert not created
        torch.testing.assert_close(projection.weight, original_projection)
    for actual, original, before, target_weight, owns in (
        (draft.model.embed_tokens, original_embed, original_embed_weight, target_embed_before, own_embed),
        (draft.lm_head, original_head, original_head_weight, target_head_before, own_head),
    ):
        if rotated and not owns:
            assert actual is not original
            assert actual.weight.data_ptr() != original.weight.data_ptr()
            torch.testing.assert_close(actual.weight, target_weight @ rotation.T)
        else:
            assert actual is original
            torch.testing.assert_close(actual.weight, before)
