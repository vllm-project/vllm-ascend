from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm.model_executor.models.transformers import base as transformers_base

import vllm_ascend.patch.worker.patch_gemma4_transformers as gemma4_patch


class _TextConfig(SimpleNamespace):
    def get_text_config(self):
        return self


class _FakeAttention:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _make_model_config():
    return SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="gemma4"),
        uses_mrope=False,
        get_num_attention_heads=lambda parallel_config: 16,
    )


def test_multimodal_forward_preserves_mm_token_type_ids():
    captured = {}
    token_type_ids = torch.tensor([[0, 1]])
    mm_token_type_ids = torch.tensor([[1, 1]])

    def fake_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs):
        captured["kwargs"] = kwargs
        return "ok"

    fake_self = SimpleNamespace(model_config=_make_model_config())
    with patch.object(transformers_base.Base, "forward", new=fake_forward):
        result = gemma4_patch._multimodal_forward(
            fake_self,
            input_ids=torch.tensor([[1, 2]]),
            positions=torch.tensor([0, 1]),
            token_type_ids=token_type_ids,
            mm_token_type_ids=mm_token_type_ids,
            ignored_kwarg="drop-me",
        )

    assert result == "ok"
    assert set(captured["kwargs"]) == {"token_type_ids", "mm_token_type_ids"}
    assert torch.equal(captured["kwargs"]["token_type_ids"], token_type_ids)
    assert torch.equal(captured["kwargs"]["mm_token_type_ids"], mm_token_type_ids)


def test_multimodal_forward_backfills_mm_token_type_ids_from_token_type_ids():
    captured = {}
    token_type_ids = torch.tensor([[2, 3]])

    def fake_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs):
        captured["kwargs"] = kwargs
        return "ok"

    fake_self = SimpleNamespace(model_config=_make_model_config())
    with patch.object(transformers_base.Base, "forward", new=fake_forward):
        gemma4_patch._multimodal_forward(
            fake_self,
            input_ids=torch.tensor([[1, 2]]),
            positions=torch.tensor([0, 1]),
            token_type_ids=token_type_ids,
        )

    assert set(captured["kwargs"]) == {"token_type_ids", "mm_token_type_ids"}
    assert torch.equal(captured["kwargs"]["token_type_ids"], token_type_ids)
    assert torch.equal(captured["kwargs"]["mm_token_type_ids"], token_type_ids)


def test_gemma4_model_forward_rebuilds_placeholder_aware_layer_inputs():
    pad_embedding = torch.tensor([10.0, 20.0, 30.0])
    merged_inputs = torch.tensor(
        [[[1.0, 1.0, 1.0], [9.0, 9.0, 9.0], [2.0, 2.0, 2.0]]]
    )
    input_ids = torch.tensor([[5, 99, 7]])
    image_mask = torch.tensor([[False, True, False]])
    empty_mask = torch.zeros_like(image_mask)
    captured = {}

    def get_per_layer_inputs(llm_input_ids, llm_inputs_embeds):
        captured["llm_input_ids"] = llm_input_ids
        captured["llm_inputs_embeds"] = llm_inputs_embeds
        return "per-layer-inputs"

    language_model = MagicMock()
    language_model.embed_tokens.weight = torch.stack(
        [pad_embedding, torch.zeros_like(pad_embedding)]
    )
    language_model.get_per_layer_inputs.side_effect = get_per_layer_inputs
    language_model.return_value = SimpleNamespace(
        last_hidden_state=torch.ones_like(merged_inputs),
        past_key_values="pkv",
        hidden_states=("hidden",),
        attentions=("attn",),
    )

    text_config = SimpleNamespace(
        pad_token_id=0,
        hidden_size_per_layer_input=True,
        use_bidirectional_attention="none",
    )
    config = SimpleNamespace(
        use_cache=True,
        text_config=text_config,
        get_text_config=lambda: text_config,
    )
    fake_self = SimpleNamespace(
        config=config,
        training=False,
        language_model=language_model,
        get_placeholder_mask=lambda input_ids: (image_mask, empty_mask, empty_mask),
    )

    result = gemma4_patch._gemma4_model_forward(
        fake_self,
        input_ids=input_ids,
        inputs_embeds=merged_inputs,
        attention_mask={"mask": torch.tensor([1])},
    )

    assert torch.equal(captured["llm_input_ids"], torch.tensor([[5, 0, 7]]))
    assert torch.equal(
        captured["llm_inputs_embeds"],
        torch.tensor([[[1.0, 1.0, 1.0], [10.0, 20.0, 30.0], [2.0, 2.0, 2.0]]]),
    )
    language_model.assert_called_once()
    assert result.past_key_values == "pkv"


def test_create_attention_instances_handles_gemma4_heterogeneous_layers():
    text_config = _TextConfig(
        layer_types=["sliding_attention", "full_attention", "full_attention"],
        num_hidden_layers=3,
        num_kv_shared_layers=1,
        head_dim=256,
        global_head_dim=512,
        sliding_window=1024,
        num_key_value_heads=8,
        num_global_key_value_heads=4,
        attention_k_eq_v=True,
        attn_logit_softcapping=30.0,
    )
    fake_self = SimpleNamespace(
        model_config=_make_model_config(),
        text_config=text_config,
        config=text_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        pp_group=SimpleNamespace(rank_in_group=0, world_size=1),
        cache_config=SimpleNamespace(),
        quant_config=SimpleNamespace(),
        model=SimpleNamespace(modules=lambda: []),
    )

    with patch.object(transformers_base, "Attention", _FakeAttention), patch.object(
        transformers_base,
        "EncoderOnlyAttention",
        _FakeAttention,
    ), patch.object(transformers_base, "get_pp_indices", return_value=(0, 3)):
        instances = gemma4_patch._create_attention_instances(fake_self)

    assert sorted(instances) == [0, 1, 2]
    assert instances[0].kwargs["head_size"] == 256
    assert instances[0].kwargs["num_kv_heads"] == 4
    assert instances[0].kwargs["per_layer_sliding_window"] == 1024
    assert instances[0].kwargs["kv_sharing_target_layer_name"] is None
    assert instances[1].kwargs["head_size"] == 512
    assert instances[1].kwargs["num_kv_heads"] == 2
    assert instances[2].kwargs["kv_sharing_target_layer_name"] == "1.attn"
