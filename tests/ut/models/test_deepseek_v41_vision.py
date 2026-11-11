# ruff: noqa: E402

from types import SimpleNamespace

import pytest

pytest.importorskip(
    "vllm.transformers_utils.configs.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

import torch
from PIL import Image
from torch import nn
from vllm.model_executor.models.interfaces import requires_raw_input_tokens, supports_multimodal
from vllm.models.deepseek_v41.common.mm_preprocess import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_SENTINEL_BASE_ID,
    IMAGE_START,
    DeepseekV4VLProcessingInfo,
    DeepseekV4VLProcessor,
    image_sentinel_mask,
    image_token_types,
)
from vllm.multimodal.processing import InputProcessingContext
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config as UpstreamDeepseekV41Config
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.model_states.interface import ModelState

from vllm_ascend.models.deepseek_v41.engram.common import (
    engram_enabled,
    valid_engram_token_mask,
)
from vllm_ascend.models.deepseek_v41.model import (
    AscendDeepseekV41LLMForCausalLM,
    DeepseekV41Model,
    _engram_enabled_for_runtime,
)
from vllm_ascend.models.deepseek_v41.vl_model import (
    AscendDeepseekV41ForCausalLM,
)
from vllm_ascend.ops.vocab_parallel_embedding import AscendVocabParallelEmbedding
from vllm_ascend.utils import normalize_deepseek_v41_config
from vllm_ascend.worker.v2.model_states.default import AscendModelState


def make_v41_config(**kwargs):
    return normalize_deepseek_v41_config(UpstreamDeepseekV41Config(**kwargs))


def test_v41_vision_wrapper_uses_v41_language_backbone():
    assert supports_multimodal(AscendDeepseekV41ForCausalLM)
    assert requires_raw_input_tokens(AscendDeepseekV41ForCausalLM)
    assert AscendDeepseekV41ForCausalLM.language_model_cls is AscendDeepseekV41LLMForCausalLM
    assert "_processor_factory" in AscendDeepseekV41ForCausalLM.__dict__


def test_v41_processing_info_accepts_v41_config():
    config = make_v41_config(
        text_config={},
        vision_config={
            "num_hidden_layers": 1,
            "patch_size": 14,
            "downsample_ratio": 3,
            "max_image_tokens": 1024,
        },
    )
    model_config = SimpleNamespace(hf_config=config)
    ctx = InputProcessingContext(model_config=model_config, tokenizer=None)

    assert DeepseekV4VLProcessingInfo(ctx).get_hf_config() is config
    assert config.image_sentinel_base_id == IMAGE_SENTINEL_BASE_ID
    # vLLM main (#56554) removed the compressor-alignment pad; the mm-prefix
    # flags moved to the model-config arch converter.
    assert config.image_pad_token_id == IMAGE_SENTINEL_BASE_ID + 1
    assert not hasattr(config, "mm_prefix_span_leading_pad_modulus")


def test_v41_image_roles_use_reference_reading_order():
    assert image_token_types(2, 3).tolist() == [
        IMAGE_START,
        IMAGE,
        IMAGE,
        IMAGE,
        IMAGE_NEW_LINE,
        IMAGE,
        IMAGE,
        IMAGE,
        IMAGE_NEW_LINE,
        IMAGE_END,
    ]


def test_v41_processor_emits_types_without_v4_perm():
    config = make_v41_config(
        text_config={},
        vision_config={
            "num_hidden_layers": 1,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "patch_size": 14,
            "rope_theta": 10000.0,
            "downsample_ratio": 3,
            "max_image_tokens": 64,
            "min_pixels": 0,
            "max_wh_ratio": None,
        },
    )
    result = DeepseekV4VLProcessor(config)(images=[Image.new("RGB", (84, 42))])

    assert result["vit_grid"].tolist() == [[3, 6]]
    assert result["llm_grid"].tolist() == [[1, 2]]
    assert result["types"].tolist() == [
        IMAGE_START,
        IMAGE,
        IMAGE,
        IMAGE_NEW_LINE,
        IMAGE_END,
    ]
    assert "perm" not in result


def test_v41_image_and_alignment_pad_are_dead_to_engram():
    # vLLM main (#56554) removed the alignment pad; only the image sentinel
    # is dead to engram.
    token_ids = torch.tensor([17, IMAGE_SENTINEL_BASE_ID, 18])
    expected = torch.tensor([True, False, True])

    torch.testing.assert_close(image_sentinel_mask(token_ids), ~expected)
    torch.testing.assert_close(
        valid_engram_token_mask(token_ids, IMAGE_SENTINEL_BASE_ID, IMAGE_SENTINEL_BASE_ID + 1),
        expected,
    )


def test_v41_engram_checkpoint_contract_is_unchanged():
    checkpoint = SimpleNamespace(engram_layer_ids=[1, 14])

    assert engram_enabled(checkpoint)
    assert not engram_enabled(SimpleNamespace(engram_layer_ids=[]))


def test_v41_a5_engram_requires_runtime_opt_in(monkeypatch):
    checkpoint = SimpleNamespace(engram_layer_ids=[1, 14])
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v41.model.DeviceOperator.get_dsv41_packed_cache_ops",
        lambda: object(),
    )

    assert not _engram_enabled_for_runtime(checkpoint, SimpleNamespace(engram_config=None))
    assert _engram_enabled_for_runtime(checkpoint, SimpleNamespace(engram_config=object()))


def test_v41_span_has_three_delimiters_and_no_image_pad_parameter():
    wrapper = object.__new__(AscendDeepseekV41ForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.image_start = nn.Parameter(torch.tensor([1.0, 1.0]))
    wrapper.image_newline = nn.Parameter(torch.tensor([2.0, 2.0]))
    wrapper.image_end = nn.Parameter(torch.tensor([3.0, 3.0]))
    image_embeds = torch.tensor([[10.0, 10.0], [20.0, 20.0]])

    span = wrapper._build_image_span(
        image_embeds,
        image_token_types(1, 2),
    )

    torch.testing.assert_close(
        span,
        torch.tensor(
            [
                [1.0, 1.0],
                [10.0, 10.0],
                [20.0, 20.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        ),
    )
    assert "image_pad" not in dict(wrapper.named_parameters())


def test_v41_alignment_pad_uses_plain_image_token_embedding():
    class LanguageModel(nn.Module):
        def embed_input_ids(self, input_ids):
            return input_ids.unsqueeze(-1)

    wrapper = object.__new__(AscendDeepseekV41ForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.language_model = LanguageModel()

    embeddings = wrapper.embed_input_ids(torch.tensor([7, IMAGE_SENTINEL_BASE_ID]))
    assert embeddings.squeeze(-1).tolist() == [7, IMAGE_SENTINEL_BASE_ID]


@pytest.mark.parametrize("publishes_length", [True, False], ids=["v41-wrapper", "model-without-hook"])
def test_mrv2_embedding_preparation_preserves_valid_rows_and_padded_buffer(monkeypatch, publishes_length):
    # Retain the real state -> wrapper -> language model -> embedding chain,
    # plus the upstream merge and persistent encoder buffer. Only image encoding
    # and cache selection are outside this test's scope.
    embedding = AscendVocabParallelEmbedding(8, 2, padding_size=4, disable_tp=True)
    weight = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    embedding.weight_loader(embedding.weight, weight)
    lookup_batches = []
    embedding.register_forward_pre_hook(
        lambda layer, args: lookup_batches.append((args[0].numel(), layer._embedding_num_tokens_padded))
    )
    backbone = DeepseekV41Model.__new__(DeepseekV41Model)
    nn.Module.__init__(backbone)
    backbone.embed_tokens = embedding
    language_model = AscendDeepseekV41LLMForCausalLM.__new__(AscendDeepseekV41LLMForCausalLM)
    nn.Module.__init__(language_model)
    language_model.model = backbone
    wrapper = AscendDeepseekV41ForCausalLM.__new__(AscendDeepseekV41ForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.language_model = language_model
    model = wrapper if publishes_length else SimpleNamespace(embed_input_ids=wrapper.embed_input_ids)
    state = AscendModelState.__new__(AscendModelState)
    state.model = model
    state.supports_mm_inputs = True
    state.mm_pruner = state.prompt_embeds_state = None
    state.encoder_runner = EncoderRunner(model, 8, 2, None, torch.float32, torch.device("cpu"))
    monkeypatch.setattr(state, "execute_mm_encoder", lambda scheduled: None)
    monkeypatch.setattr(ModelState, "gather_mm_embeddings", lambda self, batch: (batch.mm_embeds, batch.mm_mask))
    buffer_address = state.encoder_runner.inputs_embeds.data_ptr()

    # The second step must replace the first span, and an idle step must not
    # inherit it. MM fusion receives actual rows; the runner gets padded rows.
    for num_tokens, padded_tokens in [(3, 5), (1, 2), (0, 0)]:
        ids = torch.arange(padded_tokens, dtype=torch.long)
        mask = torch.zeros(num_tokens, dtype=torch.bool)
        has_image = num_tokens > 1
        if has_image:
            mask[1] = True
        image_embedding = torch.tensor([[100.0, 101.0]])
        batch = SimpleNamespace(
            input_ids=ids,
            num_tokens=num_tokens,
            num_tokens_after_padding=padded_tokens,
            mm_embeds=[image_embedding] if has_image else [],
            mm_mask=mask,
        )
        output = state.prepare_inputs_embeds({}, batch, None)
        expected = weight[ids[:num_tokens]].clone()
        if has_image:
            expected[mask] = image_embedding
        torch.testing.assert_close(output[:num_tokens], expected)
        assert output.shape == (padded_tokens, 2)
        assert output.untyped_storage().data_ptr() == buffer_address
        assert lookup_batches[-1] == (num_tokens, max(1, padded_tokens) if publishes_length else None)


def test_v41_text_only_load_skips_checkpoint_vision_weights():
    class LanguageModel(nn.Module):
        def load_weights(self, weights):
            self.loaded = list(weights)
            return {name for name, _ in self.loaded}

    wrapper = object.__new__(AscendDeepseekV41ForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.vision = None
    wrapper.language_model = LanguageModel()
    text_weight = torch.tensor([1.0])

    loaded = wrapper.load_weights(
        iter(
            [
                ("model.image_end", torch.tensor([2.0])),
                ("model.layers.0.weight", text_weight),
            ]
        )
    )

    assert wrapper.language_model.loaded == [("model.layers.0.weight", text_weight)]
    assert loaded == {"language_model.model.layers.0.weight"}
