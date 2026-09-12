from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn
from vllm.model_executor.models.interfaces import supports_multimodal
from vllm.multimodal.processing import InputProcessingContext

from vllm_ascend.deepseek_v41_config import DeepseekV41Config
from vllm_ascend.models.deepseek_v41.engram_hash import (
    valid_engram_token_mask,
)
from vllm_ascend.models.deepseek_v41.mm_preprocess import (
    COMPRESS_PAD_TO,
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD_ID,
    IMAGE_START,
    IMAGE_TOKEN_ID,
    DeepseekV41VLProcessor,
    DeepseekV41VLProcessingInfo,
    image_sentinel_mask,
    image_token_types,
    leading_compressor_pad,
)
from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41ForCausalLM
from vllm_ascend.models.deepseek_v41.vl_model import (
    AscendDeepseekV41ForConditionalGeneration,
)


def test_v41_vision_wrapper_uses_v41_language_backbone():
    assert supports_multimodal(AscendDeepseekV41ForConditionalGeneration)
    assert (
        AscendDeepseekV41ForConditionalGeneration.language_model_cls
        is AscendDeepseekV41ForCausalLM
    )
    assert "_processor_factory" in AscendDeepseekV41ForConditionalGeneration.__dict__


def test_v41_processing_info_accepts_v41_config():
    config = DeepseekV41Config(
        text_config={},
        vision_config={
            "num_hidden_layers": 1,
            "patch_size": 14,
            "downsample_ratio": 3,
            "max_num_tokens": 1024,
        },
    )
    model_config = SimpleNamespace(hf_config=config)
    ctx = InputProcessingContext(model_config=model_config, tokenizer=None)

    assert DeepseekV41VLProcessingInfo(ctx).get_hf_config() is config
    assert config.image_sentinel_base_id == IMAGE_TOKEN_ID
    assert config.image_pad_token_id == IMAGE_PAD_ID
    assert config.is_mm_prefix_lm
    assert config.mm_prefix_clamp_sliding_window
    assert config.mm_prefix_span_leading_pad_modulus == COMPRESS_PAD_TO == 2


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

    assert [leading_compressor_pad(i) for i in range(4)] == [1, 0, 1, 0]


def test_v41_processor_emits_types_without_v4_perm():
    config = DeepseekV41Config(
        text_config={},
        vision_config={
            "num_hidden_layers": 1,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "patch_size": 14,
            "rope_theta": 10000.0,
            "downsample_ratio": 3,
            "max_num_tokens": 64,
            "min_pixels": 0,
            "max_wh_ratio": None,
        },
    )
    result = DeepseekV41VLProcessor(config)(
        images=[Image.new("RGB", (84, 42))]
    )

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
    token_ids = torch.tensor([17, IMAGE_TOKEN_ID, IMAGE_PAD_ID, 18])
    expected = torch.tensor([True, False, False, True])

    torch.testing.assert_close(image_sentinel_mask(token_ids), ~expected)
    torch.testing.assert_close(
        valid_engram_token_mask(token_ids, IMAGE_TOKEN_ID, IMAGE_PAD_ID),
        expected,
    )


def test_v41_span_has_three_delimiters_and_no_image_pad_parameter():
    wrapper = object.__new__(AscendDeepseekV41ForConditionalGeneration)
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

    wrapper = object.__new__(AscendDeepseekV41ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.language_model = LanguageModel()

    embeddings = wrapper.embed_input_ids(
        torch.tensor([7, IMAGE_PAD_ID, IMAGE_TOKEN_ID])
    )
    assert embeddings.squeeze(-1).tolist() == [7, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID]
