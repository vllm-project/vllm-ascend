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

from vllm_ascend.models.deepseek_v41.engram.common import (
    valid_engram_token_mask,
)
from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41LLMForCausalLM
from vllm_ascend.models.deepseek_v41.vl_model import (
    AscendDeepseekV41ForCausalLM,
)
from vllm_ascend.utils import normalize_deepseek_v41_config


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
