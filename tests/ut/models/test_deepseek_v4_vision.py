from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from vllm.model_executor.models.interfaces import supports_eagle3

from vllm_ascend.models.deepseek_v4.mm_preprocess import IMAGE_SENTINEL_BASE_ID
from vllm_ascend.models.deepseek_v4.vl_model import (
    AscendDeepseekV4ForConditionalGeneration,
)


def test_vision_wrapper_exposes_dspark_aux_hidden_state_interface():
    model = AscendDeepseekV4ForConditionalGeneration.__new__(AscendDeepseekV4ForConditionalGeneration)
    nn.Module.__init__(model)
    language_model = MagicMock()
    model.language_model = language_model

    assert supports_eagle3(model)

    model.set_aux_hidden_state_layers((41, 42, 43))
    language_model.set_aux_hidden_state_layers.assert_called_once_with((41, 42, 43))


@pytest.fixture
def vision_wrapper():
    model = AscendDeepseekV4ForConditionalGeneration.__new__(AscendDeepseekV4ForConditionalGeneration)
    nn.Module.__init__(model)
    hidden = 8
    for i, name in enumerate(("image_start", "image_pad", "image_newline", "image_end")):
        setattr(model, name, nn.Parameter(torch.full((hidden,), float(i + 1), dtype=torch.float32)))

    language_model = MagicMock()
    inputs_embeds = torch.randn(6, hidden, dtype=torch.bfloat16)
    language_model.embed_input_ids.return_value = inputs_embeds
    model.language_model = language_model
    return model


def test_embed_input_ids_replaces_sentinels(vision_wrapper):
    ids = torch.tensor(
        [
            IMAGE_SENTINEL_BASE_ID + 0,  # start
            IMAGE_SENTINEL_BASE_ID + 1,  # pad
            IMAGE_SENTINEL_BASE_ID + 2,  # image (no mm embeddings here)
            IMAGE_SENTINEL_BASE_ID + 3,  # newline
            IMAGE_SENTINEL_BASE_ID + 4,  # end
            100,  # regular token
        ]
    )
    base = vision_wrapper.language_model.embed_input_ids.return_value.clone()
    out = vision_wrapper.embed_input_ids(ids)

    for row, name in enumerate(("image_start", "image_pad", "image_pad", "image_newline", "image_end")):
        assert torch.equal(out[row], getattr(vision_wrapper, name).to(out.dtype))
    assert torch.equal(out[5], base[5])


@pytest.mark.parametrize(
    ("marker", "rows"),
    [("image_start", [0]), ("image_pad", [1, 2]), ("image_newline", [3]), ("image_end", [4])],
)
def test_embed_input_ids_uses_updated_marker(vision_wrapper, marker, rows):
    ids = torch.tensor([*(IMAGE_SENTINEL_BASE_ID + i for i in range(5)), 100])
    before = vision_wrapper.embed_input_ids(ids).clone()
    parameter = getattr(vision_wrapper, marker)
    with torch.no_grad():
        parameter.add_(10)

    out = vision_wrapper.embed_input_ids(ids)
    expected = before.clone()
    expected[rows] = parameter.to(expected.dtype)

    assert not torch.equal(out, before)
    assert torch.equal(out, expected)


def test_sentinel_table_uses_requested_device_and_dtype(vision_wrapper):
    meta_table = vision_wrapper._get_sentinel_table(torch.bfloat16, torch.device("meta"))
    assert meta_table.device.type == "meta"
    assert meta_table.dtype == torch.bfloat16
    assert meta_table.shape == (5, 8)

    rebuilt = vision_wrapper._get_sentinel_table(torch.float32, torch.device("cpu"))
    assert rebuilt.device.type == "cpu"
    assert rebuilt.dtype == torch.float32
    for row, name in enumerate(("image_start", "image_pad", "image_pad", "image_newline", "image_end")):
        assert torch.equal(rebuilt[row], getattr(vision_wrapper, name))
    assert not rebuilt.requires_grad


def test_embed_input_ids_without_image_params_is_passthrough(vision_wrapper):
    vision_wrapper.image_start = None
    ids = torch.tensor([IMAGE_SENTINEL_BASE_ID, 100])
    out = vision_wrapper.embed_input_ids(ids)
    base = vision_wrapper.language_model.embed_input_ids.return_value
    assert torch.equal(out, base)
