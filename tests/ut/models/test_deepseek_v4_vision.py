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
    model._sentinel_table = None
    for i, name in enumerate(("image_start", "image_pad", "image_newline", "image_end")):
        setattr(model, name, nn.Parameter(torch.full((hidden,), float(i + 1), dtype=torch.float32)))

    language_model = MagicMock()
    inputs_embeds = torch.randn(6, hidden, dtype=torch.bfloat16)
    language_model.embed_input_ids.return_value = inputs_embeds
    model.language_model = language_model
    return model


def test_embed_input_ids_replaces_sentinels_and_caches_table(vision_wrapper):
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
    out = vision_wrapper.embed_input_ids(ids)

    table = vision_wrapper._sentinel_table
    assert table is not None
    assert table.dtype == torch.bfloat16
    assert table.shape == (5, 8)
    # start / pad / newline / end rows come from the cached table
    assert torch.equal(out[0], table[0])
    assert torch.equal(out[1], table[1])
    assert torch.equal(out[2], table[2])
    assert torch.equal(out[3], table[3])
    assert torch.equal(out[4], table[4])
    # regular tokens pass through unchanged
    base = vision_wrapper.language_model.embed_input_ids.return_value
    assert torch.equal(out[5], base[5])

    # Second call must reuse the cached table (no re-stacking).
    table_before = vision_wrapper._sentinel_table
    vision_wrapper.embed_input_ids(ids)
    assert vision_wrapper._sentinel_table is table_before


def test_sentinel_table_rebuilt_on_device_change(vision_wrapper):
    # The fixture mock returns a (6, 8) embedding; align it with the
    # shorter ids used below so the in-place where can broadcast.
    vision_wrapper.language_model.embed_input_ids.return_value = torch.randn(2, 8, dtype=torch.bfloat16)
    ids = torch.tensor([IMAGE_SENTINEL_BASE_ID, 100])
    vision_wrapper.embed_input_ids(ids)
    assert vision_wrapper._sentinel_table.device.type == "cpu"

    # The table is a plain attribute and does not follow nn.Module.to();
    # a device move after the first forward must invalidate the cache.
    meta_table = vision_wrapper._get_sentinel_table(torch.bfloat16, torch.device("meta"))
    assert meta_table.device.type == "meta"
    assert vision_wrapper._sentinel_table is meta_table

    # Moving back rebuilds again instead of returning the stale table.
    rebuilt = vision_wrapper._get_sentinel_table(torch.bfloat16, torch.device("cpu"))
    assert rebuilt.device.type == "cpu"
    assert rebuilt is not meta_table
    assert vision_wrapper._sentinel_table is rebuilt


def test_embed_input_ids_without_image_params_is_passthrough(vision_wrapper):
    vision_wrapper.image_start = None
    ids = torch.tensor([IMAGE_SENTINEL_BASE_ID, 100])
    out = vision_wrapper.embed_input_ids(ids)
    base = vision_wrapper.language_model.embed_input_ids.return_value
    assert torch.equal(out, base)
    assert vision_wrapper._sentinel_table is None
