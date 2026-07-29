# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from PIL import Image
from transformers import BatchFeature
from vllm.config.multimodal import ImageDummyOptions
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFlatField,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems

from vllm_ascend.models.kimi_k3 import (
    KimiK3DummyInputsBuilder,
    KimiK3MultiModalProcessor,
    KimiK3ProcessingInfo,
    navit_resize_image,
)
from vllm_ascend.transformers_utils.processors.kimi_k3 import KimiK3Processor

IMAGE_PLACEHOLDER = "<|kimi_image_placeholder|>"
MEDIA_TOKEN = "<|media_pad|>"
MEDIA_TOKEN_ID = 163605


def test_kimi_k3_processor_wraps_images_without_expanding_prompt():
    images = [object(), object()]
    image_processor = SimpleNamespace(
        preprocess=MagicMock(
            return_value={
                "pixel_values": torch.ones(2, 3),
                "grid_thws": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            }
        )
    )
    tokenizer = MagicMock(
        return_value={
            "input_ids": [[101, 102, 103]],
            "attention_mask": [[1, 1, 1]],
        }
    )
    processor = KimiK3Processor(image_processor, tokenizer)

    outputs = processor(
        text=f"before {IMAGE_PLACEHOLDER} after",
        images=images,
        return_tensors=None,
    )

    image_processor.preprocess.assert_called_once()
    media_arg = image_processor.preprocess.call_args.args[0]
    assert media_arg == [
        {"type": "image", "image": images[0]},
        {"type": "image", "image": images[1]},
    ]
    assert image_processor.preprocess.call_args.kwargs == {
        "return_tensors": None,
    }
    tokenizer.assert_called_once_with(
        [f"before {IMAGE_PLACEHOLDER} after"],
    )
    assert outputs["input_ids"] == [[101, 102, 103]]
    assert outputs["attention_mask"] == [[1, 1, 1]]


def test_kimi_k3_multimodal_fields_use_image_modality_and_grid_slices():
    processor = object.__new__(KimiK3MultiModalProcessor)
    grid_thws = torch.tensor([[1, 2, 3], [2, 3, 4]])

    assert (
        processor._hf_processor_applies_updates(
            prompt_text=IMAGE_PLACEHOLDER,
            mm_items=MagicMock(),
            hf_processor_mm_kwargs={},
            tokenization_kwargs={},
        )
        is False
    )

    fields = processor._get_mm_fields_config(
        BatchFeature({"grid_thws": grid_thws}),
        {},
    )

    pixel_values = fields["pixel_values"]
    assert pixel_values.modality == "image"
    assert isinstance(pixel_values.field, MultiModalFlatField)
    assert [(int(item[0].start), int(item[0].stop)) for item in pixel_values.field.slices] == [(0, 6), (6, 30)]

    grid = fields["grid_thws"]
    assert grid.modality == "image"
    assert isinstance(grid.field, MultiModalBatchedField)
    assert grid.field.keep_on_cpu is True


def test_kimi_k3_prompt_update_expands_original_image_size_and_media_pads():
    image = Image.new("RGB", (640, 480))
    media_tokens_calculator = MagicMock(return_value=3)
    processor = object.__new__(KimiK3MultiModalProcessor)
    processor.info = SimpleNamespace(
        media_token_id=MEDIA_TOKEN_ID,
        media_token=MEDIA_TOKEN,
        media_tokens_calculator=media_tokens_calculator,
        get_hf_config=lambda: SimpleNamespace(
            image_placeholder=IMAGE_PLACEHOLDER,
        ),
    )
    mm_items = MultiModalDataItems(
        {"image": ImageProcessorItems([image])},
    )

    updates = processor._get_prompt_updates(
        mm_items,
        {},
        MultiModalKwargsItems({}),
    )

    assert len(updates) == 1
    update = updates[0]
    assert update.modality == "image"
    assert update.target == IMAGE_PLACEHOLDER
    assert callable(update.replacement)

    details = update.replacement(0)
    assert details.full == (f"<|media_begin|>image 640x480<|media_content|>{MEDIA_TOKEN * 3}<|media_end|>")
    media_tokens_calculator.assert_called_once()
    media = media_tokens_calculator.call_args.args[0]
    assert media["type"] == "image"
    assert media["image"] is image

    tokenizer = MagicMock()
    tokenizer.encode.return_value = [10, MEDIA_TOKEN_ID, MEDIA_TOKEN_ID, MEDIA_TOKEN_ID, 11]
    assert details.is_embed is not None
    assert details.is_embed(tokenizer, details.full).tolist() == [
        False,
        True,
        True,
        True,
        False,
    ]


def test_kimi_k3_dummy_builder_profiles_true_maximum_image_shape():
    size = KimiK3ProcessingInfo.get_max_image_size(
        patch_size=14,
        merge_kernel_size=2,
        in_patch_limit=65536,
        patch_limit_on_one_side=512,
        fixed_output_tokens=None,
    )
    assert size == (1861, 7041)
    assert (
        navit_resize_image(
            size.width,
            size.height,
            patch_size=14,
            merge_kernel_size=2,
            in_patch_limit=65536,
            patch_limit_on_one_side=512,
            fixed_output_tokens=None,
        )["num_tokens"]
        == 16817
    )

    info = SimpleNamespace(
        image_processor=SimpleNamespace(
            media_proc_cfg={
                "patch_size": 14,
                "merge_kernel_size": 2,
                "in_patch_limit": 65536,
                "patch_limit_on_one_side": 512,
                "fixed_output_tokens": None,
            }
        ),
        get_hf_config=lambda: SimpleNamespace(
            image_placeholder=IMAGE_PLACEHOLDER,
        ),
        get_max_image_size=KimiK3ProcessingInfo.get_max_image_size,
    )
    builder = KimiK3DummyInputsBuilder(info)
    builder._get_dummy_images = MagicMock(return_value=["image-0", "image-1"])
    options = ImageDummyOptions(count=2, width=640, height=480)

    assert builder.get_dummy_text({"image": 2}) == IMAGE_PLACEHOLDER * 2
    assert builder.get_dummy_mm_data(
        seq_len=4096,
        mm_counts={"image": 2},
        mm_options={"image": options},
    ) == {"image": ["image-0", "image-1"]}
    builder._get_dummy_images.assert_called_once_with(
        height=7041,
        width=1861,
        num_images=2,
        overrides=options,
    )
