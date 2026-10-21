# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import torch
from PIL import Image
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails

from vllm_ascend.models.deepseek_v4 import mm_preprocess as prep
from vllm_ascend.models.deepseek_v4.mm_preprocess import (
    COMPRESS_PAD_TO,
    IMAGE,
    IMAGE_PAD,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessor,
)


class _StubInfo:
    def __init__(self):
        self.tokenizer = SimpleNamespace()

    @property
    def ctx(self):
        return SimpleNamespace(tokenizer=self.get_tokenizer())

    def get_data_parser(self):
        return MultiModalDataParser()

    def get_tokenizer(self):
        return self.tokenizer


def test_local_image_processor_builds_vit_and_llm_inputs():
    config = SimpleNamespace(
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_max_n_token=384,
        vision_min_pixels=147456,
        vision_max_wh_ratio=8,
    )
    output = DeepseekV4VLProcessor(config)(images=[Image.new("RGB", (128, 96), color=(10, 20, 30))])

    assert output["patches"].dtype == torch.bfloat16
    assert output["patches"].shape[1:] == (3, 14, 14)
    assert output["vit_grid"].shape == (1, 2)
    assert output["llm_grid"].shape == (1, 2)
    assert output["perm"].numel() == output["llm_grid"].prod().item()


def test_v027_prompt_updates_add_position_dependent_compress_pad():
    base = prep.IMAGE_SENTINEL_BASE_ID
    image_token_id = 7
    n_llm_h, n_llm_w = 3, 2
    processor = DeepseekV4VLMultiModalProcessor(_StubInfo(), None)

    types, _ = prep.build_image_block_pad_free(n_llm_h, n_llm_w)
    full = (base + types).tolist()
    update = PromptReplacement(
        modality="image",
        target=[image_token_id],
        replacement=PromptUpdateDetails.select_token_id(full, base + IMAGE),
    )
    prompt = [11, 12, image_token_id, 13, 14, 15, image_token_id, 16]
    mm_prompt_updates = {"image": [[update.resolve(0)], [update.resolve(1)]]}

    token_ids, placeholders = processor._apply_prompt_updates(
        prompt,
        mm_prompt_updates,
    )

    expected = [11, 12]
    first_types, _ = prep.build_image_block(
        n_llm_h,
        n_llm_w,
        len(expected),
    )
    expected += (base + first_types).tolist()
    expected += [13, 14, 15]
    second_types, _ = prep.build_image_block(
        n_llm_h,
        n_llm_w,
        len(expected),
    )
    expected += (base + second_types).tolist()
    expected += [16]
    assert token_ids == expected

    for placeholder in placeholders["image"]:
        pad = COMPRESS_PAD_TO - 1 - placeholder.start_idx % COMPRESS_PAD_TO
        assert placeholder.tokens[:pad] == [base + IMAGE_PAD] * pad
        assert placeholder.tokens[pad:] == full
        assert placeholder.is_embed is not None
        assert placeholder.is_embed.tolist() == [token == base + IMAGE for token in placeholder.tokens]


def test_image_processor_uses_upstream_normalized_inputs():
    config = SimpleNamespace(
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_max_n_token=384,
        vision_min_pixels=147456,
        vision_max_wh_ratio=8,
    )
    hf_processor = DeepseekV4VLProcessor(config)
    calls = []

    def call_hf_processor(processor, data, kwargs):
        calls.append((data, kwargs))
        return processor(**data, **kwargs, return_tensors="pt")

    class ImageInfo(_StubInfo):
        @property
        def ctx(self):
            return SimpleNamespace(tokenizer=self.tokenizer, call_hf_processor=call_hf_processor)

        def get_hf_processor(self, **kwargs):
            assert not kwargs
            return hf_processor

    processor = DeepseekV4VLMultiModalProcessor(ImageInfo(), None)
    image = Image.new("RGB", (128, 96))
    output = processor._call_hf_processor({"images": [image]}, {})
    assert calls == [({"images": [image]}, {})]
    assert output["patches"].dtype == torch.bfloat16
    assert "input_ids" not in output
