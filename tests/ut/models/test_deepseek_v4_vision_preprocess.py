# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import DeepseekV4Config, PreTrainedTokenizerFast
from vllm.config.multimodal import MultiModalConfig
from vllm.models.deepseek_v4.common import mm_preprocess as prep
from vllm.models.deepseek_v4.common.mm_preprocess import (
    COMPRESS_PAD_TO,
    IMAGE,
    IMAGE_PAD,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessor,
)
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails
from vllm.multimodal.processing.context import InputProcessingContext, TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
from vllm.renderers.params import TokenizeParams


class _StubInfo:
    def __init__(self):
        self.tokenizer = _StubTokenizer()

    @property
    def ctx(self):
        return SimpleNamespace(tokenizer=self.get_tokenizer())

    def get_data_parser(self):
        return MultiModalDataParser()

    def get_tokenizer(self):
        return self.tokenizer


class _StubTokenizer:
    def convert_tokens_to_ids(self, token):
        if token == prep.IMAGE_PLACEHOLDER:
            return 7
        return prep.IMAGE_SENTINEL_BASE_ID + prep.IMAGE_SENTINEL_TOKEN_NAMES.index(token)


@pytest.fixture
def vision_info():
    config = SimpleNamespace(
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_max_n_token=384,
        vision_min_pixels=147456,
        vision_max_wh_ratio=8,
    )
    ctx = SimpleNamespace(get_hf_config=lambda *args, **kwargs: config)
    return prep.DeepseekV4VLProcessingInfo(ctx)


def test_dummy_image_covers_wide_image(vision_info):
    """A square underestimates both tokens and patches (vLLM #59373)."""
    dummy = vision_info.get_image_size_with_most_features()
    processor = vision_info.get_hf_processor()
    dummy_output = processor(images=[Image.new("RGB", dummy)])
    wide_output = processor(images=[Image.new("RGB", (1932, 336))])

    assert dummy == (1932, 336)
    assert dummy_output["patches"].shape[0] == 3312
    assert dummy_output["types"].numel() == 378
    assert wide_output["patches"].shape[0] <= dummy_output["patches"].shape[0]
    assert wide_output["types"].numel() <= dummy_output["types"].numel()


@pytest.mark.parametrize("max_wh_ratio", [1, 2, 8, None])
def test_dummy_image_respects_aspect_ratio_without_resizing(vision_info, max_wh_ratio):
    config = vision_info.get_hf_config()
    config.vision_max_wh_ratio = max_wh_ratio
    dummy = vision_info.get_image_size_with_most_features()

    if max_wh_ratio is not None:
        assert dummy.width <= dummy.height * max_wh_ratio

    output = vision_info.get_hf_processor()(images=[Image.new("RGB", dummy)])
    n_vit_h, n_vit_w = output["vit_grid"][0].tolist()
    assert n_vit_h * config.vision_patch_size == dummy.height
    assert n_vit_w * config.vision_patch_size == dummy.width


@pytest.mark.parametrize("max_n_token", [384, 385])
@pytest.mark.parametrize("start_pos", range(COMPRESS_PAD_TO))
def test_dummy_image_reserves_compressor_padding(vision_info, max_n_token, start_pos):
    config = vision_info.get_hf_config()
    config.vision_max_n_token = max_n_token
    dummy = vision_info.get_image_size_with_most_features()
    output = vision_info.get_hf_processor()(images=[Image.new("RGB", dummy)])
    n_llm_h, n_llm_w = output["llm_grid"][0].tolist()
    types, _ = prep.build_image_block(n_llm_h, n_llm_w, start_pos)
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO

    assert output["types"].numel() <= max_n_token - (COMPRESS_PAD_TO - 1)
    assert types.numel() <= max_n_token
    assert torch.equal(types[compress_pad:], output["types"])


def test_dummy_builder_uses_worst_case_images(vision_info):
    builder = prep.DeepseekV4VLDummyInputsBuilder(vision_info)
    num_images = 2
    mm_data = builder.get_dummy_mm_data(seq_len=4096, mm_counts={"image": num_images}, mm_options={})

    assert len(mm_data["image"]) == num_images
    assert all(image.size == (1932, 336) for image in mm_data["image"])

    output = vision_info.get_hf_processor()(images=mm_data["image"])
    assert output["patches"].shape[0] == num_images * 3312
    assert output["types"].numel() == num_images * 378


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


@pytest.mark.parametrize("start_pos", range(COMPRESS_PAD_TO))
def test_prompt_updates_add_position_dependent_compress_pad(start_pos):
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
    prefix = [11] * start_pos
    prompt = prefix + [image_token_id, 13, 14, 15, image_token_id, 16]
    mm_prompt_updates = {"image": [[update.resolve(0)], [update.resolve(1)]]}

    token_ids, placeholders = processor._apply_prompt_updates(
        prompt,
        mm_prompt_updates,
    )

    expected = list(prefix)
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


@pytest.fixture
def processing_info():
    mm_config = MultiModalConfig(limit_per_prompt={"image": 2})
    model_config = SimpleNamespace(
        model="deepseek-v4-vision-ut",
        hf_config=DeepseekV4Config(
            vision_patch_size=14,
            vision_downsample_ratio=3,
            vision_max_n_token=384,
            vision_min_pixels=147456,
            vision_max_wh_ratio=8,
        ),
        dtype=torch.bfloat16,
        multimodal_config=mm_config,
        get_multimodal_config=lambda: mm_config,
    )
    ctx = InputProcessingContext(model_config, _StubTokenizer())
    return prep.DeepseekV4VLProcessingInfo(ctx)


@pytest.mark.parametrize("start_pos", range(COMPRESS_PAD_TO))
@pytest.mark.parametrize("image_sizes", [(), ((128, 96),), ((128, 96), (84, 168))])
def test_processor_apply_preserves_image_order_and_embedding_masks(processing_info, start_pos, image_sizes):
    processor = DeepseekV4VLMultiModalProcessor(processing_info, None)
    images = [Image.new("RGB", size) for size in image_sizes]
    image_items = processing_info.parse_mm_data({"image": images} if images else {})
    prompt = [11] * start_pos + [token for i in range(len(images)) for token in (7, 13 + i)] + [16]

    output = processor.apply(ProcessorInputs(prompt, image_items), TimingContext(enabled=False))

    expected = [11] * start_pos
    placeholders = output["mm_placeholders"].get("image", [])
    assert len(placeholders) == len(images)
    for i, (image, placeholder) in enumerate(zip(images, placeholders)):
        processed = processing_info.get_hf_processor()(images=[image])
        n_llm_h, n_llm_w = processed["llm_grid"][0].tolist()
        types, _ = prep.build_image_block(n_llm_h, n_llm_w, len(expected))
        tokens = prep.IMAGE_SENTINEL_BASE_ID + types
        assert placeholder.offset == len(expected)
        assert placeholder.length == len(tokens)
        assert torch.equal(placeholder.is_embed, types == IMAGE)
        assert placeholder.get_num_embeds() == n_llm_h * n_llm_w
        expected.extend(tokens.tolist())
        expected.append(13 + i)

    expected.append(16)
    assert output["prompt_token_ids"] == expected


class _ConcurrentTokenizer(PreTrainedTokenizerFast):
    """Hold the renderer's encoding call open and reject overlapping encoding.

    The guard makes accidental re-tokenization in the image path fail without
    depending on the timing of the fast tokenizer's Rust borrow checks.
    Read-only token-ID lookup still uses the real fast tokenizer.
    """

    def __call__(self, *args, **kwargs):
        assert self.encoding_lock.acquire(blocking=False), "concurrent tokenizer encoding"
        try:
            if self.hold_encoding:
                self.encoding_started.set()
                assert self.release_encoding.wait(timeout=10), "image processing did not finish"
            return super().__call__(*args, **kwargs)
        finally:
            self.encoding_lock.release()


@pytest.mark.parametrize("num_images", [1, 2])
@pytest.mark.parametrize("truncate_tokens", [None, 3])
def test_image_preprocessing_during_renderer_tokenization(processing_info, monkeypatch, num_images, truncate_tokens):
    # A small local vocabulary keeps this test independent of model downloads.
    monkeypatch.setattr(prep, "IMAGE_SENTINEL_BASE_ID", 5)
    tokens = ["[UNK]", "a", "b", "c", prep.IMAGE_PLACEHOLDER, *prep.IMAGE_SENTINEL_TOKEN_NAMES]
    backend = Tokenizer(WordLevel(dict(zip(tokens, range(len(tokens)), strict=True)), unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = _ConcurrentTokenizer(
        tokenizer_object=backend,
        unk_token="[UNK]",
        additional_special_tokens=tokens[4:],
    )
    tokenizer.encoding_lock = Lock()
    tokenizer.encoding_started = Event()
    tokenizer.release_encoding = Event()
    tokenizer.hold_encoding = False
    renderer = DeepseekV4Renderer.__new__(DeepseekV4Renderer)
    renderer.tokenizer = tokenizer
    params = TokenizeParams(max_total_tokens=None, truncate_prompt_tokens=truncate_tokens)
    text = {"prompt": "a b c a b c"}
    expected_text = renderer._tokenize_prompt(text, params)
    image_prompt = renderer._tokenize_prompt(
        {"prompt": "a " + f"{prep.IMAGE_PLACEHOLDER} b " * num_images},
        TokenizeParams(max_total_tokens=None),
    )["prompt_token_ids"]
    processing_info = prep.DeepseekV4VLProcessingInfo(
        InputProcessingContext(processing_info.ctx.model_config, tokenizer)
    )
    processor = DeepseekV4VLMultiModalProcessor(processing_info, None)

    def process_images():
        images = [Image.new("RGB", (128 + i * 14, 96), color=(10 + i, 20, 30)) for i in range(num_images)]
        inputs = ProcessorInputs(
            prompt=image_prompt.copy(),
            mm_data_items=processor.data_parser.parse_mm_data({"image": images}),
        )
        return processor.apply(inputs, TimingContext(enabled=False))

    expected = process_images()
    for _ in range(2):
        tokenizer.encoding_started.clear()
        tokenizer.release_encoding.clear()
        tokenizer.hold_encoding = True
        with ThreadPoolExecutor(max_workers=2) as executor:
            rendering = executor.submit(renderer._tokenize_prompt, text, params)
            try:
                assert tokenizer.encoding_started.wait(timeout=10)
                actual = executor.submit(process_images).result(timeout=10)
            finally:
                tokenizer.release_encoding.set()
            assert rendering.result(timeout=10) == expected_text
        assert actual["prompt_token_ids"] == expected["prompt_token_ids"]
        assert actual["mm_hashes"] == expected["mm_hashes"]
        for actual_item, expected_item in zip(
            actual["mm_kwargs"]["image"], expected["mm_kwargs"]["image"], strict=True
        ):
            assert actual_item.keys() == expected_item.keys()
            for name in expected_item:
                assert torch.equal(actual_item[name].data, expected_item[name].data), name
        for actual_range, expected_range in zip(
            actual["mm_placeholders"]["image"], expected["mm_placeholders"]["image"], strict=True
        ):
            assert (actual_range.offset, actual_range.length) == (expected_range.offset, expected_range.length)
            assert torch.equal(actual_range.is_embed, expected_range.is_embed)
