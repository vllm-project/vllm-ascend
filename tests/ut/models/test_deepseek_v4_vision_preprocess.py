# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch
from PIL import Image
from transformers import BatchFeature
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
    def get_data_parser(self):
        return MultiModalDataParser()

    def get_tokenizer(self):
        return None


class _NonThreadSafeTokenizer:
    def __init__(self, entered=None, release=None):
        self._state_lock = threading.Lock()
        self._active = False
        self._entered = entered
        self._release = release

    def __deepcopy__(self, memo):
        return _NonThreadSafeTokenizer()

    def __call__(self, prompt, return_tensors, **kwargs):
        with self._state_lock:
            if self._active:
                raise RuntimeError("Already borrowed")
            self._active = True
        try:
            if self._entered is not None:
                self._entered.set()
            if self._release is not None:
                self._release.wait(timeout=1)
            time.sleep(0.01)
            return {"input_ids": torch.tensor([[1]])}
        finally:
            with self._state_lock:
                self._active = False


class _ConcurrentStubInfo(_StubInfo):
    def __init__(self):
        self.tokenizer = _NonThreadSafeTokenizer()

    def get_hf_processor(self, **kwargs):
        return lambda **processor_kwargs: BatchFeature({})

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


def test_hf_tokenizer_call_is_thread_safe():
    entered = threading.Event()
    release = threading.Event()
    info = _ConcurrentStubInfo()
    info.tokenizer = _NonThreadSafeTokenizer(entered, release)
    processor = DeepseekV4VLMultiModalProcessor(info, None)

    def process():
        return processor._call_hf_processor(
            "prompt",
            {"images": []},
            {},
            {},
        )["input_ids"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        competing_call = pool.submit(
            info.tokenizer,
            "chat template",
            "pt",
        )
        assert entered.wait(timeout=1)
        outputs = list(pool.map(lambda _: process(), range(16)))
        release.set()
        competing_call.result()

    assert all(torch.equal(output, torch.tensor([[1]])) for output in outputs)
