# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
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


def _pattern_image(width: int, height: int) -> Image.Image:
    y, x = np.indices((height, width), dtype=np.uint16)
    pixels = np.stack(
        (
            (x * 3 + y * 5) % 256,
            (x * 7 + y * 11 + 17) % 256,
            (x * 13 + y * 19 + 29) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(
    ("name", "width", "height", "expected"),
    [
        (
            "square",
            392,
            392,
            (
                (784, 3, 14, 14),
                "cb9ebeb276de6b449bc942dd4d55398a2f9fbdbfdd9a28b4cea93be8cdb1847c",
                (28, 28),
                (10, 10),
                114,
                "da4b32a5c728550fcc6f88567ef03c3952aad1bba7d8f2d8ee222ec89103c556",
                100,
                "f376025fbf067fb7d845127be7b72bf0fafdc6a3510bd7b658caca8c5cd198d5",
            ),
        ),
        (
            "panorama",
            1600,
            100,
            (
                (780, 3, 14, 14),
                "2539b7bf59517689ec0b2f3e1417e5544ef617dde14ebbb88366222376d65cb0",
                (10, 78),
                (4, 26),
                110,
                "bd33de4bbf3afc05780806c320b7928f6bf774b86d6c6840a054aedfa9dc6a61",
                104,
                "a433d2bfca12832470ca925ad0403a82553a08abae9f159afacd062a1a0e3313",
            ),
        ),
        (
            "portrait",
            100,
            1600,
            (
                (920, 3, 14, 14),
                "9e22cfa3743e92e091cd86a26774175f216616aa706751c22765fa70486bdd9c",
                (115, 8),
                (39, 3),
                162,
                "74c67b7ea557942f339c7fac659f7756c13e6e8f76d7f50f267b62bb086f8b26",
                117,
                "cfae26c65300c4e315e973c22678c59372b3903a01a7a70a453bd72b990038fb",
            ),
        ),
        (
            "min_pixel",
            32,
            24,
            (
                (768, 3, 14, 14),
                "3e159e37db91587c2a7ed16cd4bc9b9e12c9e9ba3cebaf919dc97d4a943bd8ac",
                (24, 32),
                (8, 11),
                98,
                "084c2845dd3f9aa822817b17bb532621559b5d6aaa640ab612c02613e546f7a5",
                88,
                "27af7da74d9a95489b32bcdc31c4420995f81bbfe0724cd3c3bdaa40d3759e4d",
            ),
        ),
        (
            "max_token",
            4096,
            3072,
            (
                (2961, 3, 14, 14),
                "ccff1f42208ce479433d8ebbe31abadddfae727e1602d0284907d651e399af17",
                (47, 63),
                (16, 21),
                354,
                "24890234f78a3b081027729729127c131480224863bec0741864c5f7778dbc55",
                336,
                "053baba523007d1d2884962752a3df17ede2320a19c2221321c33ed38f895311",
            ),
        ),
    ],
)
def test_processor_matches_official_image_golden(name, width, height, expected):
    config = SimpleNamespace(
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_max_n_token=384,
        vision_min_pixels=147456,
        vision_max_wh_ratio=8,
    )
    output = DeepseekV4VLProcessor(config)(images=[_pattern_image(width, height)])
    (
        patch_shape,
        patch_hash,
        vit_grid,
        llm_grid,
        types_size,
        types_hash,
        perm_size,
        perm_hash,
    ) = expected

    assert tuple(output["patches"].shape) == patch_shape, name
    assert _tensor_sha256(output["patches"]) == patch_hash
    assert output["vit_grid"].tolist() == [list(vit_grid)]
    assert output["llm_grid"].tolist() == [list(llm_grid)]
    assert output["types"].numel() == types_size
    assert _tensor_sha256(output["types"]) == types_hash
    assert output["perm"].numel() == perm_size
    assert _tensor_sha256(output["perm"]) == perm_hash


@pytest.mark.parametrize(
    ("start_pos", "expected_size", "expected_hash"),
    [
        (0, 17, "79bbacd708dc7bb1a333454394bd8c3d7f8ceb34b09d034d7e5804be7e69b29f"),
        (1, 16, "f28cdc513680e49b6aa4a842f8ceabc2cf26d4f0e609b1d32ce9a62641f5a59f"),
        (2, 15, "ccae3d732e7f5df568cf40aeeb02ad044e006b4a4d265b53f1427f3fca4f0cae"),
        (3, 14, "fa1822de9eda236f7b38ebad58011b0009f335f39b295ea282f4f35cf29e3919"),
    ],
)
def test_image_block_matches_official_position_golden(
    start_pos,
    expected_size,
    expected_hash,
):
    types, perm = prep.build_image_block(3, 2, start_pos)

    assert types.numel() == expected_size
    assert _tensor_sha256(types) == expected_hash
    assert _tensor_sha256(perm) == ("9ce675ac27d3af2951b2da39d6c3dc65c4205a27bf5ecc20ab33d1d7387c7412")


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
