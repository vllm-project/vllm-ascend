# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare prompt replacements with real CPU video preprocessing, without weights."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast
from transformers.video_utils import VideoMetadata

from vllm_ascend.models.glm5next.multimodal import Glm5NextMultiModalProcessor, Glm5NextProcessingInfo
from vllm_ascend.models.glm5next.processor import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    Glm5NextVideoProcessor,
)


@pytest.fixture
def processor():
    tokens = [
        "[UNK]",
        "<|image|>",
        "<|video|>",
        "<|begin_of_image|>",
        "<|end_of_image|>",
        "<|begin_of_video|>",
        "<|end_of_video|>",
    ]
    vocab = {token: index for index, token in enumerate(tokens)}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")))
    hf_processor = Glm5NextProcessor(
        image_processor=Glm5NextImageProcessor(), video_processor=Glm5NextVideoProcessor(), tokenizer=tokenizer
    )
    config = SimpleNamespace(image_start_token_id=3, image_end_token_id=4, video_start_token_id=5, video_end_token_id=6)
    info = object.__new__(Glm5NextProcessingInfo)
    info.ctx = SimpleNamespace(get_tokenizer=lambda: tokenizer, get_hf_config=lambda *args, **kwargs: config)
    info._glm5_next_hf_processor = hf_processor
    result = object.__new__(Glm5NextMultiModalProcessor)
    result.info = info
    return result


def check_video(processor, duration, overrides=None, *, sampled=True, frames=None, indices=None):
    fps = 24.0 if duration <= 60 else 2.0
    count = int(duration * fps) if frames is None else frames
    video = torch.zeros((count, 3, 28, 28), dtype=torch.uint8)
    metadata = dict(
        fps=fps,
        total_num_frames=count,
        duration=duration,
        frames_indices=list(range(count)) if indices is None else indices,
        do_sample_frames=sampled,
    )
    kwargs = {"do_resize": False, **(overrides or {})}
    # Exercise the complete local HF wrapper, including flat/nested kwargs
    # resolution and the actual pixel sampler/patchifier.
    hf = processor.info.get_hf_processor()
    output = hf(
        text="<|video|>",
        videos=[video],
        video_metadata=[VideoMetadata(**{k: v for k, v in metadata.items() if k != "do_sample_frames"})],
        do_sample_frames=sampled,
        return_tensors="pt",
        **kwargs,
    )
    grid = output["video_grid_thw"][0]
    updates = processor._get_prompt_updates(
        {"video": [(video, metadata)]},
        kwargs,
        {"video": [{"video_grid_thw": SimpleNamespace(data=grid)}]},
    )
    replacement = next(update for update in updates if update.modality == "video").replacement(0)
    assert replacement.full[0] == 5 and replacement.full[-1] == 6
    # Each 28x28 frame has one merged embedding with the default merge_size=2.
    assert replacement.full.count(3) == int(grid[0])
    assert int(replacement.is_embed(replacement.full).sum()) == int(grid.prod()) // 4
    return replacement.full, int(grid[0])


@pytest.mark.parametrize("duration", [0.5, 2, 30, 60, 301, 600])
def test_default_sampling_matches_pixel_grid(processor, duration):
    check_video(processor, duration)


@pytest.mark.parametrize(
    "duration,overrides,expected",
    [
        (2, {"fps": 1.0}, 1),
        (2, {"fps": 4.0}, 4),
        (60, {"max_frames": 8}, 4),
        (2, {"videos_kwargs": {"fps": 4.0}}, 4),
        (60, {"videos_kwargs": {"max_frames": 8}}, 4),
        (2, {"target_fps": 1.0}, 2),
        (2, {"temporal_patch_size": 4}, 1),
    ],
)
def test_request_overrides_match_pixel_grid(processor, duration, overrides, expected):
    _, grid_t = check_video(processor, duration, overrides)
    assert grid_t == expected


@pytest.mark.parametrize("count,indices", [(1, [0]), (3, [0, 12, 24]), (4, [0, 0, 12, 12])])
def test_presampled_frames_preserve_padding_and_duplicates(processor, count, indices):
    check_video(processor, 2, {"fps": 4.0}, sampled=False, frames=count, indices=indices)


def test_request_overrides_do_not_leak_between_requests(processor):
    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(lambda kwargs: check_video(processor, 2, kwargs)[1], [{"fps": 1.0}, {"fps": 4.0}, {}]))
    assert results == [1, 4, 2]
    vp = processor.info.get_video_processor()
    assert vp.fps == 2 and vp.fps_interval == 2 and vp.max_frame_count_dynamic == 2048


@pytest.mark.parametrize("fps", [None, 0, -1, float("nan"), float("inf")])
def test_invalid_fps_is_rejected(processor, fps):
    with pytest.raises(ValueError, match="positive fps"):
        processor.info._get_video_second_idx_glm46v({"fps": fps}, 1)


def test_empty_sampling_is_rejected(processor):
    with pytest.raises(ValueError, match="selected no frames"):
        processor.info._get_video_second_idx_glm46v({"fps": 24.0, "duration": 0.25, "total_num_frames": 6}, 6)


def test_inconsistent_presampled_metadata_is_rejected(processor):
    with pytest.raises(ValueError, match="same length"):
        processor.info._get_video_second_idx_glm46v({"fps": 24.0, "do_sample_frames": False, "frames_indices": [0]}, 2)


def test_image_prompt_update_is_unchanged(processor):
    grid = torch.tensor([1, 4, 4])
    updates = processor._get_prompt_updates({}, {}, {"image": [{"image_grid_thw": SimpleNamespace(data=grid)}]})
    assert next(update for update in updates if update.modality == "image").replacement(0) == [1] * 4
