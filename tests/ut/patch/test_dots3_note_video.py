# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from vllm_ascend.patch.dots3_note_video import (
    _decode_frames,
    _flatten_seed,
    _group_bounds,
    format_timestamp,
    preprocess_video,
    solve_v2_plan,
)


def test_dots3_note_video_plan_matches_training_contract():
    num_frames, height, width = solve_v2_plan(
        visual_budget=100_000,
        duration=10.0,
        height=720,
        width=1280,
        source_fps=25.0,
    )

    assert num_frames == 10
    assert height % 28 == 0
    assert width % 28 == 0
    assert format_timestamp(3661.239) == "<01:01:01.24>"


def test_dots3_note_video_logk_grouping_is_deterministic():
    seed = _flatten_seed(b"video", "question")

    assert seed == 4055066043
    assert _group_bounds(100, 193.233, "logk", random.Random(seed)) == [
        0,
        3,
        6,
        10,
        30,
        39,
        49,
        100,
    ]


def test_dots3_note_video_seek_flushes_decoder_and_uses_frame_time(monkeypatch):
    codec_context = SimpleNamespace(flush_buffers=MagicMock())
    stream = SimpleNamespace(time_base=0.1, codec_context=codec_context)
    frame = SimpleNamespace(
        time=1.0,
        pts=None,
        time_base=0.1,
        to_image=lambda: Image.new("RGB", (4, 4)),
    )
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams.video = [stream]
    container.decode.return_value = [frame]
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _: container))

    frames = _decode_frames(b"video", [10], fps=10.0, target_h=4, target_w=4)

    assert len(frames) == 1
    container.seek.assert_called_once_with(10, stream=stream, backward=True, any_frame=False)
    codec_context.flush_buffers.assert_called_once_with()


def test_dots3_note_video_predecoded_fallback_keeps_one_composite_item():
    result = preprocess_video(
        (np.zeros((6, 32, 48, 3), dtype=np.uint8), {"fps": 2.0, "duration": 3.0}),
        prompt="<|video_pad|>question",
        question="question",
        tokenizer=None,
    )

    assert len(result.frames) == 4
    assert result.timestamps == [0.0, 1.0, 1.5, 2.5]
    assert result.audio_segments == []
    assert result.layout == [("image", index) for index in range(4)]
