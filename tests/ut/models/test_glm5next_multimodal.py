# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from types import SimpleNamespace

import pytest
from transformers.video_utils import VideoMetadata

from vllm_ascend.models.glm5next.multimodal import Glm5NextProcessingInfo
from vllm_ascend.models.glm5next.processor import Glm5NextVideoProcessor


@pytest.mark.parametrize("num_frames,fps_override", [(8, None), (8, 1), (3, None), (1, None)])
def test_video_placeholders_follow_flash_frame_sampling(num_frames, fps_override):
    kwargs = {} if fps_override is None else {"fps": fps_override}
    info = Glm5NextProcessingInfo(SimpleNamespace(get_merged_mm_kwargs=lambda _: kwargs))
    processor = Glm5NextVideoProcessor(fps_interval=2, temporal_patch_size=2)
    info._glm5_next_hf_processor = SimpleNamespace(video_processor=processor)
    metadata = dict(total_num_frames=num_frames, fps=4, duration=2, do_sample_frames=True)
    original = deepcopy(metadata)
    indices = processor.sample_frames(VideoMetadata(total_num_frames=num_frames, fps=4, duration=2), **kwargs)

    timestamps = info._get_video_second_idx_glm46v(metadata, num_frames)

    assert timestamps == [int(index / 4) for index in indices[::2]]
    assert len(timestamps) == len(indices) // processor.temporal_patch_size
    assert metadata == original


def test_video_placeholders_preserve_already_sampled_frames():
    info = Glm5NextProcessingInfo(SimpleNamespace(get_merged_mm_kwargs=lambda _: {}))
    info._glm5_next_hf_processor = SimpleNamespace(video_processor=Glm5NextVideoProcessor())
    metadata = dict(total_num_frames=8, fps=4, duration=2, do_sample_frames=False, frames_indices=[0, 7])

    assert info._get_video_second_idx_glm46v(metadata, 2) == [0]
