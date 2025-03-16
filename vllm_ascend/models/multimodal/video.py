#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from vllm/multimodal/video.py
# Copyright 2023 The vLLM team.
#
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

from __future__ import annotations

import torch
from torchvision.io import read_video
from vllm.multimodal.video import VideoMediaIO
import tempfile
import numpy.typing as npt

NUM_FRAMES = 32


def _read_video_torchvision(
    self,
    data: bytes,
) -> npt.NDArray:
    # read video using torchvision.io.read_video

    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_file:
        temp_file.write(data)
        temp_file.flush()
        video, _, _ = read_video(
            temp_file.name,
            start_pts=0.0,
            end_pts=None,
            pts_unit="sec",
            output_format="TCHW",
        )
    total_frames = video.size(0)
    idx = torch.linspace(0, total_frames - 1, NUM_FRAMES).round().long()
    video = video[idx].permute(0, 2, 3, 1)
    return video.numpy()


VideoMediaIO.load_bytes = _read_video_torchvision
