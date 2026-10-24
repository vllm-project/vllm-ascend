#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import os

import numpy as np

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = os.environ.get("QWEN35_EVS_MODEL", "/workspace/shared_assets/models/Qwen/Qwen3.5-2B")
PROMPT = "<|video_pad|>Describe what happens in this video in one sentence."


def _synthetic_video() -> tuple[np.ndarray, dict[str, object]]:
    frames = []
    for index in range(8):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = index * 24
        frame[:, :, 1] = 255 - index * 24
        frame[index * 4 : index * 4 + 16, :, :] = 255
        frames.append(frame)
    video = np.stack(frames)
    metadata = {
        "fps": 1.0,
        "duration": 8.0,
        "total_num_frames": len(video),
        "frames_indices": list(range(len(video))),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    return video, metadata


COMMON = dict(
    dtype="bfloat16",
    max_model_len=4096,
    max_num_batched_tokens=4096,
    limit_mm_per_prompt={"video": 1},
    mm_processor_kwargs={"fps": 1},
    enforce_eager=True,
    gpu_memory_utilization=0.75,
)


@wait_until_npu_memory_free()
def test_qwen35_video_without_pruning_stays_stable():
    with VllmRunner(MODEL, video_pruning_rate=0.0, **COMMON) as baseline:
        baseline_outputs = baseline.generate_greedy(
            prompts=[PROMPT],
            videos=[_synthetic_video()],
            max_tokens=32,
        )

    assert baseline_outputs and baseline_outputs[0][1]


@wait_until_npu_memory_free()
def test_qwen35_video_pruning_stays_stable():
    with VllmRunner(MODEL, video_pruning_rate=0.75, **COMMON) as pruned:
        pruned_outputs = pruned.generate_greedy(
            prompts=[PROMPT],
            videos=[_synthetic_video()],
            max_tokens=32,
        )

    assert pruned_outputs and pruned_outputs[0][1]
