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

from vllm.assets.video import VideoAsset

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = os.environ.get("QWEN35_EVS_MODEL", "/workspace/shared_assets/models/Qwen/Qwen3.5-2B")
VIDEO = VideoAsset("baby_reading", num_frames=8)
PROMPT = "Describe what happens in this video in one sentence."


@wait_until_npu_memory_free()
def test_qwen35_video_pruning_reduces_visual_tokens_and_stays_stable():
    common = dict(
        dtype="bfloat16",
        max_model_len=4096,
        max_num_batched_tokens=4096,
        limit_mm_per_prompt={"video": 1},
        mm_processor_kwargs={"fps": 1},
        enforce_eager=True,
    )
    with VllmRunner(MODEL, video_pruning_rate=0.0, **common) as baseline:
        baseline_outputs = baseline.generate_greedy(
            prompts=[PROMPT],
            videos=[VIDEO.np_ndarrays],
            max_tokens=32,
        )

    with VllmRunner(MODEL, video_pruning_rate=0.75, **common) as pruned:
        pruned_outputs = pruned.generate_greedy(
            prompts=[PROMPT],
            videos=[VIDEO.np_ndarrays],
            max_tokens=32,
        )

    assert baseline_outputs and pruned_outputs
    assert baseline_outputs[0][1]
    assert pruned_outputs[0][1]
