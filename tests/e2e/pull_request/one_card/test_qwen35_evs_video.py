#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#

import numpy as np
from modelscope import snapshot_download  # type: ignore[import-untyped]
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, qwen_prompt, wait_until_npu_memory_free


def _make_synthetic_video() -> tuple[np.ndarray, dict]:
    """Build a small deterministic video without downloading test assets."""
    palette = (
        (220, 40, 40),
        (40, 220, 40),
        (40, 40, 220),
        (220, 220, 40),
        (40, 220, 220),
        (220, 40, 220),
        (220, 140, 40),
        (140, 40, 220),
    )
    frames = np.stack([np.full((64, 64, 3), color, dtype=np.uint8) for color in palette])
    metadata = {
        "fps": 1.0,
        "total_num_frames": len(frames),
        "duration": float(len(frames)),
        "frames_indices": list(range(len(frames))),
        "do_sample_frames": False,
        "video_backend": "opencv",
    }
    return frames, metadata


@wait_until_npu_memory_free()
def test_qwen35_evs_reduces_video_prompt_tokens():
    """Verify video generation succeeds and EVS reduces prompt tokens."""
    videos = [_make_synthetic_video()]
    prompts = [
        prompt.replace("<|image_pad|>", "<|video_pad|>")
        for prompt in qwen_prompt(["Describe the colors in this video. Answer briefly."])
    ]
    model_path = snapshot_download("Qwen/Qwen3.5-2B")
    visual_token_counts = []
    nonvisual_token_ids = []
    for pruning_rate in (0.0, 0.75):
        with VllmRunner(
            model_path,
            dtype="bfloat16",
            max_model_len=4096,
            max_num_batched_tokens=4096,
            limit_mm_per_prompt={"video": 1},
            mm_processor_kwargs={"fps": 1},
            video_pruning_rate=pruning_rate,
            enforce_eager=True,
            gpu_memory_utilization=0.75,
        ) as runner:
            # RequestOutput exposes prompt_token_ids separately from generated
            # tokens, so the assertion measures pruning independently of EOS.
            hf_config = runner.model.llm_engine.vllm_config.model_config.hf_config
            video_token_id = hf_config.video_token_id
            visual_marker_ids = {
                video_token_id,
                hf_config.vision_start_token_id,
                hf_config.vision_end_token_id,
            }
            outputs = runner.model.generate(
                runner.get_inputs(prompts, videos=videos),
                sampling_params=SamplingParams(max_tokens=32, temperature=0.0),
            )

            assert len(outputs) == len(prompts)
            for output in outputs:
                assert output.outputs, "Generation should return a completion."
                assert output.outputs[0].text.strip(), "Generated output should not be empty."
                assert output.prompt_token_ids, "Video prompt should not be empty."
            visual_token_counts.append(outputs[0].prompt_token_ids.count(video_token_id))
            nonvisual_token_ids.append(
                [token_id for token_id in outputs[0].prompt_token_ids if token_id not in visual_marker_ids]
            )

    assert visual_token_counts[0] > 0, "Baseline should contain video tokens."
    # A 75% EVS rate on this eight-frame asset should remove at least half
    # the visual tokens, allowing for the first-frame retention floor.
    assert 0 < visual_token_counts[1] <= visual_token_counts[0] // 2, (
        f"Expected substantial visual-token pruning: baseline={visual_token_counts[0]}, pruned={visual_token_counts[1]}"
    )
    # Ignore visual payload and wrappers, including the replaced outer pair.
    # Text and timestamps must retain their exact token content and order.
    assert nonvisual_token_ids[1] == nonvisual_token_ids[0], "EVS changed nonvisual prompt tokens."
