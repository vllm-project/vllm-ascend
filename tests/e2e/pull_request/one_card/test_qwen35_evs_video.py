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

import huggingface_hub
import regex as re
from huggingface_hub import snapshot_download as hf_snapshot_download
from vllm import SamplingParams
from vllm.assets.video import VideoAsset

from tests.e2e.conftest import VllmRunner, qwen_prompt, wait_until_npu_memory_free


@wait_until_npu_memory_free()
def test_qwen35_evs_reduces_video_prompt_tokens():
    """Verify video generation succeeds and EVS reduces prompt tokens."""
    video = VideoAsset("baby_reading", num_frames=8)
    videos = [(video.np_ndarrays, video.metadata)]
    prompts = [
        prompt.replace("<|image_pad|>", "<|video_pad|>")
        for prompt in qwen_prompt(["What object is the baby holding and reading? Answer briefly."])
    ]
    model_path = hf_snapshot_download(
        "Qwen/Qwen3.5-2B",
        local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
    )
    prompt_token_counts = []
    visual_token_counts = []
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
            video_token_id = runner.model.llm_engine.vllm_config.model_config.hf_config.video_token_id
            outputs = runner.model.generate(
                runner.get_inputs(prompts, videos=videos),
                sampling_params=SamplingParams(max_tokens=32, temperature=0.0),
            )

            assert len(outputs) == len(prompts)
            for output in outputs:
                assert output.outputs, "Generation should return a completion."
                assert output.outputs[0].text.strip(), "Generated output should not be empty."
                assert output.prompt_token_ids, "Video prompt should not be empty."
                # This standard asset shows a baby holding and reading a book.
                assert re.search(r"\bbook\b", output.outputs[0].text, flags=re.IGNORECASE), (
                    f"Expected the visible book to be identified, got: {output.outputs[0].text!r}"
                )
            prompt_token_counts.append(len(outputs[0].prompt_token_ids))
            visual_token_counts.append(outputs[0].prompt_token_ids.count(video_token_id))

    assert prompt_token_counts[1] < prompt_token_counts[0], (
        f"EVS should reduce prompt tokens: baseline={prompt_token_counts[0]}, pruned={prompt_token_counts[1]}"
    )

    assert visual_token_counts[0] > 0, "Baseline should contain video tokens."
    # A 75% EVS rate on this eight-frame asset should remove at least half
    # the visual tokens, allowing for the first-frame retention floor.
    assert 0 < visual_token_counts[1] <= visual_token_counts[0] // 2, (
        f"Expected substantial visual-token pruning: baseline={visual_token_counts[0]}, pruned={visual_token_counts[1]}"
    )
    # The EVS processor also replaces the two outer video wrapper markers.
    # All remaining nonvisual prompt tokens must be preserved.
    nonvisual_reduction = (prompt_token_counts[0] - prompt_token_counts[1]) - (
        visual_token_counts[0] - visual_token_counts[1]
    )
    assert 0 <= nonvisual_reduction <= 2, "Unexpected removal of nonvisual prompt tokens."
