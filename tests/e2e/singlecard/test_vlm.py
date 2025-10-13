#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/test_offline_inference.py`.
"""
import os

from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset

from tests.e2e.conftest import VllmRunner

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"


def test_multimodal_vl(prompt_template):
    image = ImageAsset("cherry_blossom") \
        .pil_image.convert("RGB")
    img_questions = [
        "What is the content of this image?",
        "Describe the content of this image in detail.",
        "What's in the image?",
        "Where is this image taken?",
    ]
    images = [image] * len(img_questions)
    prompts = prompt_template(img_questions)
    with VllmRunner("Qwen/Qwen2.5-VL-3B-Instruct",
                    max_model_len=4096,
                    mm_processor_kwargs={
                        "min_pixels": 28 * 28,
                        "max_pixels": 1280 * 28 * 28,
                        "fps": 1,
                    },
                    enforce_eager=True) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts=prompts,
                                             images=images,
                                             max_tokens=64)
        assert len(outputs) == len(prompts)
        for _, output_str in outputs:
            assert output_str, "Generated output should not be empty."


def test_multimodal_audio():
    audio_prompt = "".join([
        f"Audio {idx+1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
        for idx in range(2)
    ])
    question = "What sport and what nursery rhyme are referenced?"
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n"
              f"{audio_prompt}{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    mm_data = {
        "audio": [
            asset.audio_and_sample_rate for asset in
            [AudioAsset("mary_had_lamb"),
             AudioAsset("winning_call")]
        ]
    }
    inputs = {"prompt": prompt, "multi_modal_data": mm_data}

    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=10,
                                     stop_token_ids=None)

    with VllmRunner("Qwen/Qwen2-Audio-7B-Instruct",
                    max_model_len=4096,
                    max_num_seqs=5,
                    dtype="bfloat16",
                    limit_mm_per_prompt={"audio": 2},
                    gpu_memory_utilization=0.9) as runner:
        outputs = runner.generate(inputs, sampling_params=sampling_params)

        assert outputs is not None, "Generated outputs should not be None."
        assert len(outputs) > 0, "Generated outputs should not be empty."
