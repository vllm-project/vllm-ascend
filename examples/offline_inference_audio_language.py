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
# Adapted from vllm-project/vllm/examples/offline_inference/audio_language.py
#
"""
This example shows how to use vLLM for running offline inference 
with the correct prompt format on audio language models.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""

import os

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

audio_assets = [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]
question_per_audio_count = {
    1: "What is recited in the audio?",
    2: "What sport and what nursery rhyme are referenced?"
}


def prepare_inputs(audio_count: int):
    audio_in_prompt = "".join([
        f"Audio {idx+1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
        for idx in range(audio_count)
    ])
    question = question_per_audio_count[audio_count]
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n"
              f"{audio_in_prompt}{question}<|im_end|>\n"
              "<|im_start|>assistant\n")

    mm_data = {
        "audio":
        [asset.audio_and_sample_rate for asset in audio_assets[:audio_count]]
    }

    # Merge text prompt and audio data into inputs
    inputs = {"prompt": prompt, "multi_modal_data": mm_data}
    return inputs


def main(audio_count: int):
    # NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
    # lower-end GPUs.
    # Unless specified, these settings have been tested to work on a single L4.
    # `limit_mm_per_prompt`: the max num items for each modality per prompt.
    llm = LLM(model="Qwen/Qwen2-Audio-7B-Instruct",
              max_model_len=4096,
              max_num_seqs=5,
              limit_mm_per_prompt={"audio": audio_count},
              enforce_eager=True)

    inputs = prepare_inputs(audio_count)

    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=64,
                                     stop_token_ids=None)

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    audio_count = 2
    main(audio_count)
