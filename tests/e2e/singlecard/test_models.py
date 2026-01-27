#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/entrypoints/llm/test_guided_generate.py
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
#

import os

import pytest
from modelscope import snapshot_download  # type: ignore
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset

from tests.e2e.conftest import VllmRunner

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Note: MiniCPM-2B is a MHA model, MiniCPM4-0.5B is a GQA model
MINICPM_MODELS = [
    "openbmb/MiniCPM-2B-sft-bf16",
    "OpenBMB/MiniCPM4-0.5B",
]

WHISPER_MODELS = [
    "openai-mirror/whisper-large-v3-turbo",
]


@pytest.mark.parametrize("model", MINICPM_MODELS)
def test_minicpm(model) -> None:
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5

    with VllmRunner(snapshot_download(model),
                    max_model_len=512,
                    gpu_memory_utilization=0.7) as runner:
        runner.generate_greedy(example_prompts, max_tokens)


@pytest.mark.parametrize("model", WHISPER_MODELS)
def test_whisper(model) -> None:
    prompts = ["<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"]
    audios = [AudioAsset("mary_had_lamb").audio_and_sample_rate]

    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=10,
                                     stop_token_ids=None)

    with VllmRunner(snapshot_download(model),
                    max_model_len=448,
                    max_num_seqs=5,
                    dtype="bfloat16",
                    block_size=128,
                    gpu_memory_utilization=0.9) as runner:
        outputs = runner.generate(prompts=prompts,
                                  audios=audios,
                                  sampling_params=sampling_params)

    assert outputs is not None, "Generated outputs should not be None."
    assert len(outputs) > 0, "Generated outputs should not be empty."
