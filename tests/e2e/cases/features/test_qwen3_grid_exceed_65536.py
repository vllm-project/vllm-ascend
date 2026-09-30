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

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


def test_qwen3_large_mixed_batch_exceeding_triton_grid_limit() -> None:
    """Generate a 512-request mixed batch whose block grid exceeds 65535."""
    short_prompt = "Hello, tell me a joke"
    long_prompt = "word " * 35000
    prompts = [short_prompt] * 508 + [long_prompt] * 4

    sampling_params = SamplingParams(
        max_tokens=8,
        temperature=0.5,
        repetition_penalty=1.2,
    )

    with VllmRunner(
        model_name="Qwen/Qwen3-0.6B",
        max_model_len=40960,
        max_num_seqs=512,
        enforce_eager=True,
    ) as runner:
        outputs = runner.model.generate(prompts, sampling_params)

    print(f"Generation completed: output_count={len(outputs)}", flush=True)

    assert len(outputs) == 512
    assert all(output.outputs and output.outputs[0].token_ids for output in outputs)
