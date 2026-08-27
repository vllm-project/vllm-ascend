#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""
Example: Access request-level metrics from vLLM outputs.

By default, vLLM disables log stats (disable_log_stats=True), which causes
output.metrics to be None. To populate metrics such as first_token_time,
finished_time, etc., you must explicitly set disable_log_stats=False when
creating the LLM instance.

See: https://github.com/vllm-project/vllm-ascend/issues/5027
"""

# isort: skip_file
import os

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams


def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=100, temperature=0.0)

    # IMPORTANT: Set disable_log_stats=False to enable output.metrics.
    # Without this, output.metrics will be None.
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", disable_log_stats=False)

    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        metrics = output.metrics

        print(f"Prompt: {prompt!r}")
        print(f"  Generated text: {generated_text!r}")
        if metrics is not None:
            print(f"  Arrival time: {metrics.arrival_time}")
            print(f"  First scheduled time: {metrics.first_scheduled_time}")
            print(f"  First token time: {metrics.first_token_time}")
            print(f"  Finished time: {metrics.finished_time}")
        else:
            print("  Metrics: None (set disable_log_stats=False to enable)")
        print()


if __name__ == "__main__":
    main()
