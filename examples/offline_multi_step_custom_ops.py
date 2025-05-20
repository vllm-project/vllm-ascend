#
# Copyright (c) 2025 China Merchants Bank Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/examples/offline_inference/basic.py
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

from vllm import LLM, SamplingParams

import vllm_ascend.platform as pf

pf.CUSTOM_OP_ENABLED = True  # set True for custom Ops of Multi-Step.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "China is",
]

# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
# Create an LLM.
llm = LLM(
    model="Qwen/Qwen2.5-0.5B",
    block_size=128,
    max_model_len=1024,  # max length of prompt
    tensor_parallel_size=1,  # number of NPUs to be used
    max_num_seqs=26,  # max batch number
    enforce_eager=
    True,  # Force PyTorch eager execution to debug intermediate tensors (disables graph optimizations)
    trust_remote_code=
    True,  # If the model is a cuscd tom model not yet available in the HuggingFace transformers library
    num_scheduler_steps=8,
    gpu_memory_utilization=0.5)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
