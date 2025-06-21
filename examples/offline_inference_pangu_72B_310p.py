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
# Adapted from vllm-project/vllm/examples/offline_inference/basic.py
#
import gc
import os
import torch

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()

os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


if __name__ == "__main__":
    # Update the model_path
    model_path="/home/xxx/pangu_model/pangu-pro-moe-model"

    prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(min_tokens=8, max_tokens=8, temperature=0.0)
    llm = LLM(model=model_path,
            tensor_parallel_size=8,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=0.5,
            max_num_seqs=4,
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=1024,
            disable_custom_all_reduce=True, # IMPORTANT cause 310p needed custom ops
            enable_expert_parallel=True,

            dtype="float16", # IMPORTANT cause some ATB ops cannot support bf16 on 310P
            compilation_config={"custom_ops":["+rms_norm", "+rotary_embedding"]}, # IMPORTANT cause 310p needed custom ops

            additional_config = {
                'ascend_scheduler_config': {
                    'enabled': True,
                    'enable_chunked_prefill' : False,
                    'chunked_prefill_enabled': False
                }
            }
    )

    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    del llm
    clean_up()