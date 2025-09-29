# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import torch_npu

from vllm import LLM, SamplingParams

import os


def run_offline_prompts():
    prompts = [
        "Hello, my name is",
        "The future of AI is",
        "This is a",
        "It is known that",
        "How is the",
        "If we want to",
        "He does this",
        "Maybe it is",
    ] * 10
    import random
    random.shuffle(prompts)

    sampling_params = SamplingParams(max_tokens=100, temperature=0.0, top_p=0.95)  # Use temp 0!
    llm = LLM(model="/home/kc/DeepSeek-V2-Lite", 
              tensor_parallel_size=2, 
              trust_remote_code=True, 
              enforce_eager=True, 
              max_model_len=1024, 
              gpu_memory_utilization=0.9)
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

    os.environ["RANDOM_MODE"] = "1"  # replay inputs and outputs from the cache
    os.environ["MOCK_CAPTURE_DIR"] = "/home/kc/capture/"
    os.environ["MOCK_CAPTURE_FILE"] = ".mock_cache"
    os.environ["MOCK_CAPTURE_FILE_LOCK"] = ".lock"
    
    run_offline_prompts()