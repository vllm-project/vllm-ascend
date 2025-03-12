#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import os
import gc
import multiprocessing


def run_model(dp_rank,dp_size,tp_size):
    start_id = dp_rank * tp_size
    end_id = (dp_rank + 1) * tp_size
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(start_id, end_id))

    import torch
    import torch_npu
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Now I am gonna try somthing else"
    ] * 3
    promts_per_rank = len(prompts) // dp_size
    start = dp_rank * promts_per_rank
    end = start + promts_per_rank
    prompts = prompts[start:end]
    print(f"DP rank {dp_rank} needs to process {len(prompts)} prompts")

    sampling_params = SamplingParams(temperature=0, max_tokens=16, min_tokens=16)
    llm = LLM(model="deepseek-ai/DeepSeek-V2-Lite",
              tensor_parallel_size=tp_size,
              distributed_executor_backend="mp",
              max_model_len=4096,
              trust_remote_code=True,
              enforce_eager=False,
              compilation_config=1,
              )

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    del llm
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()



def main():

    multiprocessing.set_start_method('spawn')
    dp_size = 2
    tp_size = 4
    processes = []
    for i in range(dp_size):
        proc = multiprocessing.Process(target=run_model, args=(i,dp_size,tp_size))
        processes.append(proc)

    for p in processes:
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
