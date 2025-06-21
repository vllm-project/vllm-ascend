#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/examples/offline_inference/data_parallel.py
# SPDX-License-Identifier: Apache-2.0
# usage:
# python examples/offline_inference_data_parallel.py
# we need to have a launcher to create multiple data parallel
# ranks. And each rank will create a vLLM instance to process its own prompts.

import gc
import os
from time import sleep, time


def load_prompt():
    import json
    # x = "/home/c00845552/downloads/3.5K_prompts.json"
    # x = "/home/c00845552/downloads/dsv3_prompts_rand_3584.json"
    x = "/home/d00876830/VLLM/prompts-bs16-seq3584.json"
    with open(x, "r", encoding="utf8") as f:
        prompts = json.loads(f.read())[:16]
    return prompts

def main():
    dp_rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dp_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    tp_size = 16
    etp_size = 1

    os.environ["VLLM_DP_RANK_LOCAL"] = str(0)
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = master_addr
    os.environ["VLLM_DP_MASTER_PORT"] = master_port
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(
        str(i)
        for i in range(local_rank * tp_size, (local_rank + 1) * tp_size))
    os.environ["VLLM_TORCH_PROFILER_DIR"] = "/home/d00876830/VLLM/vllm_profile/dskv3_dbo_base_bs16"
    os.environ['VLLM_USE_V1'] = '1'
    os.environ['ENABLE_MOE_ALLTOALLV'] = '1'
    os.environ['VLLM_ASCEND_ENABLE_DBO'] = '1'
    
    print(os.environ["ASCEND_RT_VISIBLE_DEVICES"])

    import torch
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment, destroy_model_parallel)

    # prompts = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ] * 4
    prompts = load_prompt()


    promts_per_rank = len(prompts) // dp_size
    start = dp_rank * promts_per_rank
    end = start + promts_per_rank
    prompts = prompts[start:end]
    if len(prompts) == 0:
        prompts = ["Placeholder"]
    print(f"DP rank {dp_rank} needs to process {len(prompts)} prompts")
    num_seqs = len(prompts)

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=10, top_k=-1, min_p=0.0, detokenize=True, logprobs=1, n=16
    )
    # Create an LLM.
    llm = LLM(model="/home/d00876830/VLLM/DeepSeek-V3-W8A8",
              tensor_parallel_size=tp_size,
              trust_remote_code=True,
              enforce_eager=True,
              enable_expert_parallel=True,
              max_model_len=8192*4,
              max_num_seqs=256,
              max_num_batched_tokens=12 * 3585,
              load_format='dummy',
              gpu_memory_utilization=0.8,
              enable_chunked_prefill=False,
              enable_prefix_caching=False,
              additional_config={
                  'expert_tensor_parallel_size': 1,
                  'enable_graph_mode': False,
                  'torchair_graph_config': {
                      'enabled': False,
                  },
                  'ascend_scheduler_config': {
                      'enabled': True,
                      'chunked_prefill_enabled': False
                  },
                  'refresh': True
              }
              )

    torch.npu.synchronize()
    start = time()
    outputs = llm.generate(prompts, sampling_params)
    torch.npu.synchronize()
    end = time()
    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"DP rank {dp_rank}, Prompt: {prompt!r}, "
    #           f"Generated text: {generated_text!r}")

    del llm
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()


if __name__ == "__main__":
    main()
