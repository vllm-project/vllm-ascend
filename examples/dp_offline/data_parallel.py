#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/examples/offline_inference/data_parallel.py
# SPDX-License-Identifier: Apache-2.0
# usage:
# python examples/offline_inference_data_parallel.py
# we need to have a launcher to create multiple data parallel
# ranks. And each rank will create a vLLM instance to process its own prompts.

import os
import gc


def main():
    dp_rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dp_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    tp_size = 4
    etp_size = 2

    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = master_addr
    os.environ["VLLM_DP_MASTER_PORT"] = master_port
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(
        str(i) for i in range(local_rank * tp_size , (local_rank + 1) * tp_size))

    import torch
    import torch_npu #noqa
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * 4


    promts_per_rank = len(prompts) // dp_size
    start = dp_rank * promts_per_rank
    end = start + promts_per_rank
    prompts = prompts[start:end]
    if len(prompts) == 0:
        prompts = ["Placeholder"]
    print(f"DP rank {dp_rank} needs to process {len(prompts)} prompts")


    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens= 4,
                                     min_tokens = 4)
    # Create an LLM.
    llm = LLM(model="deepseek-ai/DeepSeek-V2-Lite-Chat",
              tensor_parallel_size=tp_size,
              trust_remote_code=True,
              expert_tensor_parallel_size=etp_size,
              max_model_len=4096,
              enforce_eager=True)

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"DP rank {dp_rank}, Prompt: {prompt!r}, "
              f"Generated text: {generated_text!r}")

    del llm
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()


if __name__ == "__main__":
    main()
