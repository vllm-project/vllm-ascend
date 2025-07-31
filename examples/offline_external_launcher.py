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
# Adapted from vllm-project/vllm/examples/offline_inference/data_parallel.py

# Note: This script is designed to run with e2e test,
# please be careful to modify it.
"""
Usage:
Single node:
    Dense models:
        python examples/offline_external_launcher.py \
                --model="Qwen/Qwen2.5-0.5B-Instruct" \
                --tp-size=1 \
                --proc-per-node=2
    MOE models:
        python examples/offline_external_launcher.py \
                --model="Qwen/Qwen3-0.6B" \
                --tp-size=2 \
                --proc-per-node=2 \
                --enable-expert-parallel
              
Multi-node:
    Node 0 (assume the node has ip of 10.99.48.128):
            python examples/offline_external_launcher.py \
                    --model="Qwen/Qwen3-0.6B" \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=0 \
                    --proc-per-node=2 \
                    --enable-expert-parallel \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
    Node 1:
            python examples/offline_external_launcher.py \
                    --model="Qwen/Qwen3-0.6B" \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=1 \
                    --enable-expert-parallel \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
"""

import argparse
import contextlib
import gc
import os
from multiprocessing import Process
from time import sleep

import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (  # noqa E402
    destroy_distributed_environment, destroy_model_parallel, get_tp_group)
from vllm.utils import get_open_port

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def parse_args():

    parser = argparse.ArgumentParser(description="External launcher Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path",
    )
    parser.add_argument("--tp-size",
                        type=int,
                        default=1,
                        help="Tensor parallel size")
    parser.add_argument("--node-size",
                        type=int,
                        default=1,
                        help="Total number of nodes")
    parser.add_argument("--node-rank",
                        type=int,
                        default=0,
                        help="Rank of the current node")
    parser.add_argument("--proc-per-node",
                        type=int,
                        default=1,
                        help="Number of processes per node")
    parser.add_argument("--master-addr",
                        type=str,
                        default="",
                        help="Master node IP address")
    parser.add_argument("--master-port",
                        type=int,
                        default=0,
                        help="Master node port")
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="Enforce eager mode execution.")
    parser.add_argument("--trust-remote-code",
                        action="store_true",
                        help="Trust remote code.")
    parser.add_argument("--enable-expert-parallel",
                        action="store_true",
                        help="Enable expert parallel, used in MOE models.")
    return parser.parse_args()


def main(
    local_rank: int,
    rank: int,
    master_addr: str,
    master_port: int,
    model: str = "Qwen/Qwen3-0.6B",
    world_size: int = 4,
    tensor_parallel_size: int = 2,
    enable_expert_parallel: bool = False,
    enforce_eager: bool = False,
    trust_remote_code: bool = True,
):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="cpu:gloo,npu:hccl",
            world_size=world_size,
            rank=rank,
        )
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * 10
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=10,
    )
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
        enforce_eager=enforce_eager,
        trust_remote_code=trust_remote_code,
        distributed_executor_backend="external_launcher",
        seed=0,
    )
    tp_ranks = get_tp_group().ranks
    print(f'TP RANKS: {tp_ranks}')
    outputs = llm.generate(prompts, sampling_params)
    for i, output in enumerate(outputs):
        if i >= 5:
            # print only 5 outputs
            break
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Global rank: {rank}, Prompt: {prompt!r}, "
              f"Generated text: {generated_text!r}")

    # Give engines time to pause their processing loops before exiting.
    sleep(5)
    del llm
    cleanup_env_and_memory()


def cleanup_env_and_memory():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


if __name__ == "__main__":
    args = parse_args()

    tp_size = args.tp_size
    node_size = args.node_size
    proc_per_node = args.proc_per_node
    node_rank = args.node_rank

    if node_size == 1:
        master_addr = "127.0.0.1"
        master_port = get_open_port()
    else:
        master_addr = args.master_addr
        master_port = args.master_port

    world_size = node_size * proc_per_node

    procs = []
    for local_rank, rank in enumerate(
            range(proc_per_node * node_rank, proc_per_node * (node_rank + 1))):
        proc = Process(target=main,
                       args=(
                           local_rank,
                           rank,
                           master_addr,
                           master_port,
                           args.model,
                           world_size,
                           tp_size,
                           args.enable_expert_parallel,
                           args.enforce_eager,
                           args.trust_remote_code,
                       ))

        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join(timeout=600)
        if proc.exitcode is None:
            print(
                f"Killing process {proc.pid} that didn't stop within 30 minutes."
            )
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
