#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
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

import contextlib
import gc
import math
import multiprocessing
import os
import sys
from time import sleep
from unittest.mock import patch

import pytest
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (  # noqa E402
    destroy_distributed_environment, destroy_model_parallel)

MODELS = ["Qwen/Qwen3-0.6B", "vllm-ascend/DeepSeek-V2-Lite-W8A8"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [4])
@patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1"})
def test_aclgraph_capture_replay_dp2(
    model: str,
    max_tokens: int,
) -> None:
    # HCCL_OP_EXPANSION_MODE determines how max_num_batch_sizes is computed.
    if 'VLLM_WORKER_MULTIPROC_METHOD' in os.environ:
        del os.environ["VLLM_WORKER_MULTIPROC_METHOD"]
    if 'HCCL_OP_EXPANSION_MODE' in os.environ:
        del os.environ['HCCL_OP_EXPANSION_MODE']
    dp_size = 2
    tp_size = 1
    replay_counter = multiprocessing.Value("i", 0)
    capture_counter = multiprocessing.Value("i", 0)
    num_hidden_layers_shared = multiprocessing.Value("i", -1)
    num_execute_model_shared = multiprocessing.Value("i", 0)
    dp_master_ip = "127.0.0.1"
    dp_master_port = 11011

    def dp_rank_main(global_dp_rank: int, local_dp_rank: int):
        os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
        os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

        original_replay = torch.npu.NPUGraph.replay

        def replay_wrapper(self):
            with replay_counter.get_lock():
                replay_counter.value += 1
            return original_replay(self)

        original_init = torch.npu.NPUGraph.__init__

        def init_wrapper(self, *args, **kwargs):
            with capture_counter.get_lock():
                capture_counter.value += 1
            return original_init(self, *args, **kwargs)

        with patch.object(torch.npu.NPUGraph, "replay", replay_wrapper), \
             patch.object(torch.npu.NPUGraph, "__init__", init_wrapper):
            prompts = [
                "Hello, my name is", "The president of the United States is",
                "The capital of France is", "The future of AI is"
            ]
            chunk_size = len(prompts) // dp_size
            start = global_dp_rank * chunk_size
            end = start + chunk_size if global_dp_rank < dp_size - 1 else len(
                prompts)
            my_prompts = prompts[start:end]
            sampling_params = SamplingParams(max_tokens=max_tokens,
                                             temperature=0.0)

            def trace_calls(frame, event, arg):
                if event == 'call':
                    code = frame.f_code
                    func_name = code.co_name
                    file_name = code.co_filename
                    if func_name == 'execute_dummy_batch' and 'worker_v1.py' in file_name:
                        with num_execute_model_shared.get_lock():
                            num_execute_model_shared.value += 1
                return trace_calls

            sys.settrace(trace_calls)
            if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8":
                llm = LLM(
                    model=model,
                    quantization="ascend",
                    tensor_parallel_size=tp_size,
                    trust_remote_code=True,
                )
            else:
                llm = LLM(
                    model=model,
                    tensor_parallel_size=tp_size,
                    trust_remote_code=True,
                )
            num_hidden_layers_shared.value = llm.llm_engine.model_config.hf_config.num_hidden_layers
            _ = llm.generate(my_prompts, sampling_params)
            sys.settrace(None)

            # Give engines time to pause their processing loops before exiting.
            sleep(5)
            del llm
            cleanup_env_and_memory()

    processes = []
    for local_dp_rank in range(dp_size):
        global_dp_rank = local_dp_rank
        p = multiprocessing.Process(target=dp_rank_main,
                                    args=(global_dp_rank, local_dp_rank))
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=900)
        if p.exitcode != 0:
            if p.exitcode is None:
                p.kill()
                raise RuntimeError(f"Process {p.pid} timed out")
            else:
                raise RuntimeError(
                    f"Process failed with exit code {p.exitcode}")

    actual_capture = capture_counter.value
    actual_replay = replay_counter.value
    num_hidden_layers = num_hidden_layers_shared.value
    num_execute_model = num_execute_model_shared.value

    num_acl_graphs = num_hidden_layers + 1
    num_comm_groups = sum(size > 1 for size in [
        dp_size,
        tp_size,
    ])
    max_num_batch_sizes = math.floor(
        (1800 - num_comm_groups * 40) / num_acl_graphs /
        (1 + num_comm_groups * 2))
    expected_total_capture = max_num_batch_sizes * num_acl_graphs * dp_size
    assert actual_capture == expected_total_capture, (
        f"capture count mismatch. Expected: {expected_total_capture}, Got: {actual_capture}"
    )

    num_inference_steps = max_tokens + 1  # first token + max_tokens
    expected_total_replay = num_acl_graphs * num_inference_steps * dp_size + num_execute_model * num_acl_graphs
    assert actual_replay == expected_total_replay, (
        f"Replay count mismatch. Expected: {expected_total_replay}, Got: {actual_replay}"
    )
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = 'spawn'
    sleep(600)


def cleanup_env_and_memory():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
