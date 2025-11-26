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

import contextlib
import gc
import math
import multiprocessing
import os
from typing import Any
from unittest.mock import patch

import pytest
import torch
from vllm.utils.network_utils import get_open_port

MODELS = [
    "Qwen/Qwen3-0.6B",
    "vllm-ascend/DeepSeek-V2-Lite-W8A8",
]


def _install_spies(counters: dict[str, Any]) -> contextlib.ExitStack:
    """Installs thread-safe spies on NPU methods to track invocation counts."""
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    def make_spy(cls, method_name, counter):
        original = getattr(cls, method_name)

        def spy(self, *args, **kwargs):
            with counter.get_lock():
                counter.value += 1
            return original(self, *args, **kwargs)

        return spy

    stack = contextlib.ExitStack()
    hooks = [
        (torch.npu.NPUGraph, "replay", counters["replay"]),
        (torch.npu.NPUGraph, "__init__", counters["capture"]),
        (NPUModelRunner, "execute_model", counters["exec_model"]),
        (NPUModelRunner, "_dummy_run", counters["dummy_run"]),
    ]

    for cls, method, counter in hooks:
        stack.enter_context(
            patch.object(cls, method, make_spy(cls, method, counter)))

    return stack


def _run_worker_process(
    rank: int,
    local_rank: int,
    world_size: int,
    master_ip: str,
    master_port: int,
    counters: dict[str, Any],
    model_path: str,
    max_tokens: int,
):
    """Main entry point for the worker process."""
    os.environ.update({
        "VLLM_DP_RANK": str(rank),
        "VLLM_DP_RANK_LOCAL": str(local_rank),
        "VLLM_DP_SIZE": str(world_size),
        "VLLM_DP_MASTER_IP": master_ip,
        "VLLM_DP_MASTER_PORT": str(master_port),
    })

    # Import vLLM only after environment setup
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment, destroy_model_parallel)

    # Apply hooks and run inference
    with _install_spies(counters):
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

        # Simple data sharding
        chunk_size = len(prompts) // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size - 1 else len(
            prompts)
        local_prompts = prompts[start_idx:end_idx]

        llm = LLM(
            model=model_path,
            quantization="ascend" if "W8A8" in model_path else None,
            enable_expert_parallel=True if "DeepSeek" in model_path else False,
            trust_remote_code=True,
        )

        # Expose model config to the main test process
        counters["hidden_layers"].value = (
            llm.llm_engine.model_config.hf_config.num_hidden_layers)

        llm.generate(local_prompts,
                     SamplingParams(max_tokens=max_tokens, temperature=0.0))

        # Explicit cleanup is mandatory in multi-process vLLM tests
        del llm

        destroy_model_parallel()
        destroy_distributed_environment()

        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()

        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()


# @patch.dict(os.environ, clear=["HCCL_OP_EXPANSION_MODE","VLLM_WORKER_MULTIPROC_METHOD"])
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [4, 36])
@patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1"})
def test_aclgraph_capture_replay_dp2(
    model: str,
    max_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Counter doesn't work in default "spawn" mode
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)

    # Shared counters for cross-process assertion
    counters = {
        "replay": multiprocessing.Value("i", 0),
        "capture": multiprocessing.Value("i", 0),
        "exec_model": multiprocessing.Value("i", 0),
        "dummy_run": multiprocessing.Value("i", 0),
        "hidden_layers": multiprocessing.Value("i", -1),
    }

    dp_size = 2
    port = get_open_port()

    # Launch workers
    workers = []
    for rank in range(dp_size):
        p = multiprocessing.Process(
            target=_run_worker_process,
            args=(rank, rank, dp_size, "127.0.0.1", port, counters, model,
                  max_tokens),
        )
        p.start()
        workers.append(p)

    # Supervision loop
    for p in workers:
        p.join(timeout=900)
        if p.exitcode != 0:
            for k in workers:
                if k.is_alive():
                    k.kill()
            raise RuntimeError(
                f"Worker {p.pid} failed with exit code {p.exitcode}")

    actual_capture = counters["capture"].value
    actual_replay = counters["replay"].value
    num_execute_model = counters["exec_model"].value
    num_dummy_run = counters["dummy_run"].value
    num_layers = counters["hidden_layers"].value

    num_acl_graphs = num_layers + 1
    num_comm_groups = sum(1 for s in [dp_size, 1]
                          if s > 1)  # dp_size=2, tp_size=1

    # Metric 1: Graph Capture (ACL Graph Construction)
    # Ref: vllm_ascend.utils.update_aclgraph_sizes
    max_batch_sizes = math.floor((1800 - num_comm_groups * 40) /
                                 num_acl_graphs / (1 + num_comm_groups * 2))

    expected_capture = max_batch_sizes * num_acl_graphs * dp_size
    assert (
        actual_capture == expected_capture
    ), f"Capture count mismatch. Expected: {expected_capture}, Got: {actual_capture}"

    # Metric 2: Model Execution (NPUModelRunner.execute_model)
    # vLLM Step Breakdown:
    # 1. First step (prefill, 1 prompt)
    # 2. Generation steps (max_tokens)
    # 3. Final step (likely EOS/idle step), no replay here
    total_steps = max_tokens + 1  # this includes the 1 and 2 above
    expected_exec_model = (total_steps + 1) * dp_size

    assert (
        num_execute_model == expected_exec_model
    ), f"Model execution count mismatch. Expected: {expected_exec_model}, Got: {num_execute_model}"

    # Metric 3: Dummy Runs (Warmup & Alignment)
    # vLLM synchronizes globally every 32 steps.
    # Ref: vllm.v1.engine.core.DPEngineCoreProc._has_global_unfinished_reqs
    aligned_steps = (total_steps + 31) // 32 * 32

    # Part A: Warmup runs (Profile run + 2 runs per captured graph)
    warmup_runs = 1 + (2 * max_batch_sizes)

    # Part B: Alignment padding (Empty runs to hit the 32-step boundary)
    padding_runs = aligned_steps - total_steps

    expected_dummy_run = (warmup_runs + padding_runs) * dp_size

    assert (
        num_dummy_run == expected_dummy_run
    ), f"Dummy run count mismatch. Expected: {expected_dummy_run}, Got: {num_dummy_run}"

    # Metric 4: Graph Replay (Inference Execution)
    # Replays happen for every aligned step across all graphs.
    expected_replay = num_acl_graphs * aligned_steps * dp_size

    assert (
        actual_replay == expected_replay
    ), f"Replay count mismatch. Expected: {expected_replay}, Got: {actual_replay}"
