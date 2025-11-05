import contextlib
import gc
import multiprocessing
import os
import time
import types
from multiprocessing import Queue
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (  # noqa E402
    destroy_distributed_environment, destroy_model_parallel)
from vllm.utils import get_open_port

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.11.0"):
    from vllm.utils import get_open_port
else:
    from vllm.utils.network_utils import get_open_port


def cleanup_env_and_memory():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def run_with_dp(model, dp_size, local_dp_rank, global_dp_rank, dp_master_ip,
                dp_master_port, tp_size, enable_expert_parallel, enforce_eager,
                trust_remote_code, lmhead_tp_size, result_queue):
    # DP only support on V1 engine
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    # Distribute prompts into even groups.
    def start(rank):
        return rank * floor + min(rank, remainder)

    # Record the start of the current process
    current_start = start(global_dp_rank)

    prompts = prompts[start(global_dp_rank): start(global_dp_rank + 1)]
    if len(prompts) == 0:
        # if any rank has no prompts to process,
        # we need to set a placeholder prompt
        prompts = ["Placeholder"]
    print(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0,
                                     max_tokens=256,
                                     ignore_eos=False)

    # Create an LLM.
    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        enforce_eager=enforce_eager,
        enable_expert_parallel=enable_expert_parallel,
        trust_remote_code=trust_remote_code,
        max_model_len=2000,
        max_num_seqs=256,
        speculative_config={
            "num_speculative_tokens": 1,
            "method": "deepseek_mtp"
        },
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        additional_config={"lmhead_tensor_parallel_size": lmhead_tp_size})

    req_outputs = llm.generate(prompts, sampling_params)

    outputs: List[Tuple[List[List[int]], List[str]]] = []
    for req_output in req_outputs:
        prompt_str = req_output.prompt
        prompt_ids = req_output.prompt_token_ids
        req_sample_output_ids: List[List[int]] = []
        req_sample_output_strs: List[str] = []
        for sample in req_output.outputs:
            output_str = sample.text
            output_ids = list(sample.token_ids)
            req_sample_output_ids.append(prompt_ids + output_ids)
            req_sample_output_strs.append(prompt_str + output_str)
        outputs.append((req_sample_output_ids, req_sample_output_strs))

    result_queue.put({
        "global_dp_rank": global_dp_rank,
        "start": current_start,
        "results": outputs
    })

    # Give engines time to pause their processing loops before exiting.
    time.sleep(5)
    del llm
    cleanup_env_and_memory()


def run_with_lmhead_tp(args, lmhead_tp_size, result_queue):
    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    procs = []
    dp_result_queue: Queue[Dict[str, Any]] = multiprocessing.Queue()
    for local_dp_rank, global_dp_rank in enumerate(
            range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)):
        proc = multiprocessing.Process(
            target=run_with_dp,
            args=(args.model, dp_size, local_dp_rank, global_dp_rank,
                  dp_master_ip, dp_master_port, tp_size,
                  args.enable_expert_parallel, args.enforce_eager,
                  args.trust_remote_code, lmhead_tp_size, dp_result_queue))
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join(timeout=900)
        if proc.exitcode is None:
            print(
                f"Killing process {proc.pid} that didn't stop within 15 minutes."
            )
            proc.kill()
            exit_code = 1
        elif proc.exitcode != 0:
            exit_code = proc.exitcode

    if exit_code != 0:
        exit(exit_code)

    # Collect the results of all processes and record the corresponding start positions
    dp_results = []
    for _ in range(dp_size):
        res = dp_result_queue.get()
        dp_results.append((res["start"], res["results"]))

    # Sort in ascending order by start
    dp_results.sort(key=lambda x: x[0])

    # Concatenate the results in the sorted order
    final_results: List[Tuple[List[List[int]], List[str]]] = []
    for start, results in dp_results:
        final_results.extend(results)

    result_queue.put({
        "lmhead_tp_size": lmhead_tp_size,
        "results": final_results
    })
    exit(exit_code)


def check_precision(ref_outputs: List[Tuple],
                    spec_outputs: List[Tuple],
                    threshold: float = 0.66):
    matches = 0
    misses = 0
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        ref_token_ids = ref_output[0][0]
        spec_token_ids = spec_output[0][0]
        if ref_token_ids == spec_token_ids[:len(ref_token_ids)]:
            matches += 1
        else:
            misses += 1
            print(f"ref_output: {ref_output[1][0]}")
            print(f"spec_output: {spec_output[1][0]}")

    # Heuristic: expect at least 66% of the prompts to match exactly
    # Upon failure, inspect the outputs to check for inaccuracy.
    assert matches > int(threshold * len(ref_outputs))


@pytest.mark.parametrize("model", ["wemaster/deepseek_mtp_main_random_bf16"])
@patch.dict(
    os.environ, {
        "ASCEND_RT_VISIBLE_DEVICES": "0,1",
        "VLLM_USE_MODELSCOPE": "True",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "512",
    })
def test_lmhead_tp_mtp_dp2(model):
    args = types.SimpleNamespace(model=model,
                                 dp_size=2,
                                 tp_size=1,
                                 node_size=1,
                                 node_rank=0,
                                 enable_expert_parallel=True,
                                 master_addr="127.0.0.1",
                                 master_port=8055,
                                 enforce_eager=False,
                                 trust_remote_code=False)

    # When lmhead_tp_size=tp_size,lmhead_tp is not enabled
    lmhead_tp_sizes = [1, 2]

    global_result_queue: Queue[Dict[str, Any]] = multiprocessing.Queue()
    for lmhead_tp_size in lmhead_tp_sizes:
        proc = multiprocessing.Process(target=run_with_lmhead_tp,
                                       args=(args, lmhead_tp_size,
                                             global_result_queue))
        proc.start()
        print(f"Start the process with lmhead_tp_size={lmhead_tp_size}")

        proc.join()
        if proc.exitcode != 0:
            pytest.fail(
                f"The process with lmhead_tp_size={lmhead_tp_size} abnormally exited"
            )

    tp1_res, tp2_res = [global_result_queue.get() for _ in range(2)]
    check_precision(ref_outputs=tp1_res["results"],
                    spec_outputs=tp2_res["results"])
    