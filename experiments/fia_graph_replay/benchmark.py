# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare native and shared-length FIA decode on the same upstream revision."""

import argparse
import json
from collections import Counter
from functools import wraps
from time import perf_counter

import torch
import torch_npu  # noqa: F401
import vllm
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.config.compilation import PassConfig

from vllm_ascend.attention import attention_v1
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--expected-updates", type=int, choices=(0, 28))
    args = parser.parse_args()
    print("VLLM", vllm.__version__, "ASCEND", attention_v1.__file__, flush=True)
    counts = Counter()
    steps = []

    def count(method, name):
        @wraps(method)
        def counted(*args, **kwargs):
            counts[name] += 1
            return method(*args, **kwargs)

        return counted

    check = args.expected_updates is not None
    if check:
        for name in ("graph_task_update_begin", "graph_task_update_end"):
            setattr(torch.npu, name, count(getattr(torch.npu, name), name))
        torch.npu.ExternalEvent.record = count(torch.npu.ExternalEvent.record, "event_record")
        replay = ModelAclGraphManager._updatable_graph_replay

        def replay_with_counts(manager, desc, metadata):
            before = counts.copy()
            result = replay(manager, desc, metadata)
            steps.append({"size": desc.num_tokens, **(counts - before)})
            return result

        ModelAclGraphManager._updatable_graph_replay = replay_with_counts

    llm = LLM(
        args.model,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=384 * 1024**2,
        gpu_memory_utilization=0.12,
        enable_prefix_caching=False,
        disable_log_stats=False,
        additional_config={"enable_cpu_binding": False, "ascend_compilation_config": {"fuse_norm_quant": False}},
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            pass_config=PassConfig(fuse_act_quant=False, fuse_norm_quant=False),
        ),
    )
    seed_tokens = [785, 6722, 315, 9625, 374]
    lengths_by_batch = {
        1: [120],
        3: [125, 252, 508],
        8: [31, 112, 127, 125, 249, 508, 415, 369],
    }
    new_tokens = 24 if check else 64
    sampling = SamplingParams(max_tokens=new_tokens, temperature=0, ignore_eos=True, detokenize=False)
    for batch, lengths in lengths_by_batch.items():
        prompts = [
            {"prompt_token_ids": (seed_tokens * ((length + len(seed_tokens) - 1) // len(seed_tokens)))[:length]}
            for length in lengths
        ]
        llm.generate(prompts, sampling, use_tqdm=False)
        for trial in range(args.trials):
            steps.clear()
            start = perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            elapsed_ms = (perf_counter() - start) * 1000
            tpot_ms = [
                (output.metrics.last_token_ts - output.metrics.first_token_ts) * 1000 / (new_tokens - 1)
                for output in outputs
            ]
            result = {"batch": batch, "trial": trial, "e2e_ms": elapsed_ms, "tpot_ms": tpot_ms}
            if check:
                result["tokens"] = [output.outputs[0].token_ids for output in outputs]
                histogram = Counter(str(sorted(step.items())) for step in steps)
                result["replay_steps"] = histogram
                assert steps, f"No FULL decode replay for batch {batch}"
                for step in steps:
                    for name in ("graph_task_update_begin", "graph_task_update_end", "event_record"):
                        assert step.get(name, 0) == args.expected_updates, step
            print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
