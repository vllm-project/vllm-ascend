# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import os
import statistics
import time
from unittest.mock import patch

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = "Eco-Tech/GLM-5.2-w8a8"

PREFIX_TOKENS = 96 * 1024
SUFFIX_TOKENS = 4 * 1024
MAX_MODEL_LEN = PREFIX_TOKENS + SUFFIX_TOKENS + 128
NUM_SAMPLES = 3
SHARDED_MAX_REGRESSION = 0.15


def _extend_to_token_count(tokenizer, seed_text: str, token_count: int) -> list[int]:
    token_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    assert token_ids, f"seed text produced no tokens: {seed_text!r}"
    repeats = (token_count + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeats)[:token_count]


def _build_workload() -> tuple[list[int], list[list[int]]]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prefix_ids = _extend_to_token_count(
        tokenizer,
        "SFA DCP cached prefix benchmark. ",
        PREFIX_TOKENS,
    )
    suffixes = [
        _extend_to_token_count(
            tokenizer,
            f"Unique uncached suffix sample {sample_idx}. ",
            SUFFIX_TOKENS,
        )
        for sample_idx in range(NUM_SAMPLES)
    ]

    print(
        "[sfa dcp perf] token counts: "
        f"prefix={len(prefix_ids)}, "
        f"suffixes={[len(suffix) for suffix in suffixes]}, "
        f"prompt_total={len(prefix_ids) + len(suffixes[0])}"
    )
    assert len(prefix_ids) == PREFIX_TOKENS
    assert all(len(suffix) == SUFFIX_TOKENS for suffix in suffixes)
    return prefix_ids, suffixes


def _runner_kwargs(enable_sfa_dcp_sharded_indexer: bool) -> dict:
    return {
        "quantization": "ascend",
        "max_model_len": MAX_MODEL_LEN,
        "tensor_parallel_size": 16,
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 16,
        "cp_kv_cache_interleave_size": 128,
        "block_size": 128,
        "enable_prefix_caching": True,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 16384,
        "enforce_eager": True,
        "enable_expert_parallel": True,
        "gpu_memory_utilization": 0.8,
        "additional_config": {
            "enable_dsa_cp": False,
            "enable_sfa_dcp_sharded_indexer": enable_sfa_dcp_sharded_indexer,
            "enable_sparse_sfa_c8": False,
            "enable_sparse_li_c8": False,
            "enable_cpu_binding": True,
            "recompute_scheduler_enable": False,
            "ascend_compilation_config": {
                "enable_npugraph_ex": False,
            },
        },
    }


def _measure_median_ttft(
    *,
    name: str,
    prefix_ids: list[int],
    suffixes: list[list[int]],
    enable_sfa_dcp_sharded_indexer: bool,
) -> float:
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    ttfts: list[float] = []
    with VllmRunner(MODEL, **_runner_kwargs(enable_sfa_dcp_sharded_indexer)) as runner:
        runner.generate([prefix_ids], sampling_params, use_tqdm=False)

        for sample_idx, suffix_ids in enumerate(suffixes):
            prompt_ids = prefix_ids + suffix_ids
            start = time.perf_counter()
            runner.generate([prompt_ids], sampling_params, use_tqdm=False)
            ttfts.append(time.perf_counter() - start)
            print(
                f"[sfa dcp perf] {name} sample={sample_idx} "
                f"prompt_tokens={len(prompt_ids)} ttft={ttfts[-1]:.3f}s"
            )

    median_ttft = statistics.median(ttfts)
    print(
        f"[sfa dcp perf] {name} TTFTs="
        f"{[round(ttft, 3) for ttft in ttfts]} median={median_ttft:.3f}s"
    )
    return median_ttft


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="sfa_dcp_sharded_indexer",
    parallel="TP,DCP",
    deploy="single_node",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_BUFFSIZE": "768",
        "HCCL_TRANSFER_TIMEOUT": "600",
        "HCCL_EXEC_TIMEOUT": "3600",
        "HCCL_CONNECT_TIMEOUT": "3600",
        "ASCEND_A3_ENABLE": "1",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8, max_wait_seconds=300)
def test_sfa_dcp_sharded_indexer_ttft_regression() -> None:
    prefix_ids, suffixes = _build_workload()

    replicated_ttft = _measure_median_ttft(
        name="replicated",
        prefix_ids=prefix_ids,
        suffixes=suffixes,
        enable_sfa_dcp_sharded_indexer=False,
    )
    sharded_ttft = _measure_median_ttft(
        name="sharded",
        prefix_ids=prefix_ids,
        suffixes=suffixes,
        enable_sfa_dcp_sharded_indexer=True,
    )

    allowed_ttft = replicated_ttft * (1.0 + SHARDED_MAX_REGRESSION)
    print(
        "[sfa dcp perf] median comparison: "
        f"replicated={replicated_ttft:.3f}s, sharded={sharded_ttft:.3f}s, "
        f"allowed={allowed_ttft:.3f}s"
    )
    assert sharded_ttft <= allowed_ttft, (
        "SFA DCP sharded indexer TTFT regression: "
        f"sharded median {sharded_ttft:.3f}s exceeds replicated median "
        f"{replicated_ttft:.3f}s by more than {SHARDED_MAX_REGRESSION:.0%}"
    )
