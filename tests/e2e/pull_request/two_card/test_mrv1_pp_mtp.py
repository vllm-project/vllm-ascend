#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""MRV1 pipeline parallel + MTP speculative decoding accuracy (two NPUs).

Regression tests for the silent corruption tracked in vllm-ascend#17149: with
the V1 model runner and ``pipeline_parallel_size > 1`` the scheduler could
consume a draft token against stale ``Request`` state, so PP2+MTP diverged from
PP1 after a few tokens even though both runs reported speculative metrics.

The same comparison is repeated for a long prompt (cross-stage KV cache) and
for chunked prefill (micro-batch scheduling).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.model_utils import check_outputs_equal

MODEL = os.environ.get("QWEN3_5_4B_MODEL_PATH", "Qwen/Qwen3.5-4B")
PROMPTS = ["Hello, my name is", "The future of AI is"]
LONG_PROMPT = (
    "The history of computing is a story of abstraction layers: from vacuum tubes to transistors, "
    "from assembly to compilers, and from single machines to clusters that span continents. "
) * 30
MAX_TOKENS = 16


def _mtp_config() -> dict:
    return {"method": "mtp", "num_speculative_tokens": 3, "enforce_eager": True}


def _spec_metrics(runner: VllmRunner) -> tuple[int, int]:
    num_drafts = 0
    num_accepted = 0
    for metric in runner.model.get_metrics():
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            num_accepted += sum(metric.values)
    return num_drafts, num_accepted


def _run(
    pp_size: int,
    *,
    prompts: list[str] | None = None,
    max_model_len: int = 1024,
    max_num_batched_tokens: int = 2048,
) -> tuple[list[tuple[list[int], str]], tuple[int, int]]:
    with VllmRunner(
        MODEL,
        max_model_len=max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=max_num_batched_tokens,
        tensor_parallel_size=1,
        pipeline_parallel_size=pp_size,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        speculative_config=_mtp_config(),
        hf_overrides={"language_model_only": True},
    ) as runner:
        params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)
        outputs = [(token_ids[0], text[0]) for token_ids, text in runner.generate(prompts or PROMPTS, params)]
        return outputs, _spec_metrics(runner)


def _assert_pp_matches_pp1(pp1_outputs, pp1_drafts, pp2_outputs, pp2_drafts, pp2_accepted) -> None:
    check_outputs_equal(
        outputs_0_lst=pp1_outputs,
        outputs_1_lst=pp2_outputs,
        name_0="PP1+MTP3",
        name_1="PP2+MTP3",
    )
    assert pp1_drafts > 0, "PP1 did not generate draft tokens"
    assert pp2_drafts > 0, "PP2 did not generate draft tokens (speculation silently disabled)"
    assert pp2_accepted > 0, "PP2 did not accept any draft tokens"


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="spec_decode",
    parallel="PP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mrv1_pp_mtp_matches_pp1() -> None:
    pp1_outputs, (pp1_drafts, _) = _run(1)
    pp2_outputs, (pp2_drafts, pp2_accepted) = _run(2)
    _assert_pp_matches_pp1(pp1_outputs, pp1_drafts, pp2_outputs, pp2_drafts, pp2_accepted)


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="spec_decode,long_sequence",
    parallel="PP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mrv1_pp_mtp_long_sequence() -> None:
    pp1_outputs, (pp1_drafts, _) = _run(1, prompts=[LONG_PROMPT], max_model_len=4096, max_num_batched_tokens=4096)
    pp2_outputs, (pp2_drafts, pp2_accepted) = _run(
        2, prompts=[LONG_PROMPT], max_model_len=4096, max_num_batched_tokens=4096
    )
    _assert_pp_matches_pp1(pp1_outputs, pp1_drafts, pp2_outputs, pp2_drafts, pp2_accepted)


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="spec_decode,chunked_prefill",
    parallel="PP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mrv1_pp_mtp_micro_batch() -> None:
    pp1_outputs, (pp1_drafts, _) = _run(1, prompts=[LONG_PROMPT], max_model_len=4096, max_num_batched_tokens=512)
    pp2_outputs, (pp2_drafts, pp2_accepted) = _run(
        2, prompts=[LONG_PROMPT], max_model_len=4096, max_num_batched_tokens=512
    )
    _assert_pp_matches_pp1(pp1_outputs, pp1_drafts, pp2_outputs, pp2_drafts, pp2_accepted)
