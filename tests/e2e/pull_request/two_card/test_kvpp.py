#
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
#
"""Eager KV layer parallelism on two NPUs with DeepSeek-V2-Lite-W8A8.

KVPP size equals TP, so this needs at least two cards. MemFabric MTE must be
installed on the runner; otherwise the test is skipped.

Chunked prefill and prefix caching are both on. Prompts share a long prefix so
the second+ requests can hit the prefix cache; ``max_num_batched_tokens`` is
kept below that prefix length so prefill is actually chunked.

Run `pytest tests/e2e/pull_request/two_card/test_kvpp.py`.
"""

import pytest

from tests.e2e.conftest import ModelName, wait_until_npu_memory_free
from tests.e2e.pull_request.utils import compare_logprobs

MODEL = ModelName.DEEPSEEK

# Shared prefix is long enough that 128 batched tokens cannot cover it in one
# prefill chunk. Four suffixes share the prefix so later requests can hit cache.
_SHARED_PREFIX = (
    "You are a helpful assistant that answers briefly. Read the following context and continue the sentence. "
) * 16
PROMPTS = [
    _SHARED_PREFIX + "Hello, my name is",
    _SHARED_PREFIX + "The president of the United States is",
    _SHARED_PREFIX + "The capital of France is",
    _SHARED_PREFIX + "The future of AI is",
]


def _require_memfabric_mte() -> None:
    memfabric_hybrid = pytest.importorskip("memfabric_hybrid")
    if not hasattr(memfabric_hybrid, "shm"):
        pytest.skip("KVPP MTE requires memfabric_hybrid.shm")


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="kvpp,prefix_caching,chunked_prefill",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A2",
    quantization="W8A8",
    graph_mode="eager",
)
@pytest.mark.parametrize(
    "use_v2_runner",
    [False, True],
    ids=["mrv1", "mrv2"],
)
@wait_until_npu_memory_free(0.7)
def test_deepseek_v2_lite_kvpp_tp2(use_v2_runner: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_memfabric_mte()
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)
    if use_v2_runner:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    else:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)

    # `additional_config` is excluded from the eager baseline by
    # compare_logprobs, so the baseline runs without KVPP.
    compare_logprobs(
        runner_kwargs={
            "model_name": MODEL,
            "max_model_len": 1024,
            "max_num_batched_tokens": 128,
            "enforce_eager": True,
            "tensor_parallel_size": 2,
            "enable_expert_parallel": True,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "quantization": "ascend",
            "gpu_memory_utilization": 0.7,
            "distributed_executor_backend": "mp",
            "additional_config": {"enable_kvpp": True},
        },
        prompts=PROMPTS,
    )
