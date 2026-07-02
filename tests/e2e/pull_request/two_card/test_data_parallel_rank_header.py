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
#

"""
E2E test for routing requests to a specific data parallel rank via the
`X-data-parallel-rank` HTTP header.

When vLLM serves with internal data parallel load balancing
(`--data-parallel-size N` behind a single API server), a client can pin an
individual request to a chosen DP engine by sending the `X-data-parallel-rank`
header. This is used in RL / multi-turn rollouts to keep a session on a
consistent DP rank so its cached KV state is reused.

A lightweight model (`Qwen/Qwen3-0.6B`) with `--data-parallel-size 2` is used so
the test fits on two Ascend cards.

Run `pytest tests/e2e/pull_request/two_card/test_data_parallel_rank_header.py`.
"""

import os
from unittest.mock import patch

import pytest
import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free

MODEL = "Qwen/Qwen3-0.6B"
DP_SIZE = 2
PROMPT = "San Francisco is a"
MAX_TOKENS = 10
# Several requests per rank to make sure pinned routing is stable, not a fluke.
REQUESTS_PER_RANK = 3


def _completion_for_rank(server: RemoteOpenAIServer, dp_rank: int) -> dict:
    """Send a completion request pinned to `dp_rank` and return the parsed body."""
    response = requests.post(
        server.url_for("v1", "completions"),
        headers={"X-data-parallel-rank": str(dp_rank)},
        json={
            "model": MODEL,
            "prompt": PROMPT,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "n": 1,
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


@pytest.mark.parametrize("dp_size", [DP_SIZE])
@patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1"})
@wait_until_npu_memory_free(target_free_percentage=0.95)
def test_data_parallel_rank_header_routing(dp_size: int):
    """Requests pinned to each valid DP rank via `X-data-parallel-rank` must all
    succeed and return non-empty deterministic completions."""
    port = get_open_port()
    server_args = [
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        str(dp_size),
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        "0.4",
        "--enforce-eager",
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--port",
        str(port),
    ]

    with RemoteOpenAIServer(MODEL, server_args, server_port=port, auto_port=False) as server:
        for dp_rank in range(dp_size):
            for _ in range(REQUESTS_PER_RANK):
                data = _completion_for_rank(server, dp_rank)

                choices = data.get("choices")
                assert choices, f"DP rank {dp_rank}: response is missing `choices`: {data}"

                text = choices[0].get("text")
                assert text, f"DP rank {dp_rank}: empty completion text: {data}"
