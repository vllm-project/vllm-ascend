# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import requests

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free


@pytest.mark.e2e_model("Qwen/Qwen3-0.6B")
@wait_until_npu_memory_free()
def test_min_tokens_delays_stop_sequence():
    # Same generation regression as upstream vLLM #65459, through the Ascend CLI.
    with RemoteOpenAIServer(
        "Qwen/Qwen3-0.6B",
        ["--served-model-name", "anthropic-test", "--max-model-len", "1024", "--enforce-eager"],
        max_wait_seconds=60,
    ) as server:
        payload = {
            "model": "anthropic-test",
            "max_tokens": 64,
            "temperature": 0.0,
            "stop_sequences": ["\n"],
            "messages": [{"role": "user", "content": "Write two short lines of poetry."}],
        }
        baseline = requests.post(server.url_root + "/v1/messages", json=payload, timeout=10)
        baseline.raise_for_status()
        baseline_data = baseline.json()
        assert baseline_data["stop_reason"] == "stop_sequence"
        assert baseline_data["usage"]["output_tokens"] < 16

        response = requests.post(server.url_root + "/v1/messages", json={**payload, "min_tokens": 16}, timeout=10)
        response.raise_for_status()
        assert response.json()["usage"]["output_tokens"] >= 16
