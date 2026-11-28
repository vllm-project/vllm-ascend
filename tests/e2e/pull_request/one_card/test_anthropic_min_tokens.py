# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free


@pytest.mark.e2e_model("Qwen/Qwen3-0.6B")
@wait_until_npu_memory_free()
def test_min_tokens_delays_stop_sequence():
    # Same generation regression as upstream vLLM #65459, through the Ascend CLI.
    # The shared fixture does not copy its auto-selected CLI port to server.port.
    port = get_open_port()
    with RemoteOpenAIServer(
        "Qwen/Qwen3-0.6B",
        ["--served-model-name", "anthropic-test", "--max-model-len", "1024", "--enforce-eager", "--port", str(port)],
        server_host="127.0.0.1",
        server_port=port,
        auto_port=False,
        max_wait_seconds=180,
    ) as server:
        payload = {
            "model": "anthropic-test",
            "max_tokens": 64,
            "temperature": 0.0,
            "stop_sequences": ["\n"],
            "messages": [{"role": "user", "content": "Write two short lines of poetry."}],
        }
        baseline = requests.post(server.url_root + "/v1/messages", json=payload, timeout=(5, 120))
        baseline.raise_for_status()
        baseline_data = baseline.json()
        assert baseline_data["stop_reason"] == "stop_sequence"
        assert baseline_data["usage"]["output_tokens"] < 16

        response = requests.post(server.url_root + "/v1/messages", json={**payload, "min_tokens": 16}, timeout=(5, 120))
        response.raise_for_status()
        assert response.json()["usage"]["output_tokens"] >= 16
