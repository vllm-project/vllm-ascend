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

"""End-to-end test for AscendStore multiprocess transfers with Mooncake."""

import json
import time

import pytest
from vllm import SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import MooncakeLauncher, VllmRunner, wait_until_npu_memory_free

MODEL = "Qwen/Qwen3-0.6B"
TRANSFER_TIMEOUT_SECONDS = 60


def _wait_for_prefix_cache_reset(llm) -> None:
    """Wait for asynchronous stores before dropping only the local KV cache."""
    deadline = time.monotonic() + TRANSFER_TIMEOUT_SECONDS
    sampling_params = SamplingParams(max_tokens=1)
    while not llm.reset_prefix_cache():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for AscendStore transfers to finish")
        # A scheduler step collects completed stores and releases their NPU blocks.
        llm.generate(
            [TokensPrompt(prompt_token_ids=[0])],
            sampling_params,
            use_tqdm=False,
        )


@pytest.mark.e2e_model("Qwen/Qwen3-0.6B")
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="prefix_caching",
    parallel="",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_ascend_store_multiprocess_mooncake_roundtrip(tmp_path, monkeypatch) -> None:
    mooncake_port = get_open_port()
    mooncake_metrics_port = get_open_port()
    mooncake_config_path = tmp_path / "mooncake.json"
    mooncake_config_path.write_text(
        json.dumps(
            {
                "metadata_server": "P2PHANDSHAKE",
                "protocol": "ascend",
                "device_name": "",
                "master_server_address": f"127.0.0.1:{mooncake_port}",
                "global_segment_size": 1 << 30,
                "local_buffer_size": 1 << 30,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(mooncake_config_path))
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    kv_transfer_config = KVTransferConfig(
        kv_connector="AscendStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "backend": "mooncake",
            "lookup_rpc_port": "0",
            "use_layerwise": False,
            "use_multiprocess": True,
        },
    )
    sampling_params = SamplingParams(max_tokens=1, temperature=0)
    prompt = TokensPrompt(prompt_token_ids=[i % 1024 for i in range(512)])

    with (
        MooncakeLauncher(mooncake_port, mooncake_metrics_port),
        VllmRunner(
            MODEL,
            max_model_len=1024,
            gpu_memory_utilization=0.5,
            enable_prefix_caching=True,
            enforce_eager=True,
            kv_transfer_config=kv_transfer_config,
        ) as runner,
    ):
        llm = runner.model
        cold_output = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
        _wait_for_prefix_cache_reset(llm)
        loaded_output = llm.generate(prompt, sampling_params, use_tqdm=False)[0]

        assert loaded_output.num_cached_tokens and loaded_output.num_cached_tokens > 0
        assert loaded_output.outputs[0].token_ids == cold_output.outputs[0].token_ids
