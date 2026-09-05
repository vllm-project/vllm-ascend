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

"""End-to-end test for AscendStore multiprocess transfers with Memcache."""

import time
import uuid

import pytest
from vllm import SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MemcacheKVPoolConfig
from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.nightly.single_node.models.scripts.kv_pool_runtime import SingleNodeMemcacheManager

MODEL = "Qwen/Qwen3.5-27B"
TRANSFER_TIMEOUT_SECONDS = 60


def _wait_for_prefix_cache_reset(llm) -> None:
    """Wait for asynchronous stores before dropping only the local KV cache."""
    deadline = time.monotonic() + TRANSFER_TIMEOUT_SECONDS
    sampling_params = SamplingParams(max_tokens=1)
    while not llm.reset_prefix_cache():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for AscendStore transfers to finish")
        llm.generate(
            [TokensPrompt(prompt_token_ids=[0])],
            sampling_params,
            use_tqdm=False,
        )


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="mamba_ssm",
    feature="prefix_caching",
    parallel="TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_ascend_store_multiprocess_memcache_roundtrip(monkeypatch) -> None:
    meta_port = get_open_port()
    config_store_port = get_open_port()
    config = MemcacheKVPoolConfig(
        meta_service_port=meta_port,
        config_store_port=config_store_port,
        config={
            "meta": {"ock.mmc.log_level": "error"},
            "local": {
                "ock.mmc.log_level": "error",
                "ock.mmc.local_service.world_size": 256,
                "ock.mmc.local_service.protocol": "device_sdma",
                "ock.mmc.local_service.dram.size": "1GB",
            },
        },
    )
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    with SingleNodeMemcacheManager(config, f"ascend-store-transfer-mp-{uuid.uuid4().hex}") as manager:
        for name, value in manager.server_envs.items():
            monkeypatch.setenv(name, value)

        kv_transfer_config = KVTransferConfig(
            kv_connector="AscendStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "backend": "memcache",
                "lookup_rpc_port": "0",
                "use_layerwise": False,
                "use_multiprocess": True,
            },
        )
        sampling_params = SamplingParams(max_tokens=1, temperature=0)
        salt = uuid.uuid4().bytes
        prompt = TokensPrompt(prompt_token_ids=[1 + (salt[index % len(salt)] + index) % 1023 for index in range(512)])

        with VllmRunner(
            MODEL,
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
            max_model_len=1024,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            enforce_eager=True,
            kv_transfer_config=kv_transfer_config,
        ) as runner:
            llm = runner.model
            cold_output = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
            _wait_for_prefix_cache_reset(llm)
            loaded_output = llm.generate(prompt, sampling_params, use_tqdm=False)[0]

            assert loaded_output.num_cached_tokens and loaded_output.num_cached_tokens > 0
            assert loaded_output.outputs[0].token_ids == cold_output.outputs[0].token_ids
