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

"""Exercise block and layerwise AscendStore subprocess Memcache transfers."""

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
PREFIX_LENGTH = 1536  # One hybrid KV transfer block at TP2.
SUFFIX_LENGTH = 64  # Uncached tail stored by the layerwise hot request.
PROMPT_SALTS = (
    bytes.fromhex("00112233445566778899aabbccddeeff"),
    bytes.fromhex("ffeeddccbbaa99887766554433221100"),
    bytes.fromhex("6ba7b8109dad11d180b400c04fd430c8"),
    bytes.fromhex("123e4567e89b12d3a456426614174000"),
)
TRANSFER_TIMEOUT_SECONDS = 60


def _prompt(salt: bytes, suffix_seed: int | None = None) -> TokensPrompt:
    token_ids = [1 + (salt[index % len(salt)] + index) % 1023 for index in range(PREFIX_LENGTH)]
    if suffix_seed is not None:
        token_ids += [(suffix_seed + index) % 1023 for index in range(SUFFIX_LENGTH)]
    return TokensPrompt(prompt_token_ids=token_ids)


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


def _assert_block_roundtrips(llm, sampling_params) -> None:
    for case_id, salt in enumerate(PROMPT_SALTS):
        case = f"mode=block, case={case_id}, salt={salt.hex()}"
        print(f"AscendStore Memcache E2E: {case}", flush=True)
        prompt = _prompt(salt)
        cold_output = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
        _wait_for_prefix_cache_reset(llm)
        loaded_output = llm.generate(prompt, sampling_params, use_tqdm=False)[0]

        cached_tokens = loaded_output.num_cached_tokens
        assert cached_tokens and cached_tokens > 0, f"Memcache load missed: {case}"
        assert loaded_output.outputs[0].token_ids == cold_output.outputs[0].token_ids, (
            f"Memcache load changed greedy output: {case}, cached_tokens={cached_tokens}"
        )


def _assert_layerwise_hot_roundtrips(llm, sampling_params) -> None:
    for case_id, salt in enumerate(PROMPT_SALTS[:2]):
        case = f"mode=layerwise, case={case_id}, salt={salt.hex()}"
        print(f"AscendStore Memcache E2E: {case}", flush=True)

        # The cold request publishes the full prefix. The hot request loads
        # that prefix while adding a new suffix, and the final request proves
        # that the incrementally published suffix is readable.
        cold_prompt = _prompt(salt, suffix_seed=17)
        llm.generate(cold_prompt, sampling_params, use_tqdm=False)
        _wait_for_prefix_cache_reset(llm)

        hot_prompt = _prompt(salt, suffix_seed=113)
        hot_output = llm.generate(hot_prompt, sampling_params, use_tqdm=False)[0]
        assert hot_output.num_cached_tokens and hot_output.num_cached_tokens > 0, (
            f"Memcache layerwise load missed the prefix: {case}"
        )
        _wait_for_prefix_cache_reset(llm)

        reloaded_output = llm.generate(hot_prompt, sampling_params, use_tqdm=False)[0]
        assert reloaded_output.num_cached_tokens and reloaded_output.num_cached_tokens > 0, (
            f"Memcache layerwise reload missed the prompt: {case}"
        )
        assert reloaded_output.outputs[0].token_ids == hot_output.outputs[0].token_ids, (
            f"Memcache layerwise load changed greedy output: {case}, cached_tokens={reloaded_output.num_cached_tokens}"
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
def test_ascend_store_memcache_subprocess_modes(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    sampling_params = SamplingParams(max_tokens=1, temperature=0)

    for use_layerwise, assert_roundtrips in (
        (False, _assert_block_roundtrips),
        (True, _assert_layerwise_hot_roundtrips),
    ):
        mode = "layerwise" if use_layerwise else "block"
        config = MemcacheKVPoolConfig(
            meta_service_port=get_open_port(),
            config_store_port=get_open_port(),
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
        with SingleNodeMemcacheManager(config, f"ascend-store-transfer-{mode}-{uuid.uuid4().hex}") as manager:
            for name, value in manager.server_envs.items():
                monkeypatch.setenv(name, value)

            kv_transfer_config = KVTransferConfig(
                kv_connector="AscendStoreConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "backend": "memcache",
                    "lookup_rpc_port": "0",
                    "use_layerwise": use_layerwise,
                    "use_multiprocess": True,
                },
            )
            with VllmRunner(
                MODEL,
                tensor_parallel_size=2,
                distributed_executor_backend="mp",
                max_model_len=2048,
                gpu_memory_utilization=0.9,
                enable_prefix_caching=True,
                enforce_eager=True,
                kv_transfer_config=kv_transfer_config,
            ) as runner:
                assert_roundtrips(runner.model, sampling_params)
