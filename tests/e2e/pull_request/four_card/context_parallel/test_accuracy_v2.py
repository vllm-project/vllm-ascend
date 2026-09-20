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
#
"""Model Runner V2 context-parallel accuracy and feature guards.

Run `pytest tests/e2e/pull_request/four_card/context_parallel/test_accuracy_v2.py`.
"""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import patch

import pytest
import requests
from vllm import SamplingParams
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MemcacheKVPoolConfig
from tests.e2e.conftest import DPVllmRunner, RemoteOpenAIServer, VllmRunner, wait_until_npu_memory_free
from tests.e2e.nightly.single_node.models.scripts.kv_pool_runtime import SingleNodeMemcacheManager
from vllm_ascend.utils import vllm_version_is

MAX_NUM_SEQS = 4
FULL_DECODE_GRAPH = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [MAX_NUM_SEQS],
}

PCP_FULL_DECODE_GRAPH = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [4, 8],
}

DSV3_2_MODEL = os.getenv(
    "DSV3_2_MODEL_PATH",
    "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning",
)
DSV3_2_PROMPTS = [
    "The capital of France is",
    "Hello, my name is Tom, I am",
    "The president of United States is",
]
POOLING_MODEL_NAME = "pcp-pooling-test"
POOLING_PROMPT = "This is background information. " * 80 + "The capital of France is"
DSV3_2_SFA_DCP_GOLDENS = (
    [
        "The capital of France isoint054 Rund compasses",
        "Hello, my name is Tom, I am" + "ERIC slicpacelike挂",
        "The president of United States isoint054 Rund959arki",
    ],
    [
        "The capital of France isoint054 Rund959arki",
        "Hello, my name is Tom, I am" + "ERIC slicpacelike挂",
        "The president of United States isoint054 Rund959arki",
    ],
    [
        "The capital of France isorrionicALLY casmith",
        "Hello, my name is Tom, I am" + "ERIC slicpacelike挂",
        "The president of United States is平行于我 charm与技术oi",
    ],
    [
        "The capital of France isorrionic Tudefeault",
        "Hello, my name is Tom, I am" + "ERIC slicpacelike挂",
        "The president of United States is平行于我 charm与技术oi",
    ],
)
MTP_PCP_MODEL = "wemaster/deepseek_mtp_main_random_bf16"
EAGLE3_PCP_TARGET_MODEL = "Qwen/Qwen3-8B"
EAGLE3_PCP_DRAFT_MODEL = "RedHatAI/Qwen3-8B-speculator.eagle3"

LONG_CONTEXT = "This is a long context for PCP prefill validation. " * 64
PCP_PROMPTS = [
    LONG_CONTEXT + "Hello, my name is",
    LONG_CONTEXT + "The president of the United States is",
]


@dataclass(frozen=True)
class AccuracyCase:
    name: str
    model: str
    prompts: Sequence[str]
    expected_outputs: Sequence[str] | Sequence[Sequence[str]]
    max_tokens: int
    runner_kwargs: dict[str, Any]


@dataclass(frozen=True)
class InferenceCase:
    model: str
    prompts: Sequence[str]
    max_tokens: int
    runner_kwargs: dict[str, Any]


def _match_outputs_with_goldens(outputs: list[tuple[list[int], str]], goldens: Sequence[str]) -> None:
    """Helper function to compare output with golden output, ignoring whitespace differences."""
    outputs_str: Sequence[str] = [output[1] for output in outputs]
    assert len(outputs_str) == len(goldens)
    for output, golden in zip(outputs_str, goldens):
        assert isinstance(output, str) and isinstance(golden, str), "Both output and golden must be strings"
        assert output and golden, "Output and golden should not be empty"
        assert output.strip() == golden.strip()


def _run_accuracy_case(case: AccuracyCase) -> None:
    runner_cls = DPVllmRunner if case.runner_kwargs.get("data_parallel_size", 1) > 1 else VllmRunner
    with runner_cls(case.model, **case.runner_kwargs) as runner:
        outputs = runner.generate_greedy(list(case.prompts), case.max_tokens)

    if isinstance(case.expected_outputs[0], str):
        expected_outputs = cast(Sequence[str], case.expected_outputs)
        _match_outputs_with_goldens(outputs, expected_outputs)
    else:
        # If multiple expected output sets are provided, the output is considered correct if it matches any of the sets.
        multi_expected_outputs = cast(Sequence[Sequence[str]], case.expected_outputs)
        tries = []
        for expected in multi_expected_outputs:
            try:
                _match_outputs_with_goldens(outputs, expected)
            except AssertionError as exc:
                tries.append(f"Output did not match expected set:\n{exc}")
            else:
                break
        if len(tries) == len(multi_expected_outputs):
            failure_details = "\n\n".join(tries)
            raise AssertionError(f"Output did not match any of the expected output sets:\n{failure_details}")


def _run_inference_case(case: InferenceCase) -> None:
    """Verify that the configured service starts and returns generated tokens."""
    runner_cls = DPVllmRunner if case.runner_kwargs.get("data_parallel_size", 1) > 1 else VllmRunner
    with runner_cls(case.model, **case.runner_kwargs) as runner:
        outputs = runner.generate_greedy(list(case.prompts), case.max_tokens)

    assert len(outputs) == len(case.prompts)
    for token_ids, output_text in outputs:
        assert token_ids, "Each request should return at least one generated token"
        assert isinstance(output_text, str) and output_text, "Each request should return non-empty text"


def _complete_pooling_request(url: str, prompt: str | list[str]) -> list[str]:
    response = requests.post(
        url + "/v1/completions",
        json={
            "model": POOLING_MODEL_NAME,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": 16,
            "ignore_eos": True,
        },
        timeout=180,
    )
    response.raise_for_status()
    result = response.json()
    assert all(choice["finish_reason"] == "length" for choice in result["choices"])
    return [choice["text"] for choice in result["choices"]]


def _run_pcp_pooling_case(tmp_path) -> None:
    """Validate PCP graph execution and remote KV-cache reload in one service."""
    pytest.importorskip("memcache_hybrid")
    config = MemcacheKVPoolConfig(
        meta_service_port=get_open_port(),
        config_store_port=get_open_port(),
        config={
            "meta": {
                "ock.mmc.log_level": "info",
                "ock.mmc.meta_service.metrics_url": f"http://127.0.0.1:{get_open_port()}",
            },
            "local": {
                "ock.mmc.log_level": "info",
                # TP1 * PCP2 * DP2 workers participate in the local pool.
                "ock.mmc.local_service.world_size": 4,
                "ock.mmc.local_service.protocol": "device_sdma",
                "ock.mmc.local_service.dram.size": "1GB",
            },
        },
    )
    with SingleNodeMemcacheManager(config, tmp_path.name) as pool:
        port = get_open_port()
        args = [
            "--port",
            str(port),
            "--served-model-name",
            POOLING_MODEL_NAME,
            "--trust-remote-code",
            "--quantization",
            "ascend",
            "--tensor-parallel-size",
            "1",
            "--data-parallel-size",
            "2",
            "--distributed-executor-backend",
            "mp",
            "--prefill-context-parallel-size",
            "2",
            "--enable-expert-parallel",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--max-model-len",
            "1024",
            "--max-num-seqs",
            str(MAX_NUM_SEQS),
            "--max-num-batched-tokens",
            "1024",
            "--gpu-memory-utilization",
            "0.8",
            "--cp-kv-cache-interleave-size",
            "128",
            "--block-size",
            "128",
            "--seed",
            "42",
            "--generation-config",
            "vllm",
            "--compilation-config",
            json.dumps(FULL_DECODE_GRAPH),
            "--kv-transfer-config",
            json.dumps(
                {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": "kv_producer",
                    "kv_connector_extra_config": {
                        "lookup_rpc_port": "0",
                        "backend": "memcache",
                        "use_layerwise": False,
                        "load_async": True,
                    },
                }
            ),
        ]
        with RemoteOpenAIServer(
            maybe_model_redirect(DSV3_2_MODEL),
            args,
            server_port=port,
            auto_port=False,
            env_dict={
                **pool.server_envs,
                "VLLM_USE_V2_MODEL_RUNNER": "1",
                "VLLM_SERVER_DEV_MODE": "1",
            },
        ) as server:
            # Preserve the original case's multi-request inference coverage.
            smoke_outputs = _complete_pooling_request(server.url_root, DSV3_2_PROMPTS)
            assert len(smoke_outputs) == len(DSV3_2_PROMPTS)
            assert all(smoke_outputs)

            # Write a long-prefix cache entry, clear local cache, then force two
            # independent reloads from the remote pool.
            expected = _complete_pooling_request(server.url_root, POOLING_PROMPT)
            for _ in range(2):
                requests.post(server.url_for("reset_prefix_cache"), timeout=30).raise_for_status()
                assert _complete_pooling_request(server.url_root, POOLING_PROMPT) == expected

            metrics = requests.get(server.url_for("metrics"), timeout=30)
            metrics.raise_for_status()
            loaded_keys = sum(
                float(line.split()[-1])
                for line in metrics.text.splitlines()
                if line.startswith("vllm:ascend_store_load_get_keys_total{")
            )
            assert loaded_keys > 0


DSV3_2_SFA_PCP_CASE = InferenceCase(
    model=DSV3_2_MODEL,
    prompts=DSV3_2_PROMPTS,
    max_tokens=5,
    runner_kwargs={
        "max_model_len": 1024,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": 1024,
        "tensor_parallel_size": 2,
        "prefill_context_parallel_size": 2,
        "enable_expert_parallel": True,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": 0.8,
        "cp_kv_cache_interleave_size": 128,
        "block_size": 128,
        "quantization": "ascend",
        "compilation_config": FULL_DECODE_GRAPH,
    },
)

DSV3_2_SFA_PCP_DCP_CASE = AccuracyCase(
    name="dsv3_2_sfa_pcp_dcp_replicated_indexer_mrv2_tp2_pcp2_dcp4",
    model=DSV3_2_MODEL,
    prompts=DSV3_2_PROMPTS,
    expected_outputs=DSV3_2_SFA_DCP_GOLDENS,
    max_tokens=5,
    runner_kwargs={
        "max_model_len": 1024,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": 1024,
        "tensor_parallel_size": 2,
        "prefill_context_parallel_size": 2,
        "decode_context_parallel_size": 4,
        "enable_expert_parallel": True,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": 0.8,
        "cp_kv_cache_interleave_size": 128,
        "block_size": 128,
        "quantization": "ascend",
        "compilation_config": FULL_DECODE_GRAPH,
        "additional_config": {
            "enable_dsa_cp": False,
            "enable_sparse_li_c8": False,
        },
    },
)

DSV3_2_SFA_PCP_PP_MTP_CASE = InferenceCase(
    model=DSV3_2_MODEL,
    prompts=DSV3_2_PROMPTS,
    max_tokens=5,
    runner_kwargs={
        **DSV3_2_SFA_PCP_CASE.runner_kwargs,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 2,
        "async_scheduling": True,
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": 3,
        },
    },
)


@pytest.mark.e2e_model(DSV3_2_MODEL)
@pytest.mark.skipif(
    vllm_version_is("0.28.0"),
    reason="Temporary v0.28.0 SFA PCP accuracy skip; root cause is under separate investigation (PR #16009).",
)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="sfa_pcp",
    parallel="TP,EP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "768",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dsv3_2_sfa_pcp_model_runner_v2_graph() -> None:
    """Guard MRV2 SFA PCP full-decode-only graph execution."""
    _run_inference_case(DSV3_2_SFA_PCP_CASE)


@pytest.mark.e2e_model(DSV3_2_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="sfa_pcp,chunked_prefill,prefix_caching",
    parallel="DP,EP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "768",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dsv3_2_sfa_pcp_dp_model_runner_v2_graph(tmp_path) -> None:
    """Guard DP2+PCP2 graph execution and KV Cache Pooling reload."""
    _run_pcp_pooling_case(tmp_path)


@pytest.mark.e2e_model(DSV3_2_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="sfa_pcp,mtp",
    parallel="EP,PCP,PP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "768",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dsv3_2_sfa_pcp_pp_mtp_model_runner_v2_graph() -> None:
    """Guard MRV2 SFA PCP+PP+MTP graph execution with async scheduling."""
    _run_inference_case(DSV3_2_SFA_PCP_PP_MTP_CASE)


@pytest.mark.e2e_model(DSV3_2_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="sfa_pcp",
    parallel="TP,EP,PCP,DCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_BATCH_INVARIANT": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "768",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dsv3_2_sfa_pcp_dcp_model_runner_v2_graph_accuracy() -> None:
    """Guard MRV2 SFA PCP+DCP full-decode-only graph accuracy."""
    _run_accuracy_case(DSV3_2_SFA_PCP_DCP_CASE)


def _run_pcp_spec_decode(
    model: str,
    speculative_config: dict[str, object],
) -> None:
    sampling_params = SamplingParams(max_tokens=16, temperature=0.0)
    with VllmRunner(
        model,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        max_model_len=1024,
        max_num_batched_tokens=64,
        max_num_seqs=MAX_NUM_SEQS,
        disable_log_stats=False,
        distributed_executor_backend="mp",
        enable_chunked_prefill=True,
        compilation_config=PCP_FULL_DECODE_GRAPH,
        speculative_config=speculative_config,
    ) as runner:
        outputs = runner.model.generate(PCP_PROMPTS, sampling_params)
        metrics = runner.model.get_metrics()
        num_drafts = sum(metric.value for metric in metrics if metric.name == "vllm:spec_decode_num_drafts")

    token_ids = [output.outputs[0].token_ids for output in outputs]
    assert len(token_ids) == len(PCP_PROMPTS)
    assert all(token_ids)
    assert num_drafts > 0


@pytest.mark.e2e_model(MTP_PCP_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,chunked_prefill",
    parallel="TP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "1024",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mtp_mla_spec_decode_with_pcp() -> None:
    """Guard MRV2 MTP MLA PCP full-decode-only graph execution."""
    _run_pcp_spec_decode(
        MTP_PCP_MODEL,
        {
            "method": "mtp",
            "num_speculative_tokens": 3,
        },
    )


@pytest.mark.e2e_model(EAGLE3_PCP_TARGET_MODEL, EAGLE3_PCP_DRAFT_MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="eagle3,chunked_prefill",
    parallel="TP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "1024",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_eagle3_gqa_spec_decode_with_pcp() -> None:
    """Guard MRV2 Eagle3 GQA PCP full-decode-only graph execution."""
    _run_pcp_spec_decode(
        EAGLE3_PCP_TARGET_MODEL,
        {
            "method": "eagle3",
            "model": EAGLE3_PCP_DRAFT_MODEL,
            "num_speculative_tokens": 3,
        },
    )
