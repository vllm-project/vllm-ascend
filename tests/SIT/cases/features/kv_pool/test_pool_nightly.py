# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Nightly KV-pool cases for A3 single-host deployments.

The dsv4-memcache-multiprefix-64k, dsv4-memcache-dsa-pcp-dspark and
qwen38-memcache-mtp-align cases replay their guarded workloads at
short-input scale. ``test_dsv4_mooncake_dp4_multiprefix`` is plan-N12:
DSV4-Flash-0731 DP4xTP4 mooncake non-layerwise 128K multi-prefix with
token-exact A/B; performance is recorded only, and the Mooncake SSD
offload pilot (plan 2.6/7) is a separate harness follow-up that this
module does not claim. ``test_dsv4_memcache_multiprefix_64k`` is plan-N3:
DSV4.1-Flash DP4xTP4 memcache layerwise 64K multi-prefix, and
``test_qwen38_memcache_mtp_align`` is plan-N7: Qwen3.8-27B-w8a8 TP2+MTP
memcache layerwise 128K with hybrid align state checks. Both reuse the
PR-layer secret-number accuracy prompt (plan 2.7).
"""

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.utils.network_utils import get_open_port
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.common.kv_pool.config import (
    MemcacheKVPoolConfig,
    MooncakeKVPoolConfig,
)
from tests.e2e.common.single_node.kv_pool_runtime import (
    SingleNodeMemcacheManager,
    SingleNodeMooncakeManager,
)
from tests.e2e.conftest import (
    DPVllmRunner,
    VllmRunner,
    wait_until_npu_memory_free,
)
from tests.e2e.pull_request.one_card.model_runner_v2.utils import (
    calculate_acceptance_per_pos,
)
from tests.e2e.pull_request.kv_pool_watch import (
    POOL_SECRET,
    build_pool_accuracy_prompt,
    build_pool_prompts,
    extract_pool_answer,
    run_pool_ab,
    run_pool_hit_sequence,
)
from tests.e2e.pull_request.utils import PROMPTS_SHORT

DSV4_DSPARK_MODEL = os.environ.get(
    "DEEPSEEK_V41_DSPARK_MODEL_PATH",
    "Eco-Tech/DeepSeek-V4.1-Flash-w8a8",
)
DSV4_0731_MODEL = os.environ.get(
    "DEEPSEEK_V4_0731_MODEL_PATH",
    "Eco-Tech/DeepSeek-V4-Flash-0731-w8a8",
)
QWEN38_MODEL = os.environ.get("QWEN38_DENSE_MODEL", "Eco-Tech/Qwen3.8-27B-w8a8")

MEMCACHE_ENV = {
    "meta": {"ock.mmc.log_level": "error"},
    "local": {
        "ock.mmc.log_level": "error",
        "ock.mmc.local_service.world_size": 256,
        "ock.mmc.local_service.protocol": "device_sdma",
        "ock.mmc.local_service.dram.size": "20GB",
    },
}

DSV4_POOL_LCM_BLOCK_SIZE = 16384
DSV4_POOL_PROMPT_TOKENS = DSV4_POOL_LCM_BLOCK_SIZE + 4
DSV4_POOL_PERF_PROMPT_TOKENS = (
    DSV4_POOL_LCM_BLOCK_SIZE + DSV4_POOL_LCM_BLOCK_SIZE // 9
)
DSV4_POOL_MAX_MODEL_LEN = DSV4_POOL_PERF_PROMPT_TOKENS + 128
QWEN_POOL_LCM_BLOCK_SIZE = 896
# Plan-N7 128K bucket: 146 lcm blocks plus a 4-token tail retain one reusable
# block after the final block trim.
QWEN_POOL_PROMPT_TOKENS = 146 * QWEN_POOL_LCM_BLOCK_SIZE + 4
QWEN_POOL_PERF_PROMPT_TOKENS = QWEN_POOL_PROMPT_TOKENS
QWEN_POOL_MAX_MODEL_LEN = QWEN_POOL_PERF_PROMPT_TOKENS + 128

# Plan-N12 128K multi-prefix load: eight lcm blocks plus a 4-token tail; the
# measured H3 chunk shape passes max_num_batched_tokens=28672 (plan 2.1).
DSV4_DP4_POOL_PROMPT_TOKENS = 8 * DSV4_POOL_LCM_BLOCK_SIZE + 4
DSV4_DP4_POOL_MAX_NUM_BATCHED_TOKENS = 28672
DSV4_DP4_POOL_MAX_MODEL_LEN = DSV4_DP4_POOL_PROMPT_TOKENS + 128
# One prompt copy per DP rank; expected hit tokens scale by this factor.
DSV4_DP4_REPLICAS = 4

PERF_OPTIONS = {
    "perf_sample_count": 10,
    "perf_concurrency": 1,
    "perf_stability_policy": "record",
    "perf_gate": False,
}

# Plan-N3 (2026-11-20 redesign): DSV4.1-Flash single-host DP4xTP4 memcache
# layerwise. The load is the 64K bucket (four lcm blocks plus a 4-token
# tail) and every leg reuses the PR-layer secret-number accuracy prompt
# (plan 2.7): no separate dataset, the check is answer-level only and the
# main evidence is the hit and performance stages.
N3_POOL_LCM_BLOCK_SIZE = 16384
N3_POOL_PROMPT_TOKENS = 4 * N3_POOL_LCM_BLOCK_SIZE + 4
N3_POOL_MAX_MODEL_LEN = N3_POOL_PROMPT_TOKENS + 128
N3_POOL_PERF_PROMPT_TOKENS = N3_POOL_PROMPT_TOKENS
N3_DP4_REPLICAS = 4

# PR-layer secret-number prompts answer with a short numeric continuation.
SECRET_ACC_MAX_TOKENS = 32


def _build_memcache_config() -> MemcacheKVPoolConfig:
    return MemcacheKVPoolConfig(
        meta_service_port=get_open_port(),
        config_store_port=get_open_port(),
        config=MEMCACHE_ENV,
    )


def _build_mooncake_config() -> MooncakeKVPoolConfig:
    return MooncakeKVPoolConfig(
        config={
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "ascend",
            "global_segment_size": "20GB",
        },
        master_port=get_open_port(),
        metrics_port=get_open_port(),
    )


def _assert_dspark_accuracy(runner: Any) -> None:
    outputs = runner.generate_greedy(PROMPTS_SHORT, max_tokens=128)
    assert len(outputs) == len(PROMPTS_SHORT)
    for prompt, (_, text) in zip(PROMPTS_SHORT, outputs, strict=True):
        assert text, f"empty output for prompt {prompt!r}"
        expected_prefix = {
            "Hello, my name is": "Hello, my name is",
            "The president of the United States is": (
                "The president of the United States is"
            ),
            "The capital of France is": "The capital of France is Paris",
            "The future of AI is": "The future of AI is",
        }.get(prompt)
        if expected_prefix:
            assert text.startswith(expected_prefix), (
                f"unexpected output for {prompt!r}: {text[:100]!r}"
            )

    metrics = runner.model.get_metrics()
    drafts = sum(
        metric.value
        for metric in metrics
        if metric.name == "vllm:spec_decode_num_drafts"
        and isinstance(metric, Counter)
    )
    accepted = sum(
        sum(metric.values)
        for metric in metrics
        if metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos"
        and isinstance(metric, Vector)
    )
    assert drafts > 0, "DSpark produced no draft tokens"
    assert accepted > 0, "DSpark accepted no draft tokens"


@contextmanager
def _dsv4_memcache_64k_leg(use_pool: bool, case_name: str):
    with ExitStack() as stack:
        kwargs: dict[str, Any] = dict(
            data_parallel_size=4,
            tensor_parallel_size=4,
            enforce_eager=False,
            max_model_len=N3_POOL_MAX_MODEL_LEN,
            max_num_seqs=4,
            max_num_batched_tokens=N3_POOL_LCM_BLOCK_SIZE,
            gpu_memory_utilization=0.92,
            enable_expert_parallel=True,
            quantization="ascend",
            tokenizer_mode="deepseek_v41",
            block_size=128,
            attention_config={"indexer_kv_dtype": "int8"},
            enable_prefix_caching=not use_pool,
            disable_log_stats=False,
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        )
        if use_pool:
            manager = stack.enter_context(
                SingleNodeMemcacheManager(_build_memcache_config(), case_name)
            )
            stack.enter_context(patch.dict(os.environ, manager.server_envs))
            kwargs["kv_transfer_config"] = {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "backend": "memcache",
                    "lookup_rpc_port": "1",
                    "use_layerwise": True,
                },
            }
        yield stack.enter_context(DPVllmRunner(DSV4_DSPARK_MODEL, **kwargs))


@pytest.mark.e2e_model(DSV4_DSPARK_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="prefix_caching,chunked_prefill,long_sequence",
    parallel="DP,TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_BUFFSIZE": "1024",
        "TASK_QUEUE_ENABLE": "1"
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dsv4_memcache_multiprefix_64k() -> None:
    """Plan-N3: DSV4.1-Flash DP4xTP4 memcache layerwise 64K multi-prefix.

    Two distinct-secret natural-language 64K prefixes replay R1-R4 with one
    prompt copy per DP rank (H1/H2 replica hits and cross-prefix isolation)
    and exercise the per-layer write/read path. Accuracy reuses the PR-layer
    secret-number prompt (no constructed dataset, plan 2.7): each leg must
    answer both secrets exactly, while the main evidence stays the hit and
    performance stages.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        DSV4_DSPARK_MODEL, trust_remote_code=True
    )
    prefix_a = build_pool_accuracy_prompt(
        tokenizer, N3_POOL_PROMPT_TOKENS, N3_POOL_MAX_MODEL_LEN, secret="7391"
    )
    prefix_b = build_pool_accuracy_prompt(
        tokenizer, N3_POOL_PROMPT_TOKENS, N3_POOL_MAX_MODEL_LEN, secret="2846"
    )

    def accuracy(runner: Any) -> list[float]:
        # Layerwise first-pass validation: send one prompt at a time. A 32-way
        # burst deadlocked HCCL in aclnnSparseFlashMlaMetadata on the first
        # 64K prefill (plan 2.5 isolation step).
        answers = []
        for prompt, expected in ((prefix_a, "7391"), (prefix_b, "2846")):
            outputs = runner.generate_greedy(
                [prompt], max_tokens=SECRET_ACC_MAX_TOKENS
            )
            answer = extract_pool_answer(outputs[0][1])
            assert answer == expected, (answer, expected)
            answers.append(answer)
        print(f"[kv_pool_acc] secret answers={answers}", flush=True)
        return [1.0]

    def hit(runner: Any) -> dict[str, Any]:
        run_pool_hit_sequence(
            runner,
            block_size=128,
            prompt_tokens=N3_POOL_PROMPT_TOKENS,
            prompt_ids=prefix_a,
            replicas=N3_DP4_REPLICAS,
            lcm_block_size=N3_POOL_LCM_BLOCK_SIZE,
            answer_secret="7391",
            tag="nightly-n3-memcache-64k-a",
        )
        summary = run_pool_hit_sequence(
            runner,
            block_size=128,
            prompt_tokens=N3_POOL_PROMPT_TOKENS,
            prompt_ids=prefix_b,
            miss_prompt_ids=build_pool_prompts(
                N3_POOL_PROMPT_TOKENS, seed_offset=1
            )[1],
            replicas=N3_DP4_REPLICAS,
            lcm_block_size=N3_POOL_LCM_BLOCK_SIZE,
            answer_secret="2846",
            tag="nightly-n3-memcache-64k-b",
        )
        # run_pool_ab compares one representative replica against the B leg's
        # single local-hit output; per-replica identity already asserted.
        summary["hit_outputs"] = summary["hit_outputs"][:1]
        return summary

    run_pool_ab(
        case_name="nightly_n3_dsv4_memcache_multiprefix_64k",
        build_leg=_dsv4_memcache_64k_leg,
        accuracy=accuracy,
        hit=hit,
        prompt_ids=prefix_b,
        perf_prompt_tokens=N3_POOL_PERF_PROMPT_TOKENS,
        perf_prompt_ids=prefix_b,
        token_identity=True,
        acceptance_tolerance=0.0,
        # Plan 2.5: nightly performance uses one prefix, 32 concurrency and
        # 64 requests per sample batch; the first version only records.
        # Layerwise isolation step: 32-way bursts deadlock HCCL on the first
        # 64K prefill, so start at concurrency 4 with 8 requests per batch.
        perf_concurrency=4,
        perf_batch_requests=8,
        perf_sample_count=3,
        perf_warmup_count=1,
        perf_drift_count=0,
        perf_trim_count=0,
        perf_stability_policy="record",
        perf_gate=True,
    )


@contextmanager
def _dsv4_dp4_mooncake_leg(use_pool: bool, case_name: str):
    with ExitStack() as stack:
        kwargs: dict[str, Any] = dict(
            data_parallel_size=4,
            tensor_parallel_size=4,
            distributed_executor_backend="mp",
            enforce_eager=False,
            max_model_len=DSV4_DP4_POOL_MAX_MODEL_LEN,
            max_num_seqs=32,
            max_num_batched_tokens=DSV4_DP4_POOL_MAX_NUM_BATCHED_TOKENS,
            gpu_memory_utilization=0.9,
            enable_expert_parallel=True,
            quantization="ascend",
            tokenizer_mode="deepseek_v4",
            block_size=128,
            attention_config={"indexer_kv_dtype": "int8"},
            enable_prefix_caching=not use_pool,
            disable_log_stats=False,
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        )
        if use_pool:
            manager = stack.enter_context(
                SingleNodeMooncakeManager(_build_mooncake_config(), case_name)
            )
            stack.enter_context(patch.dict(os.environ, manager.server_envs))
            kwargs["kv_transfer_config"] = {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "backend": "mooncake",
                    "lookup_rpc_port": "1",
                    "use_layerwise": False,
                },
            }
        yield stack.enter_context(DPVllmRunner(DSV4_0731_MODEL, **kwargs))


@pytest.mark.e2e_model(DSV4_0731_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="prefix_caching,chunked_prefill,mixed_lengths",
    parallel="DP,TP,EP",
    deploy="pd_mix",
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
        "ASCEND_ENABLE_USE_FABRIC_MEM": "1",
        "ASCEND_AUTO_CONNECT": "1",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dsv4_mooncake_dp4_multiprefix() -> None:
    """Full-scale plan-N12 case: DP4xTP4 mooncake non-layerwise 128K multi-prefix.

    Two distinct-secret natural-language 128K prefixes each replay R1-R4 with
    one prompt copy per DP rank (object reuse scaled by replicas, DP
    replica-level hits, cross-prefix key isolation) and token-exact H3 at the
    plan's measured chunk shape. The Mooncake SSD offload pilot (plan 2.6/7)
    is a separate harness follow-up and is not claimed here.
    """
    tokenizer = AutoTokenizer.from_pretrained(DSV4_0731_MODEL, trust_remote_code=True)
    prefix_a = build_pool_accuracy_prompt(
        tokenizer, DSV4_DP4_POOL_PROMPT_TOKENS, DSV4_DP4_POOL_MAX_MODEL_LEN, secret="7391"
    )
    prefix_b = build_pool_accuracy_prompt(
        tokenizer, DSV4_DP4_POOL_PROMPT_TOKENS, DSV4_DP4_POOL_MAX_MODEL_LEN, secret="2846"
    )

    def accuracy(runner: Any) -> list[float]:
        # PR-layer secret-number check at the 128K bucket (plan 2.7).
        prompts = [prefix_a, prefix_b] * (DSV4_DP4_REPLICAS // 2)
        outputs = runner.generate_greedy(prompts, max_tokens=SECRET_ACC_MAX_TOKENS)
        assert len(outputs) == len(prompts)
        answers = [extract_pool_answer(text) for _, text in outputs]
        expected = ["7391", "2846"] * (DSV4_DP4_REPLICAS // 2)
        assert answers == expected, answers
        print(f"[kv_pool_acc] secret answers={answers}", flush=True)
        return [1.0]

    def hit(runner: Any) -> dict[str, Any]:
        # Each prefix replays R1-R4 with one prompt copy per DP rank
        # (H1/H2 replica-level hits) and misses on a disjoint random-key
        # prompt (cross-prefix key isolation). The exact hit-token
        # expectation comes from expected_hit_token_count with the 16384
        # lcm; verify the save granularity against the startup logs if this
        # assertion ever moves (plan 2.2).
        run_pool_hit_sequence(
            runner,
            block_size=128,
            prompt_tokens=DSV4_DP4_POOL_PROMPT_TOKENS,
            prompt_ids=prefix_a,
            replicas=DSV4_DP4_REPLICAS,
            lcm_block_size=DSV4_POOL_LCM_BLOCK_SIZE,
            answer_secret="7391",
            tag="nightly-dsv4-dp4-multiprefix-a",
        )
        summary = run_pool_hit_sequence(
            runner,
            block_size=128,
            prompt_tokens=DSV4_DP4_POOL_PROMPT_TOKENS,
            prompt_ids=prefix_b,
            miss_prompt_ids=build_pool_prompts(
                DSV4_DP4_POOL_PROMPT_TOKENS, seed_offset=1
            )[1],
            replicas=DSV4_DP4_REPLICAS,
            lcm_block_size=DSV4_POOL_LCM_BLOCK_SIZE,
            answer_secret="2846",
            tag="nightly-dsv4-dp4-multiprefix-b",
        )
        # run_pool_ab compares the A-leg hit outputs against the B leg's
        # single local-hit output, so hand over one representative replica.
        summary["hit_outputs"] = summary["hit_outputs"][:1]
        return summary

    run_pool_ab(
        case_name="nightly_dsv4_dp4_mooncake_multiprefix",
        build_leg=_dsv4_dp4_mooncake_leg,
        accuracy=accuracy,
        hit=hit,
        prompt_ids=prefix_b,
        perf_prompt_tokens=DSV4_DP4_POOL_PROMPT_TOKENS,
        perf_prompt_ids=prefix_b,
        acceptance_tolerance=0.0,
        perf_concurrency=32,
        perf_batch_requests=64,
        perf_sample_count=3,
        perf_warmup_count=1,
        perf_drift_count=0,
        perf_trim_count=0,
        perf_stability_policy="record",
        perf_gate=False,
    )


@contextmanager
def _dsv4_leg(use_pool: bool, case_name: str):
    with ExitStack() as stack:
        kwargs: dict[str, Any] = dict(
            max_model_len=DSV4_POOL_MAX_MODEL_LEN,
            max_num_seqs=4,
            max_num_batched_tokens=4096,
            dtype="auto",
            tensor_parallel_size=2,
            prefill_context_parallel_size=4,
            enable_expert_parallel=True,
            gpu_memory_utilization=0.9,
            quantization="ascend",
            tokenizer_mode="deepseek_v41",
            block_size=128,
            enforce_eager=False,
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
            enable_prefix_caching=not use_pool,
            disable_log_stats=False,
            speculative_config={
                "method": "dspark",
                "num_speculative_tokens": 5,
                "enable_adaptive_verification": False,
                "draft_sample_method": "greedy",
            },
            attention_config={"indexer_kv_dtype": "int8"},
            additional_config={
                "enable_prefill_mc2": True,
                "enable_pcp_o_proj_weight_sharding": False,
            },
        )
        if use_pool:
            manager = stack.enter_context(
                SingleNodeMemcacheManager(_build_memcache_config(), case_name)
            )
            stack.enter_context(patch.dict(os.environ, manager.server_envs))
            kwargs["kv_transfer_config"] = {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "backend": "memcache",
                    "lookup_rpc_port": "1",
                    "use_layerwise": False,
                },
            }
        yield stack.enter_context(VllmRunner(DSV4_DSPARK_MODEL, **kwargs))


@pytest.mark.e2e_model(DSV4_DSPARK_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dsa_pcp,dspark",
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
        "VLLM_BATCH_INVARIANT": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "2560",
        "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
        "CLOSE_MATMUL_K_SHIFT": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dsv4_memcache_dsa_pcp_dspark() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        DSV4_DSPARK_MODEL, trust_remote_code=True
    )
    prompt_ids = build_pool_accuracy_prompt(
        tokenizer, DSV4_POOL_PROMPT_TOKENS, DSV4_POOL_MAX_MODEL_LEN
    )

    def hit(runner: Any) -> dict[str, Any]:
        return run_pool_hit_sequence(
            runner,
            block_size=128,
            prompt_tokens=DSV4_POOL_PROMPT_TOKENS,
            prompt_ids=prompt_ids,
            lcm_block_size=DSV4_POOL_LCM_BLOCK_SIZE,
            check_output=False,
            answer_secret=POOL_SECRET,
            tag="nightly-dsv4-dsa-pcp-dspark",
        )

    run_pool_ab(
        case_name="nightly_dsv4_dsa_pcp_dspark_memcache",
        build_leg=_dsv4_leg,
        accuracy=_assert_dspark_accuracy,
        hit=hit,
        prompt_ids=prompt_ids,
        perf_prompt_tokens=DSV4_POOL_PERF_PROMPT_TOKENS,
        token_identity=False,
        **PERF_OPTIONS,
    )


@contextmanager
def _qwen_leg(use_pool: bool, case_name: str):
    with ExitStack() as stack:
        # Hybrid align requires the local prefix cache in both legs; the only
        # A/B difference is the pool connector.
        kwargs: dict[str, Any] = dict(
            dtype="bfloat16",
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
            max_model_len=QWEN_POOL_MAX_MODEL_LEN,
            max_num_seqs=32,
            max_num_batched_tokens=4096,
            gpu_memory_utilization=0.9,
            quantization="ascend",
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            disable_log_stats=False,
            enable_chunked_prefill=True,
            async_scheduling=True,
            seed=1024,
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": 3,
                "enforce_eager": True,
            },
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        )
        if use_pool:
            manager = stack.enter_context(
                SingleNodeMemcacheManager(_build_memcache_config(), case_name)
            )
            stack.enter_context(patch.dict(os.environ, manager.server_envs))
            kwargs["kv_transfer_config"] = {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "backend": "memcache",
                    "lookup_rpc_port": "1",
                    "use_layerwise": True,
                    "load_async": False,
                },
            }
        yield stack.enter_context(VllmRunner(QWEN38_MODEL, **kwargs))


@pytest.mark.e2e_model(QWEN38_MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="mtp,aclgraph,prefix_caching,chunked_prefill,long_sequence",
    parallel="TP",
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
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen38_memcache_mtp_align() -> None:
    """Plan-N7: Qwen3.8-27B-w8a8 TP2+MTP memcache layerwise 128K.

    Hybrid align keeps the local prefix cache in both legs, so the only A/B
    difference is the pool connector. The accuracy stage exercises the Mamba
    deferred boundary copy, checks a deterministic arithmetic answer, and
    answers the 128K secret-number prompt (plan 2.7); the hit and performance
    stages run one 128K prefix (lcm 896, H3, R4) and gate per plan 2.5.
    """
    tokenizer = AutoTokenizer.from_pretrained(QWEN38_MODEL, trust_remote_code=True)
    prompt_ids = build_pool_accuracy_prompt(
        tokenizer, QWEN_POOL_PROMPT_TOKENS, QWEN_POOL_MAX_MODEL_LEN
    )
    state: dict[str, int] = {}

    def accuracy(runner: Any) -> list[float]:
        cache_config = runner.model.llm_engine.vllm_config.cache_config
        block_size = cache_config.block_size
        state["block_size"] = block_size
        tokenizer = runner.model.get_tokenizer()
        prompts, params = [], []
        for index in range(8):
            tokens = tokenizer.encode(
                f"Request {index}: continue counting 1, 2, 3, 4. ",
                add_special_tokens=False,
            )
            prompt_len = block_size - 16 - index
            output_len = 64 + 8 * index
            assert prompt_len > 0
            assert prompt_len + output_len < QWEN_POOL_MAX_MODEL_LEN
            prompts.append(
                {"prompt_token_ids": (tokens * (prompt_len // len(tokens) + 1))[:prompt_len]}
            )
            params.append(
                SamplingParams(
                    temperature=0,
                    max_tokens=output_len,
                    ignore_eos=True,
                    detokenize=False,
                )
            )

        for _ in range(2):
            outputs = runner.model.generate(prompts, params, use_tqdm=False)
            assert len(outputs) == len(prompts)
            for output, sampling in zip(outputs, params, strict=True):
                assert output.finished
                assert len(output.outputs[0].token_ids) == sampling.max_tokens

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 4242 + 1717? Reply with only the integer."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        answer = runner.model.generate(
            [prompt], SamplingParams(temperature=0, max_tokens=32), use_tqdm=False
        )
        assert answer[0].outputs[0].text.strip() == "5959", answer

        secret_outputs = runner.generate_greedy(
            [prompt_ids], max_tokens=SECRET_ACC_MAX_TOKENS
        )
        secret_answer = extract_pool_answer(secret_outputs[0][1])
        assert secret_answer == POOL_SECRET, secret_answer
        print(f"[kv_pool_acc] secret answer={secret_answer}", flush=True)

        metrics = runner.model.get_metrics()
        drafts = sum(
            metric.value
            for metric in metrics
            if metric.name == "vllm:spec_decode_num_drafts"
        )
        assert drafts > 0, "MTP produced no draft tokens"
        rates = calculate_acceptance_per_pos(metrics, 3, Counter, Vector)
        print(f"[kv_pool_acc] acceptance per draft position: {rates}", flush=True)
        return rates

    def hit(runner: Any) -> dict[str, Any]:
        return run_pool_hit_sequence(
            runner,
            block_size=state["block_size"],
            prompt_tokens=QWEN_POOL_PROMPT_TOKENS,
            prompt_ids=prompt_ids,
            lcm_block_size=QWEN_POOL_LCM_BLOCK_SIZE,
            reset_local_cache=True,
            answer_secret=POOL_SECRET,
            tag="nightly-qwen38-mtp-align",
        )

    run_pool_ab(
        case_name="nightly_qwen38_mtp_align_memcache",
        build_leg=_qwen_leg,
        accuracy=accuracy,
        hit=hit,
        prompt_ids=prompt_ids,
        perf_prompt_tokens=QWEN_POOL_PERF_PROMPT_TOKENS,
        perf_prompt_ids=prompt_ids,
        perf_reset_local=True,
        acceptance_tolerance=0.03,
        perf_concurrency=32,
        perf_batch_requests=64,
        perf_sample_count=3,
        perf_warmup_count=1,
        perf_drift_count=0,
        perf_trim_count=0,
        perf_stability_policy="record",
        perf_gate=False,
    )

