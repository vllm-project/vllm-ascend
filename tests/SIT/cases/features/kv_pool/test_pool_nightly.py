# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Nightly KV-pool cases for A3 single-host deployments.

The dsv4-mooncake-multiprefix, dsv4-memcache-dsa-pcp-dspark and
qwen38-memcache-mtp-align cases replay their guarded workloads at
short-input scale. ``test_dsv4_mooncake_dp4_multiprefix`` is the
full-scale plan-N12 case: DP4xTP4 mooncake non-layerwise 32K
multi-prefix with token-exact A/B; performance is recorded only, and
the Mooncake SSD offload pilot (plan 2.6/7) is a separate harness
follow-up that this module does not claim.
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
from tests.e2e.pull_request.four_card.model_runner_v2.test_qwen38_pp_mtp_prefix_cache import (
    MambaAlignWorker,
)
from tests.e2e.pull_request.kv_pool_watch import (
    POOL_SECRET,
    build_pool_accuracy_prompt,
    build_pool_prompts,
    run_pool_ab,
    run_pool_hit_sequence,
)
from tests.e2e.pull_request.utils import PROMPTS_SHORT

os.environ["ASCEND_ENABLE_USE_FABRIC_MEM"] = "1"
os.environ["ASCEND_AUTO_CONNECT"] = "1"
os.environ["ASCEND_GLOBAL_RESOURCE_CONFIG"] = '{"fabric_memory.max_capacity": "2"}'
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

DSV4_DSPARK_MODEL = os.environ.get(
    "DEEPSEEK_V41_DSPARK_MODEL_PATH",
    "Eco-Tech/DeepSeek-V4.1-Flash-w8a8",
)
DSV4_MTP_MODEL = os.environ.get(
    "DEEPSEEK_V4_MTP_MODEL_PATH",
    "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
)
DSV4_0731_MODEL = os.environ.get(
    "DEEPSEEK_V4_0731_MODEL_PATH",
    "Eco-Tech/DeepSeek-V4-Flash-0731-w8a8",
)
QWEN38_MODEL = os.environ.get("QWEN38_DENSE_MODEL", "Qwen/Qwen3.8-27B")

MEMCACHE_ENV = {
    "meta": {"ock.mmc.log_level": "error"},
    "local": {
        "ock.mmc.log_level": "error",
        "ock.mmc.local_service.world_size": 256,
        "ock.mmc.local_service.protocol": "device_sdma",
        "ock.mmc.local_service.dram.size": "2GB",
    },
}

DSV4_POOL_LCM_BLOCK_SIZE = 16384
DSV4_POOL_PROMPT_TOKENS = DSV4_POOL_LCM_BLOCK_SIZE + 4
DSV4_POOL_PERF_PROMPT_TOKENS = (
    DSV4_POOL_LCM_BLOCK_SIZE + DSV4_POOL_LCM_BLOCK_SIZE // 9
)
DSV4_POOL_MAX_MODEL_LEN = DSV4_POOL_PERF_PROMPT_TOKENS + 128
QWEN_POOL_LCM_BLOCK_SIZE = 896
# Two LCM blocks plus a short tail retain one reusable block after the final
# block trim. The 8.75K performance prompt preserves a 90% hit state.
QWEN_POOL_PROMPT_TOKENS = 2 * QWEN_POOL_LCM_BLOCK_SIZE + 4
QWEN_POOL_PERF_PROMPT_TOKENS = 10 * QWEN_POOL_LCM_BLOCK_SIZE
QWEN_POOL_MAX_MODEL_LEN = QWEN_POOL_PERF_PROMPT_TOKENS + 128

# Full-scale plan-N12 32K multi-prefix load: two lcm blocks minus an
# 8-token tail keeps the second block from transferring, and the plan's
# measured H3 chunk shape passes max_num_batched_tokens=28672 (plan 2.1).
DSV4_DP4_POOL_PROMPT_TOKENS = 2 * DSV4_POOL_LCM_BLOCK_SIZE - 8
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
def _dsv4_mooncake_leg(use_pool: bool, case_name: str):
    with ExitStack() as stack:
        kwargs: dict[str, Any] = dict(
            data_parallel_size=2,
            tensor_parallel_size=4,
            distributed_executor_backend="mp",
            enforce_eager=True,
            max_model_len=DSV4_POOL_MAX_MODEL_LEN,
            max_num_seqs=4,
            max_num_batched_tokens=4096,
            gpu_memory_utilization=0.9,
            enable_expert_parallel=True,
            quantization="ascend",
            tokenizer_mode="deepseek_v4",
            block_size=128,
            attention_config={"indexer_kv_dtype": "int8"},
            enable_prefix_caching=not use_pool,
            disable_log_stats=False,
            speculative_config={"method": "mtp", "num_speculative_tokens": 1},
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
                    "use_layerwise": True,
                },
            }
        yield stack.enter_context(DPVllmRunner(DSV4_MTP_MODEL, **kwargs))


@pytest.mark.e2e_model(DSV4_MTP_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,prefix_caching,chunked_prefill,mixed_lengths",
    parallel="DP,TP,EP",
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
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "ASCEND_ENABLE_USE_FABRIC_MEM": "1",
        "ASCEND_AUTO_CONNECT": "1",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dsv4_mooncake_multiprefix() -> None:
    tokenizer = AutoTokenizer.from_pretrained(DSV4_MTP_MODEL, trust_remote_code=True)
    prefix_a = build_pool_accuracy_prompt(
        tokenizer, DSV4_POOL_PROMPT_TOKENS, DSV4_POOL_MAX_MODEL_LEN, secret="7391"
    )
    prefix_b = build_pool_accuracy_prompt(
        tokenizer, DSV4_POOL_PROMPT_TOKENS, DSV4_POOL_MAX_MODEL_LEN, secret="2846"
    )

    def accuracy(runner: Any) -> None:
        outputs = runner.generate_greedy(PROMPTS_SHORT, max_tokens=64)
        assert len(outputs) == len(PROMPTS_SHORT)
        assert all(ids and text.strip() for ids, text in outputs)

    def hit(runner: Any) -> dict[str, Any]:
        run_pool_hit_sequence(
            runner,
            block_size=128,
            prompt_tokens=DSV4_POOL_PROMPT_TOKENS,
            prompt_ids=prefix_a,
            replicas=4,
            lcm_block_size=DSV4_POOL_LCM_BLOCK_SIZE,
            answer_secret="7391",
            tag="nightly-dsv4-multiprefix-a",
        )
        summary = run_pool_hit_sequence(
            runner,
            block_size=128,
            prompt_tokens=DSV4_POOL_PROMPT_TOKENS,
            prompt_ids=prefix_b,
            miss_prompt_ids=build_pool_prompts(
                DSV4_POOL_PROMPT_TOKENS, seed_offset=1
            )[1],
            replicas=4,
            lcm_block_size=DSV4_POOL_LCM_BLOCK_SIZE,
            answer_secret="2846",
            tag="nightly-dsv4-multiprefix-b",
        )
        # run_pool_ab compares the A-leg hit outputs against the B leg's
        # single local-hit output. Per-replica cold/hit identity was already
        # asserted inside run_pool_hit_sequence, so hand over one
        # representative replica.
        summary["hit_outputs"] = summary["hit_outputs"][:1]
        return summary

    run_pool_ab(
        case_name="nightly_dsv4_multiprefix_mooncake",
        build_leg=_dsv4_mooncake_leg,
        accuracy=accuracy,
        hit=hit,
        prompt_ids=prefix_b,
        perf_prompt_tokens=DSV4_POOL_PERF_PROMPT_TOKENS,
        perf_sample_count=10,
        perf_concurrency=1,
        perf_stability_policy="record",
        perf_gate=False,
    )


@contextmanager
def _dsv4_dp4_mooncake_leg(use_pool: bool, case_name: str):
    with ExitStack() as stack:
        kwargs: dict[str, Any] = dict(
            data_parallel_size=4,
            tensor_parallel_size=4,
            distributed_executor_backend="mp",
            enforce_eager=True,
            max_model_len=DSV4_DP4_POOL_MAX_MODEL_LEN,
            max_num_seqs=4,
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
    """Full-scale plan-N12 case: DP4xTP4 mooncake non-layerwise 32K multi-prefix.

    Two distinct-secret natural-language prefixes each replay R1-R4 with one
    prompt copy per DP rank (object reuse scaled by replicas, DP replica-level
    hits, cross-prefix key isolation) and token-exact H3 at the plan's measured
    chunk shape. The Mooncake SSD offload pilot (plan 2.6/7) is a separate
    harness follow-up and is not claimed here.
    """
    tokenizer = AutoTokenizer.from_pretrained(DSV4_0731_MODEL, trust_remote_code=True)
    prefix_a = build_pool_accuracy_prompt(
        tokenizer, DSV4_DP4_POOL_PROMPT_TOKENS, DSV4_DP4_POOL_MAX_MODEL_LEN, secret="7391"
    )
    prefix_b = build_pool_accuracy_prompt(
        tokenizer, DSV4_DP4_POOL_PROMPT_TOKENS, DSV4_DP4_POOL_MAX_MODEL_LEN, secret="2846"
    )

    def accuracy(runner: Any) -> None:
        outputs = runner.generate_greedy(PROMPTS_SHORT, max_tokens=64)
        assert len(outputs) == len(PROMPTS_SHORT)
        assert all(ids and text.strip() for ids, text in outputs)

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
        **PERF_OPTIONS,
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
        kwargs: dict[str, Any] = dict(
            dtype="bfloat16",
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            distributed_executor_backend="mp",
            worker_extension_cls=(
                f"{MambaAlignWorker.__module__}.MambaAlignWorker"
            ),
            max_model_len=QWEN_POOL_MAX_MODEL_LEN,
            max_num_seqs=4,
            max_num_batched_tokens=4096,
            gpu_memory_utilization=0.9,
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
                    "use_layerwise": False,
                    "load_async": False,
                },
            }
        yield stack.enter_context(VllmRunner(QWEN38_MODEL, **kwargs))


@pytest.mark.e2e_model(QWEN38_MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="mtp,aclgraph,prefix_caching,chunked_prefill,mixed_lengths",
    parallel="PP,TP",
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
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen38_memcache_mtp_align() -> None:
    tokenizer = AutoTokenizer.from_pretrained(QWEN38_MODEL, trust_remote_code=True)
    prompt_ids = build_pool_accuracy_prompt(
        tokenizer, QWEN_POOL_PROMPT_TOKENS, QWEN_POOL_MAX_MODEL_LEN
    )
    state: dict[str, int] = {}

    def accuracy(runner: Any) -> None:
        block_sizes = runner.model.collective_rpc("install_align_probe")
        block_size = next(size for size in block_sizes if size is not None)
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
            probes = [
                probe
                for probe in runner.model.collective_rpc("read_align_probe")
                if probe is not None
            ]
            assert len(probes) == 2
            for copied, mismatched in probes:
                assert copied, "no deferred Mamba boundary copy was exercised"
                assert not mismatched, "deferred copy used another request's rows"

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
        perf_reset_local=True,
        **PERF_OPTIONS,
    )

