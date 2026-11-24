# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import json
import os
import time
from contextlib import ExitStack, contextmanager
from typing import Any
from pathlib import Path
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

DSV4_DSPARK_MODEL = os.environ.get(
    "DEEPSEEK_V41_DSPARK_MODEL_PATH",
    "Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8",
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

QWEN_POOL_PROMPT_TOKENS = 146 * QWEN_POOL_LCM_BLOCK_SIZE + 4
QWEN_POOL_PERF_PROMPT_TOKENS = QWEN_POOL_PROMPT_TOKENS
QWEN_POOL_MAX_MODEL_LEN = QWEN_POOL_PERF_PROMPT_TOKENS + 128


DSV4_DP4_POOL_PROMPT_TOKENS = 8 * DSV4_POOL_LCM_BLOCK_SIZE + 4
# 取 lcm 整块（16384）：prompt 131,076 = 8 x 16384 + 4，尾块形状与 N3 一致；
# 28672 单步激活过大，131,204 单请求需 30.67 GiB KV、可用仅 29.31 GiB，
# 引擎启动即报 KV 预算不足。
DSV4_DP4_POOL_MAX_NUM_BATCHED_TOKENS = 16384
DSV4_DP4_POOL_MAX_MODEL_LEN = DSV4_DP4_POOL_PROMPT_TOKENS + 128
# One prompt copy per DP rank; expected hit tokens scale by this factor.
DSV4_DP4_REPLICAS = 4

N3_POOL_LCM_BLOCK_SIZE = 16384
N3_POOL_PROMPT_TOKENS = 4 * N3_POOL_LCM_BLOCK_SIZE + 4
N3_POOL_MAX_MODEL_LEN = N3_POOL_PROMPT_TOKENS + 128
N3_POOL_PERF_PROMPT_TOKENS = N3_POOL_PROMPT_TOKENS
N3_DP4_REPLICAS = 4

# PR-layer secret-number prompts answer with a short numeric continuation.
SECRET_ACC_MAX_TOKENS = 32


def _diag_model(tag: str, model: str) -> None:
    """Print model provenance once per case; CI runners are unreachable."""
    env = {
        name: os.environ.get(name, "")
        for name in (
            "VLLM_USE_MODELSCOPE",
            "MODELSCOPE_CACHE",
            "HF_HOME",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        )
    }
    print(f"[kv_diag] {tag} model={model!r} env={env}", flush=True)
    path = Path(model)
    if not path.is_dir():
        try:
            from modelscope import snapshot_download as ms_snapshot_download

            path = Path(ms_snapshot_download(model, local_files_only=True))
            print(f"[kv_diag] {tag} modelscope local snapshot={path}", flush=True)
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            print(
                f"[kv_diag] {tag} not a local dir and no local snapshot: {exc!r}",
                flush=True,
            )
            return
    shards = sorted(path.glob("*.safetensors"))
    sized = sorted((item.stat().st_size, item.name) for item in shards)
    print(
        f"[kv_diag] {tag} dir={path} shards={len(shards)} "
        f"total_gib={sum(size for size, _ in sized) / 2**30:.2f}",
        flush=True,
    )
    for size, name in sized[:3]:
        print(f"[kv_diag] {tag} smallest_shard {name} bytes={size}", flush=True)
    present = {str(item.relative_to(path)) for item in path.rglob("*.safetensors")}
    for index in sorted(path.glob("*.index.json")):
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        indexed = set(weight_map.values())
        missing = sorted(indexed - present)
        print(
            f"[kv_diag] {tag} {index.name} files={len(indexed)} "
            f"missing={len(missing)} {missing[:4]}",
            flush=True,
        )
    config_path = path / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        keys = (
            "architectures",
            "model_type",
            "torch_dtype",
            "num_hidden_layers",
            "num_nextn_predict_layers",
            "max_position_embeddings",
            "quantization_config",
        )
        print(
            f"[kv_diag] {tag} config={ {key: config.get(key) for key in keys} }",
            flush=True,
        )


def _diag_leg(
    case_name: str,
    use_pool: bool,
    phase: str,
    started: float | None = None,
) -> None:
    elapsed = "" if started is None else f" elapsed_s={time.perf_counter() - started:.1f}"
    print(f"[kv_diag] {case_name} leg={'A' if use_pool else 'B'} {phase}{elapsed}", flush=True)


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


@contextmanager
def _dsv4_memcache_64k_leg(use_pool: bool, case_name: str):
    with ExitStack() as stack:
        _diag_leg(case_name, use_pool, "setup start")
        kwargs: dict[str, Any] = dict(
            data_parallel_size=4,
            tensor_parallel_size=4,
            # 单机混部按 DeepSeek-V4.1-Flash.md 5.1 A3 配置走图模式，并和 5.1 一样
            # 关闭 swa_bounded_replay。本用例刻意不开 layerwise：layerwise 的
            # AscendStoreConnector 会把 FULL_DECODE_ONLY 覆盖成 PIECEWISE，实测
            # hybrid 长 prompt 最后一个不足 block 的尾块进不了 decode（输出从尾块
            # 边界续写）；而且尾块保存 save_unaligned_tail 也只支持非 layerwise。
            # dsv4.1-p.sh 是 PD 分离的 P 节点，eager+layerwise 那套不适用于单机混部。
            enforce_eager=False,
            max_model_len=N3_POOL_MAX_MODEL_LEN,
            max_num_seqs=16,
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
            swa_bounded_replay=False,
            engram_config={"cpu_offload": True, "dp_shared_memory": True},
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
        _diag_leg(case_name, use_pool, "engine_init start")
        _engine_started = time.perf_counter()
        runner = stack.enter_context(DPVllmRunner(DSV4_DSPARK_MODEL, **kwargs))
        _diag_leg(case_name, use_pool, "engine_init done", _engine_started)
        yield runner


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
def test_dsv4_memcache_colocation_64k() -> None:
    _diag_model("n3", DSV4_DSPARK_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(
        DSV4_DSPARK_MODEL, trust_remote_code=True
    )
    prefix_a = build_pool_accuracy_prompt(
        tokenizer, N3_POOL_PROMPT_TOKENS, N3_POOL_MAX_MODEL_LEN, secret="7391"
    )

    def accuracy(runner: Any) -> list[float]:
        outputs = runner.generate_greedy(
            [prefix_a], max_tokens=SECRET_ACC_MAX_TOKENS
        )
        answer = extract_pool_answer(outputs[0][1])
        assert answer == "7391", answer
        print(f"[kv_pool_acc] secret answer={answer}", flush=True)
        return [1.0]

    def hit(runner: Any) -> dict[str, Any]:
        summary = run_pool_hit_sequence(
            runner,
            block_size=128,
            prompt_tokens=N3_POOL_PROMPT_TOKENS,
            prompt_ids=prefix_a,
            replicas=N3_DP4_REPLICAS,
            lcm_block_size=N3_POOL_LCM_BLOCK_SIZE,
            answer_secret="7391",
            tag="nightly-n3-memcache-64k",
        )
        # run_pool_ab compares one representative replica against the B leg's
        # single local-hit output; per-replica identity already asserted.
        summary["hit_outputs"] = summary["hit_outputs"][:1]
        return summary

    run_pool_ab(
        case_name="nightly_n3_dsv4_memcache_colocation_64k",
        build_leg=_dsv4_memcache_64k_leg,
        accuracy=accuracy,
        hit=hit,
        prompt_ids=prefix_a,
        perf_prompt_tokens=N3_POOL_PERF_PROMPT_TOKENS,
        perf_prompt_ids=prefix_a,
        token_identity=True,
        acceptance_ratio=None,
        perf_concurrency=4,
        perf_batch_requests=8,
        perf_sample_count=3,
        perf_warmup_count=1,
        perf_gate=True,
    )


@contextmanager
def _dsv4_dp4_mooncake_leg(use_pool: bool, case_name: str):
    with ExitStack() as stack:
        _diag_leg(case_name, use_pool, "setup start")
        kwargs: dict[str, Any] = dict(
            data_parallel_size=4,
            tensor_parallel_size=4,
            distributed_executor_backend="mp",
            enforce_eager=False,
            max_model_len=DSV4_DP4_POOL_MAX_MODEL_LEN,
            max_num_seqs=32,
            max_num_batched_tokens=DSV4_DP4_POOL_MAX_NUM_BATCHED_TOKENS,
            gpu_memory_utilization=0.94,
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
        _diag_leg(case_name, use_pool, "engine_init start")
        _engine_started = time.perf_counter()
        runner = stack.enter_context(DPVllmRunner(DSV4_0731_MODEL, **kwargs))
        _diag_leg(case_name, use_pool, "engine_init done", _engine_started)
        yield runner


@pytest.mark.e2e_model(DSV4_0731_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="prefix_caching,chunked_prefill,mixed_lengths",
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
        "ASCEND_GLOBAL_RESOURCE_CONFIG": '{"fabric_memory.max_capacity": "2", "comm_resource_config.protocol_desc":["hccs:device"]}',
        "ASCEND_ENABLE_USE_FABRIC_MEM": "1",
        "ASCEND_AUTO_CONNECT": "1",
        "ASCEND_LOCAL_COMM_RES": '{"version":"1.2"}'
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dsv4_mooncake_dp4_colocation() -> None:
    _diag_model("dp4", DSV4_0731_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(DSV4_0731_MODEL, trust_remote_code=True)
    prefix_a = build_pool_accuracy_prompt(
        tokenizer, DSV4_DP4_POOL_PROMPT_TOKENS, DSV4_DP4_POOL_MAX_MODEL_LEN, secret="7391"
    )

    def accuracy(runner: Any) -> list[float]:
        prompts = [prefix_a] * DSV4_DP4_REPLICAS
        outputs = runner.generate_greedy(prompts, max_tokens=SECRET_ACC_MAX_TOKENS)
        assert len(outputs) == len(prompts)
        answers = [extract_pool_answer(text) for _, text in outputs]
        expected = ["7391"] * DSV4_DP4_REPLICAS
        assert answers == expected, answers
        print(f"[kv_pool_acc] secret answers={answers}", flush=True)
        return [1.0]

    def hit(runner: Any) -> dict[str, Any]:
        summary = run_pool_hit_sequence(
            runner,
            block_size=128,
            prompt_tokens=DSV4_DP4_POOL_PROMPT_TOKENS,
            prompt_ids=prefix_a,
            replicas=DSV4_DP4_REPLICAS,
            lcm_block_size=DSV4_POOL_LCM_BLOCK_SIZE,
            answer_secret="7391",
            tag="nightly-dsv4-dp4-colocation",
        )
        summary["hit_outputs"] = summary["hit_outputs"][:1]
        return summary

    run_pool_ab(
        case_name="nightly_dsv4_dp4_mooncake_colocation",
        build_leg=_dsv4_dp4_mooncake_leg,
        accuracy=accuracy,
        hit=hit,
        prompt_ids=prefix_a,
        perf_prompt_tokens=DSV4_DP4_POOL_PROMPT_TOKENS,
        perf_prompt_ids=prefix_a,
        acceptance_ratio=None,
        perf_concurrency=4,
        perf_batch_requests=8,
        perf_sample_count=3,
        perf_warmup_count=1,
        perf_gate=True,
    )


@contextmanager
def _qwen_leg(use_pool: bool, case_name: str):
    with ExitStack() as stack:
        _diag_leg(case_name, use_pool, "setup start")
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
                    "use_layerwise": False,
                    "load_async": False,
                },
            }
        _diag_leg(case_name, use_pool, "engine_init start")
        _engine_started = time.perf_counter()
        runner = stack.enter_context(VllmRunner(QWEN38_MODEL, **kwargs))
        _diag_leg(case_name, use_pool, "engine_init done", _engine_started)
        yield runner


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
    _diag_model("qwen38", QWEN38_MODEL)
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
        acceptance_ratio=0.97,
        perf_concurrency=4,
        perf_batch_requests=8,
        perf_sample_count=3,
        perf_warmup_count=1,
        perf_gate=True,
    )

