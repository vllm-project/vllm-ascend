# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Nightly parity coverage for speculative-decoding refactors.

Each family is exercised in eager execution and full-decode ACL graph with
greedy sampling. Both paths must generate tokens with speculative acceptance,
and the target cache configuration must match. Exact token, acceptance, KV, and
performance comparisons for refactor-before/after are produced by
``spec_decode_refactor_benchmark.py`` from the baseline and candidate checkouts
using the same execution mode.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory

try:
    from vllm_ascend.worker.spec_decode_family import resolve_spec_decode_family
except ImportError:
    # The benchmark harness is also imported from the pre-refactor checkout.
    resolve_spec_decode_family = None

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Keep the graph parity window short enough that ACL Graph numerical drift does
# not turn this refactor gate into a long-horizon model-quality test.
MAX_TOKENS = 16


@dataclass(frozen=True)
class SpecDecodeCase:
    name: str
    model: str
    speculative_config: dict[str, Any]
    runner_kwargs: dict[str, Any]
    e2e_models: tuple[str, ...]
    expected_family: str


def _model_from_env(env_name: str, default: str) -> str:
    return os.environ.get(env_name, default)


MTP_CASE = SpecDecodeCase(
    name="mtp",
    model=_model_from_env("SPEC_DECODE_MTP_MODEL", "Qwen/Qwen3.5-4B"),
    speculative_config={"method": "qwen3_5_mtp", "num_speculative_tokens": 3},
    runner_kwargs={
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 1024,
        "max_num_seqs": len(PROMPTS),
        "gpu_memory_utilization": 0.8,
    },
    e2e_models=(_model_from_env("SPEC_DECODE_MTP_MODEL", "Qwen/Qwen3.5-4B"),),
    expected_family="mtp",
)

EAGLE3_CASE = SpecDecodeCase(
    name="eagle3",
    model=_model_from_env("SPEC_DECODE_EAGLE3_MODEL", "Qwen/Qwen3-0.6B"),
    speculative_config={
        "method": "eagle3",
        "model": _model_from_env(
            "SPEC_DECODE_EAGLE3_DRAFT_MODEL",
            "GavinLucky/SGLang-EAGLE3-Qwen3-0.6B-SpecForge",
        ),
        "num_speculative_tokens": 3,
    },
    runner_kwargs={
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 2048,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": len(PROMPTS),
        "gpu_memory_utilization": 0.8,
    },
    e2e_models=(
        _model_from_env("SPEC_DECODE_EAGLE3_MODEL", "Qwen/Qwen3-0.6B"),
        _model_from_env(
            "SPEC_DECODE_EAGLE3_DRAFT_MODEL",
            "GavinLucky/SGLang-EAGLE3-Qwen3-0.6B-SpecForge",
        ),
    ),
    expected_family="eagle",
)

DSPARK_CASE = SpecDecodeCase(
    name="dspark",
    model=_model_from_env("SPEC_DECODE_DSPARK_MODEL", "Qwen/Qwen3-0.6B"),
    speculative_config={
        "method": "dspark",
        "model": _model_from_env("SPEC_DECODE_DSPARK_DRAFT_MODEL", "yosefw/Qwen3-0.6B-DSpark"),
        "num_speculative_tokens": 4,
    },
    runner_kwargs={
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 2048,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": len(PROMPTS),
        "gpu_memory_utilization": 0.8,
    },
    e2e_models=(
        _model_from_env("SPEC_DECODE_DSPARK_MODEL", "Qwen/Qwen3-0.6B"),
        _model_from_env("SPEC_DECODE_DSPARK_DRAFT_MODEL", "yosefw/Qwen3-0.6B-DSpark"),
    ),
    expected_family="dspark",
)


@dataclass(frozen=True)
class RunResult:
    token_ids: tuple[tuple[int, ...], ...]
    acceptance_length: float
    acceptance_per_pos: tuple[float, ...]
    cache_signature: tuple[tuple[str, object], ...]
    elapsed_seconds: float
    output_tokens_per_second: float
    hbm_in_use_bytes: int
    family: str


def _cache_signature(vllm_config: Any) -> tuple[tuple[str, object], ...]:
    cache_config = getattr(vllm_config, "cache_config", None)
    fields = (
        "block_size",
        "cache_dtype",
        "enable_prefix_caching",
        "mamba_cache_mode",
        "num_gpu_blocks",
    )
    return tuple((field, getattr(cache_config, field, None)) for field in fields)


def _cache_layout(run_result: RunResult) -> tuple[tuple[str, object], ...]:
    """Return cache fields that must agree across eager and graph execution."""
    return tuple((field, value) for field, value in run_result.cache_signature if field != "num_gpu_blocks")


def _read_spec_metrics(vllm_model: Any, num_speculative_tokens: int) -> tuple[float, tuple[float, ...]]:
    num_drafts = 0
    num_accepted = 0
    accepted_per_pos = [0] * num_speculative_tokens
    for metric in vllm_model.get_metrics():
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos, value in enumerate(metric.values):
                accepted_per_pos[pos] += value

    assert num_drafts > 0
    assert sum(accepted_per_pos) == num_accepted
    acceptance_length = 1 + num_accepted / num_drafts
    rates = tuple(value / num_drafts for value in accepted_per_pos)
    return acceptance_length, rates


def _run(
    case: SpecDecodeCase,
    *,
    graph_mode: str,
    speculative: bool,
    verify_family: bool = False,
) -> RunResult:
    compilation_config = None
    enforce_eager = True
    if graph_mode == "full_decode_only":
        enforce_eager = False
        capture_size = len(PROMPTS) * (int(case.speculative_config["num_speculative_tokens"]) + 1)
        compilation_config = CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[capture_size],
        )

    speculative_config = case.speculative_config if speculative else None
    with VllmRunner(
        case.model,
        disable_log_stats=False,
        distributed_executor_backend="mp",
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,
        compilation_config=compilation_config,
        speculative_config=speculative_config,
        **case.runner_kwargs,
    ) as vllm_model:
        tokenizer = vllm_model.model.get_tokenizer()
        prompt_ids = [
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                add_special_tokens=False,
            )
            for prompt in PROMPTS
        ]
        sampling_params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

        start = time.perf_counter()
        outputs = vllm_model.model.generate(prompt_ids, sampling_params)
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start

        token_ids = tuple(tuple(output.outputs[0].token_ids) for output in outputs)
        output_token_count = sum(len(tokens) for tokens in token_ids)
        free_hbm_bytes, total_hbm_bytes = torch.npu.mem_get_info()
        hbm_in_use_bytes = int(total_hbm_bytes - free_hbm_bytes)
        cache_signature = _cache_signature(vllm_model.model.llm_engine.vllm_config)
        family = case.expected_family
        if verify_family:
            assert resolve_spec_decode_family is not None
            family = resolve_spec_decode_family(vllm_model.model.llm_engine.vllm_config).value

        if speculative:
            acceptance_length, acceptance_per_pos = _read_spec_metrics(
                vllm_model.model,
                int(case.speculative_config["num_speculative_tokens"]),
            )
        else:
            acceptance_length = 1.0
            acceptance_per_pos = ()

    cleanup_dist_env_and_memory()
    return RunResult(
        token_ids=token_ids,
        acceptance_length=acceptance_length,
        acceptance_per_pos=acceptance_per_pos,
        cache_signature=cache_signature,
        elapsed_seconds=elapsed,
        output_tokens_per_second=output_token_count / elapsed,
        hbm_in_use_bytes=hbm_in_use_bytes,
        family=family,
    )


def _assert_family_parity(case: SpecDecodeCase) -> None:
    eager = _run(case, graph_mode="eager", speculative=True, verify_family=True)
    graph = _run(case, graph_mode="full_decode_only", speculative=True, verify_family=True)

    assert _cache_layout(eager) == _cache_layout(graph)
    assert eager.family == graph.family == case.expected_family
    assert all(eager.token_ids)
    assert all(graph.token_ids)
    assert eager.acceptance_length > 1.0
    assert graph.acceptance_length > 1.0

    print(
        f"{case.name}: eager={eager.elapsed_seconds:.3f}s "
        f"({eager.output_tokens_per_second:.2f} tok/s), "
        f"graph={graph.elapsed_seconds:.3f}s "
        f"({graph.output_tokens_per_second:.2f} tok/s), "
        f"eager_acceptance={eager.acceptance_length:.4f}, "
        f"graph_acceptance={graph.acceptance_length:.4f}, "
        f"eager_hbm={eager.hbm_in_use_bytes}, graph_hbm={graph.hbm_in_use_bytes}"
    )


@pytest.mark.e2e_model(*MTP_CASE.e2e_models)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="mtp,spec_decode,aclgraph",
    parallel="TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
def test_mtp_refactor_parity() -> None:
    _assert_family_parity(MTP_CASE)


@pytest.mark.e2e_model(*EAGLE3_CASE.e2e_models)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="eagle3,spec_decode,aclgraph",
    parallel="TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
def test_eagle3_refactor_parity() -> None:
    _assert_family_parity(EAGLE3_CASE)


@pytest.mark.e2e_model(*DSPARK_CASE.e2e_models)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="dspark,spec_decode,aclgraph",
    parallel="TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
def test_dspark_refactor_parity() -> None:
    _assert_family_parity(DSPARK_CASE)
