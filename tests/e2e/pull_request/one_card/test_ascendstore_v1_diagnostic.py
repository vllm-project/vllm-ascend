# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TEMPORARY text-divergence experiment. Remove with its helper, UT and CI route after diagnosis."""

import asyncio
import json
import os
import shlex
import shutil
from contextlib import ExitStack
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_ports_list

from tests.e2e.common.kv_pool.config import MooncakeKVPoolConfig
from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.ascendstore_v1_benchmark import (
    BLOCK_SIZE,
    CONNECTORS,
    MODEL,
    PROMPT_TOKEN_COUNT,
    REUSABLE_TOKEN_COUNT,
    _read_serving_counters,
    _wait_for_prompt_tokens,
    call_worker_rpc,
    reset_local_prefix_cache,
    write_report,
)
from tests.e2e.pull_request.one_card.ascendstore_v1_diagnostic import compare_phases, token_batch_context

REQUEST_COUNT = 8
OUTPUT_TOKEN_COUNT = 32
REPEATS = 2
TARGET_REQUEST = 6
POOL_MEMORY_BYTES = 32 << 30
LOCAL_BUFFER_BYTES = 1 << 30
REQUEST_TIMEOUT_SECONDS = 600
WORKER_EXTENSION = "tests.e2e.pull_request.one_card.ascendstore_v1_diagnostic.AscendStoreDiagnosticWorkerExtension"
# Frozen from the earlier eight-request, 32-output-token report, not the newer performance workload.
EXPECTED_PROMPT_HASHES = (
    "2f61840793eaba0085b8b838b9c88dada7e44094d67fb5259c6e2829b10da628",
    "17c2952ae04d06340e38ec224f26b75ccca37514d695777bd8a350edffbbdf6d",
    "59aa4be2d043fea4d5139b6ecaf54a9ddd738e106ea89d195e669fa2bd4bad7e",
    "3da4f74abb43b39bce1736c47f2865a3cbb6212eec5dcf94ccf9af1185fb639f",
    "cf78330b5bf45cbb6493d76da69ed97c9190fac24a3c8aff1428760906faecb6",
    "16c4efca2d6d12743b2a5f18d56e276f7e70c67c2d668c3c20480cbbe6b9525d",
    "3582984ae2c543b681dbc6984a6e5dc69f131302f059ded9947d1ca603b6036e",
    "48976fb9ea83dfeaa044e6f217a8734f59f88ff8f827126c7397930c7c11131e",
)

pytestmark = pytest.mark.e2e_model(MODEL)


def test_temporary_ascendstore_text_divergence(tmp_path: Path) -> None:
    assert shutil.which("mooncake_master") is not None, "The diagnostic requires a private Mooncake master"
    model_path = maybe_model_redirect(MODEL)
    tokenizer = get_tokenizer(model_path)
    document = tokenizer.encode(
        "External KV caches retain attention keys and values for previously computed token prefixes. "
        "A request can restore those values instead of recomputing the same document. ",
        add_special_tokens=False,
    )
    question = tokenizer.encode("\nSummarize the document in one sentence.\n", add_special_tokens=False)
    prompts = []
    for index in range(REQUEST_COUNT + 1):
        name = f"benchmark-{index}" if index < REQUEST_COUNT else "warmup-0"
        header = tokenizer.encode(f"Document {name}.\n", add_special_tokens=False)
        remaining = PROMPT_TOKEN_COUNT - len(header) - len(question)
        assert document and remaining > 0
        body = (document * ((remaining + len(document) - 1) // len(document)))[:remaining]
        prompts.append(header + body + question)
    inputs, warmup = prompts[:-1], prompts[-1:]
    hashes = [sha256(json.dumps(prompt).encode()).hexdigest() for prompt in inputs]
    assert tuple(hashes) == EXPECTED_PROMPT_HASHES, "Tokenizer/input drift: this is not the original workload"
    assert len({tuple(prompt[:BLOCK_SIZE]) for prompt in prompts}) == len(prompts)
    report: dict[str, Any] = {
        "model": MODEL,
        "vllm_package_version": version("vllm"),
        "prompt_sha256": hashes,
        "prompt_tokens": PROMPT_TOKEN_COUNT,
        "output_tokens": OUTPUT_TOKEN_COUNT,
        "target_request_index": TARGET_REQUEST,
        "status": "incomplete",
        "scope": "Current code, old inputs; not a replay of the old binaries or a performance benchmark",
        "limitations": [
            "Client submission order does not force admission/model-row order; use the actual batch trace",
            "Nonstream responses and optional logprobs differ from the original streaming benchmark",
            "Returned normalized logprobs are not raw logits; only compare scores at the first divergence",
            "No KV byte checksum: small logprob gaps alone cannot establish lossless transfer",
            "A finite run with no divergence is not proof of determinism or strict correctness",
        ],
        "runs": [],
        "phases": [],
    }
    try:
        for connector in (("none", "", ""), *CONNECTORS):
            run_path = tmp_path / connector[0]
            run_path.mkdir()
            _exercise_implementation(run_path, model_path, inputs, warmup, connector, report)
        report["comparisons"] = compare_phases(report["phases"])
        report["status"] = (
            "divergence_observed" if any(item["differences"] for item in report["comparisons"]) else "not_reproduced"
        )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_report(tmp_path / "ascendstore-text-diagnostic.json", "ASCENDSTORE_TEXT_DIAGNOSTIC", report)


def _exercise_implementation(
    tmp_path: Path,
    model_path: str,
    inputs: list[list[int]],
    warmup: list[list[int]],
    connector: tuple[str, str, str],
    report: dict[str, Any],
) -> None:
    implementation, connector_name, connector_module = connector
    master_port, metrics_port, lookup_port, server_port = get_open_ports_list(4)
    server_args = shlex.split(f"""
        --host 127.0.0.1 --port {server_port} --served-model-name {MODEL}
        --dtype bfloat16 --max-model-len 16384 --max-num-batched-tokens 16384
        --max-num-seqs {REQUEST_COUNT} --block-size {BLOCK_SIZE}
        --tensor-parallel-size 1 --pipeline-parallel-size 1 --gpu-memory-utilization 0.8
        --enable-prefix-caching --prefix-caching-hash-algo sha256 --no-enable-chunked-prefill
        --enforce-eager --no-async-scheduling --seed 42
    """) + ["--worker-extension-cls", WORKER_EXTENSION]
    repository_root = str(Path(__file__).resolve().parents[4])
    server_envs = {
        "PYTHONPATH": os.pathsep.join(filter(None, (repository_root, os.environ.get("PYTHONPATH", "")))),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_KV_CACHE_LAYOUT": "NHD",
        "VLLM_SERVER_DEV_MODE": "1",
    }
    with ExitStack() as stack:
        pool = None
        if implementation != "none":
            pool_config = MooncakeKVPoolConfig(
                config={
                    "metadata_server": "P2PHANDSHAKE",
                    "protocol": "ascend",
                    "device_name": "",
                    "global_segment_size": POOL_MEMORY_BYTES,
                    "local_buffer_size": LOCAL_BUFFER_BYTES,
                },
                master_port=master_port,
                metrics_port=metrics_port,
            )
            # Native SDK imports belong to the test process on CI, never to the trace-helper import path.
            pool_runtime = import_module("tests.e2e.common.single_node.kv_pool_runtime")
            pool = stack.enter_context(pool_runtime.SingleNodeMooncakeManager(pool_config, tmp_path.name))
            server_envs.update(pool.server_envs)
            transfer_config = {
                "kv_connector": connector_name,
                "kv_connector_module_path": connector_module,
                "kv_role": "kv_both",
                "kv_load_failure_policy": "fail",
                "kv_connector_extra_config": {
                    "backend": "mooncake",
                    "lookup_rpc_port": lookup_port,
                    "use_layerwise": False,
                    "load_async": False,
                    "save_decode_cache": False,
                    "discard_partial_chunks": True,
                },
            }
            server_args += ["--kv-transfer-config", json.dumps(transfer_config)]
        report["runs"].append({"implementation": implementation, "server_args": server_args})
        server = stack.enter_context(
            RemoteOpenAIServer(
                model_path,
                server_args,
                server_host="127.0.0.1",
                server_port=server_port,
                auto_port=False,
                env_dict=server_envs,
            )
        )
        warmup_phase: dict[str, Any] = {"id": f"{implementation}-warmup"}
        _run_phase(server, warmup, [0], 1, warmup_phase, instrumented=False, external=False)
        natural_order = list(range(REQUEST_COUNT))
        cases = [("control-before", natural_order, REQUEST_COUNT), ("c1", natural_order, 1)]
        cases.append(("c8-natural", natural_order, REQUEST_COUNT))
        for slot in (1, 6, 8):
            order = [index for index in natural_order if index != TARGET_REQUEST]
            order.insert(slot - 1, TARGET_REQUEST)
            cases.append((f"c8-slot{slot}", order, REQUEST_COUNT))
        cases.append(("control-after", natural_order, REQUEST_COUNT))
        for case, order, concurrency in cases:
            instrumented = not case.startswith("control")
            for repeat in range(1, REPEATS + 1) if instrumented else (1,):
                sources = ("cold", "external") if pool is not None and case in ("c1", "c8-natural") else ("cold",)
                for source in sources:
                    phase = {
                        "id": f"{implementation}-{case}-r{repeat}-{source}",
                        "implementation": implementation,
                        "case": case,
                        "repeat": repeat,
                        "source": source,
                        "submission_order": order,
                        "concurrency": concurrency,
                        "instrumented": instrumented,
                    }
                    report["phases"].append(phase)
                    if pool is not None:
                        if source == "cold":
                            phase["pool_reset"] = call_worker_rpc(
                                server, "reset_ascendstore_benchmark_pool", args=[pool.master_address]
                            )
                        else:
                            phase["store_drain"] = call_worker_rpc(server, "drain_ascendstore_benchmark_stores")
                    phase["local_reset"] = reset_local_prefix_cache(server)
                    _run_phase(server, inputs, order, concurrency, phase, instrumented, source == "external")


def _run_phase(
    server: RemoteOpenAIServer,
    prompts: list[list[int]],
    order: list[int],
    concurrency: int,
    phase: dict[str, Any],
    instrumented: bool,
    external: bool,
) -> None:
    baseline, preemptions_before = _read_serving_counters(server)
    phase["batches"] = []
    if instrumented:
        call_worker_rpc(server, "start_ascendstore_diagnostic_trace")
    try:
        phase["observations"] = asyncio.run(_generate(server, prompts, order, concurrency, phase["id"], instrumented))
    finally:
        if instrumented:
            trace = call_worker_rpc(server, "stop_ascendstore_diagnostic_trace")["workers"][0]
            assert trace["trace_restored"] is True
            phase["batches"] = trace["batches"]
    for observation in phase["observations"]:
        assert "error" not in observation, observation
        assert len(observation["token_ids"]) == OUTPUT_TOKEN_COUNT, observation
        assert observation["finish_reason"] == "length", observation
        assert observation["usage"] == {"prompt_tokens": PROMPT_TOKEN_COUNT, "completion_tokens": OUTPUT_TOKEN_COUNT}
        if instrumented:
            logprobs = observation["logprobs"]
            assert len(logprobs["top_logprobs"]) == OUTPUT_TOKEN_COUNT
            assert logprobs["tokens"] == [f"token_id:{token}" for token in observation["token_ids"]]
            for position in range(OUTPUT_TOKEN_COUNT):
                assert token_batch_context(phase["batches"], observation["request_index"], position) is not None
    sources, preemptions_after = _wait_for_prompt_tokens(server, baseline, len(prompts) * PROMPT_TOKEN_COUNT)
    phase["prompt_tokens_by_source"] = sources
    phase["preemptions"] = preemptions_after - preemptions_before
    assert phase["preemptions"] == 0, "Unexpected preemption invalidates this diagnostic workload"
    assert sources == {
        "local_compute": len(prompts) if external else len(prompts) * PROMPT_TOKEN_COUNT,
        "local_cache_hit": 0,
        "external_kv_transfer": len(prompts) * REUSABLE_TOKEN_COUNT if external else 0,
    }, "Unexpected cache-hit provenance"
    print(f"ASCENDSTORE_DIAGNOSTIC_PHASE_COMPLETE {phase['id']}", flush=True)


async def _generate(
    server: RemoteOpenAIServer,
    prompts: list[list[int]],
    order: list[int],
    concurrency: int,
    phase_id: str,
    logprobs: bool,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)) as session:

        async def request(index: int) -> dict[str, Any]:
            observation: dict[str, Any] = {
                "request_index": index,
                "prompt_sha256": sha256(json.dumps(prompts[index]).encode()).hexdigest(),
            }
            payload = {
                "model": MODEL,
                "request_id": f"ascendstore-diag-{phase_id}-p{index}",
                "prompt": prompts[index],
                "max_tokens": OUTPUT_TOKEN_COUNT,
                "temperature": 0,
                "seed": 42,
                "ignore_eos": True,
                "stream": False,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
                "logprobs": 5 if logprobs else None,
            }
            try:
                async with semaphore, session.post(server.url_for("v1", "completions"), json=payload) as response:
                    response.raise_for_status()
                    body = await response.json()
                    choice = body["choices"][0]
                    assert choice["prompt_token_ids"] == prompts[index], "The server saw different input IDs"
                    observation.update(
                        token_ids=choice["token_ids"],
                        text=choice["text"],
                        logprobs=choice["logprobs"],
                        finish_reason=choice["finish_reason"],
                        usage={key: body["usage"][key] for key in ("prompt_tokens", "completion_tokens")},
                    )
            except Exception as error:
                observation["error"] = f"{type(error).__name__}: {error}"
            return observation

        observations = await asyncio.gather(*(request(index) for index in order))
    return sorted(observations, key=lambda observation: observation["request_index"])
