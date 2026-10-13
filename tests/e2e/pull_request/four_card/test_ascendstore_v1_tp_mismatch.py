# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare production/v1 TP=1 producer -> TP=2 consumer on three dedicated NPUs."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import time
from contextlib import ExitStack
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from statistics import mean, median
from typing import Any

import pytest
import requests
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_ports_list

from tests.e2e.common.kv_pool.config import MooncakeKVPoolConfig
from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.ascendstore_v1_benchmark import (
    BLOCK_SIZE,
    CONTROL_TIMEOUT_SECONDS,
    MODEL,
    PROMPT_TOKEN_COUNT,
    REQUEST_COUNT,
    REUSABLE_TOKEN_COUNT,
    make_requests,
    reset_local_prefix_cache,
    run_benchmark,
    write_report,
)

OUTPUT_TOKENS = (1, 8)
ROUNDS_PER_RUN = 1
PRODUCER_TP = 1
CONSUMER_TP = 2
POOL_BYTES = 32 << 30
WORKER_EXTENSION = "tests.e2e.pull_request.four_card.ascendstore_v1_tp_timing.AscendStoreTPBenchmarkWorkerExtension"
CONNECTORS = (
    (
        "production",
        "AscendStoreConnector",
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
    ),
    ("v1", "AscendStoreV1Connector", "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector"),
)
CROSSOVER_RUNS = CONNECTORS + CONNECTORS[::-1]

pytestmark = pytest.mark.e2e_model(MODEL)


def test_production_and_v1_tp1_to_tp2_serve_performance(tmp_path: Path) -> None:
    assert shutil.which("mooncake_master"), "TP serve benchmark requires mooncake_master"
    # Respect an explicitly assigned device set. The fourth CI card is deliberately left idle.
    devices = [device.strip() for device in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3").split(",")]
    assert len(devices) >= 3 and all(devices) and len(set(devices)) == len(devices), "Three distinct NPUs are required"
    model_path = maybe_model_redirect(MODEL)
    tokenizer = get_tokenizer(model_path)
    inputs = make_requests(tokenizer, "tp-benchmark")
    warmup = make_requests(tokenizer, "tp-warmup", count=1)
    first_blocks = [tuple(sample.prompt[:BLOCK_SIZE]) for sample in inputs + warmup]
    assert len(set(first_blocks)) == REQUEST_COUNT + 1, "Warmup and measured requests must not share prefixes"
    report: dict[str, Any] = {
        "model": MODEL,
        "producer_tp": PRODUCER_TP,
        "consumer_tp": CONSUMER_TP,
        "prompt_tokens": PROMPT_TOKEN_COUNT,
        "external_hit_tokens_per_request": REUSABLE_TOKEN_COUNT,
        "concurrency": 1,
        "output_tokens": OUTPUT_TOKENS,
        "requests_per_phase": REQUEST_COUNT,
        "rounds_per_run": ROUNDS_PER_RUN,
        "implementation_order": [item[0] for item in CROSSOVER_RUNS],
        "devices": {"producer": devices[:1], "consumer": devices[1:3]},
        "timing_scope": "consumer external-hit HTTP; excludes producer population, reset and Store drain",
        "backend_timing": "synchronous adapter call, including native argument conversion and transfer",
        "parameter_timing": "production keys/ranges; v1 plan plus argument materialization, reported separately",
        "diagnostics_in_control_timing": False,
        "rank_timing_note": "rank-local wall times; concurrent TP ranks must not be summed as request latency",
        "text_parity_policy": "report only; failed transfers or incorrect hit provenance still fail the test",
        "runs": [],
    }
    try:
        for run_index, connector in enumerate(CROSSOVER_RUNS, start=1):
            run_path = tmp_path / f"run-{run_index}-{connector[0]}"
            run_path.mkdir()
            run: dict[str, Any] = {"run": run_index, "implementation": connector[0], "measurements": []}
            report["runs"].append(run)
            start = time.monotonic()
            _exercise_pair(run_path, model_path, tokenizer, inputs, warmup, devices, connector, run["measurements"])
            run["duration_seconds"] = time.monotonic() - start
        report["comparisons"] = _compare_runs(report["runs"])
        report["status"] = "measured"
    except BaseException as error:
        report.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_report(tmp_path / "ascendstore-tp-mismatch-performance.json", "ASCENDSTORE_TP_PERFORMANCE", report)


def _exercise_pair(run_path, model_path, tokenizer, inputs, warmup, devices, connector, observations) -> None:
    master_port, metrics_port, producer_port, consumer_port, producer_lookup, consumer_lookup = get_open_ports_list(6)
    pool_config = MooncakeKVPoolConfig(
        config={
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "ascend",
            "device_name": "",
            "global_segment_size": POOL_BYTES,
            "local_buffer_size": 1 << 30,
        },
        master_port=master_port,
        metrics_port=metrics_port,
    )
    pool_runtime = import_module("tests.e2e.common.single_node.kv_pool_runtime")
    with ExitStack() as stack:
        pool = stack.enter_context(pool_runtime.SingleNodeMooncakeManager(pool_config, run_path.name))
        # Keep the producer (and its contributed pool segment) alive until the consumer has closed.
        producer = stack.enter_context(
            _start_server(model_path, connector, devices[:1], producer_port, producer_lookup, pool.server_envs, True)
        )
        consumer = stack.enter_context(
            _start_server(model_path, connector, devices[1:3], consumer_port, consumer_lookup, pool.server_envs, False)
        )
        run_benchmark(producer, tokenizer, [replace(warmup[0], expected_output_len=8)], 1)
        run_benchmark(consumer, tokenizer, [replace(warmup[0], expected_output_len=8)], 1)
        for output_tokens in OUTPUT_TOKENS:
            for round_index in range(1, ROUNDS_PER_RUN + 1):
                row: dict[str, Any] = {"output_tokens": output_tokens, "round": round_index}
                observations.append(row)
                row["pool_reset"] = _worker_rpc(
                    producer, "reset_ascendstore_benchmark_pool", PRODUCER_TP, [pool.master_address]
                )
                assert row["pool_reset"]["workers"][0]["removed_objects"] > 0
                row["producer_reset"] = reset_local_prefix_cache(producer)
                _worker_rpc(producer, "start_ascendstore_tp_timing", PRODUCER_TP)
                try:
                    row["producer"] = run_benchmark(producer, tokenizer, inputs, 1)
                    _assert_sources(row["producer"], external=False)
                finally:
                    row["producer_timing"] = _worker_rpc(producer, "stop_ascendstore_tp_timing", PRODUCER_TP)
                samples = [replace(sample, expected_output_len=output_tokens) for sample in inputs]
                row["control_reset"] = reset_local_prefix_cache(consumer)
                row["control"] = run_benchmark(consumer, tokenizer, samples, 1)
                _assert_sources(row["control"], external=True)
                row["diagnostic_reset"] = reset_local_prefix_cache(consumer)
                _worker_rpc(consumer, "start_ascendstore_tp_timing", CONSUMER_TP)
                try:
                    row["diagnostic"] = run_benchmark(consumer, tokenizer, samples, 1)
                    _assert_sources(row["diagnostic"], external=True)
                finally:
                    row["consumer_timing"] = _worker_rpc(consumer, "stop_ascendstore_tp_timing", CONSUMER_TP)
                row["control_diagnostic_texts_equal"] = (
                    row["control"]["generated_texts"] == row["diagnostic"]["generated_texts"]
                )
                row["diagnostic_over_control_ttft"] = (
                    row["diagnostic"]["median_ttft_ms"] / row["control"]["median_ttft_ms"]
                )
                _verify_transfers(row["producer_timing"]["workers"], row["consumer_timing"]["workers"])
                row["timing_summary"] = {
                    role: [_summarize_worker(worker) for worker in row[f"{role}_timing"]["workers"]]
                    for role in ("producer", "consumer")
                }


def _start_server(model_path, connector, devices, port, lookup_port, pool_envs, producer) -> RemoteOpenAIServer:
    _, connector_name, connector_module = connector
    transfer_config = {
        "kv_connector": connector_name,
        "kv_connector_module_path": connector_module,
        "kv_role": "kv_producer" if producer else "kv_consumer",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {
            "backend": "mooncake",
            "lookup_rpc_port": lookup_port,
            "use_layerwise": False,
            "load_async": False,
            "save_decode_cache": False,
            "discard_partial_chunks": True,
            "consumer_is_to_load": True,
            "consumer_is_to_put": False,
            "prefill_tp_size": PRODUCER_TP,
            "decode_tp_size": CONSUMER_TP,
        },
    }
    args = shlex.split(f"""
        --host 127.0.0.1 --port {port} --served-model-name {MODEL}
        --dtype bfloat16 --max-model-len 16384 --max-num-batched-tokens 16384 --max-num-seqs 8
        --block-size {BLOCK_SIZE} --tensor-parallel-size {len(devices)} --pipeline-parallel-size 1
        --gpu-memory-utilization 0.8 --enable-prefix-caching --prefix-caching-hash-algo sha256
        --enforce-eager --no-async-scheduling --no-enable-chunked-prefill --seed 42
    """) + ["--kv-transfer-config", json.dumps(transfer_config), "--worker-extension-cls", WORKER_EXTENSION]
    repository_root = str(Path(__file__).resolve().parents[4])
    return RemoteOpenAIServer(
        model_path,
        args,
        server_host="127.0.0.1",
        server_port=port,
        auto_port=False,
        max_wait_seconds=600,
        env_dict={
            **pool_envs,
            "PYTHONPATH": os.pathsep.join(filter(None, (repository_root, os.environ.get("PYTHONPATH", "")))),
            "ASCEND_RT_VISIBLE_DEVICES": ",".join(devices),
            # A2 HCCS needs device staging buffers for remote host pool memory.
            "ASCEND_BUFFER_POOL": "4:8",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_KV_CACHE_LAYOUT": "NHD",
            "VLLM_SERVER_DEV_MODE": "1",
        },
    )


def _worker_rpc(server, method: str, worker_count: int, args: list[str] | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    response = requests.post(
        server.url_for("collective_rpc"),
        json={"method": method, "args": args or [], "timeout": CONTROL_TIMEOUT_SECONDS},
        timeout=CONTROL_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    workers = response.json()["results"]
    assert len(workers) == worker_count and all(isinstance(worker, dict) for worker in workers), workers
    return {"workers": workers, "duration_ms": (time.perf_counter() - start) * 1000}


def _assert_sources(benchmark: dict[str, Any], *, external: bool) -> None:
    assert benchmark["prompt_tokens_by_source"] == {
        "local_compute": REQUEST_COUNT if external else REQUEST_COUNT * PROMPT_TOKEN_COUNT,
        "local_cache_hit": 0,
        "external_kv_transfer": REQUEST_COUNT * REUSABLE_TOKEN_COUNT if external else 0,
    }, "The TP test must consume the producer's external prefix, not local KV or recomputation"


def _verify_transfers(producers: list[dict[str, Any]], consumers: list[dict[str, Any]]) -> None:
    assert len(producers) == 1 and producers[0]["tp_size"] == PRODUCER_TP
    assert producers[0]["key_slices_per_rank"] == CONSUMER_TP
    assert {worker["tp_rank"] for worker in consumers} == {0, 1}
    stored_keys: set[str] = set()
    loaded_keys: set[str] = set()
    stored_bytes = loaded_bytes = 0
    for worker in producers + consumers:
        producer = worker["role"] == "kv_producer"
        stage = "backend_store" if producer else "backend_load"
        if not producer:
            assert worker["tp_size"] == CONSUMER_TP and worker["key_slices_per_rank"] == 1
        assert all("error" not in record for record in worker["records"]), "A diagnostic transfer failed"
        calls = [record for record in worker["records"] if record["stage"] == stage]
        assert calls, f"No {stage} on rank {worker['tp_rank']}"
        keys = [key for call in calls for key in call["keys"]]
        assert len(keys) == len(set(keys)), "Unique requests should transfer each object only once per round"
        byte_count = sum(call["byte_count"] for call in calls)
        assert byte_count > 0
        if producer:
            stored_keys.update(keys)
            stored_bytes += byte_count
        else:
            assert all(f"@head_or_tp_rank:{worker['tp_rank']}@" in key for key in keys)
            assert not loaded_keys.intersection(keys), "Consumer ranks loaded overlapping head partitions"
            assert all(
                call["result_codes"] is not None
                and len(call["result_codes"]) == len(call["keys"])
                and all(code == 0 for code in call["result_codes"])
                for call in calls
            ), "Backend did not confirm every TP Load"
            loaded_keys.update(keys)
            loaded_bytes += byte_count
    assert loaded_keys == stored_keys and loaded_bytes == stored_bytes, (
        "Consumer did not restore the producer's objects"
    )


def _summarize_worker(worker: dict[str, Any]) -> dict[str, Any]:
    stages = {}
    for stage in sorted({record["stage"] for record in worker["records"]}):
        records = [record for record in worker["records"] if record["stage"] == stage]
        values = [record["host_ms"] for record in records]
        stages[stage] = {
            "calls": len(records),
            "total_ms": sum(values),
            "mean_ms": mean(values),
            "median_ms": median(values),
            "max_ms": max(values),
            "range_count": sum(record.get("range_count", 0) for record in records),
            "byte_count": sum(record.get("byte_count", 0) for record in records),
        }
    parameter_ms = sum(stage["total_ms"] for name, stage in stages.items() if not name.startswith("backend_"))
    return {
        "tp_rank": worker["tp_rank"],
        "pid": worker["pid"],
        "parameter_ms_per_request": parameter_ms / REQUEST_COUNT,
        "stages": stages,
    }


def _compare_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = {}
    for output_tokens in OUTPUT_TOKENS:
        controls = {
            implementation: [
                row["control"]
                for run in runs
                if run["implementation"] == implementation
                for row in run["measurements"]
                if row["output_tokens"] == output_tokens
            ]
            for implementation, _, _ in CONNECTORS
        }
        production, candidate = controls["production"], controls["v1"]
        comparison: dict[str, Any] = {
            "rounds_per_implementation": len(production),
            "generated_texts_equal": all(
                actual["generated_texts"] == expected["generated_texts"]
                for expected, actual in zip(production, candidate, strict=True)
            ),
        }
        for metric in ("median_ttft_ms", "median_e2el_ms", "request_throughput"):
            reference = median(row[metric] for row in production)
            actual = median(row[metric] for row in candidate)
            comparison[metric] = {"production": reference, "v1": actual, "v1_over_production": actual / reference}
        comparisons[str(output_tokens)] = comparison
    return comparisons
