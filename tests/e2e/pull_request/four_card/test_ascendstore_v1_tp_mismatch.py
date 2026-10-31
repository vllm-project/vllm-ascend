# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare production/v1 TP=1 producer Store and TP=2 consumer Load on A2."""

import json
import os
import shlex
import shutil
from contextlib import ExitStack
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_ports_list

from tests.e2e.common.kv_pool.config import MooncakeKVPoolConfig
from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.ascendstore_v1_benchmark import (
    BLOCK_SIZE,
    CONCURRENCY,
    MODEL,
    ROUNDS_PER_RUN,
    WORKER_EXTENSION,
    assert_prompt_sources,
    call_worker_rpc,
    make_report,
    make_requests,
    reset_local_prefix_cache,
    run_benchmark,
    run_byte_accuracy,
    run_comparison,
    write_report,
)

PRODUCER_TP = 1
CONSUMER_TP = 2
# Sixteen independent BF16 8K prefixes occupy about 18 GiB, independent of their TP partitioning.
POOL_BYTES = 32 << 30

pytestmark = pytest.mark.e2e_model(MODEL)


def test_production_and_v1_tp1_to_tp2_serve_performance(tmp_path: Path) -> None:
    assert shutil.which("mooncake_master"), "TP serve benchmark requires mooncake_master"
    # Respect an explicitly assigned device set. The fourth CI card is deliberately left idle.
    devices = [device.strip() for device in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3").split(",")]
    assert len(devices) >= 3 and all(devices) and len(set(devices)) == len(devices), "Three distinct NPUs are required"
    model_path = maybe_model_redirect(MODEL)
    tokenizer = get_tokenizer(model_path)
    inputs = make_requests(tokenizer, "tp-benchmark")
    warmup = make_requests(tokenizer, "tp-warmup", count=CONCURRENCY)
    report = {
        **make_report(inputs, warmup),
        "scenario": "tp_mismatch",
        "producer_tp": PRODUCER_TP,
        "consumer_tp": CONSUMER_TP,
        "devices": {"producer": devices[:1], "consumer": devices[1:3]},
        "cold_store_role": "TP=1 producer",
        "external_load_role": "TP=2 consumer; producer remains alive but idle",
    }
    try:
        exercise = partial(
            _exercise_pair, model_path=model_path, tokenizer=tokenizer, inputs=inputs, warmup=warmup, devices=devices
        )
        run_comparison(tmp_path, report, exercise)
    except BaseException as error:
        report.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_report(tmp_path / "ascendstore-tp-mismatch-performance.json", "ASCENDSTORE_TP_PERFORMANCE", report)


def _exercise_pair(
    run_path, model_path, tokenizer, inputs, warmup, devices, connector, observations, *, accuracy_directory=None
) -> None:
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
        if accuracy_directory is not None:
            run_byte_accuracy(producer, consumer, tokenizer, accuracy_directory, observations, CONSUMER_TP)
            return
        run_benchmark(producer, tokenizer, warmup)
        call_worker_rpc(producer, "drain_ascendstore_benchmark_stores")
        run_benchmark(consumer, tokenizer, warmup)
        for round_index in range(1, ROUNDS_PER_RUN + 1):
            row: dict[str, Any] = {"round": round_index}
            observations.append(row)
            row["pool_reset"] = call_worker_rpc(
                producer, "reset_ascendstore_benchmark_pool", args=[pool.master_address]
            )
            assert row["pool_reset"]["workers"][0]["removed_objects"] > 0, "The preceding Store published no KV"
            row["producer_reset"] = reset_local_prefix_cache(producer)
            row["cold_store"] = run_benchmark(producer, tokenizer, inputs)
            assert_prompt_sources(row["cold_store"], external=False)

            # Publication is a phase boundary, not part of the producer's HTTP benchmark duration.
            row["store_drain"] = call_worker_rpc(producer, "drain_ascendstore_benchmark_stores")
            row["consumer_reset"] = reset_local_prefix_cache(consumer)
            row["external_load"] = run_benchmark(consumer, tokenizer, inputs)
            assert_prompt_sources(row["external_load"], external=True)


def _start_server(model_path, connector, devices, port, lookup_port, pool_envs, producer) -> RemoteOpenAIServer:
    _, connector_name = connector
    transfer_config = {
        "kv_connector": connector_name,
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
        --dtype bfloat16 --max-model-len 16384 --max-num-batched-tokens 16384 --max-num-seqs {CONCURRENCY}
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
