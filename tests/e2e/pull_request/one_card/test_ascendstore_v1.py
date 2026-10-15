# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare production/v1 serving with cold Store and fully External-cached prompts."""

import json
import os
import shlex
import shutil
import time
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from vllm.benchmarks.datasets import SampleRequest
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_ports_list

from tests.e2e.common.kv_pool.config import MooncakeKVPoolConfig
from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.ascendstore_v1_benchmark import (
    BLOCK_SIZE,
    CONCURRENCY,
    CROSSOVER_RUNS,
    MODEL,
    ROUNDS_PER_RUN,
    WORKER_EXTENSION,
    assert_prompt_sources,
    call_worker_rpc,
    compare_runs,
    make_report,
    make_requests,
    reset_local_prefix_cache,
    run_benchmark,
    write_report,
)

# Sixteen independent BF16 8K prefixes occupy about 18 GiB; clear this private pool between rounds.
POOL_MEMORY_BYTES = 32 << 30
LOCAL_BUFFER_BYTES = 1 << 30

pytestmark = pytest.mark.e2e_model(MODEL)


def test_production_and_v1_serve_cold_and_external_hit_performance(tmp_path: Path) -> None:
    assert shutil.which("mooncake_master") is not None, "The serve benchmark requires mooncake_master on PATH"
    model_path = maybe_model_redirect(MODEL)
    tokenizer = get_tokenizer(model_path)
    inputs = make_requests(tokenizer, "benchmark")
    warmup = make_requests(tokenizer, "warmup", count=CONCURRENCY)
    report = {**make_report(inputs, warmup), "scenario": "bulk", "tensor_parallel_size": 1}
    try:
        for run_index, connector in enumerate(CROSSOVER_RUNS, start=1):
            run_path = tmp_path / f"run-{run_index}-{connector[0]}"
            run_path.mkdir()
            run: dict[str, Any] = {"run": run_index, "implementation": connector[0], "measurements": []}
            report["runs"].append(run)
            start = time.monotonic()
            try:
                _exercise_connector(run_path, model_path, tokenizer, inputs, warmup, connector, run["measurements"])
            finally:
                run["duration_seconds"] = time.monotonic() - start
        report["comparisons"] = compare_runs(report["runs"])
        report["status"] = "measured"
    except BaseException as error:
        report.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        # The selected-test log artifact also retains earlier rounds if a later server or request fails.
        write_report(tmp_path / "ascendstore-serve-performance.json", "ASCENDSTORE_SERVE_PERFORMANCE", report)


def _exercise_connector(
    tmp_path: Path,
    model_path: str,
    tokenizer: TokenizerLike,
    inputs: list[SampleRequest],
    warmup: list[SampleRequest],
    connector: tuple[str, str, str],
    observations: list[dict[str, Any]],
    *,
    use_layerwise: bool = False,
    worker_extension: str = WORKER_EXTENSION,
    validate: Callable[[Any, Any], None] | None = None,
) -> None:
    _, connector_name, connector_module = connector
    master_port, metrics_port, lookup_port, server_port = get_open_ports_list(4)
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
    transfer_config = {
        "kv_connector": connector_name,
        "kv_connector_module_path": connector_module,
        "kv_role": "kv_both",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {
            "backend": "mooncake",
            "lookup_rpc_port": lookup_port,
            "use_layerwise": use_layerwise,
            "load_async": False,
            "save_decode_cache": False,
            "discard_partial_chunks": True,
        },
    }
    server_args = shlex.split(f"""
        --host 127.0.0.1 --port {server_port} --served-model-name {MODEL}
        --dtype bfloat16 --max-model-len 16384 --max-num-batched-tokens 16384
        --max-num-seqs {CONCURRENCY} --block-size {BLOCK_SIZE}
        --tensor-parallel-size 1 --pipeline-parallel-size 1 --gpu-memory-utilization 0.8
        --enable-prefix-caching --prefix-caching-hash-algo sha256 --no-enable-chunked-prefill
        --enforce-eager --no-async-scheduling --seed 42
    """) + ["--kv-transfer-config", json.dumps(transfer_config), "--worker-extension-cls", worker_extension]
    # This optional native dependency is needed only when executing our E2E.
    pool_runtime = import_module("tests.e2e.common.single_node.kv_pool_runtime")
    with pool_runtime.SingleNodeMooncakeManager(pool_config, tmp_path.name) as pool:
        # Worker subprocesses do not inherit pytest's sys.path; installed packages exclude tests.
        repository_root = str(Path(__file__).resolve().parents[4])
        worker_pythonpath = os.pathsep.join(filter(None, (repository_root, os.environ.get("PYTHONPATH", ""))))
        server_envs = {
            **pool.server_envs,
            "PYTHONPATH": worker_pythonpath,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_KV_CACHE_LAYOUT": "NHD",
            "VLLM_SERVER_DEV_MODE": "1",
        }
        with RemoteOpenAIServer(
            model_path,
            server_args,
            server_host="127.0.0.1",
            server_port=server_port,
            auto_port=False,
            env_dict=server_envs,
        ) as server:
            run_benchmark(server, tokenizer, warmup)
            if validate is not None:
                validate(server, pool)
                return
            for round_index in range(1, ROUNDS_PER_RUN + 1):
                row: dict[str, Any] = {"round": round_index}
                observations.append(row)
                # Pool reset drains pending Store before deleting objects; neither operation is timed by benchmark.
                row["pool_reset"] = call_worker_rpc(
                    server, "reset_ascendstore_benchmark_pool", args=[pool.master_address]
                )
                assert row["pool_reset"]["workers"][0]["removed_objects"] > 0, "The preceding Store published no KV"
                row["cold_reset"] = reset_local_prefix_cache(server)
                row["cold_store"] = run_benchmark(server, tokenizer, inputs)
                assert_prompt_sources(row["cold_store"], external=False)

                # HTTP completion need not imply Store publication. Drain explicitly before the Load phase.
                row["store_drain"] = call_worker_rpc(server, "drain_ascendstore_benchmark_stores")
                row["external_reset"] = reset_local_prefix_cache(server)
                row["external_load"] = run_benchmark(server, tokenizer, inputs)
                assert_prompt_sources(row["external_load"], external=True)
