# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate native Layerwise KeyRange before comparing production/v1 serving performance."""

import shutil
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.pull_request.ascendstore_v1_benchmark import (
    CONCURRENCY,
    CONNECTORS,
    CROSSOVER_RUNS,
    MODEL,
    assert_prompt_sources,
    call_worker_rpc,
    compare_runs,
    make_report,
    make_requests,
    reset_local_prefix_cache,
    run_benchmark,
    write_report,
)
from tests.e2e.pull_request.one_card.test_ascendstore_v1 import _exercise_connector

LAYERWISE_WORKER_EXTENSION = (
    "tests.e2e.pull_request.one_card.ascendstore_v1_layerwise.AscendStoreLayerwiseWorkerExtension"
)
# Bound retained CPU reference KV to two 8K prefixes; all sixteen prefixes are still used for performance.
VALIDATION_REQUEST_COUNT = 2

pytestmark = pytest.mark.e2e_model(MODEL)


def test_production_and_v1_layerwise_key_range(tmp_path: Path) -> None:
    assert shutil.which("mooncake_master") is not None, "The Layerwise E2E requires mooncake_master"
    backend_module = import_module("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend")
    client_type = import_module("mooncake.store").MooncakeDistributedStore
    missing = [
        name
        for name in backend_module.MOONCAKE_LAYERWISE_CLIENT_METHODS
        if not callable(getattr(client_type, name, None))
    ]
    assert not missing, f"Installed Mooncake lacks native session/range APIs: {missing}"
    model_path = maybe_model_redirect(MODEL)
    tokenizer = get_tokenizer(model_path)
    inputs = make_requests(tokenizer, "layerwise-benchmark")
    warmup = make_requests(tokenizer, "layerwise-warmup", count=CONCURRENCY)
    validation_inputs = make_requests(tokenizer, "layerwise-validation", count=VALIDATION_REQUEST_COUNT)
    report = {
        **make_report(inputs, warmup),
        "scenario": "layerwise-key-range",
        "tensor_parallel_size": 1,
        "validation": [],
        "validation_requests": VALIDATION_REQUEST_COUNT,
        "correctness_scope": "complete committed ranges; poisoned local KV; byte equality before each layer computes",
        "performance_probe_policy": "fresh servers after both correctness runs; no validation hooks or snapshots",
    }
    try:
        # No performance run starts unless both implementations complete the native chain successfully.
        for connector in CONNECTORS:
            row: dict[str, Any] = {"implementation": connector[0]}
            report["validation"].append(row)
            run_path = tmp_path / f"validation-{connector[0]}"
            run_path.mkdir()

            def validate(server: Any, pool: Any, row: dict[str, Any] = row) -> None:
                row["pool_reset"] = call_worker_rpc(
                    server, "reset_ascendstore_benchmark_pool", args=[pool.master_address]
                )
                row["cold_reset"] = reset_local_prefix_cache(server)
                row["probe"] = call_worker_rpc(server, "start_layerwise_validation")
                try:
                    row["cold_store"] = run_benchmark(server, tokenizer, validation_inputs)
                    assert_prompt_sources(row["cold_store"], external=False, request_count=VALIDATION_REQUEST_COUNT)
                    row["store_drain"] = call_worker_rpc(server, "drain_ascendstore_benchmark_stores")
                    row["external_reset"] = reset_local_prefix_cache(server)
                    row["poison"] = call_worker_rpc(server, "arm_layerwise_destination_poisoning")
                    row["external_load"] = run_benchmark(server, tokenizer, validation_inputs)
                    assert_prompt_sources(row["external_load"], external=True, request_count=VALIDATION_REQUEST_COUNT)
                    row["evidence"] = call_worker_rpc(server, "finish_layerwise_validation")
                except BaseException as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    try:
                        row["cleanup"] = call_worker_rpc(server, "remove_layerwise_validation")
                    except Exception as error:
                        row["cleanup_error"] = f"{type(error).__name__}: {error}"
                        if "error" not in row:
                            raise

            _exercise_connector(
                run_path,
                model_path,
                tokenizer,
                inputs,
                warmup,
                connector,
                [],
                use_layerwise=True,
                worker_extension=LAYERWISE_WORKER_EXTENSION,
                validate=validate,
            )

        for run_index, connector in enumerate(CROSSOVER_RUNS, start=1):
            run_path = tmp_path / f"performance-{run_index}-{connector[0]}"
            run_path.mkdir()
            run: dict[str, Any] = {"run": run_index, "implementation": connector[0], "measurements": []}
            report["runs"].append(run)
            start = time.monotonic()
            try:
                _exercise_connector(
                    run_path,
                    model_path,
                    tokenizer,
                    inputs,
                    warmup,
                    connector,
                    run["measurements"],
                    use_layerwise=True,
                    worker_extension=LAYERWISE_WORKER_EXTENSION,
                )
            finally:
                run["duration_seconds"] = time.monotonic() - start
        report["comparisons"] = compare_runs(report["runs"])
        report["status"] = "validated_and_measured"
    except BaseException as error:
        report.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_report(tmp_path / "ascendstore-layerwise-key-range.json", "ASCENDSTORE_LAYERWISE_KEY_RANGE", report)
