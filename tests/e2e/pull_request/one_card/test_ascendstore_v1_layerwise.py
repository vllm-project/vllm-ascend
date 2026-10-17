# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare production/v1 Layerwise serving outputs and performance without byte probes."""

import shutil
import time
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.pull_request.ascendstore_v1_benchmark import (
    CONCURRENCY,
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

pytestmark = pytest.mark.e2e_model(MODEL)


def test_production_and_v1_layerwise_serve_outputs_and_performance(tmp_path: Path) -> None:
    assert shutil.which("mooncake_master") is not None, "The Layerwise E2E requires mooncake_master"
    model_path = maybe_model_redirect(MODEL)
    tokenizer = get_tokenizer(model_path)
    inputs = make_requests(tokenizer, "layerwise-benchmark")
    warmup = make_requests(tokenizer, "layerwise-warmup", count=CONCURRENCY)
    report = {
        **make_report(inputs, warmup),
        "scenario": "layerwise",
        "tensor_parallel_size": 1,
    }
    try:
        for run_index, connector in enumerate(CROSSOVER_RUNS, start=1):
            run_path = tmp_path / f"performance-{run_index}-{connector[0]}"
            run_path.mkdir()
            run: dict[str, Any] = {"run": run_index, "implementation": connector[0], "measurements": []}
            report["runs"].append(run)
            start = time.monotonic()
            try:
                diagnostic = None
                if connector[0] == "v1":
                    run["external_load_diagnostic"] = {}
                    diagnostic = partial(_diagnose_external_load, tokenizer=tokenizer, inputs=inputs, run=run)
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
                    after_measurements=diagnostic,
                )
            finally:
                run["duration_seconds"] = time.monotonic() - start
        report["comparisons"] = compare_runs(report["runs"])
        report["status"] = "measured"
    except BaseException as error:
        report.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_report(tmp_path / "ascendstore-layerwise-performance.json", "ASCENDSTORE_LAYERWISE_PERFORMANCE", report)


def _diagnose_external_load(server: Any, *, tokenizer: Any, inputs: list[Any], run: dict[str, Any]) -> None:
    diagnostic = run["external_load_diagnostic"]
    diagnostic["policy"] = "separate v1-only pass after both normal rounds; excluded from ABBA comparisons"
    diagnostic["store_drain"] = call_worker_rpc(server, "drain_ascendstore_benchmark_stores")
    diagnostic["external_reset"] = reset_local_prefix_cache(server)
    diagnostic["start"] = call_worker_rpc(server, "start_ascendstore_layerwise_load_timing")
    failed = False
    try:
        diagnostic["measurement"] = run_benchmark(server, tokenizer, inputs)
        assert_prompt_sources(diagnostic["measurement"], external=True)
    except BaseException:
        failed = True
        raise
    finally:
        try:
            diagnostic["timing"] = call_worker_rpc(server, "stop_ascendstore_layerwise_load_timing")
        except BaseException as error:
            diagnostic["stop_error"] = f"{type(error).__name__}: {error}"
            if not failed:
                raise
    timing = diagnostic["timing"]["workers"][0]
    assert timing["dropped_records"] == 0 and timing["failed_calls"] == 0, "Incomplete Load timing"
    loaded_requests = [request for step in timing["steps"] for request in step["request_ids"]]
    assert len(set(loaded_requests)) == len(loaded_requests) == len(inputs), "Incomplete diagnostic request coverage"
    for stage in (
        "lookup_service",
        "lookup_backend_exists",
        "load_batch_prepare",
        "load_start_host",
        "layer_plan_prepare",
        "session_start",
        "layer_wait_host",
        "layer_task_host",
        "layer_io_host",
        "range_copy",
        "session_end",
        "execute_model_host",
        "sample_tokens_host",
    ):
        assert timing["stages"].get(stage, {}).get("calls", 0) > 0, f"Missing Load timing stage: {stage}"
    diagnostic["reference_mean_ttft_ms"] = run["measurements"][-1]["external_load"]["mean_ttft_ms"]
    diagnostic["ttft_note"] = "reference is the preceding uninstrumented round; difference also includes run drift"
