# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare production/v1 Layerwise serving outputs and performance without byte probes."""

import shutil
import time
from pathlib import Path
from typing import Any

import pytest
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.pull_request.ascendstore_v1_benchmark import (
    CONCURRENCY,
    CROSSOVER_RUNS,
    MODEL,
    compare_runs,
    make_report,
    make_requests,
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
        report["status"] = "measured"
    except BaseException as error:
        report.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_report(tmp_path / "ascendstore-layerwise-performance.json", "ASCENDSTORE_LAYERWISE_PERFORMANCE", report)
