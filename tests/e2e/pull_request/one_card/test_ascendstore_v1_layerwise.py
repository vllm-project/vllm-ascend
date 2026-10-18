# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate production/v1 Layerwise serving performance on concentrated KV byte checks."""

import shutil
from functools import partial
from pathlib import Path

import pytest
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.pull_request.ascendstore_v1_benchmark import (
    CONCURRENCY,
    MODEL,
    make_report,
    make_requests,
    run_comparison,
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
    report = {**make_report(inputs, warmup), "scenario": "layerwise", "tensor_parallel_size": 1}
    try:
        exercise = partial(
            _exercise_connector,
            model_path=model_path,
            tokenizer=tokenizer,
            inputs=inputs,
            warmup=warmup,
            use_layerwise=True,
            worker_extension=LAYERWISE_WORKER_EXTENSION,
        )
        run_comparison(tmp_path, report, exercise)
    except BaseException as error:
        report.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_report(tmp_path / "ascendstore-layerwise-performance.json", "ASCENDSTORE_LAYERWISE_PERFORMANCE", report)
