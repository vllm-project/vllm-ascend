# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 V4 Flash MTP and DSpark execution modes with full weights."""

from pathlib import Path

import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.deepseek_v4_a5_utils import (
    MODEL,
    MTP_MODEL,
    PROBE_CLASS,
    _args,
    _run_functional,
    _speculation,
)
from tests.e2e.pull_request.deepseek_v4_a5_utils import a5_runtime as a5_runtime
from tests.e2e.pull_request.deepseek_v4_a5_utils import v4_model_fixture as v4_model_fixture
from tests.e2e.pull_request.deepseek_v4_a5_utils import v4_mtp_model_fixture as v4_mtp_model_fixture


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dspark,aclgraph,chunked_prefill,mixed_lengths",
    parallel="DP,EP",
    deploy="pd_mix",
    hardware="A5",
    quantization="MXFP4,MXFP8",
    graph_mode="full_decode_only",
)
@pytest.mark.parametrize("variant", ["static", "adaptive", "dynamic"])
@wait_until_npu_memory_free()
def test_deepseek_v4_a5_dspark_graph(v4_model, variant: str, tmp_path: Path) -> None:
    model, prompts, _ = v4_model
    args, spec = _args(), _speculation()
    if variant == "adaptive":
        spec["enable_adaptive_verification"] = True
    elif variant == "dynamic":
        args["additional_config"]["dynamic_spec_config"] = {
            "method": "dspark",
            "method_params": {"initial_verify_budget_per_req": 3, "budget_update_interval": 1, "budget_threshold": 0.7},
        }
    args.update(
        speculative_config=spec,
        worker_extension_cls=PROBE_CLASS,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [6, 12, 24]},
    )
    _run_functional(model, args, prompts, tmp_path / f"dspark-{variant}.log")


@pytest.mark.e2e_model(MTP_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,aclgraph",
    parallel="DP,EP",
    deploy="pd_mix",
    hardware="A5",
    quantization="MXFP4,MXFP8",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_deepseek_v4_a5_mtp(v4_mtp_model, tmp_path: Path) -> None:
    model, prompts, _ = v4_mtp_model
    args = dict(
        _args(),
        speculative_config=_speculation("mtp"),
        worker_extension_cls=PROBE_CLASS,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8, 16]},
    )
    _run_functional(model, args, prompts, tmp_path / "mtp.log")
