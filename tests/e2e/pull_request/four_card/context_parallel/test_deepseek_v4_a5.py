# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 V4 Flash PCP and DSpark graph execution with full weights."""

import os
from pathlib import Path

import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.deepseek_a5_utils import prompt
from tests.e2e.pull_request.deepseek_v4_a5_utils import MODEL, PROBE_CLASS, _args, _run_functional, _speculation
from tests.e2e.pull_request.deepseek_v4_a5_utils import a5_runtime as a5_runtime
from tests.e2e.pull_request.deepseek_v4_a5_utils import v4_model_fixture as v4_model_fixture


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dsa_pcp,dspark,aclgraph,chunked_prefill,mixed_lengths",
    parallel="DP,EP,PCP",
    deploy="pd_mix",
    hardware="A5",
    quantization="MXFP4,MXFP8",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_deepseek_v4_a5_dspark_pcp(v4_model, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # As in A3 PCP tests, size the MC2 window for full-weight prefill dispatch.
    monkeypatch.setenv("HCCL_BUFFSIZE", "2560")
    # Match the A5 PCP deployment's HIXL resource format for MC2 transport.
    if not any(name in os.environ for name in ("ASCEND_LOCAL_COMM_RES", "ASCEND_GLOBAL_RESOURCE_CONFIG")):
        monkeypatch.setenv("ASCEND_LOCAL_COMM_RES", '{"version":"1.3"}')
    model, _, tokenizer = v4_model
    args = dict(_args(), data_parallel_size=2, data_parallel_size_local=2, prefill_context_parallel_size=2)
    args["additional_config"].update(
        enable_prefill_mc2=True, enable_pcp_o_proj_weight_sharding=False, enable_fused_mc2=0
    )
    prompts = [
        prompt(length, salt=i * 137, tokenizer=tokenizer)["prompt_token_ids"]
        for i, length in enumerate((127, 129, 769, 1025) * 2)
    ]
    args.update(
        speculative_config=_speculation(),
        worker_extension_cls=PROBE_CLASS,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [6, 12, 24]},
    )
    _run_functional(model, args, prompts, tmp_path / "pcp-dspark.log", dp_size=2)
