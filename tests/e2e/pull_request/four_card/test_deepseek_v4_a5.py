# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 V4 Flash IndexCache and data-parallel regressions with full weights."""

from pathlib import Path

import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.deepseek_a5_utils import LoggedServer, assert_invalid_request, completions, prompt
from tests.e2e.pull_request.deepseek_v4_a5_utils import MODEL, PROBE_CLASS, _args, _run_functional, _speculation
from tests.e2e.pull_request.deepseek_v4_a5_utils import a5_runtime as a5_runtime
from tests.e2e.pull_request.deepseek_v4_a5_utils import v4_model_fixture as v4_model_fixture


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="sfa_dsa,aclgraph",
    parallel="DP,EP",
    deploy="pd_mix",
    hardware="A5",
    quantization="MXFP4,MXFP8",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_deepseek_v4_a5_index_cache_freq4(v4_model, tmp_path: Path) -> None:
    model, _, tokenizer = v4_model
    prompts = [
        prompt(129, tokenizer=tokenizer)["prompt_token_ids"],
        prompt(769, salt=421, tokenizer=tokenizer)["prompt_token_ids"],
    ]
    args = dict(
        _args(),
        hf_overrides={"use_index_cache": True, "index_topk_freq": 4},
        worker_extension_cls=PROBE_CLASS,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8]},
    )
    with LoggedServer(model, args, tmp_path / "index-graph.log") as server:
        server.collective_rpc("install_graph_probe")
        # Reject an invalid request, then verify valid generation still works.
        assert_invalid_request(server)
        completions(server, prompts)
        counts = server.collective_rpc("read_graph_probe")
        assert counts and any(int(count) > 0 for count in counts), counts


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dspark,chunked_prefill,mixed_lengths",
    parallel="DP,EP",
    deploy="pd_mix",
    hardware="A5",
    quantization="MXFP4,MXFP8",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_deepseek_v4_a5_dspark_dp4(v4_model, tmp_path: Path) -> None:
    model, _, tokenizer = v4_model
    args = dict(
        _args(),
        speculative_config=_speculation(),
        worker_extension_cls=PROBE_CLASS,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [6, 12, 24]},
    )
    args["additional_config"].update(enable_shared_expert_dp=True, multistream_overlap_shared_expert=True)
    prompts = [
        prompt(length, salt=i * 137, tokenizer=tokenizer)["prompt_token_ids"]
        for i, length in enumerate((1, 127, 128, 769) * 4)
    ]
    _run_functional(model, args, prompts, tmp_path / "dp4.log", dp_size=4)
