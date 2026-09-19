# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.pd_utils import run_pd_contract

MODEL = "Qwen/Qwen3-8B"


@pytest.mark.e2e_model("Qwen/Qwen3-8B")
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="aclgraph",
    parallel="TP",
    deploy="pd_disaggregation",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free(target_free_percentage=0.95, max_wait_seconds=180)
def test_mooncake_pd_graph_matches_colocated() -> None:
    """Four requests, real transfer, unavailable-transfer failure and recovery."""
    run_pd_contract(model=MODEL, prefix=False)


@pytest.mark.e2e_model("Qwen/Qwen3-8B")
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="aclgraph,prefix_caching,chunked_prefill,mixed_lengths,kv_pool",
    parallel="TP",
    deploy="pd_disaggregation",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free(target_free_percentage=0.95, max_wait_seconds=180)
def test_mooncake_pd_prefix_chunked_graph_boundaries() -> None:
    """127/128/129/1537-token boundaries and Pool -> P -> D after local reset."""
    run_pd_contract(model=MODEL, prefix=True, kv_pool=True)
