# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.pcp_manager import PCPManager

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.utils import get_pcp_profile_num_tokens
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def make_runner(tokens=16384, pcp=16, requests=64, query=4, method="mtp", dcp=1):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.max_num_tokens = tokens
    runner.max_num_reqs = requests
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp),
        model_config=SimpleNamespace(enforce_eager=True),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE, cudagraph_capture_sizes=[]),
    )
    runner.speculative_config = SimpleNamespace(
        method=method,
        num_speculative_tokens=query - 1,
        max_num_new_slots_for_drafting=0,
    )
    runner.vllm_config.speculative_config = runner.speculative_config
    runner.speculator = None  # Non-last PP stages have no local drafter.
    return runner


@pytest.mark.parametrize(
    "tokens,pcp,requests,query,method,expected",
    [
        (16384, 1, 64, 4, "mtp", 16384),
        (16384, 16, 64, 4, "mtp", 1392),
        (16384, 16, 512, 4, "mtp", 3968),
        (17, 4, 2, 1, "mtp", 10),
        (17, 16, 64, 4, "mtp", 17),
        (16384, 8, 64, 4, "dspark", 2400),
        (16384, 8, 64, 4, "dflash", 2400),
    ],
)
def test_local_prefill_preserves_draft_requests(tokens, pcp, requests, query, method, expected):
    runner = make_runner(tokens, pcp, requests, query, method)
    assert get_pcp_profile_num_tokens(runner.vllm_config, runner.max_num_tokens, runner.max_num_reqs) == expected
    assert runner.max_num_tokens == tokens
    assert runner.max_num_reqs == requests


def test_no_draft_preserves_target_decode_requests():
    runner = make_runner(requests=2048)
    runner.speculative_config = None
    runner.vllm_config.speculative_config = None
    assert get_pcp_profile_num_tokens(runner.vllm_config, runner.max_num_tokens, runner.max_num_reqs) == 7040


@pytest.mark.parametrize("tokens,requests", [(16384, 64), (16384, 512), (17, 64)])
def test_local_dummy_preserves_actual_upstream_request_count(tokens, requests):
    runner = make_runner(tokens=tokens, requests=requests)
    local_tokens = get_pcp_profile_num_tokens(runner.vllm_config, runner.max_num_tokens, runner.max_num_reqs)
    runner.speculator = SimpleNamespace(num_query_per_req=4, num_extra_query_per_req=0)
    runner.kv_connector = Mock()
    runner.is_first_pp_rank = True
    runner.lora_config = None
    runner.maybe_dummy_run_with_lora = Mock(return_value=nullcontext())
    batches = []

    def execute(output, **kwargs):
        batches.append(output.num_scheduled_tokens)
        # Stop before any model execution or NPU allocation.
        raise RuntimeError("observed scheduler batch")

    runner.execute_model = execute
    for count in (tokens, local_tokens):
        with pytest.raises(RuntimeError, match="observed scheduler batch"):
            GPUModelRunner._dummy_run(runner, count, skip_attn=True, is_profile=True)
    assert len(batches[0]) == len(batches[1]) == min(requests, tokens // 4)
    assert sum(batches[1].values()) == local_tokens
    assert runner.max_num_tokens == tokens


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("method", ["mtp", "dspark", "dflash"])
def test_profile_temporarily_sets_budget_and_restores_on_failure(fail, method):
    runner = make_runner(requests=512, method=method)
    runner.ascend_config = SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    runner._dummy_run = Mock()

    def profile(self, **kwargs):
        assert kwargs == {"randomize_inputs": True}
        assert self.max_num_tokens == 3968
        assert self.max_num_reqs == 512
        if fail:
            raise RuntimeError("profile failed")

    with (
        patch("vllm_ascend.worker.v2.model_runner.get_mc2_tokens_capacity", return_value=None),
        patch("vllm_ascend.worker.v2.model_runner.override_mrv2_in_profile_run", return_value=nullcontext()),
        patch.object(GPUModelRunner, "profile_run", profile),
    ):
        if fail:
            with pytest.raises(RuntimeError, match="profile failed"):
                runner.profile_run(randomize_inputs=True)
        else:
            runner.profile_run(randomize_inputs=True)
    assert runner.max_num_tokens == 16384
    assert runner.max_num_reqs == 512


@pytest.mark.parametrize("capacity,reserve", [(4096, False), (1024, True)])
def test_mc2_reservation_compares_the_selected_profile_budget(capacity, reserve):
    runner = make_runner(requests=512)
    runner.ascend_config = SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    runner.get_model = Mock()
    runner._dummy_run = Mock()
    with (
        patch("vllm_ascend.worker.v2.model_runner.get_mc2_tokens_capacity", return_value=capacity),
        patch("vllm_ascend.worker.v2.model_runner.select_moe_comm_method", return_value=MoECommType.MC2),
        patch("vllm_ascend.worker.v2.model_runner.disable_compilation", return_value=nullcontext()),
        patch("vllm_ascend.worker.v2.model_runner.override_mrv2_in_profile_run", return_value=nullcontext()),
        patch.object(GPUModelRunner, "profile_run"),
    ):
        runner.profile_run()
    assert runner._dummy_run.call_count == int(reserve)
    if reserve:
        runner._dummy_run.assert_called_once_with(
            capacity, skip_attn=True, skip_eplb=True, is_profile=True, randomize_inputs=False
        )
    assert runner.max_num_tokens == 16384


@pytest.mark.parametrize("mode", [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE])
@pytest.mark.parametrize("split_grid", [False, True])
@pytest.mark.parametrize("graph_tokens,expected", [(512, 3486), (4096, 4096)])
def test_profile_budget_covers_prefill_graphs(mode, split_grid, graph_tokens, expected):
    runner = make_runner(pcp=8, requests=500, query=1)
    config = runner.vllm_config
    config.model_config.enforce_eager = False
    config.compilation_config.cudagraph_mode = mode
    config.compilation_config.cudagraph_capture_sizes = [512, 8192] if split_grid else [512, graph_tokens]
    if split_grid:
        config.compilation_config.prefill_cudagraph_capture_token_sizes = [512, graph_tokens]
    assert get_pcp_profile_num_tokens(config, runner.max_num_tokens, runner.max_num_reqs) == expected


@pytest.mark.parametrize(
    "eager,mode",
    [(True, CUDAGraphMode.PIECEWISE), (False, CUDAGraphMode.NONE), (False, CUDAGraphMode.FULL_DECODE_ONLY)],
)
def test_profile_budget_ignores_inactive_prefill_graphs(eager, mode):
    runner = make_runner(pcp=8, requests=500, query=1)
    config = runner.vllm_config
    config.model_config.enforce_eager = eager
    config.compilation_config.cudagraph_mode = mode
    config.compilation_config.prefill_cudagraph_capture_token_sizes = [4096]
    assert get_pcp_profile_num_tokens(config, runner.max_num_tokens, runner.max_num_reqs) == 3486


def test_empty_prefill_grid_does_not_use_draft_graph_sizes():
    runner = make_runner(pcp=8, requests=500, query=1)
    config = runner.vllm_config
    config.model_config.enforce_eager = False
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    config.compilation_config.prefill_cudagraph_capture_token_sizes = []
    config.compilation_config.cudagraph_capture_sizes = [8192]
    assert get_pcp_profile_num_tokens(config, runner.max_num_tokens, runner.max_num_reqs) == 3486


@pytest.mark.parametrize(
    "pcp,requests,query,lens,prefills",
    [
        (8, 500, 4, [4] * 499 + [14388], [False] * 499 + [True]),
        (8, 500, 1, [15] * 500, [True] * 500),
        (16, 500, 1, [31] * 500, [True] * 500),
        (8, 2, 1, [225, 16159], [True, True]),
        (16, 2, 1, [961, 15423], [True, True]),
        (8, 2, 256, [256, 16128], [False, True]),
        (16, 2, 8, [8, 16352], [False, True]),
    ],
)
@pytest.mark.parametrize("dcp,shard_decode", [(1, False), (1, True), (8, False)])
def test_budget_covers_actual_pcp_partitions(pcp, requests, query, lens, prefills, shard_decode, dcp):
    runner = make_runner(pcp=pcp, requests=requests, query=query, dcp=dcp)
    manager = PCPManager.__new__(PCPManager)
    manager.pcp_world_size = pcp
    manager.dcp_world_size = dcp
    manager.shard_decode_requests = shard_decode
    per_rank_tokens = [
        sum(
            count
            for _, _, count in manager._iter_rank_chunks(
                rank, np.array(lens, dtype=np.int64), np.array(prefills, dtype=np.bool_)
            )
        )
        for rank in range(pcp)
    ]
    budget = get_pcp_profile_num_tokens(runner.vllm_config, runner.max_num_tokens, runner.max_num_reqs)
    assert max(per_rank_tokens) <= budget <= runner.max_num_tokens
    if requests == 2:
        assert budget < runner.max_num_tokens


def test_dcp_budget_still_covers_prefill_graph():
    runner = make_runner(pcp=8, requests=2, query=1, dcp=8)
    config = runner.vllm_config
    config.model_config.enforce_eager = False
    config.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    config.compilation_config.prefill_cudagraph_capture_token_sizes = [4096]
    assert get_pcp_profile_num_tokens(config, runner.max_num_tokens, runner.max_num_reqs) == 4096
