# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.mrv2_utils import get_pcp_profile_num_tokens
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def make_runner(tokens=16384, pcp=16, requests=64, query=4, replicated=False):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.max_num_tokens = tokens
    runner.max_num_reqs = requests
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
        model_config=SimpleNamespace(enforce_eager=True),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE, cudagraph_capture_sizes=[]),
    )
    runner.speculative_config = SimpleNamespace(
        method="dspark" if replicated else "mtp",
        num_speculative_tokens=query - 1,
        max_num_new_slots_for_drafting=0,
    )
    runner.vllm_config.speculative_config = runner.speculative_config
    runner.speculator = None  # Non-last PP stages have no local drafter.
    return runner


@pytest.mark.parametrize(
    "tokens,pcp,requests,query,replicated,expected",
    [
        (16384, 1, 64, 4, False, 16384),
        (16384, 16, 64, 4, False, 1152),
        (16384, 16, 512, 4, False, 3072),
        (17, 4, 2, 1, False, 9),
        (17, 16, 64, 4, False, 17),
        (16384, 16, 64, 9, True, 16384),
    ],
)
def test_local_prefill_preserves_draft_requests(tokens, pcp, requests, query, replicated, expected):
    runner = make_runner(tokens, pcp, requests, query, replicated)
    assert get_pcp_profile_num_tokens(runner.vllm_config, runner.max_num_tokens, runner.max_num_reqs) == expected
    assert runner.max_num_tokens == tokens
    assert runner.max_num_reqs == requests


def test_no_draft_preserves_target_decode_requests():
    runner = make_runner(requests=2048)
    runner.speculative_config = None
    runner.vllm_config.speculative_config = None
    assert get_pcp_profile_num_tokens(runner.vllm_config, runner.max_num_tokens, runner.max_num_reqs) == 6144


def test_non_model_draft_keeps_global_budget():
    runner = make_runner()
    runner.speculative_config.method = "ngram"
    assert get_pcp_profile_num_tokens(runner.vllm_config, runner.max_num_tokens, runner.max_num_reqs) == 16384


@pytest.mark.parametrize("method", ["dflash", "dspark", "draft_model"])
def test_replicated_draft_context_is_not_reduced(method):
    runner = make_runner()
    runner.speculative_config.method = method
    assert (
        get_pcp_profile_num_tokens(runner.vllm_config, runner.max_num_tokens, runner.max_num_reqs)
        == runner.max_num_tokens
    )


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
@pytest.mark.parametrize("replicated,expected", [(False, 3072), (True, 16384)])
def test_profile_temporarily_sets_budget_and_restores_on_failure(fail, replicated, expected):
    runner = make_runner(requests=512, replicated=replicated)
    runner.ascend_config = SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    runner._dummy_run = Mock()

    def profile(self, **kwargs):
        assert kwargs == {"randomize_inputs": True}
        assert self.max_num_tokens == expected
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
@pytest.mark.parametrize("graph_tokens,expected", [(512, 3048), (4096, 4096)])
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
    assert get_pcp_profile_num_tokens(config, runner.max_num_tokens, runner.max_num_reqs) == 3048


def test_empty_prefill_grid_does_not_use_draft_graph_sizes():
    runner = make_runner(pcp=8, requests=500, query=1)
    config = runner.vllm_config
    config.model_config.enforce_eager = False
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    config.compilation_config.prefill_cudagraph_capture_token_sizes = []
    config.compilation_config.cudagraph_capture_sizes = [8192]
    assert get_pcp_profile_num_tokens(config, runner.max_num_tokens, runner.max_num_reqs) == 3048
