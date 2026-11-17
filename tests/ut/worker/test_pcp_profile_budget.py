# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def make_runner(tokens=16384, pcp=16, requests=64, query=4, replicated=False):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.max_num_tokens = tokens
    runner.max_num_reqs = requests
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp))
    runner.speculative_config = SimpleNamespace(
        method="dspark" if replicated else "mtp",
        num_speculative_tokens=query - 1,
        max_num_new_slots_for_drafting=0,
    )
    runner.speculator = None  # Non-last PP stages have no local drafter.
    return runner


@pytest.mark.parametrize(
    "tokens,pcp,requests,query,replicated,expected",
    [
        (16384, 1, 64, 4, False, 16384),
        (16384, 16, 64, 4, False, 1024),
        (16384, 16, 512, 4, False, 2048),
        (17, 4, 2, 1, False, 5),
        (17, 16, 64, 4, False, 17),
        (16384, 16, 64, 9, True, 16384),
    ],
)
def test_local_prefill_preserves_draft_requests(tokens, pcp, requests, query, replicated, expected):
    runner = make_runner(tokens, pcp, requests, query, replicated)
    assert runner._get_pcp_profile_num_tokens() == expected
    assert runner.max_num_tokens == tokens
    assert runner.max_num_reqs == requests


def test_no_draft_preserves_target_decode_requests():
    runner = make_runner(requests=2048)
    runner.speculative_config = None
    assert runner._get_pcp_profile_num_tokens() == 2048


def test_non_model_draft_keeps_global_budget():
    runner = make_runner()
    runner.speculative_config.method = "ngram"
    assert runner._get_pcp_profile_num_tokens() == 16384


@pytest.mark.parametrize("method", ["dflash", "dspark", "draft_model"])
def test_replicated_draft_context_is_not_reduced(method):
    runner = make_runner()
    runner.speculative_config.method = method
    assert runner._get_pcp_profile_num_tokens() == runner.max_num_tokens


@pytest.mark.parametrize("tokens,requests", [(16384, 64), (16384, 512), (17, 64)])
def test_local_dummy_preserves_actual_upstream_request_count(tokens, requests):
    runner = make_runner(tokens=tokens, requests=requests)
    local_tokens = runner._get_pcp_profile_num_tokens()
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
@pytest.mark.parametrize("replicated,expected", [(False, 2048), (True, 16384)])
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
