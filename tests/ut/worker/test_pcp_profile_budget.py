# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def make_runner(tokens=16384, pcp=16, requests=64, query=4, replicated=False):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.max_num_tokens = tokens
    runner.max_num_reqs = requests
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp))
    runner.speculative_config = SimpleNamespace(
        method="dspark" if replicated else "mtp", num_speculative_tokens=query - 1
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
def test_startup_dummy_keeps_global_limits_on_success_and_failure(fail):
    runner = make_runner(requests=512)
    runner.adaptive_verification = None
    runner.input_buffers = SimpleNamespace(dummy_num_tokens=None)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    runner.eplb = SimpleNamespace(suppress=nullcontext, step=Mock())
    runner._finalize_dump_data = Mock()
    seen = []

    def parent(self, num_tokens, *args, **kwargs):
        seen.append(num_tokens)
        assert self.max_num_tokens == 16384
        assert self.max_num_reqs == 512
        if fail:
            raise RuntimeError("profile failed")
        return num_tokens

    with patch.object(GPUModelRunner, "_dummy_run", parent):
        if fail:
            with pytest.raises(RuntimeError, match="profile failed"):
                runner._dummy_run(16384, skip_attn=True, is_profile=True, skip_eplb=True)
            assert seen == [2048]
        else:
            assert runner._dummy_run(16384, skip_attn=True, is_profile=True, skip_eplb=True) == 2048
            # Independent MC2 reservation and graph dummies retain their shape.
            assert runner._dummy_run(4096, skip_attn=True, is_profile=True, skip_eplb=True) == 4096
            assert runner._dummy_run(16384, is_profile=True, skip_eplb=True) == 16384
            assert runner._dummy_run(16384, skip_attn=True, skip_eplb=True) == 16384
    assert runner.max_num_tokens == 16384
    assert runner.max_num_reqs == 512
