# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.worker.v2.profile import _profile_draft, _profile_target, profile_pcp


def make_runner():
    return SimpleNamespace(
        max_num_tokens=16384,
        max_num_reqs=64,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=16),
            aux_output_config=SimpleNamespace(enabled=False),
        ),
        supports_mm_inputs=False,
        is_first_pp_rank=True,
        is_last_pp_rank=True,
        pooling_runner=None,
        model=object(),
        speculator=None,
        _dummy_sampler_run=Mock(),
        reset_encoder_cache=Mock(),
        _finalize_dump_data=Mock(),
    )


@pytest.mark.parametrize("tokens,pcp,expected", [(16384, 16, 1024), (17, 4, 5), (8, 16, 1)])
def test_target_and_draft_profile_budgets_are_independent(tokens, pcp, expected):
    runner = make_runner()
    runner.max_num_tokens = tokens
    runner.vllm_config.parallel_config.prefill_context_parallel_size = pcp
    runner.speculator = object()
    state = SimpleNamespace(hidden_states=torch.zeros(1, 8), aux_hidden_states=[torch.zeros(1, 8)])
    with (
        patch("vllm_ascend.worker.v2.profile._profile_target", return_value=state) as target,
        patch("vllm_ascend.worker.v2.profile._profile_draft") as draft,
        patch("vllm_ascend.worker.v2.profile.upstream.get_drafter_hidden_states", side_effect=lambda _, h: h),
        patch("torch.accelerator.synchronize"),
    ):
        profile_pcp(runner, randomize_inputs=True)
    target.assert_called_once_with(runner, expected, True)
    args = draft.call_args.args
    assert args[1:3] == (tokens, expected)
    assert args[3].shape == (tokens, 8)
    assert args[4][0].shape == (tokens, 8)
    assert runner._dummy_sampler_run.call_args.args[0].shape[0] == min(tokens, 64)
    assert runner.max_num_tokens == tokens
    assert runner.max_num_reqs == 64


@pytest.mark.parametrize("sharded,query,context", [(True, 1, 8), (False, 8, 128), (False, 9, 128)])
def test_mtp_local_prefill_and_block_draft_global_context(sharded, query, context):
    spec = SimpleNamespace(
        num_query_per_req=query,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=16 if sharded else 1)
        ),
        supports_mm_inputs=False,
        propose=Mock(),
    )
    runner = SimpleNamespace(
        speculator=spec,
        max_num_reqs=4,
        input_buffers=object(),
        device=torch.device("cpu"),
        sampler=SimpleNamespace(
            sampling_states=SimpleNamespace(temperature=SimpleNamespace(gpu=None), seeds=SimpleNamespace(gpu=None))
        ),
        req_states=SimpleNamespace(last_sampled_tokens=None, next_prefill_tokens=None),
        _draft_workspace_lane=object(),
    )
    with (
        patch(
            "vllm_ascend.worker.v2.profile.InputBatch.make_dummy",
            side_effect=lambda b, t, *_args, **_kwargs: SimpleNamespace(num_reqs=b, num_tokens=t, idx_mapping=None),
        ) as batch,
        patch("vllm_ascend.worker.v2.profile.upstream.use_workspace_lane", return_value=nullcontext()),
    ):
        _profile_draft(runner, 128, 8, torch.zeros(128, 16), None)
    assert batch.call_args.args[:2] == (4, context)
    kwargs = spec.propose.call_args.kwargs
    assert kwargs["last_hidden_states"].shape == (context, 16)
    assert kwargs["num_sampled"].shape == (4,)
    assert kwargs["dp_sync_state"] is None
    assert kwargs["skip_attn_for_dummy_run"] is True


@pytest.mark.parametrize("fail", [False, True])
def test_target_only_forward_restores_connector_and_does_not_draft(fail):
    runner = make_runner()
    runner.speculator = Mock()
    runner.lora_config = None
    runner.kv_connector = Mock()
    runner.maybe_dummy_run_with_lora = Mock(return_value=nullcontext())
    state = object()
    runner.execute_model_state = state

    def execute(output, **kwargs):
        assert output.total_num_scheduled_tokens == 17
        assert sum(output.num_scheduled_tokens.values()) == 17
        assert kwargs["skip_attn_for_dummy_run"] is True
        if fail:
            raise RuntimeError("target failed")

    runner.execute_model = execute
    if fail:
        with pytest.raises(RuntimeError, match="target failed"):
            _profile_target(runner, 17, False)
    else:
        assert _profile_target(runner, 17, False) is state
        assert runner.execute_model_state is None
    assert [c.args for c in runner.kv_connector.set_disabled.call_args_list] == [(True,), (False,)]
    runner.speculator.propose.assert_not_called()
