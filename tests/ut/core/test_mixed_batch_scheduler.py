# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import VllmConfig
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs

from tests.ut.kv_offload.utils import create_model_runner_output, create_request, create_scheduler, create_vllm_config
from vllm_ascend.ascend_config import SchedulerConfig
from vllm_ascend.platform import _setup_mixed_batch_scheduler


def _make_scheduler(
    async_scheduling,
    *,
    cap=2,
    max_num_seqs=16,
    chunked_prefill=True,
    max_model_len=32,
    decode_after_prefill=False,
):
    additional_config = {
        "scheduler_config": {
            "max_num_decode_seqs_in_mixed_batch": cap,
            "decode_step_after_prefill": decode_after_prefill,
        }
    }

    def configure(**kwargs):
        kwargs["model_config"].max_model_len = max_model_len
        kwargs["scheduler_config"].max_model_len = max_model_len
        kwargs["scheduler_config"].async_scheduling = async_scheduling
        kwargs["scheduler_config"].enable_chunked_prefill = chunked_prefill
        kwargs["kv_transfer_config"] = None
        kwargs["additional_config"] = additional_config
        return VllmConfig(**kwargs)

    with patch("tests.ut.kv_offload.utils.VllmConfig", new=configure):
        config = create_vllm_config(max_num_seqs=max_num_seqs, max_num_batched_tokens=32, block_size=16)

    ascend_config = SchedulerConfig.from_additional_config(additional_config)
    _setup_mixed_batch_scheduler(config, ascend_config)
    scheduler_cls = config.scheduler_config.get_scheduler_cls()
    register_all_kvcache_specs(None)
    with (
        patch("tests.ut.kv_offload.utils.Scheduler", new=scheduler_cls),
        patch(
            "vllm_ascend.core.mixed_batch_scheduler.init_ascend_config",
            return_value=SimpleNamespace(scheduler_config=ascend_config),
        ),
    ):
        return create_scheduler(config, num_blocks=128)


def _retire(scheduler, output):
    requests = [scheduler.requests[req_id] for req_id in output.num_scheduled_tokens]
    feedback = create_model_runner_output(requests)
    for i, request in enumerate(requests):
        if request.is_prefill_chunk:
            feedback.sampled_token_ids[i] = []
    scheduler.update_from_output(output, feedback)


def _add_decodes(scheduler, count=4):
    requests = [create_request(i, num_tokens=8, max_tokens=16, block_size=16) for i in range(1, count + 1)]
    for request in requests:
        scheduler.add_request(request)
    _retire(scheduler, scheduler.schedule())
    assert all(not request.is_prefill_chunk for request in requests)
    return {request.request_id for request in requests}


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("cap", [None, 0, 2])
def test_mixed_step_caps_decodes_then_holds_new_prefill(async_scheduling, cap):
    scheduler = _make_scheduler(async_scheduling, cap=cap)
    decodes = _add_decodes(scheduler)
    prefill = create_request(5, num_tokens=8)
    scheduler.add_request(prefill)

    mixed = scheduler.schedule()
    assert prefill.request_id in mixed.num_scheduled_tokens
    scheduled_decodes = decodes & mixed.num_scheduled_tokens.keys()
    assert len(scheduled_decodes) == (4 if cap is None else cap)
    _retire(scheduler, mixed)

    next_prefill = create_request(6, num_tokens=8)
    scheduler.add_request(next_prefill)
    following = scheduler.schedule()
    assert decodes <= following.num_scheduled_tokens.keys()
    assert (next_prefill.request_id in following.num_scheduled_tokens) == (cap is None)
    _retire(scheduler, following)
    if cap is not None:
        assert next_prefill.request_id in scheduler.schedule().num_scheduled_tokens


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("cap", [None, 2])
def test_prefill_step_drains_decodes_even_without_a_deferred_row(async_scheduling, cap):
    scheduler = _make_scheduler(async_scheduling, cap=cap, decode_after_prefill=True)
    decodes = _add_decodes(scheduler, count=2)
    first = create_request(5, num_tokens=8, max_tokens=1)
    scheduler.add_request(first)
    _retire(scheduler, scheduler.schedule())  # Decode opportunity after the bootstrapping prefill.
    mixed = scheduler.schedule()
    assert set(mixed.num_scheduled_tokens) == decodes | {first.request_id}
    _retire(scheduler, mixed)

    second = create_request(6, num_tokens=8, max_tokens=1)
    scheduler.add_request(second)
    drain = scheduler.schedule()
    assert set(drain.num_scheduled_tokens) == decodes
    _retire(scheduler, drain)
    assert second.request_id in scheduler.schedule().num_scheduled_tokens


def test_async_prefill_waits_for_drained_decode_result():
    scheduler = _make_scheduler(True, decode_after_prefill=True)
    first = create_request(1, num_tokens=8, max_tokens=2)
    scheduler.add_request(first)
    prefill = scheduler.schedule()
    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {first.request_id: 1}
    _retire(scheduler, prefill)

    second = create_request(2, num_tokens=8, max_tokens=1)
    scheduler.add_request(second)
    assert scheduler.schedule().total_num_scheduled_tokens == 0

    _retire(scheduler, decode)
    assert second.request_id in scheduler.schedule().num_scheduled_tokens


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("max_num_seqs", [4, 5], ids=["full", "insufficient-budget"])
def test_decodes_advance_when_waiting_prefill_cannot_run(async_scheduling, max_num_seqs):
    scheduler = _make_scheduler(async_scheduling, max_num_seqs=max_num_seqs, chunked_prefill=False)
    decodes = _add_decodes(scheduler)
    prefill = create_request(5, num_tokens=31, max_tokens=1)
    scheduler.add_request(prefill)

    output = scheduler.schedule()
    assert prefill.request_id not in output.num_scheduled_tokens
    if max_num_seqs > len(decodes):
        assert len(output.num_scheduled_tokens) == 2
        # The upstream saturation escape must still permit the decode drain.
        assert scheduler.prefill_capacity_bound
        _retire(scheduler, output)
        output = scheduler.schedule()

    assert set(output.num_scheduled_tokens) == decodes


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_running_prefill_resumes_after_decode_drain(async_scheduling):
    scheduler = _make_scheduler(async_scheduling, max_model_len=96)
    decodes = _add_decodes(scheduler)
    prefill = create_request(5, num_tokens=64, max_tokens=8)
    scheduler.add_request(prefill)

    mixed = scheduler.schedule()
    assert prefill.is_prefill_chunk
    assert len(decodes & mixed.num_scheduled_tokens.keys()) == 2
    _retire(scheduler, mixed)
    assert not scheduler.waiting

    following = scheduler.schedule()
    assert set(following.num_scheduled_tokens) == decodes
    _retire(scheduler, following)

    resumed = scheduler.schedule()
    assert prefill.request_id in resumed.num_scheduled_tokens
    assert len(decodes & resumed.num_scheduled_tokens.keys()) == 2


@pytest.mark.parametrize("cap", [-1, True, 2.5, "2"])
def test_mixed_decode_limit_accepts_only_a_nonnegative_integer_or_null(cap):
    with pytest.raises(ValueError, match="max_num_decode_seqs_in_mixed_batch"):
        SchedulerConfig.from_additional_config({"scheduler_config": {"max_num_decode_seqs_in_mixed_batch": cap}})


def test_mixed_decode_limit_preserves_an_explicit_custom_scheduler():
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(scheduler_cls="custom.Scheduler", async_scheduling=True)
    )
    with pytest.raises(ValueError, match="requires the default scheduler"):
        _setup_mixed_batch_scheduler(vllm_config, SchedulerConfig(max_num_decode_seqs_in_mixed_batch=2))
    assert vllm_config.scheduler_config.scheduler_cls == "custom.Scheduler"
