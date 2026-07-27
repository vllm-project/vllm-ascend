# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config.model import ModelConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID

from vllm_ascend.patch.platform.patch_pp_mtp import (
    _update_pp_mtp_spec_token_ids,
    _use_pp_ipc_runtime_patch,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def test_model_config_validates_local_mtp_drafter_as_single_pp_rank(monkeypatch):
    fake_registry = SimpleNamespace(
        is_pp_supported_model=lambda _architectures, _model_config: False,
    )
    monkeypatch.setattr(ModelConfig, "registry", property(lambda _self: fake_registry))

    model_config = ModelConfig.__new__(ModelConfig)
    model_config.hf_config = SimpleNamespace(model_type="qwen3_5_mtp")
    model_config.runner = "draft"
    model_config.model_arch_config = SimpleNamespace(
        total_num_attention_heads=1,
        architectures=["Qwen3_5MTP"],
    )
    model_config.multimodal_config = None

    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        pipeline_parallel_size=2,
        decode_context_parallel_size=1,
    )

    ModelConfig.verify_with_parallel_config(model_config, parallel_config)
    assert parallel_config.pipeline_parallel_size == 2


def test_model_config_keeps_target_model_pp_validation(monkeypatch):
    fake_registry = SimpleNamespace(
        is_pp_supported_model=lambda _architectures, _model_config: False,
    )
    monkeypatch.setattr(ModelConfig, "registry", property(lambda _self: fake_registry))

    model_config = ModelConfig.__new__(ModelConfig)
    model_config.hf_config = SimpleNamespace(model_type="qwen3_5_mtp")
    model_config.runner = "generate"
    model_config.model_arch_config = SimpleNamespace(
        total_num_attention_heads=1,
        architectures=["UnsupportedForPP"],
    )

    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        pipeline_parallel_size=2,
        decode_context_parallel_size=1,
    )

    with pytest.raises(NotImplementedError):
        ModelConfig.verify_with_parallel_config(model_config, parallel_config)


@pytest.mark.parametrize(
    (
        "use_pp",
        "speculative_config",
        "async_scheduling",
        "use_v2_model_runner",
        "expected",
    ),
    [
        (True, object(), False, False, True),
        (True, None, True, False, True),
        (True, None, False, False, True),
        (False, object(), True, False, False),
        (True, object(), True, True, False),
    ],
)
def test_pp_ipc_runtime_patch_enabled_for_all_v1_pp(
    use_pp,
    speculative_config,
    async_scheduling,
    use_v2_model_runner,
    expected,
):
    vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        speculative_config=speculative_config,
        use_v2_model_runner=use_v2_model_runner,
    )

    assert _use_pp_ipc_runtime_patch(vllm_config, use_pp) is expected


def test_pp_ipc_runtime_patch_skips_pd_prefill_node():
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=True,
            is_kv_consumer=False,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=object(),
        use_v2_model_runner=False,
    )

    assert _use_pp_ipc_runtime_patch(vllm_config, use_pp=True) is False


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_pp_ipc_cached_request_data_carries_confirmed_token_for_sync_and_async(
    async_scheduling,
):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.use_pp = True
    scheduler.use_v2_model_runner = False
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=async_scheduling)
    scheduler.vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        speculative_config=object(),
        use_v2_model_runner=False,
    )
    scheduler.prev_step_scheduled_req_ids = set()

    request = SimpleNamespace(
        request_id="req-0",
        all_token_ids=[11, 12, 13],
        num_computed_tokens=2,
        num_output_tokens=1,
        num_output_placeholders=0,
    )
    blocks = SimpleNamespace(get_block_ids=lambda allow_none: ([0],))

    cached_reqs_data = Scheduler._make_cached_request_data(
        scheduler,
        running_reqs=[request],
        resumed_reqs=[],
        num_scheduled_tokens={"req-0": 3},
        spec_decode_tokens={"req-0": [101, 102]},
        req_to_new_blocks={"req-0": blocks},
    )

    assert cached_reqs_data.req_ids == ["req-0"]
    assert cached_reqs_data.new_token_ids == [[13]]
    assert scheduler.scheduler_config.async_scheduling is async_scheduling


@pytest.mark.parametrize(
    ("async_scheduling", "expected_new_token_ids"),
    [(False, [[]]), (True, [[13]])],
)
def test_pp_ipc_cached_request_data_fills_empty_confirmed_token_only_for_async(
    async_scheduling,
    expected_new_token_ids,
):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.use_pp = True
    scheduler.use_v2_model_runner = False
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=async_scheduling)
    scheduler.vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        speculative_config=object(),
        use_v2_model_runner=False,
    )
    scheduler.prev_step_scheduled_req_ids = set()

    request = SimpleNamespace(
        request_id="req-0",
        all_token_ids=[11, 12, 13],
        num_computed_tokens=3,
        num_output_tokens=1,
        num_output_placeholders=0,
    )
    blocks = SimpleNamespace(get_block_ids=lambda allow_none: ([0],))

    cached_reqs_data = Scheduler._make_cached_request_data(
        scheduler,
        running_reqs=[request],
        resumed_reqs=[],
        num_scheduled_tokens={"req-0": 2},
        spec_decode_tokens={"req-0": [101, 102]},
        req_to_new_blocks={"req-0": blocks},
    )

    assert cached_reqs_data.new_token_ids == expected_new_token_ids
    assert scheduler.scheduler_config.async_scheduling is async_scheduling


def test_pp_ipc_sampled_token_handoff_advances_async_non_last_rank_state(
    monkeypatch,
):
    monkeypatch.setattr(
        "vllm_ascend.worker.model_runner_v1.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False),
    )

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.is_kv_producer = False
    runner.is_kv_consumer = False
    runner.use_async_scheduling = True
    runner.device = torch.device("cpu")
    runner.discard_request_mask = SimpleNamespace(
        np=np.zeros(2, dtype=bool),
    )
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["req-0", "req-1"],
        prev_sampled_token_ids=None,
        prev_req_id_to_index={},
        num_tokens_no_spec=np.array([3, 5], dtype=np.int64),
        is_token_ids=np.zeros((2, 8), dtype=bool),
    )
    runner.requests = {
        "req-0": SimpleNamespace(output_token_ids=[31]),
        "req-1": SimpleNamespace(output_token_ids=[41, 42]),
    }
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-0", "req-1"],
            new_token_ids=[[101], [202]],
            num_output_tokens=[1, 2],
        ),
    )

    runner._apply_pp_sampled_tokens_from_scheduler_output(scheduler_output)

    assert runner.input_batch.prev_req_id_to_index == {
        "req-0": 0,
        "req-1": 1,
    }
    assert runner.input_batch.prev_sampled_token_ids.tolist() == [[101], [202]]
    assert runner.requests["req-0"].output_token_ids == [
        31,
        PLACEHOLDER_TOKEN_ID,
    ]
    assert runner.requests["req-1"].output_token_ids == [
        41,
        42,
        PLACEHOLDER_TOKEN_ID,
    ]
    assert runner.input_batch.is_token_ids[0, 3]
    assert runner.input_batch.is_token_ids[1, 5]
    assert runner.input_batch.num_tokens_no_spec.tolist() == [4, 6]


def test_pp_ipc_sampled_token_handoff_keeps_sync_path_on_scheduler_tokens(
    monkeypatch,
):
    monkeypatch.setattr(
        "vllm_ascend.worker.model_runner_v1.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False),
    )

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.is_kv_producer = False
    runner.is_kv_consumer = False
    runner.use_async_scheduling = False
    runner.device = torch.device("cpu")
    runner.input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["req-0"],
        prev_sampled_token_ids="keep",
        prev_req_id_to_index={"keep": 0},
        num_tokens_no_spec=np.array([3], dtype=np.int64),
        is_token_ids=np.zeros((1, 8), dtype=bool),
    )
    runner.requests = {
        "req-0": SimpleNamespace(output_token_ids=[31]),
    }
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-0"],
            new_token_ids=[[101]],
            num_output_tokens=[1],
        ),
    )

    runner._apply_pp_sampled_tokens_from_scheduler_output(scheduler_output)

    assert runner.input_batch.prev_req_id_to_index == {"keep": 0}
    assert runner.input_batch.prev_sampled_token_ids == "keep"
    assert runner.requests["req-0"].output_token_ids == [31]
    assert not runner.input_batch.is_token_ids[0, 3]
    assert runner.input_batch.num_tokens_no_spec.tolist() == [3]


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_pp_mtp_spec_tokens_are_written_from_model_runner_output_for_sync_and_async(
    async_scheduling,
):
    request = SimpleNamespace(
        spec_token_ids=[],
        structured_output_request=None,
        is_finished=lambda: False,
    )
    scheduler = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        requests={"req-0": request},
        structured_output_manager=SimpleNamespace(
            should_advance=lambda _request: False,
        ),
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-0": 1})
    model_runner_output = SimpleNamespace(
        req_id_to_index={"req-0": 0},
        sampled_token_ids=[[200]],
        spec_token_ids=[[301, 302]],
    )

    _update_pp_mtp_spec_token_ids(
        scheduler,
        scheduler_output,
        model_runner_output,
    )

    assert request.spec_token_ids == [301, 302]
