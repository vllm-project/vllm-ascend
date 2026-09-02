from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.worker import model_runner_v1
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@contextmanager
def _record_forward_context(trace, *args, **kwargs):
    trace.executors.append(kwargs["device_metadata_executor"])
    trace.events.append("context-enter")
    try:
        yield
    finally:
        trace.events.append("context-exit")


def _make_executor(active: bool, trace, uses_external_events: bool = False):
    return SimpleNamespace(
        submission_in_flight=active,
        uses_external_events=uses_external_events,
        release=MagicMock(side_effect=lambda: trace.events.append("release")),
    )


def _make_execute_runner(executor, trace, forward_error=None):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    model_config = SimpleNamespace(
        enable_return_routed_experts=False,
        enforce_eager=False,
        is_encoder_decoder=False,
        use_mla=True,
    )
    runner.vllm_config = SimpleNamespace(model_config=model_config)
    runner.model_config = model_config
    runner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(need_timing=False))
    )
    runner.execute_model_state = None
    runner.speculative_config = None
    runner.use_async_scheduling = False
    runner.num_spec_tokens = 0
    runner._draft_token_ids = None
    runner.parallel_config = SimpleNamespace(
        distributed_executor_backend=None,
        data_parallel_size=1,
        enable_dbo=False,
        num_ubatches=1,
    )
    runner.cache_config = SimpleNamespace(
        kv_sharing_fast_prefill=False,
        mamba_cache_mode="none",
    )
    runner.input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["req"],
        prev_req_id_to_index=None,
        num_computed_tokens_cpu=np.zeros(1, dtype=np.int32),
    )
    runner.requests = {}
    runner.synchronize_input_prep = nullcontext
    runner._update_states = MagicMock(return_value=None)
    runner._start_dump_data = MagicMock()
    runner._prepare_inputs = MagicMock(return_value=(slice(None), None, 1))
    runner.cascade_attn_enabled = False
    batch_desc = SimpleNamespace(num_tokens=1, num_reqs=1)
    runner._determine_batch_execution_and_padding = MagicMock(
        return_value=(CUDAGraphMode.NONE, batch_desc, False, None, None)
    )
    runner.dynamic_eplb = False
    runner.use_compress = False
    runner.dcp_size = 1
    runner._build_attention_metadata = MagicMock(return_value=("metadata", None))
    runner._sanitize_placeholder_input_ids_for_forward = MagicMock()
    runner._preprocess = MagicMock(
        return_value=(torch.zeros(1, dtype=torch.int64), None, torch.zeros(1), None, {}, None)
    )
    runner.calculate_kv_scales = False
    runner.broadcast_pp_output = False
    runner.device_metadata_executor = executor
    runner._has_sinks = False
    runner.eplb_heat_collection_status = False
    runner.maybe_get_kv_connector_output = MagicMock(return_value=nullcontext(SimpleNamespace()))
    runner.model = MagicMock()

    def model_forward(*args, **kwargs):
        trace.events.append("model-forward")
        if forward_error is not None:
            raise forward_error
        return torch.zeros((1, 1))

    runner._model_forward = MagicMock(side_effect=model_forward)
    runner.use_aux_hidden_state_outputs = False
    runner.is_pooling_model = True
    runner._pool = MagicMock(return_value=SimpleNamespace(kv_connector_output=None))
    runner._finalize_dump_data = MagicMock()
    return runner


def _make_dummy_runner(executor, trace, mode, forward_error=None):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.uniform_decode_query_len = 1
    runner.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=4,
        max_num_seqs=4,
    )
    runner.max_num_tokens = 4
    runner.dynamic_eplb = False
    batch_desc = SimpleNamespace(num_tokens=1, num_reqs=1)
    runner._determine_batch_execution_and_padding = MagicMock(return_value=(mode, batch_desc, False, None, None))
    runner.dcp_size = 1
    runner.speculative_config = None
    runner.synchronize_input_prep = nullcontext
    runner.optimistic_seq_lens_cpu = torch.zeros(4, dtype=torch.int32)
    runner.seq_lens = MagicMock()
    runner.query_pos = SimpleNamespace(np=np.zeros(1, dtype=np.int32))
    runner.query_start_loc = SimpleNamespace(
        np=np.zeros(5, dtype=np.int32),
        copy_to_gpu=MagicMock(),
    )
    runner._get_cumsum_and_arange = MagicMock(return_value=np.ones(1, dtype=np.int32))
    runner._has_gdn = False
    runner._pad_query_start_loc_for_fia = MagicMock(return_value=1)
    runner.input_batch = SimpleNamespace(block_table=SimpleNamespace(commit_block_table=MagicMock()))
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])

    def build_metadata(*args, **kwargs):
        trace.events.append("metadata-build")
        executor.submission_in_flight = True
        executor.uses_external_events = kwargs["batch_descriptor"] is not None
        return "metadata", None

    runner._build_attention_metadata = MagicMock(side_effect=build_metadata)
    runner.maybe_dummy_run_with_lora = MagicMock(return_value=nullcontext())
    runner.lora_config = None
    runner.supports_mm_inputs = False
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.enable_prompt_embeds = False
    runner.input_ids = SimpleNamespace(gpu=torch.zeros(1, dtype=torch.int64))
    runner.positions = torch.zeros(1)
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.model = MagicMock()
    runner.vllm_config = SimpleNamespace(model_config=runner.model_config)
    runner.device_metadata_executor = executor
    runner._has_sinks = False
    runner.eplb_heat_collection_status = False

    def model_forward(*args, **kwargs):
        trace.events.append("model-forward")
        if forward_error is not None:
            raise forward_error
        return torch.zeros((1, 1))

    runner._model_forward = MagicMock(side_effect=model_forward)
    runner.use_aux_hidden_state_outputs = False
    runner.drafter = None
    runner.use_compress = False
    runner._finalize_dump_data = MagicMock()
    return runner


@pytest.mark.parametrize(
    ("mode", "uses_external_events", "should_raise"),
    [
        (CUDAGraphMode.FULL, False, True),
        (CUDAGraphMode.FULL, True, False),
        (CUDAGraphMode.PIECEWISE, False, False),
        (CUDAGraphMode.NONE, False, False),
    ],
)
def test_full_mode_requires_external_events(mode, uses_external_events, should_raise):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    executor = SimpleNamespace(
        submission_in_flight=True,
        uses_external_events=uses_external_events,
    )
    runner.device_metadata_executor = executor

    if should_raise:
        with pytest.raises(RuntimeError, match="requires external events"):
            runner._prepare_device_metadata_for_forward(mode)
    else:
        assert runner._prepare_device_metadata_for_forward(mode) is executor


def test_inactive_executor_is_not_forwarded():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device_metadata_executor = SimpleNamespace(
        submission_in_flight=False,
        uses_external_events=False,
    )

    assert runner._prepare_device_metadata_for_forward(CUDAGraphMode.FULL) is None


@pytest.fixture
def forward_patches(monkeypatch):
    trace = SimpleNamespace(executors=[], events=[])
    pp_group = SimpleNamespace(world_size=1, is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(model_runner_v1, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(model_runner_v1, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(model_runner_v1, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(model_runner_v1, "enable_sp", lambda *args: False)
    monkeypatch.setattr(
        model_runner_v1,
        "maybe_create_ubatch_slices",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(model_runner_v1, "update_cos_sin", lambda *args: None)
    monkeypatch.setattr(model_runner_v1, "using_paged_attention", lambda *args: False)
    monkeypatch.setattr(model_runner_v1, "lmhead_tp_enable", lambda: False)
    monkeypatch.setattr(
        model_runner_v1,
        "record_function_or_nullcontext",
        lambda *args: nullcontext(),
    )
    monkeypatch.setattr(
        model_runner_v1,
        "set_ascend_forward_context",
        lambda *args, **kwargs: _record_forward_context(trace, *args, **kwargs),
    )
    return trace


@pytest.mark.parametrize("active", [True, False])
def test_execute_model_releases_only_active_submission(forward_patches, active):
    executor = _make_executor(active, forward_patches)
    runner = _make_execute_runner(executor, forward_patches)
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1,
        num_scheduled_tokens={"req": 1},
        scheduled_cached_reqs=SimpleNamespace(new_token_ids=[]),
        scheduled_spec_decode_tokens=[],
        scheduled_encoder_inputs=[],
        num_common_prefix_blocks=0,
    )

    runner.execute_model(scheduler_output)

    assert forward_patches.executors == [executor if active else None]
    assert forward_patches.events == [
        "context-enter",
        "model-forward",
        "context-exit",
        *(["release"] if active else []),
    ]
    assert executor.release.call_count == int(active)


def test_execute_model_does_not_release_failed_forward(forward_patches):
    executor = _make_executor(True, forward_patches)
    runner = _make_execute_runner(executor, forward_patches, RuntimeError("forward failed"))
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1,
        num_scheduled_tokens={"req": 1},
        scheduled_cached_reqs=SimpleNamespace(new_token_ids=[]),
        scheduled_spec_decode_tokens=[],
        scheduled_encoder_inputs=[],
        num_common_prefix_blocks=0,
    )

    with pytest.raises(RuntimeError, match="forward failed"):
        runner.execute_model(scheduler_output)

    assert forward_patches.executors == [executor]
    assert forward_patches.events == [
        "context-enter",
        "model-forward",
        "context-exit",
    ]
    executor.release.assert_not_called()


@pytest.mark.parametrize(
    ("is_profile", "mode", "is_graph_capturing", "expected_events"),
    [
        (
            False,
            CUDAGraphMode.FULL,
            True,
            ["metadata-build", "context-enter", "model-forward", "context-exit", "release"],
        ),
        (True, CUDAGraphMode.NONE, False, ["context-enter", "model-forward", "context-exit"]),
        (False, CUDAGraphMode.PIECEWISE, True, ["context-enter", "model-forward", "context-exit"]),
    ],
)
def test_dummy_run_releases_only_active_submission(
    forward_patches,
    is_profile,
    mode,
    is_graph_capturing,
    expected_events,
):
    executor = _make_executor(False, forward_patches)
    runner = _make_dummy_runner(executor, forward_patches, mode)

    runner._dummy_run(
        1,
        cudagraph_runtime_mode=mode,
        is_profile=is_profile,
        is_graph_capturing=is_graph_capturing,
    )

    active = mode == CUDAGraphMode.FULL
    assert forward_patches.executors == [executor if active else None]
    assert forward_patches.events == expected_events
    assert executor.release.call_count == int(active)


def test_dummy_run_does_not_release_failed_forward(forward_patches):
    executor = _make_executor(False, forward_patches)
    runner = _make_dummy_runner(
        executor,
        forward_patches,
        CUDAGraphMode.FULL,
        RuntimeError("forward failed"),
    )

    with pytest.raises(RuntimeError, match="forward failed"):
        runner._dummy_run(
            1,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            is_graph_capturing=True,
        )

    assert forward_patches.executors == [executor]
    assert forward_patches.events == [
        "metadata-build",
        "context-enter",
        "model-forward",
        "context-exit",
    ]
    executor.release.assert_not_called()
